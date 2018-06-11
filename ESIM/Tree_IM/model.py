import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import gc
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def ortho_weight(ndim):
	"""
	Random orthogonal weights
	Used by norm_weights(below), in which case, we
	are ensuring that the rows are orthogonal
	(i.e W = U \Sigma V, U has the same
	# of rows, V has the same # of cols)
	"""
	W = np.random.randn(ndim, ndim)
	u, s, v = np.linalg.svd(W)
	return u.astype('float32')

# mapping from scalar to vector
def map_label_to_target(label,num_classes):
	target = torch.Tensor(1,num_classes)
	ceil = int(math.ceil(label))
	floor = int(math.floor(label))
	if ceil==floor:
		target[0][floor-1] = 1
	else:
		target[0][floor-1] = ceil - label
		target[0][ceil-1] = label - floor
	return target

def map_label_to_target_sentiment(label, num_classes = 0 ,fine_grain = False):
	# num_classes not use yet
	target = torch.LongTensor(1)
	target[0] = int(label) # nothing to do here as we preprocess data
	return target

class BinaryTreeLeafModule(nn.Module):

	def __init__(self, cuda, in_dim, mem_dim):
		super(BinaryTreeLeafModule, self).__init__()
		self.cudaFlag = cuda
		self.in_dim = in_dim
		self.mem_dim = mem_dim

		self.cx = nn.Linear(self.in_dim, self.mem_dim)
		self.ox = nn.Linear(self.in_dim, self.mem_dim)
		if self.cudaFlag:
			self.cx = self.cx.cuda()
			self.ox = self.ox.cuda()

	def forward(self, input):
		c = self.cx(input)
		o = F.sigmoid(self.ox(input))
		h = o * F.tanh(c)
		return c, h

class BinaryTreeComposer(nn.Module):

	def __init__(self, cuda, in_dim, mem_dim):
		super(BinaryTreeComposer, self).__init__()
		self.cudaFlag = cuda
		self.in_dim = in_dim
		self.mem_dim = mem_dim

		def new_gate():
			lh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
			rh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
			lh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
			rh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
			return lh, rh

		def new_W():
			w = nn.Linear(self.in_dim, self.mem_dim)
			w.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
			return w

		self.ilh, self.irh = new_gate()
		self.lflh, self.lfrh = new_gate()
		self.rflh, self.rfrh = new_gate()
		self.ulh, self.urh = new_gate()
		self.olh, self.orh = new_gate()

		self.cx = new_W()
		self.ox = new_W()
		self.fx = new_W()
		self.ix = new_W()

		if self.cudaFlag:
			self.ilh = self.ilh.cuda()
			self.irh = self.irh.cuda()
			self.lflh = self.lflh.cuda()
			self.lfrh = self.lfrh.cuda()
			self.rflh = self.rflh.cuda()
			self.rfrh = self.rfrh.cuda()
			self.ulh = self.ulh.cuda()
			self.urh = self.urh.cuda()
			self.olh = self.olh.cuda()
			self.orh = self.orh.cuda()

	def forward(self, input, lc, lh , rc, rh):
		u = F.tanh(self.cx(input) + self.ulh(lh) + self.urh(rh))
		i = F.sigmoid(self.ix(input) + self.ilh(lh) + self.irh(rh))
		lf = F.sigmoid(self.fx(input) + self.lflh(lh) + self.lfrh(rh))
		rf = F.sigmoid(self.fx(input) + self.rflh(lh) + self.rfrh(rh))
		c =  i* u + lf*lc + rf*rc
		o = F.sigmoid(self.ox(input) + self.olh(lh) + self.orh(rh))
		h = o * F.tanh(c)
		return c, h

class BinaryTreeLSTM(nn.Module):
	def __init__(self, cuda, in_dim, mem_dim, word_embedding, num_words):
		super(BinaryTreeLSTM, self).__init__()
		self.cudaFlag = cuda
		self.in_dim = in_dim
		self.mem_dim = mem_dim
		self.word_embedding=word_embedding
		self.num_words=num_words

		#self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
		self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
		self.output_module = None
		self.all_ststes=[]
		self.all_words=[]

	def set_output_module(self, output_module):
		self.output_module = output_module

	def getParameters(self):
		"""
		Get flatParameters
		note that getParameters and parameters is not equal in this case
		getParameters do not get parameters of output module
		:return: 1d tensor
		"""
		params = []
		for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh, self.ux, self.uh]:
			# we do not get param of output module
			l = list(m.parameters())
			params.extend(l)

		one_dim = [p.view(p.numel()) for p in params]
		params = F.torch.cat(one_dim)
		return params

	def forward(self, tree, embs, PAD):

		if tree.num_children == 0:
			lc = Variable(torch.zeros(1, self.mem_dim))
			lh = Variable(torch.zeros(1, self.mem_dim))
			rc = Variable(torch.zeros(1, self.mem_dim))
			rh = Variable(torch.zeros(1, self.mem_dim))
			if torch.cuda.is_available():
				lc = lc.cuda()
				lh = lh.cuda()
				rc = rc.cuda()
				rh = rh.cuda()
			tree.state = self.composer.forward(embs[tree.idx-1], lc, lh, rc, rh)
			self.all_ststes.append(tree.state[1].view(1, self.mem_dim))
			#self.all_words.append(embs[tree.idx-1])
		else:
			for idx in xrange(tree.num_children):
				_ = self.forward(tree.children[idx], embs, PAD)

			lc, lh, rc, rh = self.get_child_state(tree)
			if PAD:
				index = Variable(torch.LongTensor([self.num_words-1]))
				if torch.cuda.is_available():
					index=index.cuda()
				tree.state = self.composer.forward(self.word_embedding(index),lc, lh, rc, rh)
			else:
				tree.state = self.composer.forward(embs[tree.idx - 1], lc, lh, rc, rh)
			self.all_ststes.append(tree.state[1].view(1,self.mem_dim))
			#self.all_words.append(self.word_embedding[self.num_words-1])

		return tree.state#, loss

	def get_child_state(self, tree):
		lc, lh = tree.children[0].state
		rc, rh = tree.children[1].state
		return lc, lh, rc, rh

class ESIM(nn.Module):
	"""
		Implementation of the multi feed forward network model described in
		the paper "A Decomposable Attention Model for Natural Language
		Inference" by Parikh et al., 2016.

		It applies feedforward MLPs to combinations of parts of the two sentences,
		without any recurrent structure.
	"""
	def __init__(self, num_units, num_classes, vocab_size, embedding_size, pretrained_emb, num_words):
		super(ESIM, self).__init__()
		self.vocab_size=vocab_size
		self.num_units = num_units
		self.num_classes = num_classes
		self.embedding_size=embedding_size
		self.pretrained_emb=pretrained_emb

		self.dropout = nn.Dropout(p=0.5)
		self.word_embedding=nn.Embedding(vocab_size,embedding_size)

		self.tree_lstm_intra=BinaryTreeLSTM(torch.cuda.is_available(),embedding_size,num_units, self.word_embedding, num_words)

		self.linear_layer_compare = nn.Sequential(nn.Linear(4*num_units, num_units), nn.ReLU(), nn.Dropout(p=0.5))
		#                                          nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU())

		self.tree_lstm_compare=BinaryTreeLSTM(torch.cuda.is_available(),embedding_size,num_units, self.word_embedding, num_words)

		self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(4*num_units, num_units), nn.ReLU(),
													nn.Dropout(p=0.5), nn.Linear(num_units, num_classes))

		self.init_weight()

	def init_weight(self):
		#nn.init.normal(self.linear_layer_project,mean=0,std=0.1)
		#print(self.linear_layer_attend[3])
		#self.linear_layer_attend[1].weight.data.normal_(0, 0.01)
		#self.linear_layer_attend[1].bias.data.fill_(0)
		#self.linear_layer_attend[4].weight.data.normal_(0, 0.01)
		#self.linear_layer_attend[4].bias.data.fill_(0)
		self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
		self.linear_layer_compare[0].bias.data.fill_(0)
		#self.linear_layer_compare[4].weight.data.normal_(0, 0.01)
		#self.linear_layer_compare[4].bias.data.fill_(0)
		self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
		self.linear_layer_aggregate[1].bias.data.fill_(0)
		self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
		self.linear_layer_aggregate[4].bias.data.fill_(0)
		self.word_embedding.weight.data.copy_(torch.from_numpy(self.pretrained_emb))

	def attention_softmax3d(self,raw_attentions):
		reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
		out=nn.functional.softmax(reshaped_attentions, dim=1)
		return out.view(raw_attentions.size(0),raw_attentions.size(1),raw_attentions.size(2))

	def _transformation_input(self,embed_sent, tree, PAD=True):

		embed_sent = self.word_embedding(embed_sent)
		embed_sent = self.dropout(embed_sent)
		#print('intra:')
		#print(embed_sent)
		_=self.tree_lstm_intra(tree, embed_sent, PAD)
		#print(len(self.tree_lstm_intra.all_ststes))
		output=torch.cat(self.tree_lstm_intra.all_ststes,0)
		#embed_sent=torch.cat(self.tree_lstm_intra.all_words,0)
		del self.tree_lstm_intra.all_ststes[:]
		#del self.tree_lstm_intra.all_words[:]
		#gc.collect()
		return output

	def attend(self,sent1,sent2):

		repr2=torch.transpose(sent2,1,2)
		self.raw_attentions = torch.matmul(sent1, repr2)
		att_sent1 = self.attention_softmax3d(self.raw_attentions)
		beta = torch.matmul(att_sent1, sent2)

		raw_attentions_t = torch.transpose(self.raw_attentions, 1, 2).contiguous()
		att_sent2 = self.attention_softmax3d(raw_attentions_t)
		alpha = torch.matmul(att_sent2, sent1)

		return alpha, beta

	def compare(self,sentence,soft_alignment, tree, PAD=False):

		sent_alignment=torch.cat([sentence, soft_alignment, sentence-soft_alignment, sentence * soft_alignment],2)
		sent_alignment = self.linear_layer_compare(sent_alignment)
		sent_alignment = self.dropout(sent_alignment)

		#print('compare:')
		#print(sent_alignment)
		sent_alignment=sent_alignment[0]
		_=self.tree_lstm_compare(tree, sent_alignment,PAD)
		output = torch.cat(self.tree_lstm_compare.all_ststes, 0)
		del self.tree_lstm_compare.all_ststes[:]
		#gc.collect()
		return output

	def aggregate(self,v1, v2):

		v1_mean = torch.mean(v1, 1)
		v2_mean = torch.mean(v2, 1)
		v1_max, _ = torch.max(v1, 1)
		v2_max, _ = torch.max(v2, 1)
		out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max, v2_mean, v2_max), 1))

		return out

	def forward(self,sent1, sent2,tree1, tree2):
		sent1=self._transformation_input(sent1, tree1)
		sent2=self._transformation_input(sent2, tree2)
		sent1=torch.unsqueeze(sent1,0)
		sent2=torch.unsqueeze(sent2,0)
		alpha, beta = self.attend(sent1, sent2)
		v1=self.compare(sent1,beta, tree1)
		v2=self.compare(sent2,alpha, tree2)
		v1 = torch.unsqueeze(v1, 0)
		v2 = torch.unsqueeze(v2, 0)
		logits=self.aggregate(v1,v2)
		return logits