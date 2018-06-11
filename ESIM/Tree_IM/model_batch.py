import sys
import math
import torch
import numpy as np
import torch.nn as nn
from util import *
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
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

def norm_weight(nin, nout=None, scale=0.01, ortho=True):
	"""
	Random weights drawn from a Gaussian
	"""
	if nout is None:
		nout = nin
	if nout == nin and ortho:
		W = ortho_weight(nin)
	else:
		W = scale * np.random.randn(nin, nout)
	return W.astype('float32')

def generate_mask_2(values, sent_sizes):
	mask_matrix = np.zeros((len(sent_sizes), max(sent_sizes), values.size(2)))
	for i in range(len(sent_sizes)):
		mask_matrix[i][:sent_sizes[i]][:]=1
	if torch.cuda.is_available():
		mask_matrix = torch.Tensor(mask_matrix).cuda()
	else:
		mask_matrix = torch.Tensor(mask_matrix)
	return values*Variable(mask_matrix)

def generate_mask(lsent_sizes, rsent_sizes):
	mask_matrix=np.zeros((len(lsent_sizes),max(lsent_sizes),max(rsent_sizes)))
	for i in range(len(lsent_sizes)):
		mask_matrix[i][:lsent_sizes[i]][:rsent_sizes[i]]=1
	if torch.cuda.is_available():
		mask_matrix = torch.Tensor(mask_matrix).cuda()
	else:
		mask_matrix = torch.Tensor(mask_matrix)
	return Variable(mask_matrix)

class BinaryTreeCell(nn.Module):

	def __init__(self, cuda, in_dim, mem_dim):
		super(BinaryTreeCell, self).__init__()
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
	def __init__(self, cuda, in_dim, mem_dim):
		super(BinaryTreeLSTM, self).__init__()
		self.cudaFlag = cuda
		self.in_dim = in_dim
		self.mem_dim = mem_dim

		#self.leaf_module = BinaryTreeLeafModule(cuda,in_dim, mem_dim)
		self.TreeCell = BinaryTreeCell(cuda, in_dim, mem_dim)
		self.output_module = None
		self.all_ststes=[]
		self.all_words=[]

	def forward(self, x, x_mask, x_left_mask, x_right_mask):
		"""

		:param x: #step x #sample x dim_emb
		:param x_mask: #step x #sample
		:param x_left_mask: #step x #sample x #step
		:param x_right_mask: #step x #sample x #step
		:return:
		"""
		h = Variable(torch.zeros(x.size(1), x.size(0), x.size(2)))
		c = Variable(torch.zeros(x.size(1), x.size(0), x.size(2)))
		if torch.cuda.is_available():
			h=h.cuda()
			c=c.cuda()
		for step in range(x.size(0)):
			input=x[step] # #sample x dim_emb
			lh=torch.sum(x_left_mask[step][:,:,None]*h,1)
			rh=torch.sum(x_right_mask[step][:,:,None]*h,1)
			lc=torch.sum(x_left_mask[step][:,:,None]*c,1)
			rc=torch.sum(x_right_mask[step][:,:,None]*c,1)
			step_c, step_h=self.TreeCell(input, lc, lh , rc, rh)
			if step==0:
				new_h = torch.cat((torch.unsqueeze(step_h, 1), h[:,step + 1:, :]), 1)
				new_c = torch.cat((torch.unsqueeze(step_c, 1), c[:,step + 1:, :]), 1)
			elif step==(x.size(0)-1):
				new_h = torch.cat((h[:, :step, :], torch.unsqueeze(step_h, 1)), 1)
				new_c = torch.cat((c[:, :step, :], torch.unsqueeze(step_c, 1)), 1)
			else:
				new_h=torch.cat((h[:,:step,:], torch.unsqueeze(step_h,1), h[:,step+1:,:]),1)
				new_c=torch.cat((c[:,:step,:], torch.unsqueeze(step_c,1), c[:,step+1:,:]),1)
			h=x_mask[step][:,None,None]*new_h + (1-x_mask[step][:,None,None])*h
			c=x_mask[step][:,None,None]*new_c + (1-x_mask[step][:,None,None])*c
		return h

class ESIM(nn.Module):
	"""
		Implementation of the multi feed forward network model described in
		the paper "A Decomposable Attention Model for Natural Language
		Inference" by Parikh et al., 2016.

		It applies feedforward MLPs to combinations of parts of the two sentences,
		without any recurrent structure.
	"""
	def __init__(self, num_units, num_classes, vocab_size, embedding_size, pretrained_emb,
				 training=True, project_input=True,
				 use_intra_attention=False, distance_biases=10, max_sentence_length=30):
		"""
		Create the model based on MLP networks.

		:param num_units: size of the networks
		:param num_classes: number of classes in the problem
		:param vocab_size: size of the vocabulary
		:param embedding_size: size of each word embedding
		:param use_intra_attention: whether to use intra-attention model
		:param training: whether to create training tensors (optimizer)
		:param project_input: whether to project input embeddings to a
			different dimensionality
		:param distance_biases: number of different distances with biases used
			in the intra-attention model
		"""
		super(ESIM, self).__init__()
		self.vocab_size=vocab_size
		self.num_units = num_units
		self.num_classes = num_classes
		self.project_input = project_input
		self.embedding_size=embedding_size
		self.distance_biases=distance_biases
		self.max_sentence_length=max_sentence_length
		self.pretrained_emb=pretrained_emb

		self.dropout = nn.Dropout(p=0.5)
		self.word_embedding=nn.Embedding(vocab_size,embedding_size)

		self.tree_lstm_intra=BinaryTreeLSTM(torch.cuda.is_available(),embedding_size, num_units)

		#self.lstm_intra = nn.LSTM(embedding_size, num_units, num_layers=1, batch_first=True)
		#self.lstm_intra_r = nn.LSTM(embedding_size, num_units, num_layers=1, batch_first=True)

		self.linear_layer_compare = nn.Sequential(nn.Linear(4*num_units, num_units), nn.ReLU(), nn.Dropout(p=0.5))
		#                                          nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU())

		#self.lstm_compare = nn.LSTM(num_units, num_units, num_layers=1, batch_first=True)
		#self.lstm_compare_r = nn.LSTM(num_units, num_units, num_layers=1, batch_first=True)
		self.tree_lstm_compare=BinaryTreeLSTM(torch.cuda.is_available(), embedding_size, num_units)

		self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(4*num_units, num_units), nn.ReLU(),
													nn.Dropout(p=0.5), nn.Linear(num_units, num_classes))

		self.init_weight()

	def ortho_weight(self):
		"""
		Random orthogonal weights
		Used by norm_weights(below), in which case, we
		are ensuring that the rows are orthogonal
		(i.e W = U \Sigma V, U has the same
		# of rows, V has the same # of cols)
		"""
		ndim=self.num_units
		W = np.random.randn(ndim, ndim)
		u, s, v = np.linalg.svd(W)
		return u.astype('float32')

	def initialize_lstm(self):
		if torch.cuda.is_available():
			init=torch.Tensor(np.concatenate([self.ortho_weight(),self.ortho_weight(),self.ortho_weight(),self.ortho_weight()], 0)).cuda()
		else:
			init = torch.Tensor(
				np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
		return init

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

	def _transformation_input(self,embed_sent, x1_mask, x1_left_mask, x1_right_mask):
		embed_sent = self.word_embedding(embed_sent)
		embed_sent = self.dropout(embed_sent)
		hidden=self.tree_lstm_intra(embed_sent, x1_mask, x1_left_mask, x1_right_mask)
		return hidden


	def aggregate(self,v1, v2):
		"""
		Aggregate the representations induced from both sentences and their
		representations

		:param v1: tensor with shape (batch, time_steps, num_units)
		:param v2: tensor with shape (batch, time_steps, num_units)
		:return: logits over classes, shape (batch, num_classes)
		"""
		v1_mean = torch.mean(v1, 1)
		v2_mean = torch.mean(v2, 1)
		v1_max, _ = torch.max(v1, 1)
		v2_max, _ = torch.max(v2, 1)
		out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max, v2_mean, v2_max), 1))

		#v1_sum=torch.sum(v1,1)
		#v2_sum=torch.sum(v2,1)
		#out=self.linear_layer_aggregate(torch.cat([v1_sum,v2_sum],1))

		return out

	def forward(self, x1, x1_mask, x1_left_mask, x1_right_mask, x2, x2_mask, x2_left_mask, x2_right_mask):
		sent1=self._transformation_input(x1,x1_mask, x1_left_mask, x1_right_mask)
		sent2=self._transformation_input(x2,x2_mask, x2_left_mask, x2_right_mask)

		ctx1=torch.transpose(sent1,0,1)
		ctx2=torch.transpose(sent2,0,1)
		# ctx1: #step1 x #sample x #dimctx
		# ctx2: #step2 x #sample x #dimctx
		ctx1 = ctx1 * x1_mask[:, :, None]
		ctx2 = ctx2 * x2_mask[:, :, None]

		# weight_matrix: #sample x #step1 x #step2
		weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1, 2, 0))
		weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0]).permute(1, 2, 0)
		weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0]).permute(1, 2, 0)

		# weight_matrix_1: #step1 x #step2 x #sample
		weight_matrix_1 = weight_matrix_1 * x1_mask[:, None, :]
		weight_matrix_2 = weight_matrix_2 * x2_mask[None, :, :]

		alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
		beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)

		ctx2_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha,3)).sum(0)
		ctx1_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta,3)).sum(1)

		inp1 = torch.cat([ctx1, ctx1_, ctx1 * ctx1_, ctx1 - ctx1_], 2)
		inp2 = torch.cat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], 2)
		inp1=self.dropout(self.linear_layer_compare(inp1))
		inp2=self.dropout(self.linear_layer_compare(inp2))
		v1=self.tree_lstm_compare(inp1, x1_mask, x1_left_mask, x1_right_mask)
		v2=self.tree_lstm_compare(inp2, x2_mask, x2_left_mask, x2_right_mask)
		logits = self.aggregate(v1, v2)
		return logits
