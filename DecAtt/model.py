import sys
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime


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

class DecAtt(nn.Module):
	"""
	    Implementation of the multi feed forward network model described in
	    the paper "A Decomposable Attention Model for Natural Language
	    Inference" by Parikh et al., 2016.

	    It applies feedforward MLPs to combinations of parts of the two sentences,
	    without any recurrent structure.
	"""
	def __init__(self, num_units, num_classes, vocab_size, embedding_size,
	             pretrained_emb, training=True, project_input=True,
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
		super(DecAtt, self).__init__()
		self.vocab_size=vocab_size
		self.num_units = num_units
		self.num_classes = num_classes
		self.project_input = project_input
		self.embedding_size=embedding_size
		self.distance_biases=distance_biases
		self.intra_attention=False
		self.max_sentence_length=max_sentence_length
		self.pretrained_emb=pretrained_emb

		self.bias_embedding=nn.Embedding(max_sentence_length,1)
		self.word_embedding=nn.Embedding(vocab_size,embedding_size)
		#self.linear_layer_project = Variable(torch.Tensor(embedding_size, projected_embedding_size))
		#if torch.cuda.is_available():
		#	self.linear_layer_project=self.linear_layer_project.cuda()
		self.linear_layer_project = nn.Linear(embedding_size, num_units, bias=False)
		#self.linear_layer_intra = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, num_units), nn.ReLU())

		self.linear_layer_attend = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU(),
		                                         nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU())

		self.linear_layer_compare = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_units*2, num_units), nn.ReLU(),
		                                          nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU())

		self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(num_units*2, num_units), nn.ReLU(),
		                                            nn.Dropout(p=0.2), nn.Linear(num_units, num_units), nn.ReLU(),
		                                            nn.Linear(num_units, num_classes), nn.LogSoftmax())
		self.init_weight()

	def init_weight(self):
		#print(self.linear_layer_attend[3])
		self.linear_layer_project.weight.data.normal_(0, 0.01)
		self.linear_layer_attend[1].weight.data.normal_(0, 0.01)
		self.linear_layer_attend[1].bias.data.fill_(0)
		self.linear_layer_attend[4].weight.data.normal_(0, 0.01)
		self.linear_layer_attend[4].bias.data.fill_(0)
		self.linear_layer_compare[1].weight.data.normal_(0, 0.01)
		self.linear_layer_compare[1].bias.data.fill_(0)
		self.linear_layer_compare[4].weight.data.normal_(0, 0.01)
		self.linear_layer_compare[4].bias.data.fill_(0)
		self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
		self.linear_layer_aggregate[1].bias.data.fill_(0)
		self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
		self.linear_layer_aggregate[4].bias.data.fill_(0)
		#self.word_embedding.weight.data.copy_(torch.from_numpy(self.pretrained_emb))

	def attention_softmax3d(self,raw_attentions):
		reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
		out=nn.functional.softmax(reshaped_attentions, dim=1)
		return out.view(raw_attentions.size(0),raw_attentions.size(1),raw_attentions.size(2))

	def _transformation_input(self,embed_sent):
		#embed_sent=torch.matmul(embed_sent,self.linear_layer_project)
		#embed_sent=self.word_embedding(embed_sent)
		embed_sent=self.linear_layer_project(embed_sent)
		result=embed_sent
		#result, (state, _) = self.lstm_intra(embed_sent)
		if self.intra_attention:
			f_intra = self.linear_layer_intra(embed_sent)
			f_intra_t = torch.transpose(f_intra, 1, 2)
			raw_attentions = torch.matmul(f_intra, f_intra_t)
			time_steps=embed_sent.size(1)
			r = torch.arange(0, time_steps)
			r_matrix=r.view(1,-1).expand(time_steps,time_steps)
			raw_index=r_matrix-r.view(-1,1)
			clipped_index=torch.clamp(raw_index,0,self.distance_biases-1)
			clipped_index=Variable(clipped_index.long())
			if torch.cuda.is_available():
				clipped_index=clipped_index.cuda()
			bias=self.bias_embedding(clipped_index)
			bias=torch.squeeze(bias)
			raw_attentions += bias
			attentions = self.attention_softmax3d(raw_attentions)
			attended = torch.matmul(attentions, embed_sent)
			result =torch.cat([embed_sent,attended],2)
		return result

	def attend(self,sent1,sent2, lsize_list, rsize_list):
		"""
		Compute inter-sentence attention. This is step 1 (attend) in the paper

		:param sent1: tensor in shape (batch, time_steps, num_units),
		    the projected sentence 1
		:param sent2: tensor in shape (batch, time_steps, num_units)
		:return: a tuple of 3-d tensors, alfa and beta.
		"""
		repr1=self.linear_layer_attend(sent1)
		repr2=self.linear_layer_attend(sent2)
		repr2=torch.transpose(repr2,1,2)
		raw_attentions = torch.matmul(repr1, repr2)

		#self.mask = generate_mask(lsize_list, rsize_list)
		# masked = mask(self.raw_attentions, rsize_list)
		#masked = raw_attentions * self.mask
		att_sent1 = self.attention_softmax3d(raw_attentions)
		beta=torch.matmul(att_sent1,sent2) #input2_soft

		raw_attentions_t=torch.transpose(raw_attentions,1,2).contiguous()
		#self.mask_t = torch.transpose(self.mask, 1, 2).contiguous()
		# masked = mask(raw_attentions_t, lsize_list)
		#masked = raw_attentions_t * self.mask_t
		att_sent2 = self.attention_softmax3d(raw_attentions_t)
		alpha=torch.matmul(att_sent2,sent1) #input1_soft

		return alpha, beta

	def compare(self,sentence,soft_alignment):
		"""
		Apply a feed forward network to compare o   ne sentence to its
		soft alignment with the other.

		:param sentence: embedded and projected sentence,
		    shape (batch, time_steps, num_units)
		:param soft_alignment: tensor with shape (batch, time_steps, num_units)
		:return: a tensor (batch, time_steps, num_units)
		"""
		sent_alignment=torch.cat([sentence, soft_alignment],2)
		out = self.linear_layer_compare(sent_alignment)
		#out, (state, _) = self.lstm_compare(out)
		return out

	def aggregate(self,v1, v2):
		"""
		Aggregate the representations induced from both sentences and their
		representations

		:param v1: tensor with shape (batch, time_steps, num_units)
		:param v2: tensor with shape (batch, time_steps, num_units)
		:return: logits over classes, shape (batch, num_classes)
		"""
		v1_sum=torch.sum(v1,1)
		v2_sum=torch.sum(v2,1)
		out=self.linear_layer_aggregate(torch.cat([v1_sum,v2_sum],1))
		return out

	def forward(self,sent1, sent2, lsize_list, rsize_list):
		sent1=self._transformation_input(sent1)
		sent2=self._transformation_input(sent2)
		alpha, beta = self.attend(sent1, sent2, lsize_list, rsize_list)
		v1=self.compare(sent1,beta)
		v2=self.compare(sent2,alpha)
		#v1 = generate_mask_2(v1, lsize_list)
		#v2 = generate_mask_2(v2, rsize_list)
		logits=self.aggregate(v1,v2)
		return logits

