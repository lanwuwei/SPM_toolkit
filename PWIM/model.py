import sys
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import math
import torch.nn as nn
from util import *
import numpy
import random


class DeepPairWiseWord(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, num_layers,task,granularity,num_class,dict,fake_dict, dict_char_ngram, oov,tokens, word_freq,
	             feature_maps, kernels, charcnn_embedding_size, charcnn_max_word_length, character_ngrams,c2w_mode,character_ngrams_overlap,word_mode,
	             combine_mode,lm_mode, deep_CNN):#,corpus):
		super(DeepPairWiseWord, self).__init__()
		self.task = task
		if task=='pit':
			self.limit = 32
		else:
			self.limit = 48
		# Language model parameters--------------------------
		if lm_mode:
			self.lm_loss=nn.NLLLoss()
			self.lm_softmax=nn.LogSoftmax()
			self.lm_tanh=nn.Tanh()
			self.lm_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
			self.lm_Wm_forward=Variable(torch.rand(hidden_dim,hidden_dim))
			self.lm_Wm_backward = Variable(torch.rand(hidden_dim, hidden_dim))
			self.lm_Wq_forward=Variable(torch.rand(hidden_dim,len(tokens)))
			self.lm_Wq_backword=Variable(torch.rand(hidden_dim, len(tokens)))

		# end of LM parameters-------------------------------
		self.granularity=granularity
		self.num_class=num_class
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.embedding_dim=embedding_dim
		self.dict=dict
		self.fake_dict=fake_dict
		self.dict_char_ngram=dict_char_ngram
		self.word_freq=word_freq
		self.oov=oov
		self.tokens=tokens
		word2id={}
		index=0
		for word in tokens:
			word2id[word]=index
			index+=1
		self.word2id=word2id
		self.feature_maps=feature_maps
		self.kernels=kernels
		self.charcnn_max_word_length=charcnn_max_word_length
		self.character_ngrams=character_ngrams
		self.c2w_mode=c2w_mode
		self.character_ngrams_overlap=character_ngrams_overlap
		self.word_mode=word_mode
		self.combine_mode=combine_mode
		self.lm_mode=lm_mode
		self.deep_CNN=deep_CNN
		if granularity=='char':
			self.df = Variable(torch.rand(embedding_dim, embedding_dim))
			self.db = Variable(torch.rand(embedding_dim, embedding_dim))
			self.bias = Variable(torch.rand(embedding_dim))
			self.Wx=Variable(torch.rand(embedding_dim, embedding_dim))
			self.W1=Variable(torch.rand(embedding_dim, embedding_dim))
			self.W2=Variable(torch.rand(embedding_dim, embedding_dim))
			self.W3=Variable(torch.rand(embedding_dim, embedding_dim))
			self.vg = Variable(torch.rand(embedding_dim, 1))
			self.bg = Variable(torch.rand(1, 1))
			#--self.string_sim_weight=Variable(torch.rand(13,48,48))
			#--self.string_sim_bias = Variable(torch.rand(13,48, 48))
			if torch.cuda.is_available():
				self.df = self.df.cuda()
				self.db = self.db.cuda()
				self.bias = self.bias.cuda()
				self.Wx=self.Wx.cuda()
				self.W1=self.W1.cuda()
				self.W2=self.W2.cuda()
				self.W3=self.W3.cuda()
				self.vg=self.vg.cuda()
				self.bg=self.bg.cuda()
				#--self.string_sim_weight=self.string_sim_weight.cuda()
				#--self.string_sim_bias=self.string_sim_bias.cuda()
				if lm_mode:
					self.lm_Wm_forward=self.lm_Wm_forward.cuda()
					self.lm_Wm_backward=self.lm_Wm_backward.cuda()
					self.lm_Wq_forward=self.lm_Wq_forward.cuda()
					self.lm_Wq_backword=self.lm_Wq_backword.cuda()
				pass
			self.c2w_embedding=nn.Embedding(len(dict_char_ngram),50)
			self.char_cnn_embedding=nn.Embedding(len(dict_char_ngram),charcnn_embedding_size)
			self.lstm_c2w = nn.LSTM(50, embedding_dim, 1, bidirectional=True)
			self.charCNN_filter1 = nn.Sequential(
				nn.Conv2d(1, feature_maps[0], (kernels[0], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[0] + 1, 1), stride=1)
			)
			self.charCNN_filter2 = nn.Sequential(
				nn.Conv2d(1, feature_maps[1], (kernels[1], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[1] + 1, 1), stride=1)
			)
			self.charCNN_filter3 = nn.Sequential(
				nn.Conv2d(1, feature_maps[2], (kernels[2], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[2] + 1, 1), stride=1)
			)
			self.charCNN_filter4 = nn.Sequential(
				nn.Conv2d(1, feature_maps[3], (kernels[3], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[3] + 1, 1), stride=1)
			)
			self.charCNN_filter5 = nn.Sequential(
				nn.Conv2d(1, feature_maps[4], (kernels[4], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[4] + 1, 1), stride=1)
			)
			self.charCNN_filter6 = nn.Sequential(
				nn.Conv2d(1, feature_maps[5], (kernels[5], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[5] + 1, 1), stride=1)
			)
			self.charCNN_filter7 = nn.Sequential(
				nn.Conv2d(1, feature_maps[6], (kernels[6], charcnn_embedding_size)),
				nn.Tanh(),
				nn.MaxPool2d((charcnn_max_word_length - kernels[6] + 1, 1), stride=1)
			)
			self.transform_gate = nn.Sequential(nn.Linear(1100, 1100), nn.Sigmoid())
			self.char_cnn_mlp = nn.Sequential(nn.Linear(1100, 1100), nn.Tanh())
			# --
			self.down_sampling_200 = nn.Linear(1100, 200)
			self.down_sampling_300 = nn.Linear(1100, 300)
		elif granularity=='word':
			#pass
			''''''
			self.word_embedding=nn.Embedding(len(tokens),embedding_dim)
			self.copied_word_embedding=nn.Embedding(len(tokens),embedding_dim)
			pretrained_weight = numpy.zeros(shape=(len(self.tokens), self.embedding_dim))
			for word in self.tokens:
				pretrained_weight[self.tokens.index(word)] = self.dict[word].numpy()
			self.copied_word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
			''''''
		#self.lstm = nn.LSTM(embedding_dim*2, hidden_dim, num_layers, bidirectional=True)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)
		if not deep_CNN:
			self.mlp_layer = nn.Sequential(nn.Linear(self.limit*self.limit*13, 16),nn.Linear(16, num_class), nn.LogSoftmax())
		else:
			self.layer1 = nn.Sequential(
				nn.Conv2d(13, 128, kernel_size=3,stride=1, padding=1),
				#nn.Conv2d(26, 128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2,stride=2, ceil_mode=True))
			self.layer1_ = nn.Sequential(
				#nn.Conv2d(13, 128, kernel_size=3, stride=1, padding=1),
				nn.Conv2d(26, 128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
			self.layer2 = nn.Sequential(
				nn.Conv2d(128, 164, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
			self.layer3 = nn.Sequential(
				nn.Conv2d(164, 192, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
			self.layer4 = nn.Sequential(
				nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
			self.layer5 = nn.Sequential(
				nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
			self.layer5_0 = nn.Sequential(
				nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=True))
			self.fc1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
			self.fc2 = nn.Sequential(nn.Linear(128, num_class), nn.LogSoftmax())
		self.init_weight()

	def init_weight(self):
		if self.deep_CNN:
			self.layer1[0].weight.data.normal_(0, math.sqrt(2.0/(3*3*128)))
			self.layer1[0].bias.data.fill_(0)
			self.layer2[0].weight.data.normal_(0, math.sqrt(2.0/(3*3*164)))
			self.layer2[0].bias.data.fill_(0)
			self.layer3[0].weight.data.normal_(0, math.sqrt(2.0/(3*3*192)))
			self.layer3[0].bias.data.fill_(0)
			self.layer4[0].weight.data.normal_(0, math.sqrt(2.0/(3*3*192)))
			self.layer4[0].bias.data.fill_(0)
			self.layer5[0].weight.data.normal_(0, math.sqrt(2.0/(3*3*128)))
			self.layer5[0].bias.data.fill_(0)
			self.fc1[0].weight.data.uniform_(-0.1, 0.1)
			self.fc1[0].bias.data.fill_(0)
			self.fc2[0].weight.data.uniform_(-0.1, 0.1)
			self.fc2[0].bias.data.fill_(0)
		if self.granularity=='word':
			pretrained_weight=numpy.zeros(shape=(len(self.tokens),self.embedding_dim))
			for word in self.tokens:
				pretrained_weight[self.tokens.index(word)]=self.dict[word].numpy()
			self.copied_word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))


	def unpack(self,bi_hidden, half_dim):
		#print(bi_hidden[0])
		for i in range(bi_hidden.size(0)):
			vec=bi_hidden[i][:]
			if i==0:
				h_fw=vec[:half_dim].view(1,-1)
				h_bw=vec[half_dim:].view(1,-1)
			else:
				h_fw_new = vec[:half_dim].view(1, -1)
				h_bw_new = vec[half_dim:].view(1, -1)
				h_fw=torch.cat((h_fw,h_fw_new),0)
				h_bw=torch.cat((h_bw,h_bw_new),0)
		#print(h_fw.size())
		#print(h_fw[0])
		#print(h_bw[0])
		#sys.exit()
		return (h_fw, h_bw)

	def pairwise_word_interaction(self,out0,out1, target_A, target_B):
		extra_loss=0
		h_fw_0, h_bw_0 = self.unpack(out0.view(out0.size(0),out0.size(2)),half_dim=self.hidden_dim)
		h_fw_1, h_bw_1 = self.unpack(out1.view(out1.size(0),out1.size(2)),half_dim=self.hidden_dim)
		#print(h_fw_0)
		#print(h_bw_0)
		#print(h_fw_1)
		#print(h_bw_1)
		#sys.exit()
		h_bi_0 = out0.view(out0.size(0),out0.size(2))
		h_bi_1 = out1.view(out1.size(0),out1.size(2))
		h_sum_0 = h_fw_0 + h_bw_0
		h_sum_1 = h_fw_1 + h_bw_1
		len0 = h_fw_0.size(0)
		len1 = h_fw_1.size(0)
		i=0
		j=0
		#simCube1 = torch.mm(h_fw_0[i].view(1, -1), h_fw_1[j].view(-1, 1))
		#simCube2 = torch.mm(h_bw_0[i].view(1, -1), h_bw_1[j].view(-1, 1))
		#simCube3 = torch.mm(h_bi_0[i].view(1, -1), h_bi_1[j].view(-1, 1))
		#simCube4 = torch.mm(h_sum_0[i].view(1, -1), h_sum_1[j].view(-1, 1))
		#simCube5 = F.pairwise_distance(h_fw_0[i].view(1, -1), h_fw_1[j].view(1, -1))
		simCube5_0 = h_fw_0[i].view(1, -1)
		simCube5_1 = h_fw_1[j].view(1, -1)
		#simCube6 = F.pairwise_distance(h_bw_0[i].view(1, -1), h_bw_1[j].view(1, -1))
		simCube6_0 = h_bw_0[i].view(1, -1)
		simCube6_1 = h_bw_1[j].view(1, -1)
		#simCube7 = F.pairwise_distance(h_bi_0[i].view(1, -1), h_bi_1[j].view(1, -1))
		simCube7_0 = h_bi_0[i].view(1, -1)
		simCube7_1 = h_bi_1[j].view(1, -1)
		#simCube8 = F.pairwise_distance(h_sum_0[i].view(1, -1), h_sum_1[j].view(1, -1))
		simCube8_0 = h_sum_0[i].view(1,-1)
		simCube8_1 = h_sum_1[j].view(1,-1)
		#simCube9 = F.cosine_similarity(h_fw_0[i].view(1, -1), h_fw_1[j].view(1, -1))
		#simCube10 = F.cosine_similarity(h_bw_0[i].view(1, -1), h_bw_1[j].view(1, -1))
		#simCube11 = F.cosine_similarity(h_bi_0[i].view(1, -1), h_bi_1[j].view(1, -1))
		#simCube12 = F.cosine_similarity(h_sum_0[i].view(1, -1), h_sum_1[j].view(1, -1))
		for i in range(len0):
			for j in range(len1):
				if not(i==0 and j==0):
					simCube5_0 = torch.cat((simCube5_0, h_fw_0[i].view(1, -1)))
					simCube5_1 = torch.cat((simCube5_1, h_fw_1[j].view(1, -1)))
					simCube6_0 = torch.cat((simCube6_0, h_bw_0[i].view(1, -1)))
					simCube6_1 = torch.cat((simCube6_1, h_bw_1[j].view(1, -1)))
					simCube7_0 = torch.cat((simCube7_0, h_bi_0[i].view(1, -1)))
					simCube7_1 = torch.cat((simCube7_1, h_bi_1[j].view(1, -1)))
					simCube8_0 = torch.cat((simCube8_0, h_sum_0[i].view(1, -1)))
					simCube8_1 = torch.cat((simCube8_1, h_sum_1[j].view(1, -1)))
		simCube1 = torch.unsqueeze(torch.mm(h_fw_0, torch.transpose(h_fw_1, 0, 1)), 0)
		simCube2 = torch.unsqueeze(torch.mm(h_bw_0, torch.transpose(h_bw_1, 0, 1)), 0)
		simCube3 = torch.unsqueeze(torch.mm(h_bi_0, torch.transpose(h_bi_1, 0, 1)), 0)
		simCube4 = torch.unsqueeze(torch.mm(h_sum_0, torch.transpose(h_sum_1, 0, 1)), 0)
		simCube5 = torch.neg(F.pairwise_distance(simCube5_0, simCube5_1))
		simCube5 = torch.unsqueeze(simCube5.view(len0, len1), 0)
		simCube6 = torch.neg(F.pairwise_distance(simCube6_0, simCube6_1))
		simCube6 = torch.unsqueeze(simCube6.view(len0, len1), 0)
		simCube7 = torch.neg(F.pairwise_distance(simCube7_0, simCube7_1))
		simCube7 = torch.unsqueeze(simCube7.view(len0, len1), 0)
		simCube8 = torch.neg(F.pairwise_distance(simCube8_0,simCube8_1))
		simCube8 = torch.unsqueeze(simCube8.view(len0, len1), 0)

		simCube9 = F.cosine_similarity(simCube5_0,simCube5_1)
		simCube9 = torch.unsqueeze(simCube9.view(len0,len1), 0)
		simCube10 = F.cosine_similarity(simCube6_0, simCube6_1)
		simCube10 = torch.unsqueeze(simCube10.view(len0, len1), 0)
		simCube11 = F.cosine_similarity(simCube7_0, simCube7_1)
		simCube11 = torch.unsqueeze(simCube11.view(len0, len1), 0)
		simCube12 = F.cosine_similarity(simCube8_0, simCube8_1)
		simCube12= torch.unsqueeze(simCube12.view(len0, len1), 0)
		''''''
		if torch.cuda.is_available():
			simCube13 = torch.unsqueeze(Variable(torch.zeros(len0,len1)).cuda()+1,0)
		else:
			simCube13 = torch.unsqueeze(Variable(torch.zeros(len0,len1))+1,0)
		simCube=torch.cat((simCube9,simCube5,simCube1,simCube10,simCube6,simCube2,simCube12,simCube8,simCube4,simCube11,simCube7,simCube3,simCube13),0)
		#simCube=torch.unsqueeze(simCube,0)
		#simCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, self.limit - simCube.size(2)))[0]
		#print(simCube1)
		#print(simCube)
		#print(simCube8)
		#sys.exit()
		return simCube, extra_loss

	def similarity_focus(self,simCube):
		if torch.cuda.is_available():
			mask=torch.mul(torch.ones(simCube.size(0),simCube.size(1),simCube.size(2)).cuda(),0.1)
		else:
			mask=torch.mul(torch.ones(simCube.size(0),simCube.size(1),simCube.size(2)),0.1)
		s1tag=torch.zeros(simCube.size(1))
		s2tag=torch.zeros(simCube.size(2))
		sorted, indices=torch.sort(simCube[6].view(1,-1),descending=True)
		record=[]
		for indix in indices[0]:
			pos1=torch.div(indix,simCube.size(2)).data[0]
			pos2=(indix-simCube.size(2)*pos1).data[0]
			if s1tag[pos1]+s2tag[pos2]<=0:
				s1tag[pos1]=1
				s2tag[pos2]=1
				record.append((pos1,pos2))
				mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
				mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
				mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
				mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
				mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
				mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
				mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
				mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
				mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
				mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
				mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
				mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
			mask[12][pos1][pos2] = mask[12][pos1][pos2] + 0.9
		s1tag = torch.zeros(simCube.size(1))
		s2tag = torch.zeros(simCube.size(2))
		sorted, indices = torch.sort(simCube[7].view(1, -1), descending=True)
		counter=0
		for indix in indices[0]:
			pos1 = torch.div(indix, simCube.size(2)).data[0]
			pos2 = (indix-simCube.size(2)*pos1).data[0]
			if s1tag[pos1] + s2tag[pos2] <= 0:
				counter+=1
				if (pos1,pos2) in record:
					continue
				else:
					s1tag[pos1] = 1
					s2tag[pos2] = 1
					#record.append((pos1,pos2))
					mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
					mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
					mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
					mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
					mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
					mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
					mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
					mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
					mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
					mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
					mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
					mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
			if counter>=len(record):
				break
		focusCube=torch.mul(simCube,Variable(mask))
		return focusCube

	def deep_cnn(self,focusCube):
		simCube = torch.unsqueeze(focusCube, 0)
		focusCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, self.limit - simCube.size(2)))[0]
		focusCube=torch.unsqueeze(focusCube,0)
		out=self.layer1(focusCube)
		out=self.layer2(out)
		out=self.layer3(out)
		if self.limit==16:
			out=self.layer5(out)
		elif self.limit==32:
			out = self.layer4(out)
			out=self.layer5(out)
		elif self.limit==48:
			out = self.layer4(out)
			out=self.layer5_0(out)
			#out=self.layer5_1(out)
		#print('debug 6: (out size)')
		#print(out.size())
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		#print(out)
		return out

	def mlp(self,focusCube):
		simCube = torch.unsqueeze(focusCube, 0)
		focusCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, self.limit - simCube.size(2)))[0]
		#print(focusCube.view(-1))
		result=self.mlp_layer(focusCube.view(-1))
		#sys.exit()
		return result

	def language_model(self,out0,out1, target_A, target_B):
		extra_loss=0
		h_fw_0, h_bw_0 = self.unpack(out0.view(out0.size(0),out0.size(2)),half_dim=self.hidden_dim)
		h_fw_1, h_bw_1 = self.unpack(out1.view(out1.size(0),out1.size(2)),half_dim=self.hidden_dim)
		''''''
		m_fw_0=self.lm_tanh(torch.mm(h_fw_0, self.lm_Wm_forward))
		m_bw_0=self.lm_tanh(torch.mm(h_bw_0, self.lm_Wm_backward))
		m_fw_1=self.lm_tanh(torch.mm(h_fw_1, self.lm_Wm_forward))
		m_bw_1=self.lm_tanh(torch.mm(h_bw_1, self.lm_Wm_backward))
		q_fw_0=self.lm_softmax(torch.mm(m_fw_0, self.lm_Wq_forward))
		q_bw_0=self.lm_softmax(torch.mm(m_bw_0, self.lm_Wq_backword))
		q_fw_1 = self.lm_softmax(torch.mm(m_fw_1, self.lm_Wq_forward))
		q_bw_1 = self.lm_softmax(torch.mm(m_bw_1, self.lm_Wq_backword))
		target_fw_0=Variable(torch.LongTensor(target_A[1:]+[self.tokens.index('</s>')]))
		target_bw_0=Variable(torch.LongTensor([self.tokens.index('<s>')]+target_A[:-1]))
		target_fw_1=Variable(torch.LongTensor(target_B[1:]+[self.tokens.index('</s>')]))
		target_bw_1=Variable(torch.LongTensor([self.tokens.index('<s>')]+target_B[:-1]))
		if torch.cuda.is_available():
			target_fw_0=target_fw_0.cuda()
			target_bw_0=target_bw_0.cuda()
			target_fw_1=target_fw_1.cuda()
			target_bw_1=target_bw_1.cuda()
		loss1=self.lm_loss(q_fw_0, target_fw_0)
		loss2=self.lm_loss(q_bw_0, target_bw_0)
		loss3=self.lm_loss(q_fw_1, target_fw_1)
		loss4=self.lm_loss(q_bw_1, target_bw_1)
		extra_loss=loss1+loss2+loss3+loss4
		''''''
		return extra_loss

	def word_layer(self, lsents, rsents):
		glove_mode=self.word_mode[0]
		update_inv_mode=self.word_mode[1]
		update_oov_mode=self.word_mode[2]
		if glove_mode==True and update_inv_mode==False and update_oov_mode==False:
			try:
				sentA = torch.cat([self.dict[word].view(1, self.embedding_dim) for word in lsents], 0)
				sentA = Variable(sentA)  # .cuda()
				sentB = torch.cat([self.dict[word].view(1, self.embedding_dim) for word in rsents], 0)
				sentB = Variable(sentB)  # .cuda()
			except:
				print(lsents)
				print(rsents)
				sys.exit()
			if torch.cuda.is_available():
				sentA = sentA.cuda()
				sentB = sentB.cuda()
		elif glove_mode==True and update_inv_mode==False and update_oov_mode==True:
			firstFlag=True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						indice=Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice=indice.cuda()
						output=self.word_embedding(indice)
						firstFlag=False
					else:
						output=Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output=output.cuda()
						firstFlag=False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new=Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new=output_new.cuda()
						output = torch.cat((output, output_new), 0)
			sentA=output
			firstFlag = False
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==True and update_inv_mode==True and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output=output.view(1,-1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.copied_word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new=Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1,-1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.copied_word_embedding(indice)
						output = torch.cat((output, output_new), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.copied_word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1,-1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.copied_word_embedding(indice)
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==True and update_inv_mode==True and update_oov_mode==True:
			tmp=[]
			for word in lsents:
				try:
					tmp.append(self.word2id[word])
				except:
					tmp.append(self.word2id['oov'])
			indices = Variable(torch.LongTensor(tmp))
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentA = self.copied_word_embedding(indices)
			tmp = []
			for word in rsents:
				try:
					tmp.append(self.word2id[word])
				except:
					tmp.append(self.word2id['oov'])
			indices = Variable(torch.LongTensor(tmp))
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentB = self.copied_word_embedding(indices)
		elif glove_mode==False and update_inv_mode==False and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output = output.cuda()
					output = output.view(1, -1)
					firstFlag = False
				else:
					output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output_new = output_new.cuda()
					output = torch.cat((output, output_new.view(1, -1)), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output = output.cuda()
					output = output.view(1, -1)
					firstFlag = False
				else:
					output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
					if torch.cuda.is_available():
						output_new = output_new.cuda()
					output = torch.cat((output, output_new.view(1, -1)), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==False and update_oov_mode==True:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output_new = output_new.view(1,-1)
						output = torch.cat((output, output_new), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
					else:
						output = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
				else:
					if word in self.oov:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new), 0)
					else:
						output_new = Variable(self.fake_dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output_new = output_new.view(1, -1)
						output = torch.cat((output, output_new), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==True and update_oov_mode==False:
			firstFlag = True
			for word in lsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(self.dict[word].view(1, self.embedding_dim))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(torch.Tensor(self.dict[word].view(1, self.embedding_dim)))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1, -1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new.view(1, -1)), 0)
			sentA = output
			firstFlag = True
			for word in rsents:
				if firstFlag:
					if word in self.oov:
						output = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output = output.cuda()
						output = output.view(1, -1)
						firstFlag = False
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output = self.word_embedding(indice)
						firstFlag = False
				else:
					if word in self.oov:
						output_new = Variable(
							torch.Tensor([random.uniform(-0.05, 0.05) for i in range(self.embedding_dim)]))
						if torch.cuda.is_available():
							output_new = output_new.cuda()
						output = torch.cat((output, output_new.view(1, -1)), 0)
					else:
						indice = Variable(torch.LongTensor([self.tokens.index(word)]))
						if torch.cuda.is_available():
							indice = indice.cuda()
						output_new = self.word_embedding(indice)
						output = torch.cat((output, output_new.view(1, -1)), 0)
			sentB = output
		elif glove_mode==False and update_inv_mode==True and update_oov_mode==True:
			indices=Variable(torch.LongTensor([self.tokens.index(word) for word in lsents]))
			#print(indices)
			if torch.cuda.is_available():
				indices=indices.cuda()
			sentA=self.word_embedding(indices)
			indices = Variable(torch.LongTensor([self.tokens.index(word) for word in rsents]))
			#print(indices)
			if torch.cuda.is_available():
				indices = indices.cuda()
			sentB = self.word_embedding(indices)
		sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
		sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
		return (sentA, sentB)

	def c2w_cell(self,indices,h,c):
		input=Variable(torch.LongTensor(indices))
		if torch.cuda.is_available():
			input = input.cuda()
		input=self.c2w_embedding(input)
		input=input.view(-1,1,50)
		out,(state,_)=self.lstm_c2w(input,(h,c))
		output_char = torch.mm(self.df, state[0][0][:].view(-1, 1)) + torch.mm(self.db, state[1][0][:].view(-1, 1)) + self.bias.view(-1, 1)
		output_char = output_char.view(1, -1)
		return output_char

	def charCNN_cell(self,indices):
		input = Variable(torch.LongTensor(indices))
		if torch.cuda.is_available():
			input = input.cuda()
		input = self.char_cnn_embedding(input)
		input=torch.unsqueeze(input,0)
		out1 = self.charCNN_filter1(input)
		out2 = self.charCNN_filter2(input)
		out3 = self.charCNN_filter3(input)
		out4 = self.charCNN_filter4(input)
		out5 = self.charCNN_filter5(input)
		out6 = self.charCNN_filter6(input)
		out7 = self.charCNN_filter7(input)
		final_output=torch.cat([torch.squeeze(out1),torch.squeeze(out2),torch.squeeze(out3),torch.squeeze(out4),torch.squeeze(out5),torch.squeeze(out6),torch.squeeze(out7)])
		final_output=final_output.view(1,-1)
		transform_gate=self.transform_gate(final_output)
		final_output=transform_gate * self.char_cnn_mlp(final_output)+(1-transform_gate) * final_output
		return final_output

	def generate_word_indices(self,word):
		if self.task=='hindi':
			#char_gram = syllabifier.orthographic_syllabify(word, 'hi')
			char_gram = list(splitclusters(word))
			indices=[]
			if self.character_ngrams == 1:
				indices=[self.dict_char_ngram[char] for char in char_gram]
			elif self.character_ngrams == 2:
				if self.character_ngrams_overlap:
					if len(char_gram) <= 2:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(len(char_gram) - 1):
							indices.append(self.dict_char_ngram[char_gram[i]+char_gram[i+1]])
				else:
					if len(char_gram) <= 2:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(0, len(char_gram) - 1, 2):
							indices.append(self.dict_char_ngram[char_gram[i] + char_gram[i + 1]])
						if len(char_gram)%2==1:
							indices.append(self.dict_char_ngram[char_gram[len(char_gram)-1]])
		else:
			indices=[]
			if self.character_ngrams == 1:
				for char in word:
					try:
						indices.append(self.dict_char_ngram[char])
					except:
						continue
				#indices = [self.dict_char_ngram[char] for char in word]
			elif self.character_ngrams == 2:
				if self.character_ngrams_overlap:
					if len(word) <= 2:
						try:
							indices = [self.dict_char_ngram[word]]
						except:
							indices = [self.dict_char_ngram[' ']]
					else:
						for i in range(len(word) - 1):
							try:
								indices.append(self.dict_char_ngram[word[i:i + 2]])
							except:
								indices.append(self.dict_char_ngram[' '])
				else:
					if len(word) <= 2:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(0, len(word) - 1, 2):
							indices.append(self.dict_char_ngram[word[i:i + 2]])
						if len(word) % 2 == 1:
							indices.append(self.dict_char_ngram[word[len(word) - 1]])
			elif self.character_ngrams == 3:
				if self.character_ngrams_overlap:
					if len(word) <= 3:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(len(word) - 2):
							indices.append(self.dict_char_ngram[word[i:i + 3]])
				else:
					if len(word) <= 3:
						indices = [self.dict_char_ngram[word]]
					else:
						for i in range(0, len(word) - 2, 3):
							indices.append(self.dict_char_ngram[word[i:i + 3]])
						if len(word) % 3 == 1:
							indices.append(self.dict_char_ngram[word[len(word) - 1]])
						elif len(word) % 3 == 2:
							indices.append(self.dict_char_ngram[word[len(word) - 2:]])
		return indices

	def c2w_or_cnn_layer(self,lsents, rsents):
		h = Variable(torch.zeros(2, 1, self.embedding_dim))  # 2 for bidirection
		c = Variable(torch.zeros(2, 1, self.embedding_dim))
		if torch.cuda.is_available():
			h=h.cuda()
			c=c.cuda()
		firstFlag=True
		for word in lsents:
			indices=self.generate_word_indices(word)
			if not self.c2w_mode:
				if len(indices)<20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
			if firstFlag:
				if self.c2w_mode:
					output=self.c2w_cell([indices], h, c)
				else:
					output=self.charCNN_cell([indices])
				firstFlag=False
			else:
				if self.c2w_mode:
					output_new=self.c2w_cell([indices], h, c)
				else:
					output_new=self.charCNN_cell([indices])
				output=torch.cat((output,output_new),0)
		#print(output)
		#sys.exit()
		sentA=output
		firstFlag = True
		for word in rsents:
			# print word
			indices = self.generate_word_indices(word)
			if not self.c2w_mode:
				if len(indices)<20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
			# print(indices)
			if firstFlag:
				if self.c2w_mode:
					output=self.c2w_cell([indices], h, c)
				else:
					output=self.charCNN_cell([indices])
				firstFlag = False
			else:
				if self.c2w_mode:
					output_new=self.c2w_cell([indices], h, c)
				else:
					output_new=self.charCNN_cell([indices])
				output = torch.cat((output, output_new), 0)
		#print(output)
		#sys.exit()
		sentB=output
		sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
		sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
		return (sentA,sentB)

	def mix_cell(self,word, output_word, output_char):
		result=None
		extra_loss=0
		indices_reduce_dim = Variable(torch.LongTensor([i * 2 for i in range(self.embedding_dim)]))
		if torch.cuda.is_available():
			indices_reduce_dim=indices_reduce_dim.cuda()
		if self.combine_mode == 'concat':
			result = torch.cat((output_word, output_char), 1)
			result = torch.index_select(result, 1, indices_reduce_dim)
		elif self.combine_mode == 'g_0.25':
			result = 0.25 * output_word + 0.75 * output_char
		elif self.combine_mode == 'g_0.50':
			result = 0.5 * output_word + 0.5 * output_char
		elif self.combine_mode == 'g_0.75':
			result = 0.75 * output_word + 0.25 * output_char
		elif self.combine_mode == 'adaptive':
			gate = self.sigmoid(torch.mm(output_word, self.vg) + self.bg)
			gate=gate.expand(1,self.embedding_dim)
			result = (1-gate) * output_word + gate * output_char
		elif self.combine_mode == 'attention':
			gate = self.sigmoid(torch.mm(self.tanh(torch.mm(output_word, self.W1) + torch.mm(output_char, self.W2)), self.W3))
			result = gate*output_word+(1-gate)*output_char
			if word not in self.oov:
				extra_loss+=(1-F.cosine_similarity(output_word,output_char))
		elif self.combine_mode == 'backoff':
			if word in self.oov:
				result=output_char
			else:
				result=output_word
		return (result, extra_loss)

	def mix_layer(self,lsents,rsents):
		h = Variable(torch.zeros(2, 1, self.embedding_dim))  # 2 for bidirection
		c = Variable(torch.zeros(2, 1, self.embedding_dim))
		if torch.cuda.is_available():
			h=h.cuda()
			c=c.cuda()
		firstFlag = True
		#if (index + 1) % (int(42200 / 4)) == 0:
		#	extra_loss=0
		#else:
		#	extra_loss=self.language_model(index, h,c)
		extra_loss=0
		# print(lsents)
		# print(rsents)
		# sys.exit()
		for word in lsents:
			indices = self.generate_word_indices(word)
			if self.c2w_mode:
				output_char = self.c2w_cell([indices], h, c)
			else:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
				output_char = self.charCNN_cell([indices])
				if self.task=='sts':
					output_char=self.down_sampling_300(output_char)
				else:
					output_char=self.down_sampling_200(output_char)
			#indice=Variable(torch.LongTensor([self.tokens.index(word)]))
			#if torch.cuda.is_available():
			#	indice=indice.cuda()
			#output_word = self.copied_word_embedding(indice).view(1,-1)
			output_word = Variable(torch.Tensor(self.dict[word])).view(1,-1)
			if torch.cuda.is_available():
				output_word = output_word.cuda()
			if firstFlag:
				output, extra_loss=self.mix_cell(word, output_word, output_char)
				output2 = output_char
				firstFlag = False
			else:
				output_new, extra_loss = self.mix_cell(word, output_word, output_char)
				output_new2 = output_char
				output = torch.cat((output, output_new), 0)
				output2 = torch.cat((output2, output_new2), 0)
		sentA = output
		sentA2 = output2
		firstFlag=True
		for word in rsents:
			indices = self.generate_word_indices(word)
			if self.c2w_mode:
				output_char = self.c2w_cell([indices], h, c)
			else:
				if len(indices) < 20:
					indices = indices + [0 for i in range(self.charcnn_max_word_length - len(indices))]
				else:
					indices = indices[0:20]
				output_char = self.charCNN_cell([indices])
				if self.task=='sts':
					output_char=self.down_sampling_300(output_char)
				else:
					output_char=self.down_sampling_200(output_char)
			#indice = Variable(torch.LongTensor([self.tokens.index(word)]))
			#if torch.cuda.is_available():
			#	indice = indice.cuda()
			#output_word = self.copied_word_embedding(indice).view(1, -1)
			output_word = Variable(torch.Tensor(self.dict[word])).view(1,-1)
			if torch.cuda.is_available():
				output_word = output_word.cuda()
			if firstFlag:
				output, extra_loss=self.mix_cell(word, output_word, output_char)
				output2 = output_char
				firstFlag = False
			else:
				output_new, extra_loss = self.mix_cell(word, output_word, output_char)
				output_new2 = output_char
				output = torch.cat((output, output_new), 0)
				output2 = torch.cat((output2, output_new2), 0)
		sentB = output
		sentB2 = output2
		sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)#*2)
		sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)#*2)
		sentA2 = torch.unsqueeze(sentA2, 0).view(-1, 1, self.embedding_dim)  # *2)
		sentB2 = torch.unsqueeze(sentB2, 0).view(-1, 1, self.embedding_dim)  # *2)
		return (sentA, sentA2, sentB, sentB2, extra_loss)

	def forward(self,input_A, input_B, index):
		extra_loss1=0
		extra_loss2=0
		raw_input_A=input_A
		raw_input_B=input_B
		if torch.cuda.is_available():
			h0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim)).cuda() # 2 for bidirection
			#initial_h0 = Variable(torch.cat((input_A.data[0][0].view(1,300),input_A.data[-1][0].view(1,300)),0).view(2,1,300)).cuda()
			#initial_c0 = Variable(torch.cat((input_A.data[0][0].view(1,300),input_A.data[-1][0].view(1,300)),0).view(2,1,300)).cuda()
			c0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim)).cuda()
		else:
			h0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
			c0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
		if self.granularity=='word':
			input_A, input_B = self.word_layer(input_A, input_B)
		elif self.granularity=='char':
			input_A, input_B=self.c2w_or_cnn_layer(input_A,input_B)
			if self.lm_mode:
				target_A = []  # [self.tokens.index(word) for word in input_A]
				for word in raw_input_A:
					if self.word_freq[word] >= 4:
						target_A.append(self.tokens.index(word))
					else:
						target_A.append(self.tokens.index('oov'))
				target_B = []  # [self.tokens.index(word) for word in input_B]
				for word in raw_input_B:
					if self.word_freq[word] >= 4:
						target_B.append(self.tokens.index(word))
					else:
						target_B.append(self.tokens.index('oov'))
				lm_out0, _ = self.lm_lstm(input_A, (h0, c0))
				lm_out1, _ = self.lm_lstm(input_B, (h0, c0))
				extra_loss2 = self.language_model(lm_out0, lm_out1, target_A, target_B)
		elif self.granularity=='mix':
			input_A, input_A2, input_B, input_B2, extra_loss1 = self.mix_layer(input_A, input_B)
			if self.lm_mode:
				target_A = []  # [self.tokens.index(word) for word in input_A]
				for word in raw_input_A:
					if self.word_freq[word] >= 4:
						target_A.append(self.tokens.index(word))
					else:
						target_A.append(self.tokens.index('oov'))
				target_B = []  # [self.tokens.index(word) for word in input_B]
				for word in raw_input_B:
					if self.word_freq[word] >= 4:
						target_B.append(self.tokens.index(word))
					else:
						target_B.append(self.tokens.index('oov'))
				lm_out0, _ = self.lm_lstm(input_A2, (h0, c0))
				lm_out1, _ = self.lm_lstm(input_B2, (h0, c0))
				extra_loss2 = self.language_model(lm_out0, lm_out1, target_A, target_B)

		out0, (state0,_) = self.lstm(input_A, (h0, c0))
		out1, (state1,_) = self.lstm(input_B, (h0, c0))
		simCube, _=self.pairwise_word_interaction(out0,out1, target_A=None, target_B=None)
		focusCube=self.similarity_focus(simCube)
		if self.deep_CNN:
			output = self.deep_cnn(focusCube)
		else:
			output = self.mlp(focusCube)
		output=output.view(1,2)
		return (output,extra_loss1+ extra_loss2)

