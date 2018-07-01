from __future__ import division
import torch
import os
import random
from model import *
from torch import optim
import torch.nn as nn
from datetime import datetime
from torch_util import *
import tqdm
import data_loader
import sys
import time
from datetime import timedelta
from os.path import expanduser
from torchtext.vocab import load_word_vectors

def create_batch(data,from_index, to_index):
	if to_index>len(data):
		to_index=len(data)
	lsize=0
	rsize=0
	lsize_list=[]
	rsize_list=[]
	for i in range(from_index, to_index):
		length=len(data[i][0])+2
		lsize_list.append(length)
		if length>lsize:
			lsize=length
		length=len(data[i][1])+2
		rsize_list.append(length)
		if length>rsize:
			rsize=length
	#lsize+=1
	#rsize+=1
	lsent = data[from_index][0]
	lsent = ['bos']+lsent + ['oov' for k in range(lsize -1 - len(lsent))]
	#print(lsent)
	left_sents = [[word2id[word] for word in lsent]]
	#left_sents = torch.cat((dict[word].view(1, -1) for word in lsent))
	#left_sents = torch.unsqueeze(left_sents,0)

	rsent = data[from_index][1]
	rsent = ['bos']+rsent + ['oov' for k in range(rsize -1 - len(rsent))]
	#print(rsent)
	right_sents = [[word2id[word] for word in rsent]]
	#right_sents = torch.cat((dict[word].view(1, -1) for word in rsent))
	#right_sents = torch.unsqueeze(right_sents,0)

	labels=[data[from_index][2]]

	for i in range(from_index+1, to_index):

		lsent=data[i][0]
		lsent=['bos']+lsent+['oov' for k in range(lsize -1 - len(lsent))]
		#print(lsent)
		left_sents.append([word2id[word] for word in lsent])
		#left_sent = torch.cat((dict[word].view(1,-1) for word in lsent))
		#left_sent = torch.unsqueeze(left_sent, 0)
		#left_sents = torch.cat([left_sents, left_sent])

		rsent=data[i][1]
		rsent=['bos']+rsent+['oov' for k in range(rsize -1 - len(rsent))]
		#print(rsent)
		right_sents.append([word2id[word] for word in rsent])
		#right_sent = torch.cat((dict[word].view(1,-1) for word in rsent))
		#right_sent = torch.unsqueeze(right_sent, 0)
		#right_sents = torch.cat((right_sents, right_sent))

		labels.append(data[i][2])

	left_sents=Variable(torch.LongTensor(left_sents))
	right_sents=Variable(torch.LongTensor(right_sents))
	labels=Variable(torch.LongTensor(labels))
	lsize_list=torch.LongTensor(lsize_list)
	rsize_list =torch.LongTensor(rsize_list)

	if torch.cuda.is_available():
		left_sents=left_sents.cuda()
		right_sents=right_sents.cuda()
		labels=labels.cuda()
		lsize_list=lsize_list.cuda()
		rsize_list=rsize_list.cuda()
	#print(left_sents)
	#print(right_sents)
	return left_sents, right_sents, labels, lsize_list, rsize_list

if __name__ == '__main__':
	task='quora'
	print('task: '+task)
	print('model: SSE')
	torch.manual_seed(6)

	num_class = 2
	if torch.cuda.is_available():
		print('CUDA is available!')
		basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/quora'
		embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
	else:
		basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/quora'
		embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
	train_pairs = readQuoradata(basepath + '/train/')
	dev_pairs = readQuoradata(basepath + '/dev/')
	test_pairs = readQuoradata(basepath + '/test/')

	print('# of train pairs: %d' % len(train_pairs))
	print('# of dev pairs: %d' % len(dev_pairs))
	print('# of test pairs: %d' % len(test_pairs))

	tokens = []
	dict={}
	word2id={}
	vocab = set()
	for pair in train_pairs:
		left = pair[0]
		right = pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in dev_pairs:
		left = pair[0]
		right = pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in test_pairs:
		left = pair[0]
		right = pair[1]
		vocab |= set(left)
		vocab |= set(right)
	tokens=list(vocab)
	#for line in open(basepath + '/vocab.txt'):
	#	tokens.append(line.strip().decode('utf-8'))
	wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', 300)
	#embedding = []
	tokens.append('oov')
	tokens.append('bos')
	pretrained_emb = np.zeros(shape=(len(tokens), 300))
	id=0
	for word in tokens:
		try:
			dict[word] = wv_arr[wv_dict[word]]
		except:
			dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(300)])
		word2id[word]=id
		pretrained_emb[id] = dict[word].numpy()
		id+=1

	model = StackBiLSTMMaxout(h_size=[512, 1024, 2048], v_size=10, d=300, mlp_d=1600, dropout_r=0.1, max_l=60, num_class=num_class)
	if torch.cuda.is_available():
		pretrained_emb=torch.Tensor(pretrained_emb).cuda()
	else:
		pretrained_emb = torch.Tensor(pretrained_emb)
	model.Embd.weight.data = pretrained_emb

	if torch.cuda.is_available():
		model.cuda()

	start_lr = 2e-4
	batch_size=32
	report_interval=10000
	optimizer = optim.Adam(model.parameters(), lr=start_lr)
	criterion = nn.CrossEntropyLoss()

	iterations = 0

	best_dev_loss = 10e10

	model=torch.load(basepath + '/model_SSE_' + task + '.pkl')
	test_batch_index = 0
	test_num_correct = 0
	#msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
	accumulated_loss = 0
	test_batch_i = 0
	pred = []
	gold = []
	while test_batch_i < len(test_pairs):
		left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(test_pairs, test_batch_i,
		                                                                       test_batch_i + batch_size)
		left_sents = torch.transpose(left_sents, 0, 1)
		right_sents = torch.transpose(right_sents, 0, 1)
		test_batch_i += len(labels)
		output = F.softmax(model(left_sents, lsize_list, right_sents, rsize_list))
		result = output.data.cpu().numpy()
		loss = criterion(output, labels)
		accumulated_loss += loss.data[0]
		a = np.argmax(result, axis=1)
		b = labels.data.cpu().numpy()
		test_num_correct += np.sum(a == b)
		pred.extend(result[:,1])
	msg = '\t test loss: %f' % accumulated_loss
	test_acc = test_num_correct / len(test_pairs)
	msg += '\t test accuracy: %f' % test_acc
	print(msg)
	with open(basepath+'/quora_SSE_prob.txt','w') as f:
		for item in pred:
			f.writelines(str(1 - item)+'\t'+str(item)+'\n')
	sys.exit()
	print('start training...')
	for epoch in range(100):
		batch_counter = 0
		accumulated_loss = 0
		print('--' * 20)
		start_time = time.time()
		i_decay = epoch / 2
		lr = start_lr / (2 ** i_decay)
		print(lr)
		train_pairs = np.array(train_pairs)
		rand_idx = np.random.permutation(len(train_pairs))
		train_pairs = train_pairs[rand_idx]
		train_batch_i = 0
		train_num_correct=0
		train_sents_scaned = 0
		while train_batch_i < len(train_pairs):
			model.train()
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(train_pairs, train_batch_i, train_batch_i+batch_size)
			train_sents_scaned += len(labels)
			train_batch_i+=len(labels)
			left_sents=torch.transpose(left_sents,0,1)
			right_sents=torch.transpose(right_sents,0,1)
			output=model(left_sents,lsize_list,right_sents,rsize_list)
			result = output.data.cpu().numpy()
			a = np.argmax(result, axis=1)
			b = labels.data.cpu().numpy()
			train_num_correct += np.sum(a == b)
			loss = criterion(output, labels)
			optimizer.zero_grad()
			for pg in optimizer.param_groups:
				pg['lr'] = lr
			loss.backward()
			optimizer.step()
			batch_counter += 1
			accumulated_loss += loss.data[0]
			if batch_counter % report_interval == 0:
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t train batch loss: %f' % accumulated_loss
				msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
				print(msg)
				# valid after each report interval
				model.eval()
				dev_batch_index = 0
				dev_num_correct = 0
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				accumulated_loss = 0
				dev_batch_i = 0
				pred=[]
				gold=[]
				while dev_batch_i < len(dev_pairs):
					left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(dev_pairs, dev_batch_i,
					                                                                       dev_batch_i+batch_size)
					dev_batch_i += len(labels)
					left_sents = torch.transpose(left_sents, 0, 1)
					right_sents = torch.transpose(right_sents, 0, 1)
					output = model(left_sents, lsize_list,right_sents, rsize_list)
					result = output.data.cpu().numpy()
					loss = criterion(output, labels)
					accumulated_loss += loss.data[0]
					a = np.argmax(result, axis=1)
					b = labels.data.cpu().numpy()
					dev_num_correct += np.sum(a == b)
				msg += '\t dev loss: %f' % accumulated_loss
				dev_acc = dev_num_correct / len(dev_pairs)
				msg += '\t dev accuracy: %f' % dev_acc
				print(msg)
				if accumulated_loss < best_dev_loss:
					best_dev_loss=accumulated_loss
					torch.save(model, basepath+'/model_SSE_'+task+'.pkl')
					test_batch_index = 0
					test_num_correct = 0
					msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
					accumulated_loss = 0
					test_batch_i = 0
					pred = []
					gold = []
					while test_batch_i < len(test_pairs):
						left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(test_pairs, test_batch_i, test_batch_i+batch_size)
						left_sents = torch.transpose(left_sents, 0, 1)
						right_sents = torch.transpose(right_sents, 0, 1)
						test_batch_i+=len(labels)
						output = model(left_sents, lsize_list, right_sents, rsize_list)
						result = output.data.cpu().numpy()
						loss = criterion(output, labels)
						accumulated_loss += loss.data[0]
						a = np.argmax(result, axis=1)
						b = labels.data.cpu().numpy()
						test_num_correct += np.sum(a == b)
					msg += '\t test loss: %f' % accumulated_loss
					test_acc = test_num_correct / len(test_pairs)
					msg += '\t test accuracy: %f' % test_acc
					print(msg)
		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))
