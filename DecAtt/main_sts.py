from __future__ import division

from datetime import datetime
from torch.autograd import Variable
from model import *
import argparse
import sys
from util import *
import time
import torch
import random
import logging
import numpy as np
import cPickle as pickle
from datetime import timedelta
from os.path import expanduser
from torch.autograd import Variable
from torchtext.vocab import load_word_vectors

def prepare_data(pairs):
	batch_list = []
	batch_index = 0
	while batch_index < len(pairs):
		try:
			subset = pairs[batch_index:batch_index + batch_size]
		except:
			subset = pairs[batch_index:]
		tmp_a = np.array([len(item[0]) for item in subset])
		tmp_b = np.array([len(item[1]) for item in subset])
		batch_index2 = batch_index + min(len(np.where(tmp_a == tmp_a[0])[0]), len(np.where(tmp_b == tmp_b[0])[0]))
		batch_list.append([batch_index, batch_index2])
		batch_index = batch_index2
	return batch_list

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
	left_sents = torch.cat((dict[word].view(1, -1) for word in lsent))
	left_sents = torch.unsqueeze(left_sents,0)

	rsent = data[from_index][1]
	rsent = ['bos']+rsent + ['oov' for k in range(rsize -1 - len(rsent))]
	#print(rsent)
	right_sents = torch.cat((dict[word].view(1, -1) for word in rsent))
	right_sents = torch.unsqueeze(right_sents,0)

	labels=[data[from_index][2]]

	for i in range(from_index+1, to_index):

		lsent=data[i][0]
		lsent=['bos']+lsent+['oov' for k in range(lsize -1 - len(lsent))]
		#print(lsent)
		left_sent = torch.cat((dict[word].view(1,-1) for word in lsent))
		left_sent = torch.unsqueeze(left_sent, 0)
		left_sents = torch.cat([left_sents, left_sent])

		rsent=data[i][1]
		rsent=['bos']+rsent+['oov' for k in range(rsize -1 - len(rsent))]
		#print(rsent)
		right_sent = torch.cat((dict[word].view(1,-1) for word in rsent))
		right_sent = torch.unsqueeze(right_sent, 0)
		right_sents = torch.cat((right_sents, right_sent))

		labels.append(data[i][2])

	left_sents=Variable(left_sents)
	right_sents=Variable(right_sents)
	if task=='sts':
		labels=Variable(torch.Tensor(labels))
	else:
		labels=Variable(torch.LongTensor(labels))
	lsize_list=Variable(torch.LongTensor(lsize_list))
	rsize_list = Variable(torch.LongTensor(rsize_list))

	if torch.cuda.is_available():
		left_sents=left_sents.cuda()
		right_sents=right_sents.cuda()
		labels=labels.cuda()
		lsize_list=lsize_list.cuda()
		rsize_list=rsize_list.cuda()
	#print(left_sents)
	#print(right_sents)
	#print(labels)
	return left_sents, right_sents, labels, lsize_list, rsize_list

if __name__ == '__main__':
	task='sts'
	print('task: '+task)
	EMBEDDING_DIM = 300
	PROJECTED_EMBEDDING_DIM = 300

	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--e', dest='num_epochs', default=5000, type=int, help='Number of epochs')
	parser.add_argument('--b', dest='batch_size', default=32, help='Batch size', type=int)
	parser.add_argument('--u', dest='num_units', help='Number of hidden units', default=100, type=int)
	parser.add_argument('--r', help='Learning rate', type=float, default=0.05, dest='rate')
	parser.add_argument('--lower', help='Lowercase the corpus', default=True, action='store_true')
	parser.add_argument('--model', help='Model selection', default='DecAtt', type=str)
	parser.add_argument('--optim', help='Optimizer algorithm', default='adagrad', choices=['adagrad', 'adadelta', 'adam'])
	parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
									   to have the norm equal to max_grad_norm', type=float, default=5)
	if task=='snli':
		num_class=3
		parser.add_argument('--train', help='JSONL or TSV file with training corpus', default='/data/snli_1.0_train.jsonl')
		parser.add_argument('--dev', help='JSONL or TSV file with development corpus', default='/data/snli_1.0_dev.jsonl')
		parser.add_argument('--test', help='JSONL or TSV file with testing corpus', default='/data/snli_1.0_test.jsonl')
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DecAtt/data/snli'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DecAtt/data/snli'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		train_pairs = pickle.load(open(basepath + "/train_pairs.p", "rb"))
		dev_pairs = pickle.load(open(basepath + "/dev_pairs.p", "rb"))
		test_pairs = pickle.load(open(basepath + "/test_pairs.p", "rb"))
	elif task=='mnli':
		num_class = 3
		parser.add_argument('--train', help='JSONL or TSV file with training corpus', default='/data/mnli/multinli_1.0_train.jsonl')
		parser.add_argument('--dev_m', help='JSONL or TSV file with development corpus', default='/data/mnli/multinli_1.0_dev_matched.jsonl')
		parser.add_argument('--dev_um', help='JSONL or TSV file with development corpus',
		                   default='/data/mnli/multinli_1.0_dev_mismatched.jsonl')
		parser.add_argument('--test_m', help='JSONL or TSV file with testing corpus', default='/data/mnli/multinli_1.0_test_matched.jsonl')
		parser.add_argument('--test_um', help='JSONL or TSV file with testing corpus',
		                    default='/data/mnli/multinli_1.0_test_mismatched.jsonl')
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DecAtt/data/mnli'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DecAtt/data/mnli'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		'''
		train_pairs = util.read_corpus(expanduser("~")+'/Documents/research/pytorch/DeepPairWiseWord' + args.train, True)
		dev_pairs_m = util.read_corpus(expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord' + args.dev_m, True)
		dev_pairs_um = util.read_corpus(expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord' + args.dev_um, True)
		test_pairs_m = util.read_corpus(expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord' + args.test_m, True)
		test_pairs_um = util.read_corpus(expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord' + args.test_um, True)
		pickle.dump(train_pairs, open("data/mnli/train_pairs.p", "wb"))
		pickle.dump(dev_pairs_m, open("data/mnli/dev_pairs_m.p", "wb"))
		pickle.dump(dev_pairs_um, open("data/mnli/dev_pairs_um.p", "wb"))
		pickle.dump(test_pairs_m, open("data/mnli/test_pairs_m.p", "wb"))
		pickle.dump(test_pairs_um, open("data/mnli/test_pairs_um.p", "wb"))
		'''
		train_pairs = pickle.load(open(basepath + "/train_pairs.p", "rb"))
		dev_pairs_m = pickle.load(open(basepath + "/dev_pairs_m.p", "rb"))
		dev_pairs_um = pickle.load(open(basepath + "/dev_pairs_um.p", "rb"))
		test_pairs_m = pickle.load(open(basepath + "/test_pairs_m.p", "rb"))
		test_pairs_um = pickle.load(open(basepath + "/test_pairs_um.p", "rb"))
	elif task=='quora':
		num_class = 2
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/quora'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/quora'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		train_pairs = readQuoradata(basepath+'/train/')
		dev_pairs=readQuoradata(basepath+'/dev/')
		test_pairs=readQuoradata(basepath+'/test/')
	elif task=='url':
		num_class = 2
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/url'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/url'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		train_pairs = readQuoradata(basepath + '/train/')
		dev_pairs = None#readQuoradata(basepath + '/dev/')
		test_pairs = readQuoradata(basepath + '/test_9324/')
		if dev_pairs==None:
			dev_pairs=test_pairs
	elif task=='pit':
		num_class = 2
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/pit'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/pit'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		train_pairs = readQuoradata(basepath + '/train/')
		test_pairs = readQuoradata(basepath + '/test/')
		dev_pairs=test_pairs
	elif task=='sts':
		num_class = 6
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/sts'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/sts'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		train_pairs = readSTSdata(basepath + '/train/')
		test_pairs = readSTSdata(basepath + '/test/')
		dev_pairs=test_pairs
	elif task=='wikiqa':
		num_class=2
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/wikiqa'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/wikiqa'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
			train_pairs = readQuoradata(basepath + '/train/')
			dev_pairs = readQuoradata(basepath + '/dev/')
			test_pairs = readQuoradata(basepath + '/test/')
	elif task=='trecqa':
		num_class = 2
		if torch.cuda.is_available():
			print('CUDA is available!')
			basepath = expanduser("~") + '/pytorch/DeepPairWiseWord/data/trecqa'
			embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
		else:
			basepath = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/trecqa'
			embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
			train_pairs = readQuoradata(basepath + '/train-all/')
			dev_pairs = readQuoradata(basepath + '/raw-dev/')
			test_pairs = readQuoradata(basepath + '/raw-test/')
	#sys.exit()
	args = parser.parse_args()
	print('Model: %s' % args.model)
	print('Read data ...')
	print('Number of training pairs: %d' % len(train_pairs))
	#print('Number of development pairs: %d' % len(dev_pairs))
	#print('Number of testing pairs: %d' % len(test_pairs))
	batch_size = args.batch_size
	num_epochs = args.num_epochs
	'''
	vocab=set()
	for pair in train_pairs:
		left=pair[0]
		right=pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in dev_pairs_m:
		left=pair[0]
		right=pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in dev_pairs_um:
		left=pair[0]
		right=pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in test_pairs_m:
		left=pair[0]
		right=pair[1]
		vocab |= set(left)
		vocab |= set(right)
	for pair in test_pairs_um:
		left=pair[0]
		right=pair[1]
		vocab |= set(left)
		vocab |= set(right)
	with open('vocab.txt', 'w') as f:
		for word in vocab:
			f.write(word.encode('utf-8')+'\n')
	sys.exit()
	'''
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
	wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', EMBEDDING_DIM)
	#embedding = []
	tokens.append('oov')
	tokens.append('bos')
	#embedding.append(dict[word].numpy())
	#print(len(embedding))
	#np.save('embedding',np.array(embedding))
	#sys.exit()
	pretrained_emb = np.zeros(shape=(len(tokens), EMBEDDING_DIM))
	oov={}
	for id in range(100):
		oov[id]=torch.normal(torch.zeros(EMBEDDING_DIM),std=1)
	id=0
	for word in tokens:
		try:
			dict[word] = wv_arr[wv_dict[word]]/torch.norm(wv_arr[wv_dict[word]])
		except:
			#if args.model=='DecAtt':
			#	dict[word]=oov[np.random.randint(100)]
			#else:
			dict[word] = torch.normal(torch.zeros(EMBEDDING_DIM),std=1)
		word2id[word]=id
		pretrained_emb[id] = dict[word].numpy()
		id+=1

	if task=='sts':
		criterion = nn.KLDivLoss()
	else:
		criterion = torch.nn.NLLLoss(size_average=True)
	#criterion = torch.nn.CrossEntropyLoss()
	model=DecAtt(200,num_class,len(tokens),EMBEDDING_DIM, PROJECTED_EMBEDDING_DIM, pretrained_emb)
	if torch.cuda.is_available():
		model = model.cuda()
		criterion = criterion.cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05, weight_decay=5e-5)

	print('Start training...')
	batch_counter = 0
	best_dev_loss=10e10
	best_dev_loss_m=10e10
	best_dev_loss_um=10e10
	accumulated_loss=0
	report_interval = 1000
	model.train()
	'''
	train_pairs = np.array(train_pairs)
	both_lengths = np.array([(len(train_pairs[i][0]), len(train_pairs[i][1])) for i in range(len(train_pairs))],
	                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	for epoch in range(num_epochs):
		rand_idx = np.random.permutation(len(train_pairs))
		train_pairs = train_pairs[rand_idx]
		both_lengths = both_lengths[rand_idx]
		sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
		train_pairs=train_pairs[sorted_lengths]
		both_lengths=both_lengths[sorted_lengths]
		print('--' * 20)
		batch_index = 0
		batch_counter=0
		while batch_index < len(train_pairs):
			batch_index2=batch_index+batch_size
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(train_pairs, batch_index,
			                                                                       batch_index2)
			batch_index = batch_index2
			optimizer.zero_grad()
			output = model(left_sents, right_sents, lsize_list, rsize_list)
			# print('forward finished'+str(datetime.now()))
			loss = criterion(output, labels)
			loss.backward()
			optimizer.step()
			# print('backword finished' + str(datetime.now()))
			batch_counter += 1
			# print(batch_counter, loss.data[0])
			accumulated_loss += loss.data[0]
			if batch_counter % report_interval == 0:
				model.eval()
				dev_batch_index = 0
				dev_num_correct = 0
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t training batch loss: %f' % accumulated_loss
				accumulated_loss = 0
				while dev_batch_index<len(test_pairs):
					dev_batch_index2=dev_batch_index+1
					left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(test_pairs, dev_batch_index,
					                                                                       dev_batch_index2)
					dev_batch_index=dev_batch_index2
					output = model(left_sents, right_sents, lsize_list, rsize_list)
					result = output.data.cpu().numpy()
					loss = criterion(output, labels)
					accumulated_loss += loss.data[0]
					a = np.argmax(result, axis=1)
					b = labels.data.cpu().numpy()
					dev_num_correct += np.sum(a == b)
				msg += '\t dev loss: %f' % accumulated_loss
				dev_acc = dev_num_correct / len(test_pairs)
				msg += '\t dev accuracy: %f' % dev_acc
				print(msg)
				accumulated_loss = 0
			model.train()
	'''
	train_pairs=np.array(train_pairs)
	rand_idx = np.random.permutation(len(train_pairs))
	train_pairs = train_pairs[rand_idx]
	both_lengths = np.array([(len(train_pairs[i][0]), len(train_pairs[i][1])) for i in range(len(train_pairs))],
	                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
	train_pairs = train_pairs[sorted_lengths]
	train_batch_list=prepare_data(train_pairs)


	#dev_pairs=np.array(dev_pairs)
	#both_lengths = np.array([(len(dev_pairs[i][0]), len(dev_pairs[i][1])) for i in range(len(dev_pairs))],
	#                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	#sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
	#dev_pairs = dev_pairs[sorted_lengths]
	dev_batch_list=prepare_data(dev_pairs)

	#test_pairs=np.array(test_pairs)
	#both_lengths = np.array([(len(test_pairs[i][0]), len(test_pairs[i][1])) for i in range(len(test_pairs))],
	#                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	#sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
	#test_pairs = test_pairs[sorted_lengths]
	test_batch_list=prepare_data(test_pairs)
	'''

	dev_pairs_m = np.array(dev_pairs_m)
	both_lengths = np.array([(len(dev_pairs_m[i][0]), len(dev_pairs_m[i][1])) for i in range(len(dev_pairs_m))],
	                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
	dev_pairs_m = dev_pairs_m[sorted_lengths]
	dev_batch_list_m = prepare_data(dev_pairs_m)

	dev_pairs_um = np.array(dev_pairs_um)
	both_lengths = np.array([(len(dev_pairs_um[i][0]), len(dev_pairs_um[i][1])) for i in range(len(dev_pairs_um))],
	                        dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
	sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
	dev_pairs_um = dev_pairs_um[sorted_lengths]
	dev_batch_list_um = prepare_data(dev_pairs_um)
	'''
	batch_index=0
	for epoch in range(num_epochs):
		batch_counter=0
		accumulated_loss = 0
		model.train()
		print('--' * 20)
		start_time = time.time()
		train_rand_i=np.random.permutation(len(train_batch_list))
		train_batch_i=0
		train_sents_scaned=0
		train_num_correct=0
		while train_batch_i<len(train_batch_list):
			batch_index, batch_index2=train_batch_list[train_rand_i[train_batch_i]]
			train_batch_i+=1
			#batch_index2=batch_index+batch_size
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(train_pairs, batch_index, batch_index2)
			train_sents_scaned+=len(labels)
			#print(lsize_list)
			#print(rsize_list)
			#batch_index=batch_index2
			optimizer.zero_grad()

			output = model(left_sents, right_sents, lsize_list, rsize_list)
			result = output.data.cpu().numpy()
			a = np.argmax(result, axis=1)
			b = labels.data.cpu().numpy()
			train_num_correct += np.sum(a == b)
			#print('forward finished'+str(datetime.now()))
			#print(output)
			#print(labels)
			#sys.exit()
			loss = criterion(output, labels)
			loss.backward()

			'''
			grad_norm = 0.
			#para_norm = 0.

			for m in model.modules():
				if isinstance(m, nn.Linear):
					# print(m)
					grad_norm += m.weight.grad.data.norm() ** 2
					#para_norm += m.weight.data.norm() ** 2
					if m.bias is not None:
						grad_norm += m.bias.grad.data.norm() ** 2
						#para_norm += m.bias.data.norm() ** 2

			grad_norm ** 0.5
			#para_norm ** 0.5

			try:
				shrinkage = args.max_grad_norm / grad_norm
			except:
				pass
			if shrinkage < 1:
				for m in model.modules():
					# print m
					if isinstance(m, nn.Linear):
						m.weight.grad.data = m.weight.grad.data * shrinkage
						if m.bias is not None:
							m.bias.grad.data = m.bias.grad.data * shrinkage
			'''
			optimizer.step()
			#print('backword finished' + str(datetime.now()))
			batch_counter += 1
			#print(batch_counter, loss.data[0])
			accumulated_loss += loss.data[0]
			if batch_counter % report_interval ==0:
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t train batch loss: %f' % (accumulated_loss/train_sents_scaned)
				#msg += '\t train accuracy: %f' % (train_num_correct/train_sents_scaned)
				print(msg)
		'''
		# valid after each epoch
		model.eval()
		dev_batch_index=0
		dev_num_correct=0
		msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
		accumulated_loss=0
		dev_batch_i = 0
		while dev_batch_i<len(dev_batch_list_m):
			dev_batch_index, dev_batch_index2 = dev_batch_list_m[dev_batch_i]
			dev_batch_i += 1
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(dev_pairs_m, dev_batch_index, dev_batch_index2)
			#left_sents, right_sents, labels = create_batch(dev_pairs, 0, len(dev_pairs))
			output = model(left_sents, right_sents, lsize_list, rsize_list)
			result=output.data.cpu().numpy()
			loss = criterion(output, labels)
			accumulated_loss += loss.data[0]
			a = np.argmax(result, axis=1)
			b = labels.data.cpu().numpy()
			dev_num_correct+=np.sum(a == b)
		msg += '\t dev_m loss: %f' % accumulated_loss
		dev_acc=dev_num_correct/len(dev_pairs_m)
		msg += '\t dev_m accuracy: %f' % dev_acc
		print(msg)
		if accumulated_loss < best_dev_loss_m:
			best_dev_loss_m=accumulated_loss
			test_batch_index = 0
			pred=[]
			while test_batch_index < len(test_pairs_m):
				left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(test_pairs_m, test_batch_index,
				                                                                       test_batch_index+1)
				test_batch_index+=1
				output = model(left_sents, right_sents, lsize_list, rsize_list)
				result = output.data.cpu().numpy()
				a = np.argmax(result, axis=1)
				pred.extend(a)
			with open(basepath + '/sub_m.csv', 'w+') as f:
				index = ['neutral','contradiction','entailment']
				f.write("pairID,gold_label\n")
				for i, k in enumerate(pred):
					f.write(str(i + 9847) + "," + index[k] + "\n")
			torch.save(model, basepath + '/model_DecAtt_m' + '.pkl')

		dev_batch_index=0
		dev_num_correct=0
		msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
		accumulated_loss=0
		dev_batch_i = 0
		while dev_batch_i<len(dev_batch_list_um):
			dev_batch_index, dev_batch_index2 = dev_batch_list_um[dev_batch_i]
			dev_batch_i += 1
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(dev_pairs_um, dev_batch_index, dev_batch_index2)
			#left_sents, right_sents, labels = create_batch(dev_pairs, 0, len(dev_pairs))
			output = model(left_sents, right_sents, lsize_list, rsize_list)
			result=output.data.cpu().numpy()
			loss = criterion(output, labels)
			accumulated_loss += loss.data[0]
			a = np.argmax(result, axis=1)
			b = labels.data.cpu().numpy()
			dev_num_correct+=np.sum(a == b)
		msg += '\t dev_um loss: %f' % accumulated_loss
		dev_acc=dev_num_correct/len(dev_pairs_um)
		msg += '\t dev_um accuracy: %f' % dev_acc
		print(msg)
		if accumulated_loss < best_dev_loss_um:
			best_dev_loss_um=accumulated_loss
			test_batch_index = 0
			pred=[]
			while test_batch_index < len(test_pairs_um):
				left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(test_pairs_um, test_batch_index,
				                                                                       test_batch_index+1)
				test_batch_index+=1
				output = model(left_sents, right_sents, lsize_list, rsize_list)
				result = output.data.cpu().numpy()
				a = np.argmax(result, axis=1)
				pred.extend(a)
			with open(basepath + '/sub_um.csv', 'w+') as f:
				index = ['neutral','contradiction','entailment']
				f.write("pairID,gold_label\n")
				for i, k in enumerate(pred):
					f.write(str(i) + "," + index[k] + "\n")
			torch.save(model, basepath + '/model_DecAtt_um' + '.pkl')
		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))
		'''
		# valid after each epoch
		model.eval()
		dev_batch_index = 0
		dev_num_correct = 0
		msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
		accumulated_loss = 0
		dev_batch_i = 0
		pred=[]
		gold=[]
		while dev_batch_i < len(dev_batch_list):
			dev_batch_index, dev_batch_index2 = dev_batch_list[dev_batch_i]
			dev_batch_i += 1
			left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(dev_pairs, dev_batch_index,
			                                                                       dev_batch_index2)
			# left_sents, right_sents, labels = create_batch(dev_pairs, 0, len(dev_pairs))
			output = model(left_sents, right_sents, lsize_list, rsize_list)
			result = np.exp(output.data.cpu().numpy())
			loss = criterion(output, labels)
			accumulated_loss += loss.data[0]
			a = np.argmax(result, axis=1)
			b = labels.data.cpu().numpy()
			dev_num_correct += np.sum(a == b)
			if task=='pit' or task=='url' or task=='wikiqa' or task=='trecqa':
				pred.extend(result[:,1])
				gold.extend(b)
			if task=='sts':
				pred.extend(0*result[:,0]+1*result[:,1]+2*result[:,2]+3*result[:,3]+4*result[:,4]+5*result[:,5])
				gold.extend(0*b[:,0]+1*b[:,1]+2*b[:,2]+3*b[:,3]+4*b[:,4]+5*b[:,5])
		msg += '\t dev loss: %f' % accumulated_loss
		dev_acc = dev_num_correct / len(dev_pairs)
		#msg += '\t dev accuracy: %f' % dev_acc
		print(msg)
		result1 = pearson(pred[0:450], gold[0:450])
		result2 = pearson(pred[450:750], gold[450:750])
		result3 = pearson(pred[750:1500], gold[750:1500])
		result4 = pearson(pred[1500:2250], gold[1500:2250])
		result5 = pearson(pred[2250:3000], gold[2250:3000])
		result6 = pearson(pred[3000:3750], gold[3000:3750])
		wt_mean = 0.12 * result1 + 0.08 * result2 + 0.2 * result3 + 0.2 * result4 + 0.2 * result5 + 0.2 * result6
		print('weighted pearson mean: %.6f' % wt_mean)
		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))
