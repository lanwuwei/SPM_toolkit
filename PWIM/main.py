#Author: Wuwei Lan

from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchtext.vocab import load_word_vectors
import random
import time
from datetime import datetime
from datetime import timedelta
import pickle
import numpy
from model import DeepPairWiseWord
from util import *
import argparse
import os

def main(args):
	#torch.manual_seed(123)
	EMBEDDING_DIM = 200
	HIDDEN_DIM = 250
	num_epochs = 20
	task=args.task
	granularity=args.granularity
	dict={}
	dict_char_ngram={}
	word_freq={}
	fake_dict={}
	oov=[]
	feature_maps = [50, 100, 150, 200, 200, 200, 200]
	kernels = [1, 2, 3, 4, 5, 6, 7]
	charcnn_embedding_size = 15
	max_word_length = 20
	c2w_mode = False
	character_ngrams = 3
	character_ngrams_2 = None
	character_ngrams_overlap = False
	glove_mode = None
	update_inv_mode = None
	update_oov_mode = None
	combine_mode=None
	lm_mode=None
	word_mode = (glove_mode, update_inv_mode, update_oov_mode)

	basepath= os.path.dirname(os.path.abspath(__file__))

	if task=='url':
		num_class = 2
		trainset = readURLdata(basepath+'/data/url/train/',granularity)
		testset = readURLdata(basepath + '/data/url/test/', granularity)
	elif task=='msrp':
		num_class = 2
		trainset = readURLdata(basepath+'/data/msrp/train/', granularity)
		testset = readURLdata(basepath + '/data/msrp/test/', granularity)
	elif task=='pit':
		num_class = 2
		trainset = readPITdata(basepath+'/data/pit/train/', granularity)
		testset = readPITdata(basepath+'/data/pit/test/', granularity)
	else:
		print('wrong input for the first argument!')
		sys.exit()

	if granularity=='char':
		# charcnn parameters
		feature_maps = [50, 100, 150, 200, 200, 200, 200]
		kernels = [1, 2, 3, 4, 5, 6, 7]
		charcnn_embedding_size = 15
		max_word_length = 20

		# c2w parameters
		if args.language_model:
			lm_mode = True
		else:
			lm_mode = False
		if args.char_assemble=='c2w':
			c2w_mode = True
		else:
			c2w_mode = False
		character_ngrams = args.char_ngram
		character_ngrams_overlap = False

		#tokens = []
		#for line in open(basepath + '/data/' + task + '/vocab.txt'):
		#	tokens.append(line.strip())
		tokens=set()
		lsents, rsents, labels = trainset
		for sent in lsents:
			for word in sent:
				tokens.add(word)
		for sent in rsents:
			for word in sent:
				tokens.add(word)
		lsents, rsents, labels = testset
		for sent in lsents:
			for word in sent:
				tokens.add(word)
		for sent in rsents:
			for word in sent:
				tokens.add(word)
		tokens=list(tokens)
		org_tokens = tokens[:]
		tokens.append('<s>')
		tokens.append('</s>')
		tokens.append('oov')
		# word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
		word_freq = {}
		files = ['/train/a.toks', '/train/b.toks', '/test/a.toks', '/test/b.toks']
		for filename in files:
			for line in open(basepath + '/data/' + task + filename):
				line = line.strip()
				for word in line.split():
					# if word not in oov:
					try:
						word_freq[word] += 1
					except:
						word_freq[word] = 1
		if c2w_mode:
			EMBEDDING_DIM = 200
		else:
			EMBEDDING_DIM = 1100
		if character_ngrams == 1:
			# dict_char_ngram = pickle.load(open(base_path+ '/char_dict.p', "rb"))
			dict_char_ngram = set()
			for word in tokens:
				for i in range(len(word)):
					dict_char_ngram.add(word[i])
			ngrams_list = list(dict_char_ngram)
			dict_char_ngram = {}
			count = 0
			for unit in ngrams_list:
				dict_char_ngram[unit] = count
				count += 1
			pickle.dump(dict_char_ngram, open( "./saved_dir/dict_char_unigram.p", "wb" ) )
		elif character_ngrams == 2 and character_ngrams_overlap:
			# dict_char_ngram = pickle.load(open(base_path+ '/bigram_dict.p', "rb"))
			dict_char_ngram = set()
			for word in tokens:
				if len(word) <= 2:
					dict_char_ngram.add(word)
				else:
					for i in range(len(word) - 1):
						dict_char_ngram.add(word[i:i + 2])
			ngrams_list = list(dict_char_ngram)
			dict_char_ngram = {}
			count = 0
			for unit in ngrams_list:
				dict_char_ngram[unit] = count
				count += 1
		elif character_ngrams == 2 and not character_ngrams_overlap:
			# dict_char_ngram = pickle.load(open(base_path+ '/bigram_dict_no_overlap.p', "rb"))
			dict_char_ngram = set()
			for word in tokens:
				if len(word) <= 2:
					dict_char_ngram.add(word)
				else:
					for i in range(0, len(word) - 1, 2):
						dict_char_ngram.add(word[i:i + 2])
					if len(word) % 2 == 1:
						dict_char_ngram.add(word[len(word) - 1])
			ngrams_list = list(dict_char_ngram)
			dict_char_ngram = {}
			count = 0
			for unit in ngrams_list:
				dict_char_ngram[unit] = count
				count += 1
		elif character_ngrams == 3 and character_ngrams_overlap:
			# dict_char_ngram = pickle.load(open(base_path+ '/trigram_dict.p', "rb"))
			dict_char_ngram = set()
			for word in tokens:
				if len(word) <= 3:
					dict_char_ngram.add(word)
				else:
					for i in range(len(word) - 2):
						dict_char_ngram.add(word[i:i + 3])
			ngrams_list = list(dict_char_ngram)
			dict_char_ngram = {}
			count = 0
			for unit in ngrams_list:
				dict_char_ngram[unit] = count
				count += 1
		elif character_ngrams == 3 and not character_ngrams_overlap:
			# dict_char_ngram = pickle.load(open(base_path+ '/trigram_dict_no_overlap.p', "rb"))
			dict_char_ngram = set()
			for word in tokens:
				if len(word) <= 3:
					dict_char_ngram.add(word)
				else:
					for i in range(0, len(word) - 2, 3):
						dict_char_ngram.add(word[i:i + 3])
					if len(word) % 3 == 1:
						dict_char_ngram.add(word[len(word) - 1])
					elif len(word) % 3 == 2:
						dict_char_ngram.add(word[len(word) - 2:])
			ngrams_list = list(dict_char_ngram)
			dict_char_ngram = {}
			count = 0
			for unit in ngrams_list:
				dict_char_ngram[unit] = count
				count += 1
		dict_char_ngram[' '] = len(dict_char_ngram)
		print('current task: ' + task + ', lm mode: ' + str(lm_mode) + ', c2w mode: ' + str(c2w_mode) + ', n = ' + str(
			character_ngrams) + ', overlap = ' + str(character_ngrams_overlap) + '.')
	elif granularity == 'word':
		tokens = []
		count = 0
		num_inv = 0
		num_oov = 0
		if args.pretrained:
			glove_mode = True
		else:
			glove_mode = False
		update_inv_mode = False
		update_oov_mode = False
		word_mode = (glove_mode, update_inv_mode, update_oov_mode)
		if task == 'msrp':
			#for line in open(basepath + '/data/' + task + '/vocab.txt'):
			#	tokens.append(line.strip())
			tokens=set()
			lsents, rsents, labels = trainset
			for sent in lsents:
				for word in sent:
					tokens.add(word)
			for sent in rsents:
				for word in sent:
					tokens.add(word)
			lsents, rsents, labels = testset
			for sent in lsents:
				for word in sent:
					tokens.add(word)
			for sent in rsents:
				for word in sent:
					tokens.add(word)
			tokens=list(tokens)
			tokens.append('oov')
			dict = {}
			EMBEDDING_DIM = 300
			# wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
			wv_dict, wv_arr, wv_size = load_word_vectors(expanduser("~")+'/Documents/research/pytorch/DeepPairWiseWord' + '/VDPWI-NN-Torch/data/glove', 'glove.840B',
			                                             EMBEDDING_DIM)
			# wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
			# wv_dict={}
			# wv_arr={}
			for word in tokens:
				fake_dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
				try:
					dict[word] = wv_arr[wv_dict[word]]
					num_inv += 1
				except:
					num_oov += 1
					# print(word)
					oov.append(word)
					dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
		elif task == 'url' or task == 'pit':
			for line in open(basepath + '/data/' + task + '/vocab.txt'):
				tokens.append(line.strip())
			# print(len(tokens))
			tokens.append('oov')
			dict = {}
			EMBEDDING_DIM = 200
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove',
			                                             'glove.twitter.27B', EMBEDDING_DIM)
			num_oov = 0
			num_inv = 0
			for word in tokens:
				fake_dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
				try:
					dict[word] = wv_arr[wv_dict[word]]
					num_inv += 1
				except:
					num_oov += 1
					oov.append(word)
					dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/char_dict.p', "rb"))
			word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
		print('finished loading word vector, there are ' + str(num_inv) + ' INV words and ' + str(
			num_oov) + ' OOV words.')
		print('current task: ' + task + ', glove mode = ' + str(glove_mode) + ', update_inv_mode = ' + str(
			update_inv_mode) + ', update_oov_mode = ' + str(update_oov_mode))
		saved_file = 'current task: ' + task + ', glove mode = ' + str(glove_mode) + ', update_inv_mode = ' + str(
			update_inv_mode) + ', update_oov_mode = ' + str(update_oov_mode) + '.txt'
	else:
		print('wrong input for the second argument!')
		sys.exit()

	model=DeepPairWiseWord(EMBEDDING_DIM,HIDDEN_DIM,1,task,granularity,num_class,dict,fake_dict, dict_char_ngram, oov,tokens, word_freq,
	                       feature_maps,kernels,charcnn_embedding_size,max_word_length,character_ngrams,c2w_mode,character_ngrams_overlap, word_mode,
	                       combine_mode, lm_mode, args.deep_CNN)#, corpus)
	if torch.cuda.is_available():
		model=model.cuda()
	lsents, rsents, labels = trainset
	criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=True)
	if torch.cuda.is_available():
		criterion = criterion.cuda()
	optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)#, momentum=0.1, weight_decay=0.05)#,momentum=0.9,weight_decay=0.95)
	#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	# Train the Model
	#print(oov)
	print('start training')
	max_result=-1
	batch_size=32
	report_interval=50000
	for epoch in range(num_epochs):
		print('--'*20)
		model.train()
		optimizer.zero_grad()
		start_time = time.time()
		data_loss = 0
		indices = torch.randperm(len(lsents))
		train_correct=0
		#print(len(indices))
		for index, i in enumerate(indices):
			#print(index)
			#start_time = time.time()
			sentA = lsents[i]
			sentB = rsents[i]
			if task=='sick' or task=='sts' or task=='snli' or task=='wiki':
				label=Variable(torch.Tensor(labels[i]))
			else:
				label = Variable(torch.LongTensor(labels[i]))#.cuda()
			if torch.cuda.is_available():
				label=label.cuda()
			output,extra_loss = model(sentA, sentB, index)
			#tmp_output = np.exp(output.data[0].cpu().numpy())
			#print index, 'gold: ', labels[i][0], 'predict: ', np.argmax(tmp_output)
			#print(extra_loss)
			loss = criterion(output, label)+extra_loss
			loss.backward()
			data_loss += loss.data[0]
			output = np.exp(output.data[0].cpu().numpy())
			if labels[i][0] == np.argmax(output):
				train_correct += 1
			#print(loss-extra_loss)
			#print('*'*20)
			if (index+1) % batch_size == 0:
				optimizer.step()
				optimizer.zero_grad()

			if (index+1) % report_interval == 0:
				msg = '%d completed epochs, %d batches' % (epoch, index+1)
				msg += '\t train batch loss: %f' % (data_loss / (index+1))
				train_acc=train_correct/(index+1)
				print(msg)

			if (index + 1) % (int(len(lsents)/2)) == 0:
				model.eval()
				# test on URL dataset
				#print('testing on URL dataset:')
				#testset = readURLdata(basepath + '/data/url/test_9324/', granularity)
				test_lsents, test_rsents, test_labels = testset
				predicted = []
				gold = []
				correct = 0
				for test_i in range(len(test_lsents)):
					sentA = test_lsents[test_i]
					sentB = test_rsents[test_i]
					output, _ = model(sentA, sentB, index)
					output = np.exp(output.data[0].cpu().numpy())
					if test_labels[test_i][0] == np.argmax(output):
						correct += 1
					predicted.append(output[1])
					gold.append(test_labels[test_i][0])
				_,result=URL_maxF1_eval(predict_result=predicted, test_data_label=gold)
				if result > max_result:
					max_result = result
					torch.save(model,'./saved_dir/char_CNN_unigram.pkl')
				elapsed_time = time.time() - start_time
				print('Epoch ' + str(epoch + 1) + ' finished within ' + str(timedelta(seconds=elapsed_time))+', and current time:'+ str(datetime.now()))
				print('Best result until now: %.6f' % max_result)
				model.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, default='msrp',
						help='Currently supported tasks: pit, url, msrp')
	parser.add_argument('--granularity', type=str, default='word',
						help='Currently supported granularities: char and word.')
	parser.add_argument('--pretrained', type=bool, default=True,
	                    help='Use pretrained word embedding or not')
	parser.add_argument('--char_ngram', type=int, default=1,
	                    help='unigram (1), bigram (2) or trigram (3)')
	parser.add_argument('--char_assemble', type=str, default='cnn',
	                    help='Assemble char embedding into word embedding: c2w or cnn')
	parser.add_argument('--language_model', type=bool, default=False,
	                    help='Use multi task language model or not')
	parser.add_argument('--deep_CNN', type=bool, default=True,
	                    help='use 19 layer CNN or not')
	args = parser.parse_args()
	print(args)
	main(args)
