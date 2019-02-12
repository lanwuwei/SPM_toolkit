#Author: Wuwei Lan
#This model is for this paper: Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement, by Hua He and Jimmy Lin

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
from os.path import expanduser
from gensim.models.keyedvectors import KeyedVectors
import cPickle

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

	if torch.cuda.is_available():
		basepath = expanduser("~") + '/pytorch/DeepPairWiseWord'
	else:
		basepath=expanduser("~")+'/Documents/research/pytorch/DeepPairWiseWord'

	if task=='url':
		num_class = 2
		trainset = readURLdata(basepath+'/data/url/train/',granularity)
		testset = readURLdata(basepath + '/data/url/test_9324/', granularity)
	elif task=='quora':
		num_class = 2
		trainset = readURLdata(basepath+'/data/quora/train/', granularity)
		testset = readURLdata(basepath + '/data/quora/test/', granularity)
	elif task=='msrp':
		num_class = 2
		trainset = readURLdata(basepath+'/data/msrp/train/', granularity)
		testset = readURLdata(basepath + '/data/msrp/test/', granularity)
	elif task=='sick':
		num_class = 5
		trainset = readSICKdata(basepath+'/data/sick/train/',granularity)
		devset = readSICKdata(basepath+'/data/sick/dev/',granularity)
		testset = readSICKdata(basepath+'/data/sick/test/',granularity)
	elif task=='pit':
		num_class = 2
		trainset = readPITdata(basepath+'/data/pit/train/', granularity)
		#devset = readPITdata(basepath+'/data/pit/dev/',granularity)
		testset = readPITdata(basepath+'/data/pit/test/', granularity)
	elif task=='hindi':
		num_class=2
		trainset = read_Hindi_data(basepath+'/data/hindi/train/', granularity)
		testset = read_Hindi_data(basepath + '/data/hindi/test/', granularity)
	elif task=='sts':
		num_class = 6
		trainset = readSTSdata(basepath+'/data/sts/train/',granularity)
		testset = readSTSdata(basepath+'/data/sts/test/',granularity)
	elif task=='snli':
		num_class = 3
		trainset = readSNLIdata(basepath+'/data/snli/train/',granularity)
		testset = readSNLIdata(basepath+'/data/snli/test/', granularity)
	elif task=='mnli':
		num_class = 3
		trainset = readMNLIdata(basepath+'/data/mnli/train/',granularity)
		devset_m = readMNLIdata(basepath+'/data/mnli/dev_m/', granularity)
		devset_um = readMNLIdata(basepath + '/data/mnli/dev_um/', granularity)
		testset_m = readMNLIdata(basepath + '/data/mnli/test_m/', granularity)
		testset_um = readMNLIdata(basepath + '/data/mnli/test_um/', granularity)
	elif task=='wiki':
		'''
		_name_to_id = {
        'counter-vandalism': 0,
        'fact-update': 1,
        'refactoring': 2,
        'copy-editing': 3,
        'other': 4,
        'wikification': 5,
        'vandalism': 6,
        'simplification': 7,
        'elaboration': 8,
        'verifiability': 9,
        'process': 10,
        'clarification': 11,
        'disambiguation': 12,
        'point-of-view': 13
    }
		'''
		num_class = 14
		data = pickle.load(open(basepath+"/data/wiki/data.cpickle", "rb"))
		left=[]
		right=[]
		label=[]
		id=[]
		for i in range(2976):
			id.append(data[i][0])
			label.append([int(item) for item in data[i][3][0]])
			left_sent=[item.encode('utf-8') for item in data[i][1][0]]
			right_sent=[item.encode('utf-8') for item in data[i][2][0]]
			shared=[]
			for item in left_sent:
				if item in right_sent:
					shared.append(item)
			for item in shared:
				if item in left_sent and item in right_sent:
					left_sent.remove(item)
					right_sent.remove(item)
			if len(left_sent) == 0:
				left_sent=['<EMPTY-EDIT>']
			if len(right_sent) == 0:
				right_sent=['<EMPTY-EDIT>']
			left.append(left_sent)
			right.append(right_sent)
			#print(left_sent)
			#print(right_sent)
			#print(id[0])
			#print('*'*20)
		trainset=(left, right, label)
		#sys.exit()
		left = []
		right = []
		label = []
		for i in range(2376, 2976):
			id.append(data[i][0])
			label.append([int(item) for item in data[i][3][0]])
			left_sent = [item.encode('utf-8') for item in data[i][1][0]]
			right_sent = [item.encode('utf-8') for item in data[i][2][0]]
			shared = []
			for item in left_sent:
				if item in right_sent:
					shared.append(item)
			for item in shared:
				if item in left_sent and item in right_sent:
					left_sent.remove(item)
					right_sent.remove(item)
			if len(left_sent) == 0:
				left_sent = ['<EMPTY-EDIT>']
			if len(right_sent) == 0:
				right_sent = ['<EMPTY-EDIT>']
			left.append(left_sent)
			right.append(right_sent)
		testset = (left, right, label)
	elif task=='wikiqa':
		num_class=2
		trainset=readURLdata(basepath+'/data/wikiqa/train/',granularity)
		testset=readURLdata(basepath+'/data/wikiqa/test/',granularity)
	elif task=='trecqa':
		num_class=2
		trainset=readURLdata(basepath+'/data/trecqa/train-all/',granularity)
		testset=readURLdata(basepath+'/data/trecqa/raw-test/',granularity)
	else:
		print('wrong input for the first argument!')
		sys.exit()

	if granularity=='word':
		tokens = []
		count=0
		num_inv=0
		num_oov=0
		glove_mode = True
		update_inv_mode = True
		update_oov_mode = True
		word_mode=(glove_mode,update_inv_mode,update_oov_mode)
		if task == 'sick' or task=='quora' or task=='msrp':
			for line in open(basepath+'/data/'+task+'/vocab.txt'):
				tokens.append(line.strip())
			dict = {}
			EMBEDDING_DIM = 300
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath+'/VDPWI-NN-Torch/data/glove', 'glove.840B', EMBEDDING_DIM)
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
			#wv_dict={}
			#wv_arr={}
			for word in tokens:
				fake_dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
				try:
					dict[word] = wv_arr[wv_dict[word]]
					num_inv += 1
				except:
					num_oov += 1
					#print(word)
					oov.append(word)
					dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
		elif task=='sts':
			for line in open(basepath+'/data/'+task+'/vocab.txt'):
				tokens.append(line.strip())
			dict = {}
			#EMBEDDING_DIM = 200
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
			#EMBEDDING_DIM = 300
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.840B', EMBEDDING_DIM)
			EMBEDDING_DIM = 300
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath+'/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
			#wv_dict={}
			#wv_arr={}
			#oov = []
			#for line in open(basepath + '/data/' + task + '/oov.txt'):
			#	line = line.strip()
			#	oov.append(line)
			#inv = []
			#for line in open(basepath + '/data/' + task + '/inv_14000.txt'):
			#	line = line.strip()
			#	inv.append(line)
			# count=len(oov)+len(inv)
			#inv = tokens
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
		elif task == 'snli' or task=='wikiqa' or task=='trecqa' or task=='mnli':
			for line in open(basepath+'/data/'+task+'/vocab.txt'):
				tokens.append(line.strip())
			dict = {}
			#EMBEDDING_DIM = 200
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
			EMBEDDING_DIM = 300
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.840B', EMBEDDING_DIM)
			#EMBEDDING_DIM = 300
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath+'/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
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
			#dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/char_dict.p', "rb"))
			#word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
			dict_char_ngram = {}
			word_freq= {}
		elif task == 'hindi':
			#words, embeddings = pickle.load(open(basepath+'/data/hindi/polyglot-hi.pkl', 'rb'))
			#print("Emebddings shape is {}".format(embeddings.shape))
			#print words[777], embeddings[777]
			embeddings_file_bin = basepath+'/data/hindi/hi/hi.bin'
			model_bin = KeyedVectors.load(embeddings_file_bin)
			#print(words[777], model_bin[words[777]])
			#sys.exit()
			for line in open(basepath+'/data/'+task+'/vocab.txt'):
				tokens.append(line.strip().decode('utf-8'))
			dict = {}
			EMBEDDING_DIM = 300
			for word in tokens:
				fake_dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
				try:
					dict[word] = model_bin[word]
					num_inv += 1
				except:
					num_oov += 1
					oov.append(word)
					dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
		elif task == 'url' or task=='pit':
			for line in open(basepath+'/data/'+task+'/vocab.txt'):
				tokens.append(line.strip())
			# print(len(tokens))
			dict = {}
			EMBEDDING_DIM = 200
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath+'/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
			#EMBEDDING_DIM = 300
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
			#wv_dict={}
			#wv_arr={}
			# print(len(wv_dict))
			#oov = []
			#for line in open(basepath+'/data/'+task+'/oov.txt'):
			#	line = line.strip()
			#	oov.append(line)
			#inv=[]
			#for line in open(basepath+'/data/'+task+'/inv_4000.txt'):
			#	line = line.strip()
			#	inv.append(line)
			#count=len(oov)+len(inv)
			#inv = tokens
			num_oov=0
			num_inv=0
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
		print('finished loading word vector, there are '+str(num_inv)+' INV words and '+str(num_oov)+' OOV words.')
		print('current task: '+task+', glove mode = ' + str(glove_mode) + ', update_inv_mode = ' + str(update_inv_mode) + ', update_oov_mode = ' + str(update_oov_mode))
		saved_file='current task: '+task+', glove mode = ' + str(glove_mode) + ', update_inv_mode = ' + str(update_inv_mode) + ', update_oov_mode = ' + str(update_oov_mode)+'.txt'
	#subprocess.call(['echo','finished loading word vector, there are ',str(num_inv),' INV words and ',str(len(oov)),' OOV words.'])
	elif granularity=='char':
		# charcnn parameters
		feature_maps=[50,100,150,200,200,200,200]
		kernels=[1,2,3,4,5,6,7]
		charcnn_embedding_size=15
		max_word_length=20

		#c2w parameters
		lm_mode = False
		c2w_mode = False
		character_ngrams=1
		character_ngrams_overlap=True

		tokens=[]
		if task!='wiki':
			if task=='hindi':
				for line in open(basepath+'/data/' + task +'/vocab.txt'):
					tokens.append(line.strip().decode('utf-8'))
				tokens.append('<s>'.decode())
				tokens.append('</s>'.decode())
				tokens.append('oov'.decode())
			else:
				for line in open(basepath+'/data/' + task + '/vocab.txt'):
					tokens.append(line.strip())
				org_tokens=tokens[:]
				tokens.append('<s>')
				tokens.append('</s>')
				tokens.append('oov')
			word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
		if c2w_mode:
			EMBEDDING_DIM = 200
		else:
			EMBEDDING_DIM = 1100
		if character_ngrams==1:
			dict_char_ngram = pickle.load(open(basepath+'/data/' + task + '/char_dict.p', "rb"))
		elif character_ngrams==2 and character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/bigram_dict.p', "rb"))
		elif character_ngrams==2 and not character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/bigram_dict_no_overlap.p', "rb"))
		elif character_ngrams==3 and character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/trigram_dict.p', "rb"))
		elif character_ngrams==3 and not character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/trigram_dict_no_overlap.p', "rb"))
		print('current task: '+task+', lm mode: '+str(lm_mode)+', c2w mode: '+str(c2w_mode)+', n = '+str(character_ngrams)+', overlap = '+str(character_ngrams_overlap)+'.')
		saved_file='current task: '+task+', lm mode: '+str(lm_mode)+', c2w mode: '+str(c2w_mode)+', n = '+str(character_ngrams)+', overlap = '+str(character_ngrams_overlap)+'.txt'
	elif granularity=='mix':
		tokens = []
		num_oov=0
		num_inv=0
		for line in open(basepath+'/data/'+task+'/vocab.txt'):
			tokens.append(line.strip())
		tokens.append('<s>')
		tokens.append('</s>')
		tokens.append('oov')
		# print(len(tokens))
		dict = {}
		#oov=[]
		if task=='sts':
			EMBEDDING_DIM = 300
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
			#wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/VDPWI-NN-Torch/data/glove', 'glove.840B', EMBEDDING_DIM)
		else:
			EMBEDDING_DIM = 200
			wv_dict, wv_arr, wv_size = load_word_vectors(basepath+'/VDPWI-NN-Torch/data/glove', 'glove.twitter.27B', EMBEDDING_DIM)
		'''
		EMBEDDING_DIM = 300
		wv_dict, wv_arr, wv_size = load_word_vectors(basepath + '/data/paragram/paragram_300_sl999/', 'paragram', EMBEDDING_DIM)
		'''
		oov=[]
		for word in tokens:
			'''
			if word in oov or word in inv:
				count+=1
				dict[word] = torch.Tensor([0 for i in range(EMBEDDING_DIM)])
			else:
				dict[word] = wv_arr[wv_dict[word]]
				num_inv+=1
			'''
			try:
				dict[word] = wv_arr[wv_dict[word]]
				num_inv+=1
			except:
				num_oov+=1
				oov.append(word)
				# print(word)
				dict[word] = torch.Tensor([random.uniform(-0.05, 0.05) for i in range(EMBEDDING_DIM)])
				#dict[word] = torch.Tensor([0 for i in range(EMBEDDING_DIM)])

		lm_mode = False
		combine_mode='g_0.75' # 'concat', 'g_0.25', 'g_0.50', 'g_0.75', 'adaptive', 'attention', 'backoff'
		# c2w parameters
		c2w_mode = False
		character_ngrams = 1
		#character_ngrams_2 = 3
		character_ngrams_overlap = False
		if character_ngrams == 1:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/char_dict.p', "rb"))
		elif character_ngrams == 2 and character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/bigram_dict.p', "rb"))
		elif character_ngrams == 2 and not character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/bigram_dict_no_overlap.p', "rb"))
		elif character_ngrams == 3 and character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/trigram_dict.p', "rb"))
		elif character_ngrams == 3 and not character_ngrams_overlap:
			dict_char_ngram = pickle.load(open(basepath + '/data/' + task + '/trigram_dict_no_overlap.p', "rb"))
		'''
		if character_ngrams_2 == 1:
			dict_char_ngram_2 = pickle.load(open(basepath + '/data/' + task + '/char_dict.p', "rb"))
		elif character_ngrams_2 == 2 and character_ngrams_overlap:
			dict_char_ngram_2 = pickle.load(open(basepath + '/data/' + task + '/bigram_dict.p', "rb"))
		elif character_ngrams_2 == 2 and not character_ngrams_overlap:
			dict_char_ngram_2 = pickle.load(open(basepath + '/data/' + task + '/bigram_dict_no_overlap.p', "rb"))
		elif character_ngrams_2 == 3 and character_ngrams_overlap:
			dict_char_ngram_2 = pickle.load(open(basepath + '/data/' + task + '/trigram_dict.p', "rb"))
		elif character_ngrams_2 == 3 and not character_ngrams_overlap:
			dict_char_ngram_2 = pickle.load(open(basepath + '/data/' + task + '/trigram_dict_no_overlap.p', "rb"))
		'''
		word_freq = pickle.load(open(basepath + '/data/' + task + '/word_freq.p', "rb"))
		print('current task: '+task+', lm mode: '+str(lm_mode)+', combination mode: '+combine_mode+', c2w mode: '+str(c2w_mode)+', n = '+str(character_ngrams)+', overlap = '+str(character_ngrams_overlap)+'.')
		print('finished loading word & char table, there are '+str(num_inv)+' INV words and '+str(num_oov)+' OOV words.')
	elif granularity=='cross':
		oov=[]
		dict_char=[]
		tokens=[]
		word_freq=[]
		overlap=True
		if overlap:
			dict_ngram = pickle.load(open(basepath + '/data/' + task + '/cross_trigram_dict.p', "rb"))
		else:
			dict_ngram = pickle.load(open(basepath + '/data/' + task + '/cross_trigram_dict_no_overlap.p', "rb"))
	else:
		print('wrong input for the second argument!')
		sys.exit()

	model=DeepPairWiseWord(EMBEDDING_DIM,HIDDEN_DIM,1,task,granularity,num_class,dict,fake_dict, dict_char_ngram, oov,tokens, word_freq,
	                       feature_maps,kernels,charcnn_embedding_size,max_word_length,character_ngrams,c2w_mode,character_ngrams_overlap, word_mode,
	                       combine_mode, lm_mode)#, corpus)
	#print(get_n_params(model))
	#sys.exit()
	#print(model.lm_train_data)
	#sys.exit()
	#premodel=DeepPairWiseWord(EMBEDDING_DIM,HIDDEN_DIM,1,task,granularity,num_class,dict,dict_char,oov)
	#premodel.load_state_dict(torch.load('model_char_only.pkl'))
	#premodel=torch.load('model_char_only.pkl')
	#model.embedding=premodel.embedding
	#model.lstm_c2w=premodel.lstm_c2w
	#model.df=premodel.df
	#model.db=premodel.db
	#model.bias=premodel.bias
	if torch.cuda.is_available():
		model=model.cuda()
	lsents, rsents, labels = trainset
	#print(len(lsents))
	#threshold=40000
	#lsents = lsents[:threshold]
	#rsents = rsents[:threshold]
	#labels = labels[:threshold]
	# Loss and Optimizer
	if task=='sick' or task=='sts' or task=='snli':
		indices = torch.randperm(len(lsents))
		print('indices:')
		print(indices[:10])
		#for line in open('./data/sick/order.txt'):
		#	indices.append(int(line.strip()) - 1)
		criterion = nn.KLDivLoss()
		if torch.cuda.is_available():
			criterion=criterion.cuda()
	elif task=='url' or task=='pit' or task=='hindi' or task=='quora' or task=='msrp' or task=='wikiqa' or task=='trecqa' or task=='mnli':
		'''
		indices = torch.randperm(len(trainset[0]))
		with open('./data/'+task+'/order.txt','w') as f:
			for item in indices:
				f.writelines(str(item)+'\n')
		'''
		#indices = []
		#for line in open('./data/'+task+'/order.txt'):
		#	indices.append(int(line.strip()))
		indices = torch.randperm(len(lsents))
		#print('indices:')
		#print(indices[:10])
		criterion = nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=True)
		if torch.cuda.is_available():
			criterion = criterion.cuda()
	elif task=='wiki':
		indices = torch.randperm(len(lsents))
		print('indices:')
		print(indices[:10])
		criterion = nn.MultiLabelSoftMarginLoss()
		if torch.cuda.is_available():
			criterion = criterion.cuda()
	optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)#, momentum=0.1, weight_decay=0.05)#,momentum=0.9,weight_decay=0.95)
	#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
	# Train the Model
	#print(oov)
	print('start training')
	#subprocess.call(['echo','start training'])
	gold=[]
	gold_um=[]
	if task=='url':
		for line in open(basepath+'/data/' + task + '/test_9324/sim.txt'):
			gold.append(int(line.strip()))
	elif task=='snli':
		for line in open(basepath+'/data/' + task + '/test/sim.txt'):
			gold.append(line.strip())
	elif task=='trecqa':
		for line in open(basepath+'/data/' + task + '/raw-test/sim.txt'):
			gold.append(float(line.strip()))
	elif task=='mnli':
		pass
		'''
		for line in open(basepath+'/data/' + task + '/dev_m/sim.txt'):
			gold.append(float(['neutral', 'entailment','contradiction'].index(line.strip())))
		for line in open(basepath+'/data/' + task + '/dev_um/sim.txt'):
			gold_um.append(float(['neutral', 'entailment','contradiction'].index(line.strip())))
		'''
	else:
		for line in open(basepath+'/data/' + task + '/test/sim.txt'):
			gold.append(float(line.strip()))
	max_result=-1
	max_result_um=-1
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
			if granularity=='word':
				sentA = lsents[i]
				sentB = rsents[i]
				'''
				#print(lsents[i])
				try:
					sentA = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in lsents[i]), 0)
					sentA = Variable(sentA)#.cuda()
					#print(lsents[i])
					#print(sentA)
					#print(rsents[i])
					sentB = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in rsents[i]), 0)
					sentB = Variable(sentB)#.cuda()
				except:
					print(lsents[i])
					print(rsents[i])
					sys.exit()
				#print(rsents[i])
				#print(sentB)
				#sys.exit()
				if torch.cuda.is_available():
					sentA=sentA.cuda()
					sentB=sentB.cuda()
				sentA = torch.unsqueeze(sentA, 0).view(-1, 1, EMBEDDING_DIM)
				sentB = torch.unsqueeze(sentB, 0).view(-1, 1, EMBEDDING_DIM)
				# label=torch.unsqueeze(label,0)
				'''
			elif granularity=='char' or granularity=='mix' or granularity=='cross':
				#sentA=[]
				#sentB=[]
				#for word in lsents[i]:
				#	sentA.append([dict[char] for char in word])
				#for word in rsents[i]:
				#	sentB.append([dict[char] for char in word])
				#print(i)
				sentA=lsents[i]
				sentB=rsents[i]
			if task=='sick' or task=='sts' or task=='snli' or task=='wiki':
				label=Variable(torch.Tensor(labels[i]))
			else:
				label = Variable(torch.LongTensor(labels[i]))#.cuda()
			if torch.cuda.is_available():
				label=label.cuda()
			# Forward + Backward + Optimize
			#elapsed_time = time.time() - start_time
			#print('data preparation time: '+str(timedelta(seconds=elapsed_time)))
			#print(sentA)
			#print(sentB)
			#print(id[i])
			#print('*'*20)
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
				msg += '\t train accuracy: %f' % train_acc
				print(msg)

			if (index + 1) % (int(len(lsents)/2)) == 0:
				#print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f'
				#	   % (epoch + 1, num_epochs, index + 1, len(lsents) // 1, data_loss))#loss.data[0]))
				#subprocess.call(['echo','Epoch ',str(epoch+1),'Loss: ',str(data_loss)])
				#break
				#data_loss = 0
				#torch.save(model.state_dict(), 'model.pkl')
				#model.load_state_dict(torch.load('model_char_only.pkl'))

				if task=='sick' or task=='sts' or task=='snli' or task=='wiki':
					model.eval()
					test_lsents, test_rsents, test_labels = testset
					predicted=[]
					tmp_result=0
					#gold=[]
					#for line in open('./data/sick/test/sim.txt'):
					#	gold.append(float(line.strip()))
					for test_i in range(len(test_lsents)):
						if granularity == 'word':
							'''
							sentA = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in test_lsents[test_i]), 0)
							sentA = Variable(sentA)
							# print(sentA)
							sentB = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in test_rsents[test_i]), 0)
							sentB = Variable(sentB)
							if torch.cuda.is_available():
								sentA = sentA.cuda()
								sentB = sentB.cuda()
							#label = torch.unsqueeze(label, 0)
							sentA = torch.unsqueeze(sentA, 0).view(-1, 1, EMBEDDING_DIM)
							sentB = torch.unsqueeze(sentB, 0).view(-1, 1, EMBEDDING_DIM)
							'''
							sentA = test_lsents[test_i]
							sentB = test_rsents[test_i]
						elif granularity == 'char' or granularity=='mix':
							sentA = test_lsents[test_i]
							sentB = test_rsents[test_i]
						raw_output,_ = model(sentA, sentB, index)
						#print(output)
						if task=='sick':
							output=raw_output
							output = np.exp(output.data[0].cpu().numpy())
							predicted.append(1*output[0]+2*output[1]+3*output[2]+4*output[3]+5*output[4])
						elif task=='snli':
							output = raw_output
							output = np.exp(output.data[0].cpu().numpy())
							output=[output[0],output[1],output[2]]
							tmp_output=output.index(max(output))
							predicted.append(tmp_output)
							if test_labels[test_i].index(max(test_labels[test_i]))==tmp_output:
								tmp_result+=1
						elif task=='wiki':
							output = torch.sigmoid(raw_output).data > 0.5
							output = output.cpu()
							predicted = list(output.numpy()[0])
							if predicted==test_labels[test_i]:
								tmp_result+=1
						else:
							output = raw_output
							output = np.exp(output.data[0].cpu().numpy())
							predicted.append(0 * output[0] + 1 * output[1] + 2 * output[2] + 3 * output[3] + 4 * output[4] + 5*output[5])
					#print(predicted)
					#print(gold)
					if task=='sick':
						result = pearson(predicted, gold)
						print('Test Correlation: %.6f' % result)
						if result>max_result:
							max_result=result
					elif task=='snli' or task=='wiki':
						result=tmp_result/len(test_lsents)
						print('Test Accuracy: %.6f' % result)
						if result>max_result:
							max_result=result
					else:
						result1=pearson(predicted[0:450],gold[0:450])
						result2=pearson(predicted[450:750],gold[450:750])
						result3=pearson(predicted[750:1500],gold[750:1500])
						result4=pearson(predicted[1500:2250],gold[1500:2250])
						result5=pearson(predicted[2250:3000],gold[2250:3000])
						result6=pearson(predicted[3000:3750],gold[3000:3750])
						print('deft-forum: %.6f, deft-news: %.6f, headlines: %.6f, images: %.6f, OnWN: %.6f, tweet-news: %.6f' %(result1, result2, result3, result4, result5, result6))
						wt_mean=0.12*result1+0.08*result2+0.2*result3+0.2*result4+0.2*result5+0.2*result6
						print('weighted mean: %.6f' % wt_mean)
						if wt_mean>max_result:
							max_result=wt_mean
						if task=='sts':
							with open(basepath+'/data/sts/sts_PWIM_prob.txt', 'w') as f:
								for item in predicted:
									f.writelines(str(item)+'\n')
						#else:
						#	with open('SICK_with_paragram_result.txt', 'w') as f:
						#		for item in predicted:
						#			f.writelines(str(item)+'\n')
				else:
					model.eval()
					msg = '%d completed epochs, %d batches' % (epoch, index+1)
					if task=='mnli':
						test_lsents, test_rsents, test_labels = devset_m
					else:
						test_lsents, test_rsents, test_labels = testset
					predicted = []
					correct=0
					#gold=gold[:3000]
					#print(len(gold))
					for test_i in range(len(test_lsents)):
						# start_time = time.time()
						if granularity == 'word':
							sentA = test_lsents[test_i]
							sentB = test_rsents[test_i]
							'''
							sentA = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in test_lsents[test_i]), 0)
							sentA = Variable(sentA)#.cuda()
							# print(sentA)
							sentB = torch.cat((dict[word].view(1, EMBEDDING_DIM) for word in test_rsents[test_i]), 0)
							sentB = Variable(sentB)#.cuda()
							# print(sentB)
							if torch.cuda.is_available():
								sentA=sentA.cuda()
								sentB=sentB.cuda()
							sentA = torch.unsqueeze(sentA, 0).view(-1, 1, EMBEDDING_DIM)
							sentB = torch.unsqueeze(sentB, 0).view(-1, 1, EMBEDDING_DIM)
						# label=torch.unsqueeze(label,0)
							'''
						elif granularity == 'char' or granularity=='mix':
							sentA = test_lsents[test_i]
							sentB = test_rsents[test_i]
						output, _ = model(sentA, sentB, index)
						#print(output)
						output=np.exp(output.data[0].cpu().numpy())
						if test_labels[test_i][0]==np.argmax(output):
							correct+=1
						predicted.append(output[1])
					#result=float(correct)/len(test_lsents)
					#print('Test Accuracy: %.4f'% result)
					#result_acc, result_f1=URL_maxF1_eval(predict_result=predicted,test_data_label=gold)
					result = correct / len(test_lsents)
					msg += '\t dev m accuracy: %f' % result
					print(msg)
					if result>max_result:
						max_result=result
						test_lsents, test_rsents, test_labels = testset_m
						predicted=[]
						for test_i in range(len(test_lsents)):
							# start_time = time.time()
							if granularity == 'word':
								sentA = test_lsents[test_i]
								sentB = test_rsents[test_i]
							output, _ = model(sentA, sentB, index)
							output = np.exp(output.data[0].cpu().numpy())
							predicted.append(np.argmax(output))
						with open(basepath + '/sub_m.csv', 'w+') as f:
							label_dict = [ 'neutral', 'entailment','contradiction']
							f.write("pairID,gold_label\n")
							for i, k in enumerate(predicted):
								f.write(str(i + 9847) + "," + label_dict[k] + "\n")
						#with open(basepath+'/PWIM_prob_result_'+task, 'w') as f:
						#	for item in predicted:
						#		f.writelines(str(item)+'\n')
					if task=='mnli':
						msg = '%d completed epochs, %d batches' % (epoch, index + 1)
						test_lsents, test_rsents, test_labels = devset_um
						predicted = []
						correct = 0
						for test_i in range(len(test_lsents)):
							# start_time = time.time()
							if granularity == 'word':
								sentA = test_lsents[test_i]
								sentB = test_rsents[test_i]
							output, _ = model(sentA, sentB, index)
							# print(output)
							output = np.exp(output.data[0].cpu().numpy())
							if test_labels[test_i][0] == np.argmax(output):
								correct += 1
							predicted.append(output[1])
						#result_acc, result_f1 = URL_maxF1_eval(predict_result=predicted, test_data_label=gold_um)
						result_acc = correct / len(test_lsents)
						msg += '\t dev um accuracy: %f' % result_acc
						print(msg)
						if result_acc > max_result_um:
							max_result_um = result_acc
							test_lsents, test_rsents, test_labels = testset_um
							predicted = []
							for test_i in range(len(test_lsents)):
								# start_time = time.time()
								if granularity == 'word':
									sentA = test_lsents[test_i]
									sentB = test_rsents[test_i]
								output, _ = model(sentA, sentB, index)
								output = np.exp(output.data[0].cpu().numpy())
								predicted.append(np.argmax(output))
							with open(basepath + '/sub_um.csv', 'w+') as f:
								label_dict = ['neutral', 'entailment','contradiction']
								f.write("pairID,gold_label\n")
								for i, k in enumerate(predicted):
									f.write(str(i) + "," + label_dict[k] + "\n")
						#with open('current task: '+task+', lm mode: '+str(lm_mode)+', combination mode: '+combine_mode+', c2w mode: '+str(c2w_mode)+', n = '+str(character_ngrams)+', overlap = '+str(character_ngrams_overlap)+'.txt','w') as f:
						#	for item in predicted:
						#		f.writelines(str(item)+'\n')
						#torch.save(model, 'model_URL_unigram_CNN.pkl')
						#torch.save(model, 'model_word_inv_18k.pkl')
						#torch.save(model, 'model_word_inv_3k.pkl')
						#torch.save(model, 'model_char_only.pkl')
						#torch.save(model, 'model_word_only_pit.pkl')
						#torch.save(model, 'model_word_char_backoff.pkl')
						#torch.save(model, 'model_word_char_g_0.5.pkl')
						#torch.save(model, 'model_word_char_adaptive.pkl')
						#torch.save(model, 'model_word_char_attention.pkl')
						#with open('model_word_inv_0k_result.txt', 'w') as f:
						#with open('sts_model_word_only_inv_17k_result.txt', 'w') as f:
						#with open('model_word_inv_3k_result.txt', 'w') as f:
						#with open('model_char_only_result.txt', 'w') as f:
						#with open('model_word_only_result_pit.txt', 'w') as f:
						#with open('model_word_char_g_0.5_result.txt', 'w') as f:
						#with open('model_word_char_backoff_result.txt', 'w') as f:
						#with open('model_word_char_adaptive.txt', 'w') as f:
						#with open('model_word_char_attention_result.txt','w') as f:
						#	for item in predicted:
						#		f.writelines(str(item)+'\n')
						'''
						h = Variable(torch.zeros(2, 1, model.embedding_dim))  # 2 for bidirection
						c = Variable(torch.zeros(2, 1, model.embedding_dim))
						if torch.cuda.is_available():
							h = h.cuda()
							c = c.cuda()
						subword_embedding={}
						for word in org_tokens:
							tmp_indices = model.generate_word_indices(word)
							if not model.c2w_mode:
								if len(tmp_indices) < 20:
									tmp_indices = tmp_indices + [0 for i in range(model.charcnn_max_word_length - len(tmp_indices))]
								else:
									tmp_indices = tmp_indices[0:20]
							if model.c2w_mode:
								output = model.c2w_cell([tmp_indices], h, c)
							else:
								output = model.charCNN_cell([tmp_indices])
							subword_embedding[word]=output.data[0].cpu().numpy()
						pickle.dump(subword_embedding, open('URL_subword_lm_embedding.p', "wb"))
						'''
				elapsed_time = time.time() - start_time
				print('Epoch ' + str(epoch + 1) + ' finished within ' + str(timedelta(seconds=elapsed_time))+', and current time:'+ str(datetime.now()))
				print('Best result until now: %.6f' % max_result)
				print('Best um result until now: %.6f' % max_result_um)
				#subprocess.call(['echo','Epoch ' , str(epoch + 1) , ' finished within ' , str(timedelta(seconds=elapsed_time)),', and current time:', str(datetime.now())])
				#subprocess.call(['echo','Best result until now: ',str(max_result)])
				model.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--task', type=str, default='sts',
						help='Currently supported tasks: sick, pit, sts, url, snli, mnli, hindi, quora, msrp, wiki wikiqa and trecqa')
	parser.add_argument('--granularity', type=str, default='word',
						help='Currently supported granularities: char, word, mix and cross.')
	args = parser.parse_args()
	print(args)
	main(args)
