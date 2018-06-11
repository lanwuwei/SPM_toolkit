from __future__ import division
import json
#import nltk
import sys
import torch
import math
import logging
import numpy as np
from os.path import expanduser
from numpy import linalg as LA

#tokenizer = nltk.tokenize.TreebankWordTokenizer()


def pearson(x,y):
	x=np.array(x)
	y=np.array(y)
	x=x-np.mean(x)
	y=y-np.mean(y)
	return x.dot(y)/(LA.norm(x)*LA.norm(y))

def URL_maxF1_eval(predict_result,test_data_label):
	test_data_label=[item>=1 for item in test_data_label]
	counter = 0
	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0

	for i, t in enumerate(predict_result):

		if t>0.5:
			guess=True
		else:
			guess=False
		label = test_data_label[i]
		#print guess, label
		if guess == True and label == False:
			fp += 1.0
		elif guess == False and label == True:
			fn += 1.0
		elif guess == True and label == True:
			tp += 1.0
		elif guess == False and label == False:
			tn += 1.0
		if label == guess:
			counter += 1.0
		#else:
			#print label+'--'*20
			# if guess:
			# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

	try:
		P = tp / (tp + fp)
		R = tp / (tp + fn)
		F = 2 * P * R / (P + R)
	except:
		P=0
		R=0
		F=0

	#print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
	#print "ACCURACY: %s" % (counter/len(predict_result))
	accuracy=counter/len(predict_result)

	#print "# true pos:", tp
	#print "# false pos:", fp
	#print "# false neg:", fn
	#print "# true neg:", tn
	maxF1=0
	P_maxF1=0
	R_maxF1=0
	probs = predict_result
	sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
	sortedindex.reverse()

	truepos=0
	falsepos=0
	for sortedi in sortedindex:
		if test_data_label[sortedi]==True:
			truepos+=1
		elif test_data_label[sortedi]==False:
			falsepos+=1
		precision=0
		if truepos+falsepos>0:
			precision=truepos/(truepos+falsepos)

		recall=truepos/(tp+fn)
		f1=0
		if precision+recall>0:
			f1=2*precision*recall/(precision+recall)
			if f1>maxF1:
				#print probs[sortedi]
				maxF1=f1
				P_maxF1=precision
				R_maxF1=recall
	print "PRECISION: %s, RECALL: %s, max_F1: %s" % (P_maxF1, R_maxF1, maxF1)
	return (accuracy, maxF1)

def tokenize(text):
	"""
	Tokenize a piece of text using the Treebank tokenizer

	:return: a list of strings
	"""
	return tokenizer.tokenize(text)
	#return text.split()

def shuffle_arrays(*arrays):
	"""
	Shuffle all given arrays with the same RNG state.

	All shuffling is in-place, i.e., this function returns None.
	"""
	rng_state = np.random.get_state()
	for array in arrays:
		np.random.shuffle(array)
		np.random.set_state(rng_state)

def readSTSdata(dir):
	#print(len(dict))
	#print(dict['bmxs'])
	lsents=[]
	rsents=[]
	labels=[]
	for line in open(dir+'a.toks'):
		line = line.decode('utf-8')
		pieces=line.strip().split()
		lsents.append(pieces)
	for line in open(dir+'b.toks'):
		line = line.decode('utf-8')
		pieces = line.strip().split()
		rsents.append(pieces)
	for line in open(dir + 'sim.txt'):
		sim = float(line.strip())
		ceil = int(math.ceil(sim))
		floor = int(math.floor(sim))
		tmp = [0, 0, 0, 0, 0, 0]
		if floor != ceil:
			tmp[ceil] = sim - floor
			tmp[floor] = ceil - sim
		else:
			tmp[floor] = 1
		labels.append(tmp)
	#data=(lsents,rsents,labels)
	if not len(lsents)==len(rsents)==len(labels):
		print('error!')
		sys.exit()
	clean_data = []
	for i in range(len(lsents)):
		clean_data.append((lsents[i], rsents[i], labels[i]))
	return clean_data

def readQuoradata(dir):
	lsents = []
	rsents = []
	labels = []
	for line in open(dir+'a.toks'):
		line = line.decode('utf-8')
		pieces=line.strip().split()
		lsents.append(pieces)
	for line in open(dir+'b.toks'):
		line = line.decode('utf-8')
		pieces = line.strip().split()
		rsents.append(pieces)
	for line in open(dir + 'sim.txt'):
		labels.append(int(line.strip()))
	clean_data = []
	for i in range(len(lsents)):
		clean_data.append((lsents[i],rsents[i],labels[i]))
	return clean_data

def readSNLIdata(dir):
	#print(len(dict))
	#print(dict['bmxs'])
	lsents=[]
	rsents=[]
	labels=[]
	for line in open(dir+'a.toks'):
		pieces=line.strip().split()
		lsents.append(pieces)
	for line in open(dir+'b.toks'):
		pieces = line.strip().split()
		rsents.append(pieces)
	for line in open(dir + 'sim.txt'):
		sim = line.strip()
		if sim=='neutral':
			labels.append([0,1,0])
		elif sim=='entailment':
			labels.append([1,0,0])
		elif sim=='contradiction':
			labels.append([0,0,1])
		else:
			labels.append([0, 0, 0])
	clean_data = []
	for i in range(len(lsents)):
		if labels[i]!=[0,0,0]:
			clean_data.append((lsents[i],rsents[i],labels[i].index(max(labels[i]))))
	return clean_data

def read_corpus(filename, lowercase):
	"""
	Read a JSONL or TSV file with the SNLI corpus

	:param filename: path to the file
	:param lowercase: whether to convert content to lower case
	:return: a list of tuples (first_sent, second_sent, label)
	"""
	logging.info('Reading data from %s' % filename)
	# we are only interested in the actual sentences + gold label
	# the corpus files has a few more things
	useful_data = []

	# the SNLI corpus has one JSON object per line
	with open(filename, 'rb') as f:

		if filename.endswith('.tsv') or filename.endswith('.txt'):

			for line in f:
				line = line.decode('utf-8').strip()
				if lowercase:
					line = line.lower()
				if len(line.split('\t')) == 10:
					(label, _, _, _, _, sent1, sent2, _, _, _) = line.split('\t')
				elif len(line.split('\t')) == 14:
					(label, _, _, _, _, sent1, sent2, _, _, _, _, _, _, _) = line.split('\t')
				#sent1, sent2, label = line.split('\t')
				if label not in ('contradiction','neutral','entailment'):
					continue
				tokens1 = tokenize(sent1)
				tokens2 = tokenize(sent2)
				useful_data.append((tokens1, tokens2, ('contradiction','neutral','entailment').index(label)))
		else:
			for line in f:
				line = line.decode('utf-8')
				if lowercase:
					line = line.lower()
				data = json.loads(line)
				if data['gold_label'] == '-':
					# ignore items without a gold label
					continue

				sentence1_parse = data['sentence1_parse']
				sentence2_parse = data['sentence2_parse']
				label = data['gold_label']

				tree1 = nltk.Tree.fromstring(sentence1_parse)
				tree2 = nltk.Tree.fromstring(sentence2_parse)
				tokens1 = tree1.leaves()
				tokens2 = tree2.leaves()
				t = (tokens1, tokens2, ('neutral','contradiction','entailment', 'hidden').index(label))
				useful_data.append(t)

	return useful_data
