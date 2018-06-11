from __future__ import division
import sys
import numpy as np
from numpy import linalg as LA
import math
import os
import torch
import unicodedata
from os.path import expanduser
from collections import *

def splitclusters(s):
    """Generate the grapheme clusters for the string s. (Not the full
    Unicode text segmentation algorithm, but probably good enough for
    Devanagari.)

    """
    virama = u'\N{DEVANAGARI SIGN VIRAMA}'
    cluster = u''
    last = None
    for c in s:
        cat = unicodedata.category(c)[0]
        if cat == 'M' or cat == 'L' and last == virama:
            cluster += c
        else:
            if cluster:
                yield cluster
            cluster = c
        last = c
    if cluster:
        yield cluster

def jaccard_index(set_1, set_2):
    return len(set_1.intersection(set_2)) / float(len(set_1.union(set_2)))

def PINC(source,candidate,N=3):
    tokenize_source=[char for char in source]
    tokenize_candidate=[char for char in candidate]
    if len(tokenize_candidate)<N or len(tokenize_source)<N:
        return 1.0
    sum=0
    for i in range(N):
        n_gram_s=[]
        n_gram_c=[]
        for k in range(len(tokenize_source)-i):
            n_gram_s.append(tokenize_source[k:k+i+1])
        for k in range(len(tokenize_candidate)-i):
            n_gram_c.append(tokenize_candidate[k:k+i+1])
        #print(n_gram_s)
        #print(n_gram_c)
        counter=0
        for element in n_gram_s:
            if element in n_gram_c:
                counter+=1
        sum+=float(counter)/len(n_gram_c)
        #print(counter)
        #print(sum/N)
    return sum/N

def intersect(list1, list2):
    cnt1 = Counter()
    cnt2 = Counter()
    for tk1 in list1:
        cnt1[tk1] += 1
    for tk2 in list2:
        cnt2[tk2] += 1
    inter = cnt1 & cnt2
    return list(inter.elements())

def ngram_overlap_features(s_word, t_word):
	s1grams = [w for w in s_word]
	t1grams = [w for w in t_word]
	s2grams = []
	t2grams = []
	s3grams = []
	t3grams = []
	for i in range(0, len(s1grams) - 1):
		if i < len(s1grams) - 1:
			s2gram = s1grams[i] + " " + s1grams[i + 1]
			s2grams.append(s2gram)
		if i < len(s1grams) - 2:
			s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
			s3grams.append(s3gram)

	for i in range(0, len(t1grams) - 1):
		if i < len(t1grams) - 1:
			t2gram = t1grams[i] + " " + t1grams[i + 1]
			t2grams.append(t2gram)
		if i < len(t1grams) - 2:
			t3gram = t1grams[i] + " " + t1grams[i + 1] + " " + t1grams[i + 2]
			t3grams.append(t3gram)

	f1gram = 0
	precision1gram = len(set(intersect(s1grams, t1grams))) / len(set(s1grams))
	recall1gram = len(set(intersect(s1grams, t1grams))) / len(set(t1grams))
	if (precision1gram + recall1gram) > 0:
		f1gram = 2 * precision1gram * recall1gram / (precision1gram + recall1gram)
	if len(set(s2grams)) > 0 and len(set(t2grams)) > 0:
		precision2gram = len(set(intersect(s2grams, t2grams))) / len(set(s2grams))
		recall2gram = len(set(intersect(s2grams, t2grams))) / len(set(t2grams))
	else:
		precision2gram = 0
		recall2gram = 0
	f2gram = 0
	if (precision2gram + recall2gram) > 0:
		f2gram = 2 * precision1gram * recall2gram / (precision2gram + recall2gram)
	if len(set(s3grams)) > 0 and len(set(t3grams)):
		precision3gram = len(set(intersect(s3grams, t3grams))) / len(set(s3grams))
		recall3gram = len(set(intersect(s3grams, t3grams))) / len(set(t3grams))
	else:
		precision3gram = 0
		recall3gram = 0
	f3gram = 0
	if (precision3gram + recall3gram) > 0:
		f3gram = 2 * precision3gram * recall3gram / (precision3gram + recall3gram)
	features=[f1gram, precision1gram, recall1gram, f2gram, precision2gram, recall2gram, f3gram, precision3gram, recall3gram]
	return features

def pearson(x,y):
	x=np.array(x)
	y=np.array(y)
	x=x-np.mean(x)
	y=y-np.mean(y)
	return x.dot(y)/(LA.norm(x)*LA.norm(y))

def readURLdata(dir,granularity):
	lsents = []
	rsents = []
	labels = []
	if granularity=='word' or granularity =='mix' or granularity == 'char' or granularity=='multi-task':
		for line in open(dir + 'a.toks'):
			pieces = line.strip().split()
			lsents.append(pieces)
		for line in open(dir + 'b.toks'):
			pieces = line.strip().split()
			rsents.append(pieces)
		for line in open(dir + 'sim.txt'):
			pieces = line.strip().split()
			labels.append([int(item) for item in pieces])
	'''
	elif granularity=='char':
		if 'train' in dir:
			filename='Twitter_URL_Corpus_train.txt'
		else:
			filename='Twitter_URL_Corpus_test.txt'#'new_testing_5136.txt'
		for line in open(dir+filename):
			if len(line.strip().split('\t')) == 4:
				a, b, sim, url = line.strip().split('\t')
				score, _ = eval(sim)
				if score >= 4:
					score = 1
				elif score <= 2:
					score = 0
				if int(score) != 3:
					lsents.append(a.split())
					rsents.append(b.split())
					labels.append([int(score)])
	'''
	data = (lsents, rsents, labels)
	if not len(lsents) == len(rsents) == len(labels):
		print('error!')
		sys.exit()
	return data

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

		if (tp+fn)>0:
			recall=truepos/(tp+fn)
		else:
			recall=0
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

def readPITdata(dir,granularity):
	# print(len(dict))
	# print(dict['bmxs'])
	lsents = []
	rsents = []
	labels = []
	if granularity == 'word' or granularity == 'mix' or granularity=='char' or granularity=='multi-task':
		for line in open(dir + 'a.toks'):
			pieces = line.strip().split()
			lsents.append(pieces)
		for line in open(dir + 'b.toks'):
			pieces = line.strip().split()
			rsents.append(pieces)
	elif granularity=='cross':
		for line in open(dir + 'a.toks'):
			pieces = line.strip()
			lsents.append(pieces)
		for line in open(dir + 'b.toks'):
			pieces = line.strip()
			rsents.append(pieces)
	'''
	elif granularity == 'char':
		filename=''
		if 'train' in dir:
			filename = 'train.data'
		elif 'dev' in dir:
			filename = 'dev.data'
		elif 'test' in dir:
			filename = 'test.data'
		for line in open(dir + filename):
			if filename=='train.data':
				(trendid, trendname, origsent, candsent, judge, origsenttag, candsenttag) = line.strip().split('\t')
			else:
				(_ , trendid, trendname, origsent, candsent, judge, tag) = line.strip().split('\t')
			expert_label = -1
			if judge[0] == '(':  # labelled by Amazon Mechanical Turk in format like "(2,3)"
				nYes = eval(judge)[0]
				if nYes >= 3:
					expert_label = 1
				elif nYes <= 1:
					expert_label = 0
			elif judge[0].isdigit():  # labelled by expert in format like "2"
				nYes = int(judge[0])
				if nYes >= 4:
					expert_label = 1
				elif nYes <= 2:
					expert_label = 0
			if expert_label != -1:
				lsents.append(origsent.split())
				rsents.append(candsent.split())
	'''
	for line in open(dir + 'sim.txt'):
		labels.append([int(line.strip())])
	data = (lsents, rsents, labels)
	if not len(lsents) == len(rsents) == len(labels):
		print('error!')
		sys.exit()
	# print(lsents[0])
	# print(rsents[0])
	# print(labels[0])
	return data

def trim_list_pair(l1, l2):
	start_index=None
	end_index=None
	for i in range(len(l1)):
		if i < len(l2):
			if l1[i] == l2[i]:
				continue
			else:
				start_index = i
				break
		else:
			start_index=i
			break
	if start_index==None:
		start_index=len(l1)
	for i in range(len(l1)):
		if i < len(l2):
			if l1[-1 - i] == l2[-1 - i]:
				continue
			else:
				end_index = i
				break
		else:
			end_index=i
			break
	if end_index == None:
		end_index=len(l1)
	if end_index==0:
		end_index=None
		return (l1[start_index:end_index], l2[start_index:end_index])
	else:
		return (l1[start_index:-end_index], l2[start_index:-end_index])

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp