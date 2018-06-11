from __future__ import division

import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from tree import Tree
import Constants

def print_tree(tree, level):
	indent = ''
	for i in range(level):
		indent += '| '
	line = indent + str(tree.idx)
	print (line)
	for i in xrange(tree.num_children):
		print_tree(tree.children[i], level+1)
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

class Dataset(data.Dataset):
	def __init__(self, path, vocab, num_classes):
		super(Dataset, self).__init__()
		self.vocab = vocab
		self.num_classes = num_classes

		self.lsentences = self.read_sentences(os.path.join(path,'a.toks'))
		self.rsentences = self.read_sentences(os.path.join(path,'b.toks'))

		self.ltrees = self.read_trees(os.path.join(path,'a.cparents'))
		self.rtrees = self.read_trees(os.path.join(path,'b.cparents'))

		self.labels = self.read_labels(os.path.join(path,'sim.txt'))

		self.size = self.labels.size(0)

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		ltree = deepcopy(self.ltrees[index])
		rtree = deepcopy(self.rtrees[index])
		lsent = deepcopy(self.lsentences[index])
		rsent = deepcopy(self.rsentences[index])
		label = deepcopy(self.labels[index])
		return (ltree,lsent,rtree,rsent,label)

	def read_sentences(self, filename):
		with open(filename,'r') as f:
			sentences = [self.read_sentence(line) for line in f.readlines()]
		return sentences

	def read_sentence(self, line):
		indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
		return torch.LongTensor(indices)

	def read_trees(self, filename):
		with open(filename,'r') as f:
			trees = [self.read_tree(line) for line in f.readlines()]
		return trees

	def read_tree(self, line):
		parents = map(int,line.split())
		trees = dict()
		root = None
		for i in xrange(1,len(parents)+1):
			#if not trees[i-1] and parents[i-1]!=-1:
			if i-1 not in trees.keys() and parents[i-1]!=-1:
				idx = i
				prev = None
				while True:
					parent = parents[idx-1]
					if parent == -1:
						break
					tree = Tree()
					if prev is not None:
						tree.add_child(prev)
					trees[idx-1] = tree
					tree.idx = idx-1
					#if trees[parent-1] is not None:
					if parent-1 in trees.keys():
						trees[parent-1].add_child(tree)
						break
					elif parent==0:
						root = tree
						break
					else:
						prev = tree
						idx = parent
		return root

	def read_labels(self, filename):
		with open(filename,'r') as f:
			labels = map(lambda x: float(x), f.readlines())
			labels = torch.Tensor(labels)
		return labels