import cPickle as pkl
import gzip
import os
import re
import sys
import numpy
import math
import random

from binary_tree import BinaryTree

def convert_ptb_to_tree(line):
	index = 0
	tree = None
	line = line.rstrip()

	stack = []
	parts = line.split()
	for p_i, p in enumerate(parts):
		# opening of a bracket, create a new node, take parent from top of stack
		if p == '(':
			if tree is None:
				tree = BinaryTree(index)
			else:
				add_descendant(tree, index, stack[-1])
			# add the newly created node to the stack and increment the index
			stack.append(index)
			index += 1
		# close of a bracket, pop node on top of the stack
		elif p == ')':
			stack.pop(-1)
		# otherwise, create a new node, take parent from top of stack, and set word
		else:
			add_descendant(tree, index, stack[-1])
			tree.set_word(index, p)
			index += 1
	return tree

def add_descendant(tree, index, parent_index):
	# add to the left first if possible, then to the right
	if tree.has_left_descendant_at_node(parent_index):
		if tree.has_right_descendant_at_node(parent_index):
			sys.exit("Node " + str(parent_index) + " already has two children")
		else:
			tree.add_right_descendant(index, parent_index)
	else:
		tree.add_left_descendant(index, parent_index)

def fopen(filename, mode='r'):
	if filename.endswith('.gz'):
		return gzip.open(filename, mode)
	return open(filename, mode)

class TextIterator:
	"""Simple Bitext iterator."""
	def __init__(self, source, target, label,
				 dict,
				 batch_size=128,
				 n_words=-1,
				 maxlen=500,
				 shuffle=True, task_type='classification'):
		self.source = fopen(source, 'r')
		self.target = fopen(target, 'r')
		self.label = fopen(label, 'r')
		with open(dict, 'rb') as f:
			self.dict = pkl.load(f)
		self.batch_size = batch_size
		self.n_words = n_words
		self.maxlen = maxlen
		self.shuffle = shuffle
		self.end_of_data = False
		self.task_type=task_type

		self.source_buffer = []
		self.target_buffer = []
		self.label_buffer = []
		self.k = batch_size * 20

	def __iter__(self):
		return self

	def reset(self):
		self.source.seek(0)
		self.target.seek(0)
		self.label.seek(0)

	def next(self):
		if self.end_of_data:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		source = []
		target = []
		label = []

		# fill buffer, if it's empty
		assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
		assert len(self.source_buffer) == len(self.label_buffer), 'Buffer size mismatch!'

		if len(self.source_buffer) == 0:
			for k_ in xrange(self.k):
				ss = self.source.readline()
				if ss == "":
					break
				tt = self.target.readline()
				if tt == "":
					break
				ll = self.label.readline()
				if ll == "":
					break
				if self.task_type!='classification':
					try:
						sim = float(ll.strip())
					except:
						print(ss)
						sys.exit()
					ceil = int(math.ceil(sim))
					floor = int(math.floor(sim))
					tmp = [0, 0, 0, 0, 0, 0]
					if floor != ceil:
						tmp[ceil] = sim - floor
						tmp[floor] = ceil - sim
					else:
						tmp[floor] = 1
					ll=tmp

				try:
					ss = convert_ptb_to_tree(ss)
				except:
					print('ss')
					ss = convert_ptb_to_tree('( null null )')
				#ss.print_tree()
				words_ss, left_mask_ss, right_mask_ss = ss.convert_to_sequence_and_masks(ss.root)
				words_ss = [self.dict[w] if w in self.dict else 1
					  for w in words_ss]
				if self.n_words > 0:
					words_ss = [w if w < self.n_words else 1 for w in words_ss]
				ss = (words_ss, left_mask_ss, right_mask_ss)

				try:
					tt = convert_ptb_to_tree(tt)
				except:
					print('ss')
					tt = convert_ptb_to_tree('( null null )')
				words_tt, left_mask_tt, right_mask_tt = tt.convert_to_sequence_and_masks(tt.root)
				words_tt = [self.dict[w] if w in self.dict else 1
					  for w in words_tt]
				if self.n_words > 0:
					words_tt = [w if w < self.n_words else 1 for w in words_tt]
				tt = (words_tt, left_mask_tt, right_mask_tt)

				if len(words_ss) > self.maxlen or len(words_tt) > self.maxlen:
					continue

				self.source_buffer.append(ss)
				self.target_buffer.append(tt)
				if self.task_type!='classification':
					self.label_buffer.append(ll)
				else:
					self.label_buffer.append(ll.strip())

			if self.shuffle:
				# sort by target buffer
				tlen = numpy.array([len(t[0]) for t in self.target_buffer])
				tidx = tlen.argsort()
				# shuffle mini-batch
				tindex = []
				small_index = range(int(math.ceil(len(tidx)*1./self.batch_size)))
				random.shuffle(small_index)
				for i in small_index:
					if (i+1)*self.batch_size > len(tidx):
						tindex.extend(tidx[i*self.batch_size:])
					else:
						tindex.extend(tidx[i*self.batch_size:(i+1)*self.batch_size])

				tidx = tindex

				_sbuf = [self.source_buffer[i] for i in tidx]
				_tbuf = [self.target_buffer[i] for i in tidx]
				_lbuf = [self.label_buffer[i] for i in tidx]

				self.source_buffer = _sbuf
				self.target_buffer = _tbuf
				self.label_buffer = _lbuf

		if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.label_buffer) == 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		try:

			# actual work here
			while True:

				# read from source file and map to word index
				try:
					ss = self.source_buffer.pop(0)
					tt = self.target_buffer.pop(0)
					ll = self.label_buffer.pop(0)
				except IndexError:
					break

				source.append(ss)
				target.append(tt)
				label.append(ll)

				if len(source) >= self.batch_size or \
						len(target) >= self.batch_size or \
						len(label) >= self.batch_size:
					break
		except IOError:
			self.end_of_data = True

		if len(source) <= 0 or len(target) <= 0 or len(label) <= 0:
			self.end_of_data = False
			self.reset()
			raise StopIteration

		return source, target, label
