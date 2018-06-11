from __future__ import division
import sys

import gc
import numpy
import torch
import time
from data_iterator import TextIterator
import cPickle as pkl
from model_batch import *
from os.path import expanduser
from datetime import timedelta
from torchtext.vocab import load_word_vectors

# batch preparation
def prepare_data(group_x, group_y, labels):
	lengths_x = [len(s[0]) for s in group_x]
	lengths_y = [len(s[0]) for s in group_y]

	n_samples = len(group_x)
	maxlen_x = numpy.max(lengths_x)
	maxlen_y = numpy.max(lengths_y)

	x_seq = numpy.zeros((maxlen_x, n_samples)).astype('int64')
	y_seq = numpy.zeros((maxlen_y, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
	y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
	x_left_mask = numpy.zeros((maxlen_x, n_samples, maxlen_x)).astype('float32')
	x_right_mask = numpy.zeros((maxlen_x, n_samples, maxlen_x)).astype('float32')
	y_left_mask = numpy.zeros((maxlen_y, n_samples, maxlen_y)).astype('float32')
	y_right_mask = numpy.zeros((maxlen_y, n_samples, maxlen_y)).astype('float32')
	l = numpy.zeros((n_samples,)).astype('int64')

	for idx, [s_x, s_y, ll] in enumerate(zip(group_x, group_y, labels)):
		x_seq[-lengths_x[idx]:, idx] = s_x[0]
		x_mask[-lengths_x[idx]:, idx] = 1.
		x_left_mask[-lengths_x[idx]:, idx, -lengths_x[idx]:] = s_x[1]
		x_right_mask[-lengths_x[idx]:, idx, -lengths_x[idx]:] = s_x[2]
		y_seq[-lengths_y[idx]:, idx] = s_y[0]
		y_mask[-lengths_y[idx]:, idx] = 1.
		y_left_mask[-lengths_y[idx]:, idx, -lengths_y[idx]:] = s_y[1]
		y_right_mask[-lengths_y[idx]:, idx, -lengths_y[idx]:] = s_y[2]
		l[idx] = ll

	if torch.cuda.is_available():
		x_seq=Variable(torch.LongTensor(x_seq)).cuda()
		y_seq = Variable(torch.LongTensor(y_seq)).cuda()
		x_mask = Variable(torch.FloatTensor(x_mask)).cuda()
		y_mask = Variable(torch.FloatTensor(y_mask)).cuda()
		x_left_mask=Variable(torch.FloatTensor(x_left_mask)).cuda()
		y_left_mask=Variable(torch.FloatTensor(y_left_mask)).cuda()
		x_right_mask=Variable(torch.FloatTensor(x_right_mask)).cuda()
		y_right_mask=Variable(torch.FloatTensor(y_right_mask)).cuda()
		l=Variable(torch.LongTensor(l)).cuda()
	else:
		x_seq = Variable(torch.LongTensor(x_seq))
		y_seq = Variable(torch.LongTensor(y_seq))
		x_mask = Variable(torch.FloatTensor(x_mask))
		y_mask = Variable(torch.FloatTensor(y_mask))
		x_left_mask = Variable(torch.FloatTensor(x_left_mask))
		y_left_mask = Variable(torch.FloatTensor(y_left_mask))
		x_right_mask = Variable(torch.FloatTensor(x_right_mask))
		y_right_mask = Variable(torch.FloatTensor(y_right_mask))
		l = Variable(torch.LongTensor(l))

	x = (x_seq, x_mask, x_left_mask, x_right_mask)
	y = (y_seq, y_mask, y_left_mask, y_right_mask)

	return x, y, l

print('task: SNLI')
print('model: Tree_IM')
if torch.cuda.is_available():
	print('CUDA is available!')
	base_path = expanduser("~") + '/pytorch/ESIM'
	embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
else:
	base_path = expanduser("~") + '/Documents/research/pytorch/ESIM'
	embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'

datasets = [base_path+'/data/binary_tree/premise_snli_1.0_train.txt',
            base_path+'/data/binary_tree/hypothesis_snli_1.0_train.txt',
            base_path+'/data/binary_tree/label_snli_1.0_train.txt']
valid_datasets = [base_path+'/data/binary_tree/premise_snli_1.0_dev.txt',
                  base_path+'/data/binary_tree/hypothesis_snli_1.0_dev.txt',
                  base_path+'/data/binary_tree/label_snli_1.0_dev.txt']
test_datasets = [base_path+'/data/binary_tree/premise_snli_1.0_test.txt',
                 base_path+'/data/binary_tree/hypothesis_snli_1.0_test.txt',
                 base_path+'/data/binary_tree/label_snli_1.0_test.txt']
dictionary = base_path+'/data/binary_tree/snli_vocab_cased.pkl'

maxlen=150
batch_size=32
max_epochs=1000
dim_word=300
with open(dictionary, 'rb') as f:
	worddicts = pkl.load(f)
n_words=len(worddicts)
wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', dim_word)
pretrained_emb=norm_weight(n_words, dim_word)
for word in worddicts.keys():
	try:
		pretrained_emb[worddicts[word]]=wv_arr[wv_dict[word]].numpy()
	except:
		pretrained_emb[worddicts[word]] = torch.normal(torch.zeros(dim_word),std=1).numpy()
print('load data...')
train = TextIterator(datasets[0], datasets[1], datasets[2],
						 dictionary,
						 n_words=n_words,
						 batch_size=batch_size,
						 maxlen=maxlen, shuffle=True)
valid = TextIterator(valid_datasets[0], valid_datasets[1], valid_datasets[2],
					 dictionary,
					 n_words=n_words,
					 batch_size=batch_size,
					 shuffle=False)
test = TextIterator(test_datasets[0], test_datasets[1], test_datasets[2],
					 dictionary,
					 n_words=n_words,
					 batch_size=batch_size,
					 shuffle=False)
criterion = torch.nn.CrossEntropyLoss()
model = ESIM(dim_word, 3, n_words, dim_word, pretrained_emb)
if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0004)
print('start training...')
clip_c=10
max_result=0
report_interval=1000
for epoch in xrange(max_epochs):
	model.train()
	print('--' * 20)
	start_time = time.time()
	batch_counter=0
	train_batch_i = 0
	train_sents_scaned = 0
	train_num_correct = 0
	accumulated_loss=0
	for x1, x2, y in train:
		#print(x1[0][0])
		#for item in x1[0][0]:
		#	print(worddicts.keys()[item])
		x1, x2, y = prepare_data(x1, x2, y)
		train_sents_scaned += len(y)
		optimizer.zero_grad()
		#if x1[0].size(0)>100:
		#	print(x1[0].size(0))
		#continue
		output = model(x1[0], x1[1], x1[2], x1[3], x2[0], x2[1], x2[2], x2[3])
		result = output.data.cpu().numpy()
		a = np.argmax(result, axis=1)
		b = y.data.cpu().numpy()
		train_num_correct += np.sum(a == b)
		loss = criterion(output, y)
		loss.backward()
		grad_norm = 0.

		for m in list(model.parameters()):
			grad_norm+=m.grad.data.norm() ** 2

		for m in list(model.parameters()):
			if grad_norm>clip_c**2:
				try:
					m.grad.data= m.grad.data / torch.sqrt(grad_norm) * clip_c
				except:
					pass

		optimizer.step()
		accumulated_loss += loss.data[0]
		batch_counter += 1
		del x1,x2,y
		if batch_counter % report_interval == 0:
			gc.collect()
			msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
			msg += '\t training batch loss: %f' % (accumulated_loss / train_sents_scaned)
			msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
			print(msg)
	# valid after each epoch
	model.eval()
	msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
	accumulated_loss = 0
	dev_num_correct = 0
	n_done = 0
	for dev_x1, dev_x2, dev_y in valid:
		x1, x2, y = prepare_data(dev_x1, dev_x2, dev_y)
		with torch.no_grad():
			output = F.softmax(model(x1[0], x1[1], x1[2], x1[3], x2[0], x2[1], x2[2], x2[3]))
		n_done += len(y)
		result = output.data.cpu().numpy()
		loss = criterion(output, y)
		accumulated_loss += loss.data[0]
		a = numpy.argmax(result, axis=1)
		b = y.data.cpu().numpy()
		dev_num_correct += numpy.sum(a == b)
	msg += '\t dev loss: %f' % (accumulated_loss / n_done)
	dev_acc = dev_num_correct / n_done
	msg += '\t dev accuracy: %f' % dev_acc
	# test after each epoch
	print(msg)
	msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
	accumulated_loss = 0
	dev_num_correct = 0
	n_done = 0
	pred=[]
	for dev_x1, dev_x2, dev_y in test:
		x1, x2, y = prepare_data(dev_x1, dev_x2, dev_y)
		with torch.no_grad():
			output = F.softmax(model(x1[0], x1[1], x1[2], x1[3], x2[0], x2[1], x2[2], x2[3]))
		n_done += len(y)
		result = output.data.cpu().numpy()
		loss = criterion(output, y)
		accumulated_loss += loss.data[0]
		a = numpy.argmax(result, axis=1)
		b = y.data.cpu().numpy()
		dev_num_correct += numpy.sum(a == b)
		pred.extend(result)
	msg += '\t test loss: %f' % (accumulated_loss / n_done)
	dev_acc = dev_num_correct / n_done
	msg += '\t test accuracy: %f' % dev_acc
	print(msg)
	if dev_acc>max_result:
		max_result=dev_acc
		with open(base_path+'/snli_Tree_IM_prob.txt','w') as f:
			for item in pred:
				f.writelines(str(item[0])+'\t'+str(item[1])+'\t'+str(item[2])+'\n')
	elapsed_time = time.time() - start_time
	print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))