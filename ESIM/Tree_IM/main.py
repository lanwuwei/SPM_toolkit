from __future__ import division
import sys
import cPickle as pkl
from os.path import expanduser
import torch
import numpy
from model import *
import time
import gc
from datetime import timedelta
from vocab import Vocab
from util import *
from torchtext.vocab import load_word_vectors


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
	"""
	Random weights drawn from a Gaussian
	"""
	if nout is None:
		nout = nin
	if nout == nin and ortho:
		W = ortho_weight(nin)
	else:
		W = scale * numpy.random.randn(nin, nout)
	return W.astype('float32')

if torch.cuda.is_available():
	print('CUDA is available!')
	base_path = expanduser("~") + '/pytorch/DeepPairWiseWord/data/url/'
	embedding_path = expanduser("~") + '/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'
else:
	base_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/data/url/'
	embedding_path = expanduser("~") + '/Documents/research/pytorch/DeepPairWiseWord/VDPWI-NN-Torch/data/glove'

task='url'
print('task: '+task)
print('model: TreeIM')
num_classes=2
vocab = Vocab(filename=base_path+'vocab.txt')
vocab.add('__PAD__')
num_words=vocab.size()
train_dataset = Dataset(base_path+'train/', vocab, num_classes)
test_dataset = Dataset(base_path+'test_9324/', vocab, num_classes)
#test_dataset = dev_dataset
dim_word=300
batch_size=32
num_epochs=500
valid_batch_size=32
print 'Loading data'
n_words=vocab.size()
pretrained_emb=norm_weight(n_words, dim_word)
wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', dim_word)
for word in vocab.labelToIdx.keys():
	try:
		pretrained_emb[vocab.labelToIdx[word]]=wv_arr[wv_dict[word]].numpy()
	except:
		pretrained_emb[vocab.labelToIdx[word]]=np.random.uniform(-0.05,0.05,dim_word)

criterion = torch.nn.CrossEntropyLoss()
model = ESIM(dim_word, num_classes, n_words, dim_word, pretrained_emb, num_words)
if torch.cuda.is_available():
	model = model.cuda()
	criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0004)
print('start training...')
accumulated_loss=0
batch_counter=0
report_interval = 5000
best_dev_loss=10e10
best_dev_loss2=10e10
clip_c=10
model.train()

gold=[]
for line in open(base_path+'/test_9324/sim.txt'):
	gold.append(int(line.strip()))

for epoch in range(num_epochs):
	accumulated_loss = 0
	model.train()
	optimizer.zero_grad()
	print('--' * 20)
	start_time = time.time()
	train_sents_scaned = 0
	train_num_correct = 0
	batch_counter=0
	indices = torch.randperm(len(train_dataset))
	#for idx in tqdm(xrange(len(train_dataset)), desc='Training epoch ' + str(epoch + 1) + ''):
	for idx in range(len(train_dataset)):
		#print(idx)
		train_sents_scaned+=1
		ltree, lsent, rtree, rsent, label = train_dataset[idx]#indices[idx]]
		#print(lsent)
		#for item in lsent:
		#	print(vocab.idxToLabel[item])
		#print_tree(ltree, 0)
		#sys.exit()
		linput, rinput = Variable(lsent), Variable(rsent)
		#target = Variable(map_label_to_target(label, num_classes))
		target=Variable(torch.LongTensor([int(label)]))
		if torch.cuda.is_available():
			linput, rinput = linput.cuda(), rinput.cuda()
			#ltree, rtree = ltree.cuda(), rtree.cuda()
			target = target.cuda()
		output = model(linput, rinput, ltree, rtree)
		del ltree, linput
		del rtree, rinput
		#gc.collect()
		result = output.data.cpu().numpy()
		a = np.argmax(result, axis=1)
		b = target.data.cpu().numpy()
		train_num_correct += np.sum(a == b)
		loss = criterion(output, target)
		loss.backward()
		#optimizer.step()
		accumulated_loss += loss.data[0]
		#print(loss.data[0])
		batch_counter += 1
		if train_sents_scaned%batch_size==0:
			optimizer.step()
			optimizer.zero_grad()
		if batch_counter % report_interval == 0:
			msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
			msg += '\t train batch loss: %f' % (accumulated_loss / train_sents_scaned)
			#msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
			print(msg)
			gc.collect()
	# valid after each epoch
	model.eval()
	msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
	accumulated_loss = 0
	test_num_correct = 0
	n_done=0
	pred=[]
	#for idx in tqdm(xrange(len(test_dataset)), desc='Testing epoch ' + str(epoch + 1) + ''):
	for idx in range(len(test_dataset)):
		ltree, lsent, rtree, rsent, label = test_dataset[idx]
		linput, rinput = Variable(lsent), Variable(rsent)
		#target = Variable(map_label_to_target(label, num_classes))
		target=Variable(torch.LongTensor([int(label)]))
		if torch.cuda.is_available():
			linput, rinput = linput.cuda(), rinput.cuda()
			#ltree, rtree = ltree.cuda(), rtree.cuda()
			target = target.cuda()
		with torch.no_grad():
			output = model(linput, rinput, ltree, rtree)
		loss = criterion(output, target)
		accumulated_loss+=loss
		result = output.data.cpu().numpy()
		a = np.argmax(result, axis=1)
		b = target.data.cpu().numpy()
		test_num_correct += np.sum(a == b)
		pred.append(output.data[0].cpu().numpy()[1])
		del linput, rinput, ltree, rtree, target
	msg += '\t test loss: %f' % (accumulated_loss/len(test_dataset))
	test_acc = test_num_correct / len(test_dataset)
	#msg += '\t test accuracy: %f' % test_acc
	print(msg)
	gc.collect()
	URL_maxF1_eval(predict_result=pred, test_data_label=gold)
	with open(base_path + '/result_prob_TreeIM_' + task, 'w') as f:
		for i in range(len(pred)):
			f.writelines(str(pred[i]) + '\n')