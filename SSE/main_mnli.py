from __future__ import division
import torch
import os
from model import *
from torch import optim
import torch.nn as nn
from datetime import datetime
from torch_util import *
import config
import tqdm
import data_loader
import sys
import time
from datetime import timedelta
from os.path import expanduser

def build_kaggle_submission_file(model_path):
	print('start testing...')
	torch.manual_seed(6)

	snli_d, mnli_d, embd = data_loader.load_data_sm(
		config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 32, 32, 32, 32), device=0)

	m_train, m_dev_m, m_dev_um, m_test_m, m_test_um = mnli_d

	m_test_um.shuffle = False
	m_test_m.shuffle = False
	m_test_um.sort = False
	m_test_m.sort = False

	model = StackBiLSTMMaxout(h_size=[512, 1024, 2048], v_size=10, d=300, mlp_d=1600, dropout_r=0.1, max_l=60)
	model.Embd.weight.data = embd
	# model.display()

	if torch.cuda.is_available():
		embd.cuda()
		model.cuda()

	criterion = nn.CrossEntropyLoss()

	model.load_state_dict(torch.load(model_path))

	m_pred = model_eval(model, m_test_m, criterion, pred=True)
	um_pred = model_eval(model, m_test_um, criterion, pred=True)

	model.max_l = 150
	print(um_pred)
	print(m_pred)

	with open(basepath+'/sub_um.csv', 'w+') as f:
		index = ['entailment', 'contradiction', 'neutral']
		f.write("pairID,gold_label\n")
		for i, k in enumerate(um_pred):
			f.write(str(i) + "," + index[k] + "\n")

	with open(basepath+'/sub_m.csv', 'w+') as f:
		index = ['entailment', 'contradiction', 'neutral']
		f.write("pairID,gold_label\n")
		for j, k in enumerate(m_pred):
			f.write(str(j + 9847) + "," + index[k] + "\n")

def train(combined_set=False):
	torch.manual_seed(6)

	if torch.cuda.is_available():
		print('CUDA is available!')
		base_path = expanduser("~") + '/pytorch/SSE'
	else:
		base_path = expanduser("~") + '/Documents/research/pytorch/SSE'

	if torch.cuda.is_available():
		snli_d, mnli_d, embd = data_loader.load_data_sm(
			config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 200, 200, 30, 30), device=None)
	else:
		snli_d, mnli_d, embd = data_loader.load_data_sm(
			config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 200, 200, 30, 30), device=-1)

	s_train, s_dev, s_test = snli_d
	m_train, m_dev_m, m_dev_um, m_test_m, m_test_um = mnli_d

	s_train.repeat = False
	m_train.repeat = False

	model = StackBiLSTMMaxout(h_size=[512, 1024, 2048], v_size=10, d=300, mlp_d=1600, dropout_r=0.1, max_l=60)
	model.Embd.weight.data = embd

	if torch.cuda.is_available():
		embd.cuda()
		model.cuda()

	start_lr = 2e-4

	optimizer = optim.Adam(model.parameters(), lr=start_lr)
	criterion = nn.CrossEntropyLoss()

	best_m_dev = -1
	best_um_dev = -1
	report_interval = 1000
	threshold=1

	print('start training...')
	for epoch in range(20):
		print('--' * 20)
		accumulated_loss = 0
		batch_counter = 0
		train_sents_scaned = 0
		s_train.init_epoch()
		m_train.init_epoch()
		start_time = time.time()

		if not combined_set:
			#train_iter, dev_iter, test_iter = s_train, s_dev, s_test
			train_iter=m_train
			train_iter.repeat = False
			print(len(train_iter))
		else:
			train_iter = data_loader.combine_two_set(s_train, m_train, rate=[0.15, 1], seed=epoch)
			dev_iter, test_iter = s_dev, s_test

		#start_perf = model_eval(model, dev_iter, criterion)
		i_decay = epoch / 2
		lr = start_lr / (2 ** i_decay)
		print(lr)
		model.train()
		train_num_correct = 0

		for batch_idx, batch in enumerate(train_iter):

			s1, s1_l = batch.premise
			s2, s2_l = batch.hypothesis
			y = batch.label - 1

			train_sents_scaned += len(y)
			out = model(s1, s1_l - 1, s2, s2_l - 1)
			loss = criterion(out, y)
			result = out.data.cpu().numpy()
			a = np.argmax(result, axis=1)
			b = y.data.cpu().numpy()
			train_num_correct += np.sum(a == b)
			optimizer.zero_grad()

			for pg in optimizer.param_groups:
				pg['lr'] = lr

			loss.backward()
			optimizer.step()

			accumulated_loss += loss.data[0]
			batch_counter += 1
			if batch_counter % report_interval == 0:
				msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
				msg += '\t training batch loss: %f' % (accumulated_loss / train_sents_scaned)
				msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
				print(msg)

			if batch_counter > (len(train_iter) // threshold):
				break
		msg = '%d completed epochs, %d batches' % (epoch, batch_counter)
		msg += '\t training batch loss: %f' % (accumulated_loss / train_sents_scaned)
		msg += '\t train accuracy: %f' % (train_num_correct / train_sents_scaned)
		print(msg)
		# valid after each epoch
		model.max_l = 150
		mdm_score, mdm_loss, validm_prob = model_eval(model, m_dev_m, criterion)
		mdum_score, mdum_loss, validum_prob = model_eval(model, m_dev_um, criterion)

		print(' MNLI_M:{}/{}'.format(mdm_score, mdm_loss))
		print(' MNLI_UM:{}/{}'.format(mdum_score, mdum_loss))
		model.max_l = 60

		if mdm_score>best_m_dev:
			best_m_dev = mdm_score
			with open(base_path + '/prob_SSE_' + task+'_m', 'w') as f:
				for item in validm_prob:
					f.writelines(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')

		if mdum_score>best_um_dev:
			best_um_dev = mdum_score
			with open(base_path + '/prob_SSE_' + task+'_um', 'w') as f:
				for item in validum_prob:
					f.writelines(str(item[0]) + '\t' + str(item[1]) + '\t' + str(item[2]) + '\n')

		elapsed_time = time.time() - start_time
		print('Epoch ' + str(epoch) + 'Batch number ' + str(batch_counter) + ' finished within ' + str(timedelta(seconds=elapsed_time)))

if __name__ == "__main__":
	print('model: SSE')
	print('task: mnli')
	task='mnli'
	if torch.cuda.is_available():
		print('CUDA is available!')
		basepath = expanduser("~") + '/pytorch/SSE'
	else:
		basepath = expanduser("~") + '/Documents/research/pytorch/SSE'
	train(False)
	#build_kaggle_submission_file(basepath+'/saved_model/model_saved')