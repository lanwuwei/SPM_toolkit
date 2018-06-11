from __future__ import division
import os
import sys
import time
import torch
import random
import pickle
from model import *
import torch.nn as nn
from torch import optim
from torch_util import *
from datetime import datetime
from datetime import timedelta
from collections import Counter
from torchtext.vocab import load_word_vectors

import pdb


def create_batch(data,from_index, to_index):
  if to_index > len(data):
    to_index = len(data)
  lsize, rsize = 0, 0
  lsize_list, rsize_list = [], []
  for i in range(from_index, to_index):
    length = len(data[i][0])
    lsize_list.append(length)
    if length > lsize:
      lsize = length
    length = len(data[i][1])
    rsize_list.append(length)
    if length > rsize:
      rsize = length

  left_sents, right_sents, labels = [], [], []
  for i in range(from_index, to_index):
    lsent = data[i][0]
    lsent = lsent + ['<pad>' for k in range(lsize - len(lsent))]
    left_sents.append([word2id[word] for word in lsent])
    rsent = data[i][1]
    rsent = rsent + ['<pad>' for k in range(rsize - len(rsent))]
    right_sents.append([word2id[word] for word in rsent])
    labels.append(data[i][2])

  left_sents=Variable(torch.LongTensor(left_sents))
  right_sents=Variable(torch.LongTensor(right_sents))
  labels=Variable(torch.LongTensor(labels))
  lsize_list=torch.LongTensor(lsize_list)
  rsize_list =torch.LongTensor(rsize_list)

  if torch.cuda.is_available():
    left_sents=left_sents.cuda()
    right_sents=right_sents.cuda()
    labels=labels.cuda()
    lsize_list=lsize_list.cuda()
    rsize_list=rsize_list.cuda()
  return left_sents, right_sents, labels, lsize_list, rsize_list


def make_vocab(train_pairs, dev_pairs, test_pairs):
  vocab_counter = Counter()
  for left, right, _ in train_pairs + dev_pairs + test_pairs:
    vocab_counter.update(left)
    vocab_counter.update(right)
  
  tokens = ['<pad>', '<unk>'] + [w for w, _ in vocab_counter.most_common()]
  word2id={}
  for _, word in enumerate(tokens):
    word2id[word] = _

  return tokens, word2id


if __name__ == '__main__':
  task='quora'
  print('task: '+task)
  torch.manual_seed(6)

  num_class = 2
  if torch.cuda.is_available():
    print('CUDA is available!')
  
  basepath = './data'
  embedding_path = '../data/glove'
  train_pairs = readQuoradata(basepath + '/train/')
  dev_pairs = readQuoradata(basepath + '/dev/')
  test_pairs = readQuoradata(basepath + '/test/')

  print('# of train pairs: %d' % len(train_pairs))
  print('# of dev pairs: %d' % len(dev_pairs))
  print('# of test pairs: %d' % len(test_pairs))

  tokens, word2id = make_vocab(train_pairs, dev_pairs, test_pairs)
  with open(os.path.join('./results', 'vocab.pkl'), 'wb') as f:
    pickle.dump((tokens, word2id), f, protocol=pickle.HIGHEST_PROTOCOL)

  wv_dict, wv_arr, wv_size = load_word_vectors(embedding_path, 'glove.840B', 300)
  pretrained_emb = []
  for _, word  in enumerate(tokens):
    if word in wv_dict:
      pretrained_emb.append(wv_arr[wv_dict[word]].numpy())
    else:
      pretrained_emb.append(np.random.uniform(-0.05, 0.05, size=[300]))
  pretrained_emb = np.stack(pretrained_emb)
  assert pretrained_emb.shape == (len(tokens), 300)


  model = StackBiLSTMMaxout(h_size=[512, 1024, 2048], 
                            v_size=len(tokens), 
                            d=300, 
                            mlp_d=1600, 
                            dropout_r=0.1, 
                            max_l=60, 
                            num_class=num_class)

  if torch.cuda.is_available():
    pretrained_emb=torch.Tensor(pretrained_emb).cuda()
  else:
    pretrained_emb = torch.Tensor(pretrained_emb)
  model.Embd.weight.data = pretrained_emb

  if torch.cuda.is_available():
    model.cuda()

  start_lr = 2e-4
  batch_size=32
  optimizer = optim.Adam(model.parameters(), lr=start_lr)
  criterion = nn.CrossEntropyLoss()

  # ckpt_path = os.path.join('./results', '%s_%d.pkl' % (task, 0))
  # #  model = torch.load(ckpt_path)
  # model.load_state_dict(torch.load(ckpt_path))
  # model.eval()
  # 
  # # Test
  # start_time = time.time()
  # test_batch_index = 0
  # test_num_correct = 0
  # msg = ''
  # accumulated_loss = 0
  # test_batch_i = 0
  # pred = []
  # gold = []
  # while test_batch_i < len(test_pairs):
  #   left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(
  #       test_pairs, test_batch_i, test_batch_i+batch_size)
  #   left_sents = torch.transpose(left_sents, 0, 1)
  #   right_sents = torch.transpose(right_sents, 0, 1)
  #   test_batch_i+=len(labels)
  #   output = model(left_sents, lsize_list, right_sents, rsize_list)
  #   result = output.data.cpu().numpy()
  #   loss = criterion(output, labels)
  #   # accumulated_loss += loss.data[0]
  #   accumulated_loss += loss.item()
  #   a = np.argmax(result, axis=1)
  #   b = labels.data.cpu().numpy()
  #   test_num_correct += np.sum(a == b)
  # 
  # elapsed_time = time.time() - start_time
  # print('Test finished within ' + str(timedelta(seconds=elapsed_time)))
  # msg += '\t test loss: %.4f' % accumulated_loss
  # test_acc = test_num_correct / len(test_pairs)
  # msg += '\t test accuracy: %.4f' % test_acc
  # print(msg)

  # sys.exit()


  
  print('start training...')
  best_dev_acc = 0.0
  for epoch in range(100):
    batch_counter = 0
    accumulated_loss = 0
    print('--' * 20)
    start_time = time.time()
    i_decay = epoch / 2
    lr = start_lr / (2 ** i_decay)
    print('lr: %f' % lr)
    train_pairs = np.array(train_pairs)
    rand_idx = np.random.permutation(len(train_pairs))
    train_pairs = train_pairs[rand_idx]
    train_batch_i = 0
    train_num_correct=0
    train_sents_scaned = 0
    while train_batch_i < len(train_pairs):
      model.train()
      left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(
          train_pairs, train_batch_i, train_batch_i+batch_size)
      train_sents_scaned += len(labels)
      train_batch_i+=len(labels)
      left_sents=torch.transpose(left_sents,0,1)
      right_sents=torch.transpose(right_sents,0,1)
      output=model(left_sents,lsize_list,right_sents,rsize_list)
      result = output.data.cpu().numpy()
      a = np.argmax(result, axis=1)
      b = labels.data.cpu().numpy()
      train_num_correct += np.sum(a == b)
      loss = criterion(output, labels)
      optimizer.zero_grad()
      for pg in optimizer.param_groups:
        pg['lr'] = lr
      loss.backward()
      optimizer.step()
      batch_counter += 1
      # accumulated_loss += loss.data[0]
      accumulated_loss += loss.item()

    elapsed_time = time.time() - start_time
    print('Epoch ' + str(epoch) + ' finished within ' + str(timedelta(seconds=elapsed_time)))
    msg = ''
    msg += '\t train loss: %.4f' % accumulated_loss
    msg += '\t train accuracy: %.4f' % (train_num_correct / train_sents_scaned)
    print(msg)
      
    # Validation
    start_time = time.time()
    model.eval()
    dev_batch_index = 0
    dev_num_correct = 0
    msg = ''
    accumulated_loss = 0
    dev_batch_i = 0
    pred=[]
    gold=[]
    while dev_batch_i < len(dev_pairs):
      left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(
          dev_pairs, dev_batch_i, dev_batch_i+batch_size)
      dev_batch_i += len(labels)
      left_sents = torch.transpose(left_sents, 0, 1)
      right_sents = torch.transpose(right_sents, 0, 1)
      output = model(left_sents, lsize_list,right_sents, rsize_list)
      result = output.data.cpu().numpy()
      loss = criterion(output, labels)
      accumulated_loss += loss.item()
      a = np.argmax(result, axis=1)
      b = labels.data.cpu().numpy()
      dev_num_correct += np.sum(a == b)
    
    elapsed_time = time.time() - start_time
    print('Validation finished within ' + str(timedelta(seconds=elapsed_time)))
    msg += '\t dev loss: %.4f' % accumulated_loss
    dev_acc = dev_num_correct / len(dev_pairs)
    msg += '\t dev accuracy: %.4f' % dev_acc
    
    # if accumulated_loss < best_dev_loss:
    if dev_acc > best_dev_acc:
      ckpt_path = os.path.join('./results', '%s_%d.pkl' % (task, epoch))
      msg += '\t | checkpoint: ' + ckpt_path
      best_dev_acc = dev_acc
      torch.save(model.state_dict(), ckpt_path)
    print(msg)

    # Test
    start_time = time.time()
    test_batch_index = 0
    test_num_correct = 0
    msg = ''
    accumulated_loss = 0
    test_batch_i = 0
    pred = []
    gold = []
    while test_batch_i < len(test_pairs):
      left_sents, right_sents, labels, lsize_list, rsize_list = create_batch(
          test_pairs, test_batch_i, test_batch_i+batch_size)
      left_sents = torch.transpose(left_sents, 0, 1)
      right_sents = torch.transpose(right_sents, 0, 1)
      test_batch_i+=len(labels)
      output = model(left_sents, lsize_list, right_sents, rsize_list)
      result = output.data.cpu().numpy()
      loss = criterion(output, labels)
      # accumulated_loss += loss.data[0]
      accumulated_loss += loss.item()
      a = np.argmax(result, axis=1)
      b = labels.data.cpu().numpy()
      test_num_correct += np.sum(a == b)
    
    elapsed_time = time.time() - start_time
    print('Test finished within ' + str(timedelta(seconds=elapsed_time)))
    msg += '\t test loss: %.4f' % accumulated_loss
    test_acc = test_num_correct / len(test_pairs)
    msg += '\t test accuracy: %.4f' % test_acc
    print(msg)
