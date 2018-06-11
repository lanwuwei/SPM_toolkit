import torch
import config
from torchtext import data, vocab
from torchtext import datasets
from mnli import MNLI
import numpy as np
import itertools
from torch.autograd import Variable


class RParsedTextLField(data.Field):
	def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
		super(RParsedTextLField, self).__init__(
			eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
				t for t in parse if t not in ('(', ')')],
			postprocessing=lambda parse, _, __: [
				list(reversed(p)) for p in parse])


class ParsedTextLField(data.Field):
	def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
		super(ParsedTextLField, self).__init__(
			eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
				t for t in parse if t not in ('(', ')')])


def load_data(data_root, embd_file, reseversed=True, batch_sizes=(32, 32, 32), device=-1):
	if reseversed:
		testl_field = RParsedTextLField()
	else:
		testl_field = ParsedTextLField()

	transitions_field = datasets.snli.ShiftReduceField()
	y_field = data.Field(sequential=False)

	train, dev, test = datasets.SNLI.splits(testl_field, y_field, transitions_field, root=data_root)
	testl_field.build_vocab(train, dev, test)
	y_field.build_vocab(train)

	if torch.cuda.is_available():
		testl_field.vocab.vectors = torch.load(embd_file)
	else:
		testl_field.vocab.vectors = torch.load(embd_file, map_location=lambda storage, loc: storage)

	train_iter, dev_iter, test_iter = data.Iterator.splits(
		(train, dev, test), batch_sizes=batch_sizes, device=device, shuffle=False)

	return train_iter, dev_iter, test_iter, testl_field.vocab.vectors


def load_data_sm(data_root, embd_file, reseversed=True, batch_sizes=(32, 32, 32, 32, 32), device=-1):
	if reseversed:
		testl_field = RParsedTextLField()
	else:
		testl_field = ParsedTextLField()

	transitions_field = datasets.snli.ShiftReduceField()
	y_field = data.Field(sequential=False)
	g_field = data.Field(sequential=False)

	train_size, dev_size, test_size, m_dev_size, m_test_size = batch_sizes

	snli_train, snli_dev, snli_test = datasets.SNLI.splits(testl_field, y_field, transitions_field, root=data_root)

	mnli_train, mnli_dev_m, mnli_dev_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
													  train='train.jsonl',
													  validation='dev_matched.jsonl',
													  test='dev_mismatched.jsonl')

	mnli_test_m, mnli_test_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
											train=None,
											validation='test_matched_unlabeled.jsonl',
											test='test_mismatched_unlabeled.jsonl')

	testl_field.build_vocab(snli_train, snli_dev, snli_test,
							mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)

	g_field.build_vocab(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)
	y_field.build_vocab(snli_train)
	#print('Important:', y_field.vocab.itos)
	if torch.cuda.is_available():
		testl_field.vocab.vectors = torch.load(embd_file)
	else:
		testl_field.vocab.vectors = torch.load(embd_file, map_location=lambda storage, loc: storage)

	snli_train_iter, snli_dev_iter, snli_test_iter = data.Iterator.splits(
		(snli_train, snli_dev, snli_test), batch_sizes=batch_sizes, device=device, shuffle=False)

	mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter = data.Iterator.splits(
		(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um),
		batch_sizes=(train_size, m_dev_size, m_test_size, m_dev_size, m_test_size),
		device=device, shuffle=False, sort=False)

	# if random_combined:
	#     snli_train.examples = list(np.random.choice(snli_train.examples, round(len(snli_train) * rate), replace=False)) + mnli_train.examples
	#     train = snli_train
	#     train_iter = data.Iterator.splits(train, batch_sizes=train_size, device=device, shuffle=False, sort=False)
	#     mnli_train_iter, snli_train_iter = train_iter, train_iter

	return (snli_train_iter, snli_dev_iter, snli_test_iter), (mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter), testl_field.vocab.vectors


def load_data_with_dict(data_root, embd_file, reseversed=True, batch_sizes=(32, 32, 32, 32, 32), device=-1):
	if reseversed:
		testl_field = RParsedTextLField()
	else:
		testl_field = ParsedTextLField()

	transitions_field = datasets.snli.ShiftReduceField()
	y_field = data.Field(sequential=False)
	g_field = data.Field(sequential=False)

	train_size, dev_size, test_size, m_dev_size, m_test_size = batch_sizes

	snli_train, snli_dev, snli_test = datasets.SNLI.splits(testl_field, y_field, transitions_field, root=data_root)

	mnli_train, mnli_dev_m, mnli_dev_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
													  train='train.jsonl',
													  validation='dev_matched.jsonl',
													  test='dev_mismatched.jsonl')

	mnli_test_m, mnli_test_um = MNLI.splits(testl_field, y_field, transitions_field, g_field, root=data_root,
											train=None,
											validation='test_matched_unlabeled.jsonl',
											test='test_mismatched_unlabeled.jsonl')

	testl_field.build_vocab(snli_train, snli_dev, snli_test,
							mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)

	g_field.build_vocab(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um)
	y_field.build_vocab(snli_train)
	print('Important:', y_field.vocab.itos)
	testl_field.vocab.vectors = torch.load(embd_file, map_location=lambda storage, loc: storage)

	snli_train_iter, snli_dev_iter, snli_test_iter = data.Iterator.splits(
		(snli_train, snli_dev, snli_test), batch_sizes=batch_sizes, device=device, shuffle=False)

	mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter = data.Iterator.splits(
		(mnli_train, mnli_dev_m, mnli_dev_um, mnli_test_m, mnli_test_um),
		batch_sizes=(train_size, m_dev_size, m_test_size, m_dev_size, m_test_size),
		device=device, shuffle=False, sort=False)

	return (snli_train_iter, snli_dev_iter, snli_test_iter), (mnli_train_iter, mnli_dev_m_iter, mnli_dev_um_iter, mnli_test_m_iter, mnli_test_um_iter), testl_field.vocab.vectors, testl_field.vocab


def raw_input(ws, dict, device=-1):
	# ws = ['I', 'like', 'research', '.']
	ws_t = Variable(torch.from_numpy(np.asarray([[dict.stoi[w]] for w in ws], dtype=np.int64)))
	wl_t = torch.LongTensor(1).zero_()
	wl_t[0] = len(ws)

	if device != -1 and torch.cuda.is_available():
		wl_t.cuda()
		ws_t.cuda()

	return ws_t, wl_t


def combine_two_set(set_1, set_2, rate=(1, 1), seed=0):
	np.random.seed(seed)
	len_1 = len(set_1)
	len_2 = len(set_2)
	# print(len_1, len_2)
	p1, p2 = rate
	c_1 = np.random.choice([0, 1], len_1, p=[1 - p1, p1])
	c_2 = np.random.choice([0, 1], len_2, p=[1 - p2, p2])
	iter_1 = itertools.compress(iter(set_1), c_1)
	iter_2 = itertools.compress(iter(set_2), c_2)
	for it in itertools.chain(iter_1, iter_2):
		yield it


if __name__ == '__main__':
	pass
  