import torch.nn as nn
import torch_util
import torch
import torch.nn.functional as F

class StackBiLSTMMaxout(nn.Module):
	def __init__(self, h_size, v_size=10, d=300, mlp_d=1600, dropout_r=0.1, max_l=60, num_class=3):
		super(StackBiLSTMMaxout, self).__init__()
		self.Embd = nn.Embedding(v_size, d)

		self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
							num_layers=1, bidirectional=True)

		self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
							  num_layers=1, bidirectional=True)

		self.lstm_2 = nn.LSTM(input_size=(d + (h_size[0] + h_size[1]) * 2), hidden_size=h_size[2],
							  num_layers=1, bidirectional=True)

		self.max_l = max_l
		self.h_size = h_size

		self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
		self.mlp_2 = nn.Linear(mlp_d, mlp_d)
		self.sm = nn.Linear(mlp_d, num_class)

		self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
										  self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
										  self.sm])

	def display(self):
		for param in self.parameters():
			print(param.data.size())

	def forward(self, s1, l1, s2, l2):
		if self.max_l:
			l1 = l1.clamp(max=self.max_l)
			l2 = l2.clamp(max=self.max_l)
			if s1.size(0) > self.max_l:
				s1 = s1[:self.max_l, :]
			if s2.size(0) > self.max_l:
				s2 = s2[:self.max_l, :]

		p_s1 = self.Embd(s1)
		p_s2 = self.Embd(s2)

		s1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
		s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

		# Length truncate
		len1 = s1_layer1_out.size(0)
		len2 = s2_layer1_out.size(0)
		p_s1 = p_s1[:len1, :, :] # [T, B, D]
		p_s2 = p_s2[:len2, :, :] # [T, B, D]

		# Using residual connection
		s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
		s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

		s1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
		s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

		s1_layer3_in = torch.cat([p_s1, s1_layer1_out, s1_layer2_out], dim=2)
		s2_layer3_in = torch.cat([p_s2, s2_layer1_out, s2_layer2_out], dim=2)

		s1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
		s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

		s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, l1)
		s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

		# Only use the last layer
		features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
							  torch.abs(s1_layer3_maxout - s2_layer3_maxout),
							  s1_layer3_maxout * s2_layer3_maxout],
							 dim=1)

		out = self.classifier(features)
		return out

def model_eval(model, data_iter, criterion, pred=False):
	model.eval()
	data_iter.init_epoch()
	n_correct = loss = 0
	totoal_size = 0
	prob=[]
	if not pred:
		for batch_idx, batch in enumerate(data_iter):

			s1, s1_l = batch.premise
			s2, s2_l = batch.hypothesis
			y = batch.label.data - 1

			pred = model(s1, s1_l - 1, s2, s2_l - 1)
			n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == y).sum()

			loss += criterion(pred, batch.label - 1).data[0] * batch.batch_size
			totoal_size += batch.batch_size

			output=F.softmax(pred)
			result = output.data.cpu().numpy()
			prob.extend(result)

		avg_acc = 100. * n_correct / totoal_size
		avg_loss = loss / totoal_size

		return avg_acc, avg_loss, prob
	else:
		pred_list = []
		for batch_idx, batch in enumerate(data_iter):

			s1, s1_l = batch.premise
			s2, s2_l = batch.hypothesis

			pred = model(s1, s1_l - 1, s2, s2_l - 1)
			pred_list.append(torch.max(pred, 1)[1].view(batch.label.size()).data)

		return torch.cat(pred_list, dim=0)