import torch
import random
from scipy.stats import norm
from torch import nn
from torch import optim
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, dict_size, hidden_dim, lstm_layer, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(dict_size, hidden_dim).to(self.device)
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer, 
                            bidirectional=True).to(self.device)
  
    
    def forward(self, sentence):
        x = self.embedding(sentence).view(sentence.size(0), 1, -1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        pooled = torch.max(lstm_out, 0)[0]
        return pooled

class DNN(nn.Module):
  def __init__(self, input_size, hidden_size, device):
    super(DNN, self).__init__()
    self.device = device
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.linear1 = torch.nn.Linear(self.input_size, self.hidden_size).to(self.device)
    self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
    self.linear3 = torch.nn.Linear(self.hidden_size, 3).to(self.device)
    self.ReLU = torch.nn.ReLU(self.hidden_size)

  def forward(self, vector):
    output = self.linear3(self.ReLU(self.linear2(self.ReLU(self.linear1(vector)))))
    return output

class BiLSTMDNN(nn.Module):
  def __init__(self, dict_size, matching, device, hid_lstm=2048, lstm_layer=2, hid_dnn=4096):
    super(BiLSTMDNN, self).__init__()
    self.device = device
    self.dict_size = dict_size
    self.hid_lstm = hid_lstm
    self.lstm_layer = lstm_layer
    self.hid_dnn = hid_dnn
    self.matching = matching
    self.biLSTM_prem = BiLSTM(self.dict_size, self.hid_lstm, self.lstm_layer, device=self.device)
    self.biLSTM_hyp = BiLSTM(self.dict_size, self.hid_lstm, self.lstm_layer, device=self.device)
    if self.matching == 'conc':
      self.DNN = DNN(self.hid_lstm*2*2, self.hid_dnn, device=self.device)
    else:
      self.DNN = DNN(self.hid_lstm*2, self.hid_dnn, device=self.device)

  def forward(self, premise, hypothesis):
    emb1 = self.biLSTM_prem(premise)
    emb2 = self.biLSTM_hyp(hypothesis)
    if self.matching == 'conc':
      input = torch.cat((emb1, emb2), 1)
    elif self.matching == 'prod':
      input = emb1*emb2
    elif self.matching == 'diff':
      input = torch.abs(emb1-emb2)
    output = self.DNN(input)
    return output

def get_num(num=0):
  num = int(abs(norm.rvs(loc=2.5, scale=1, size=1, random_state=None)*100))
  #print(num)
  if 0 < num < 501:
    return num
  else:
    #print(num)
    return get_num(num)

def train(model, dataset, valid, w2i, l2i, epoches=1):
  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  for epoch in range(epoches):
    t = tqdm(dataset)
    for sentence1, sentence2, label in t:
      optimizer.zero_grad()
      sent1 = torch.tensor([w2i[x] for x in sentence1], dtype=torch.long, device=device)
      sent2 = torch.tensor([w2i[x] for x in sentence2], dtype=torch.long, device=device)
      output = model(sent1, sent2)
      lbl = torch.tensor([l2i[label[0]]], dtype=torch.long, device=device)
      loss = loss_function(output, lbl)
      t.set_description(f"loss: {round(float(loss), 3)}")
      t.refresh()
      loss.backward()
      optimizer.step()
    acc = 0
    for elem in valid:
      sent1 = torch.tensor([w2i.get(x, w2i['UNK']) for x in elem[0]], dtype=torch.long, device=device)
      sent2 = torch.tensor([w2i.get(x, w2i['UNK']) for x in elem[1]], dtype=torch.long, device=device)
      pred = int(torch.max(model(sent1, sent2), 1)[1])
      if pred == l2i[elem[2][0]]:
        acc += 1
    print(f'accuracy: {round(acc/len(valid), 3)}')






