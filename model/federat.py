class Federation:
  """ 
  
  dataset : list - list of N tuples where the last element is label, previous are task dependent
  task : string - type of task to train the model. possible values: snli
  num_devices : int - number on how many parts dataset should be divided (and, consequently, how many models should be trained)
  num_train : int - number of devices to use for training

  """
  def __init__(self, dataset, task, num_devices=500, num_train=5):
    self.full_dataset = dataset
    self.task = task
    self.num_devices = num_devices
    self.num_train = num_train
    self.datasets = None
    self.state_dict = None
    self.l2i = {}
    self.w2i = {}
    self.model = None
    
    

  def get_num(self, num=0):
    loc = 2.5 if self.num_devices > 100 else 0
    num = int(abs(norm.rvs(loc=loc, scale=1, size=1, random_state=None)*100))
    if 0 < num < self.num_devices+1:
      return num
    else:
      return self.get_num(num)

  def get_dict(self, sentences):
    w2i = {}
    i = 0
    for sentence in sentences:
      for word in sentence:
        if word not in w2i:
          w2i[word] = i
          i += 1
    w2i['UNK'] = len(w2i)
    return w2i

  def divide_dataset(self, dataset, num_devices):
    datasets = [[] for _ in range(num_devices)]
    for obj in dataset:
      num = self.get_num()
      datasets[num-1].append(obj)
    return datasets

  def training(self, model, epoches=8, valid=None, **params):
    print('Dataset division')
    self.datasets = self.divide_dataset(self.full_dataset, self.num_devices)
    self.datasets.sort(reverse=True)
    print('Training start')
    t = tqdm(self.datasets[:self.num_train])
    for i, dataset in enumerate(t):
      if self.task == 'snli':
        self.l2i = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        self.w2i = self.get_dict([x[0] for x in dataset]+[x[1] for x in dataset])
        self.model = model(len(self.w2i), params['matching'], params['device'], params['hid_lstm'], params['hid_dnn'])
        print(torch.cuda.max_memory_allocated())
        if self.state_dict:
          next_state_dict = self.model.state_dict()
          self.state_dict['biLSTM_prem.embedding.weight'] = next_state_dict['biLSTM_prem.embedding.weight']
          self.state_dict['biLSTM_hyp.embedding.weight'] = next_state_dict['biLSTM_hyp.embedding.weight']
          self.model.load_state_dict(self.state_dict)
        self.model.train()
        self.train_model(dataset, self.w2i, self.l2i, epoches=epoches)
        t.set_description(f"loss: {round(float(loss), 3)}")
        t.refresh()
        self.state_dict = self.model.state_dict()
        if valid:
          for elem in valid:
            sent1 = torch.tensor([self.w2i.get(x, self.w2i['UNK']) for x in elem[0]], dtype=torch.long, device=device)
            sent2 = torch.tensor([self.w2i.get(x, self.w2i['UNK']) for x in elem[1]], dtype=torch.long, device=device)
            pred = int(torch.max(self.model(sent1, sent2), 1)[1])
            if pred == self.l2i[elem[2][0]]:
              acc += 1
          print(f'accuracy for the {i} device: {round(acc/len(valid), 3)}')


  
  def train_model(self, dataset, w2i, l2i, epoches):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    for epoch in range(epoches):
      for sentence1, sentence2, label in dataset:
        optimizer.zero_grad()
        sent1 = torch.tensor([w2i[x] for x in sentence1], dtype=torch.long, device=device)
        sent2 = torch.tensor([w2i[x] for x in sentence2], dtype=torch.long, device=device)
        output = self.model(sent1, sent2)
        lbl = torch.tensor([l2i[label[0]]], dtype=torch.long, device=device)
        loss = loss_function(output, lbl)
        loss.backward()
        optimizer.step()