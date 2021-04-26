import copy
from collections import OrderedDict
from sklearn.exceptions import NotFittedError
import torch

class Client:
    def __init__(self, dataloader):
        self.model = None
        self.dataloader = dataloader
    
    def update_model(self, optimizer, lr, loss_fn, epoches):
        self.model.train()
        optimizer = optimizer(self.model.parameters(), lr=lr)
        for epoch in range(epoches):
            for sentence, label in self.dataloader:
              print(sentence)
              print(label)
              pred = self.model(sentence)
              loss = loss_fn(pred, label)
              loss.backward()
              optimizer.step()
              optimizer.zero_grad()
        
class Server:
    def __init__(self, model, dataset, optimizer, lr, loss_fn, num_clients, device):
      self.model = model
      self.dataset = dataset
      self.optimizer = optimizer
      self.lr = lr
      self.num_clients = num_clients
      self.loss_fn = loss_fn
      self.device = device
      self.fitted = False
    
    def fit(self):
      fract = len(self.dataset)//self.num_clients
      datasets = [self.dataset[fract*i:fract*(i+1)] for i in range(int(len(self.dataset)/fract) + 1)]
      dataloaders = [torch.utils.data.DataLoader(dataset) for dataset in datasets]
      self.aggregation_weights = [len(dataset)/len(self.dataset) for dataset in datasets]
      self.clients = [Client(dataloader) for dataloader in dataloaders]
      for client in self.clients:
          client.model = copy.deepcopy(self.model)
      print(f'{self.num_clients} clients are ready.\nEach client has {len(datasets[0])} examples.')
      self.fitted = True

    def train(self, global_epoches, local_epoches):
      if not self.fitted:
        raise NotFittedError('method fit should be called before the training')
      for num in range(global_epoches):
          update_state = OrderedDict()
          for n, client in enumerate(self.clients):
              client.update_model(self.optimizer, self.lr, self.loss_fn, local_epoches)
              local_state = client.model.state_dict()
              for key in self.model.state_dict().keys():
                  if k == 0:
                      update_state[key] = local_state[key] * aggregation_weights[k]
                  else:
                      update_state[key] += local_state[key] * aggregation_weights[k]
          self.model.load_state_dict(update_state)
          for client in self.clients:
              client.model = copy.deepcopy(self.model)
