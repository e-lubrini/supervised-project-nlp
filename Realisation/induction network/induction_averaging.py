import configparser
import copy
import os
import pickle
import random

import numpy as np
import torch
from criterion import Criterion
from model import FewShotInduction
from torch import optim

config = configparser.ConfigParser()
config.read("config.ini")

seed = int(config['model']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

log_interval = int(config['model']['log_interval'])
dev_interval = int(config['model']['dev_interval'])

train_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['train_loader']), 'rb'))
dev_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'rb'))
test_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['test_loader']), 'rb'))

vocabulary = pickle.load(open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'rb'))

weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
support = int(config['model']['support'])
model = FewShotInduction(C=int(config['model']['class']),
                         S=support,
                         vocab_size=len(vocabulary),
                         embed_size=int(config['model']['embed_dim']),
                         hidden_size=int(config['model']['hidden_dim']),
                         d_a=int(config['model']['d_a']),
                         iterations=int(config['model']['iterations']),
                         outsize=int(config['model']['relation_dim']),
                         weights=weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=float(config['model']['lr']))
criterion = Criterion(way=int(config['model']['class']),
                      shot=int(config['model']['support']))

first_node = ['apparel', 'office_products', 'automotive', 'toys_games', 'computer_video_games', 'software']
second_node = ['grocery', 'beauty', 'magazines', 'jewelry_watches', 'sports_outdoors', 'cell_phones_service', 'baby']
third_node = ['outdoor_living', 'video', 'camera_photo', 'health_personal_care', 'gourmet_food', 'music']


def train(episode, node):
    model.train()
    data, target = train_loader.get_batch()
    needed = False
    while not needed:
        if train_loader.filenames[train_loader.index][:-9] not in node:
            data, target = train_loader.get_batch()
        else:
            needed = True
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    predict = model(data)
    loss, acc = criterion(predict, target)
    loss.backward()
    optimizer.step()
    if episode % log_interval == 0:
        print('Train Episode: {} Loss: {} Acc: {}'.format(episode, loss.item(), acc))


def dev(model, episode):
    model.eval()
    correct = 0.
    count = 0.
    for data, target in dev_loader:
        data = data.to(device)
        target = target.to(device)
        predict = model(data)
        _, acc = criterion(predict, target)
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    print('Dev Episode: {} Acc: {}'.format(episode, acc))
    return acc


def test(model):
    model.eval()
    correct = 0.
    count = 0.
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        predict = model(data)
        _, acc = criterion(predict, target)
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    print('Test Acc: {}'.format(acc))
    return acc


def main(model, node):
    best_episode, best_acc = 0, 0.
    episodes = 10000
    early_stop = int(config['model']['early_stop']) * dev_interval
    for episode in range(1, episodes + 1):
        train(episode, node)
        if episode % dev_interval == 0:
            acc = dev(model, episode)
            if acc > best_acc:
                print('Better acc! Saving model!')
                torch.save(model.state_dict(), config['model']['model_path'])
                best_episode, best_acc = episode, acc
            if episode - best_episode >= early_stop:
                print('Early stop at episode', episode)
                break

    print('Reload the best model on episode', best_episode, 'with best acc', best_acc.item())
    ckpt = torch.load(config['model']['model_path'])
    model.load_state_dict(ckpt)
    test(model)
    return model


def test_ensemble(model1, model2, model3):
    model1.eval()
    model2.eval()
    model3.eval()
    correct = 0.
    count = 0.
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        pred1 = model1(data)
        pred2 = model2(data)
        pred3 = model3(data)
        predict = torch.stack([pred1, pred2, pred3]).mean(0)
        _, acc = criterion(predict, target)
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    print('Test Acc: {}'.format(acc))
    return acc


model1 = main(copy.deepcopy(model), first_node)
model2 = main(copy.deepcopy(model), second_node)
model3 = main(copy.deepcopy(model), third_node)

test_ensemble(model1, model2, model3)
