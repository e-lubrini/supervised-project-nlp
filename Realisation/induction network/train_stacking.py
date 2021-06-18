import os
import configparser
import pickle
import random
import numpy as np
import torch
from model import FewShotInduction
from criterion import Criterion
from tensorboardX import SummaryWriter
from torch import optim
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

config = configparser.ConfigParser()
config.read("config.ini")

# seed
seed = int(config['model']['seed'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# log_interval
log_interval = int(config['model']['log_interval'])
dev_interval = int(config['model']['dev_interval'])

# data loaders
train_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['train_loader']), 'rb'))
dev_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'rb'))
test_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['test_loader']), 'rb'))

vocabulary = pickle.load(open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'rb'))

# word2vec weights
weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))

# model & optimizer & criterion
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

# writer
os.makedirs(config['model']['log_path'], exist_ok=True)
writer = SummaryWriter(config['model']['log_path'])


#first_node = ['apparel','office_products','automotive','toys_games','computer_video_games','software']
#second_node = ['grocery','beauty','magazines','jewelry_watches','sports_outdoors','cell_phones_service','baby']
#third_node = ['outdoor_living','video','camera_photo','health_personal_care','gourmet_food', 'music']


def test():
    model1.eval()
    model2.eval()
    model3.eval()
    correct = 0.
    count = 0.
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        #predict = model(data)
        pred1 = model1(data)
        pred2 = model2(data)
        pred3 = model3(data)
        predict = np.concatenate((pred1.tolist(), pred2.tolist(), pred3.tolist()), axis=1)
        pred = predict.tolist()
        predict = torch.tensor(mlp.predict_proba(pred), device='cuda')
        _, acc = criterion(predict, target)
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    writer.add_scalar('test_acc', acc)
    print('Test Acc: {}'.format(acc))
    return acc

model1 = FewShotInduction(C=int(config['model']['class']),
                          S=support,
                          vocab_size=len(vocabulary),
                          embed_size=int(config['model']['embed_dim']),
                          hidden_size=int(config['model']['hidden_dim']),
                          d_a=int(config['model']['d_a']),
                          iterations=int(config['model']['iterations']),
                          outsize=int(config['model']['relation_dim']),
                          weights=weights).to(device)

model2 = FewShotInduction(C=int(config['model']['class']),
                          S=support,
                          vocab_size=len(vocabulary),
                          embed_size=int(config['model']['embed_dim']),
                          hidden_size=int(config['model']['hidden_dim']),
                          d_a=int(config['model']['d_a']),
                          iterations=int(config['model']['iterations']),
                          outsize=int(config['model']['relation_dim']),
                          weights=weights).to(device)

model3 = FewShotInduction(C=int(config['model']['class']),
                          S=support,
                          vocab_size=len(vocabulary),
                          embed_size=int(config['model']['embed_dim']),
                          hidden_size=int(config['model']['hidden_dim']),
                          d_a=int(config['model']['d_a']),
                          iterations=int(config['model']['iterations']),
                          outsize=int(config['model']['relation_dim']),
                          weights=weights).to(device)

ckpt1 = torch.load('fewshot_model1.pt')
model1.load_state_dict(ckpt1)
ckpt2 = torch.load('fewshot_model6.pt')
model2.load_state_dict(ckpt2)
ckpt3 = torch.load('fewshot_model3.pt')
model3.load_state_dict(ckpt3)

predicted = []
true = []

data, target = train_loader.get_batch()
needed = False
while not needed:
    if train_loader.filenames[train_loader.index][:-9] not in ['baby']:
        data, target = train_loader.get_batch()
    else:
      needed = True

for model in [model1, model2, model3]:
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    predicted.append(model(data).tolist())
    true = target.tolist()[10:]

X = np.concatenate(predicted, axis=1)

mlp = LogisticRegression(max_iter=500)
mlp.fit(X, true)



test()
"""
# random search logistic regression model on the sonar dataset
#from scipy.stats import loguniform
from loguniform import LogUniform
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# define model
mlp = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = np.logspace(-3,3,10)
# define search
search = GridSearchCV(mlp, space,  scoring='accuracy', n_jobs=-1, cv=cv,  verbose=1)
# execute search
result = search.fit(X, true)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

"""