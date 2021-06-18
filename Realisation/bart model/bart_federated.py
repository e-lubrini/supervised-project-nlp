import copy
from collections import OrderedDict

import torch
from transformers import BartForSequenceClassification, BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

domains_test = {}
domains_train = {}

training_domains = ('books.t2.train', 'dvd.t2.train', 'electronics.t2.train', 'kitchen_housewares.t2.train',
                    'books.t4.train', 'dvd.t4.train', 'electronics.t4.train', 'kitchen_housewares.t4.train',
                    'books.t5.train', 'dvd.t5.train', 'electronics.t5.train', 'kitchen_housewares.t5.train',
                    'books.t2.test', 'dvd.t2.test', 'electronics.t2.test', 'kitchen_housewares.t2.test',
                    'books.t4.test', 'dvd.t4.test', 'electronics.t4.test', 'kitchen_housewares.t4.test',
                    'books.t5.test', 'dvd.t5.test', 'electronics.t5.test', 'kitchen_housewares.t5.test')

for file in training_domains:
    with open('../DiverseFewShot_Amazon/Amazon_few_shot/' + file) as f:
        f = f.readlines()
        if file.endswith('train'):
            if file[:-6] not in domains_train:
                domains_train[file[:-6]] = []
        elif file.endswith('test'):
            if file[:-5] not in domains_test:
                domains_test[file[:-5]] = []
        for text in f:
            sentence = text.split('\t')[0]
            label = int(text.split('\t')[1].strip())
            if label == -1:
                label = 0
            positive = 'This text is positive'
            negative = 'This text is negative'
            pos_tokens = tokenizer.encode(sentence, positive, max_length=512, truncation=True, return_tensors='pt')
            neg_tokens = tokenizer.encode(sentence, negative, max_length=512, truncation=True, return_tensors='pt')
            tokens = torch.stack((pos_tokens, neg_tokens), dim=0)
            if file.endswith('train'):
                domains_train[file[:-6]].append((tokens, label))
            elif file.endswith('test'):
                domains_test[file[:-5]].append((tokens, label))


def do_step(model, dataset, optimizer, crossentropy):
    for posneg, label in dataset:
        optimizer.zero_grad()
        output = model(posneg)[0]
        posneg_output = output[:, [0, 2]]
        probs = posneg_output.softmax(dim=1)
        if label == 1:
            y = torch.LongTensor([1, 0])
        else:
            y = torch.LongTensor([0, 1])
        loss = crossentropy(probs, y)
        loss.backward()
        optimizer.step()
    return model


mod1 = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
opt1 = torch.optim.SGD(mod1.parameters(), lr=0.0001)
loss1 = torch.nn.CrossEntropyLoss()
mod1.train()

mod2 = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
opt2 = torch.optim.SGD(mod2.parameters(), lr=0.0001)
loss2 = torch.nn.CrossEntropyLoss()
mod2.train()

mod3 = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
opt3 = torch.optim.SGD(mod3.parameters(), lr=0.0001)
loss3 = torch.nn.CrossEntropyLoss()
mod3.train()

mod4 = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
opt4 = torch.optim.SGD(mod4.parameters(), lr=0.0001)
loss4 = torch.nn.CrossEntropyLoss()
mod4.train()

models_list = [[mod1, opt1, loss1, 'books'], [mod2, opt2, loss2, 'dvd'],
               [mod3, opt3, loss3, 'electronics'], [mod4, opt4, loss4, 'kitchen_housewares']]

model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')

for ep in range(100):
    local_states = []
    for num_mod, (mod, opt, lossf, domain) in enumerate(models_list):
        models_list[num_mod][0] = do_step(mod, domains_train[domain], opt, lossf)
        print(f'FineTune{domain} Epoch {ep}')
        local_states.append(models_list[num_mod][0].state_dict())

    update_state = OrderedDict()
    for n, local_state in enumerate(local_states):
        for key in model.state_dict().keys():
            if n == 0:
                update_state[key] = local_state[key] * 0.25
            else:
                update_state[key] += local_state[key] * 0.25
    model.load_state_dict(update_state)
    if ep != 99:
        print('update')
        for num in range(len(models_list)):
            models_list[num][0] = copy.deepcopy(model)


def test(model, dataset):
    total_acc = []
    for domain, examples in dataset.items():
        cor = 0
        for posneg, label in examples:
            output = model(posneg)[0]
            posneg_output = output[:, [0, 2]]
            probs = posneg_output.softmax(dim=1)
            pos = probs[0][1].item()
            neg = probs[1][1].item()
            if pos >= neg and label == 1:
                cor += 1
            elif pos < neg and label == 0:
                cor += 1
        accuracy = cor / len(examples)
        print(f'{domain} ACCURACY : {accuracy} ({round(accuracy * 100, 1)})')
        total_acc.append(accuracy)
    print(f'TOTAL ACCURACY : {round(sum(total_acc) / len(total_acc) * 100, 1)} ({sum(total_acc) / len(total_acc)})')


model.eval()
print('Evaluate server')
test(model, domains_test)
