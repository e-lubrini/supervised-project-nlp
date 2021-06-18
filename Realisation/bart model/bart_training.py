import torch
from transformers import BartForSequenceClassification, BartTokenizer

model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

domains_train = {}

training_domains = ('books.t2.train', 'dvd.t2.train', 'electronics.t2.train', 'kitchen_housewares.t2.train',
                    'books.t4.train', 'dvd.t4.train', 'electronics.t4.train', 'kitchen_housewares.t4.train',
                    'books.t5.train', 'dvd.t5.train', 'electronics.t5.train', 'kitchen_housewares.t5.train')

for file in training_domains:
    with open('../DiverseFewShot_Amazon/Amazon_few_shot/' + file) as f:
        f = f.readlines()
        if file[:-9] not in domains_train:
            domains_train[file[:-9]] = []
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
            domains_train[file[:-9]].append((tokens, label))

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
crossentropy = torch.nn.CrossEntropyLoss()
model.train()


def train_model(model, dataset, num_epoch=100):
    for epoch in range(num_epoch):
        print('epoch', epoch)
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


mod = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
print('Train books')
model = train_model(domains_train['books'], mod, 100)
torch.save(model.state_dict(), '../models/bart_model_books.pt')

mod = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
print('Train dvd')
model = train_model(domains_train['dvd'], mod, 100)
torch.save(model.state_dict(), '../models/bart_model_dvd.pt')

mod = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
print('Train electronics')
model = train_model(domains_train['electronics'], mod, 100)
torch.save(model.state_dict(), '../models/bart_model_elec.pt')

mod = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
print('Train kitchen')
model = train_model(domains_train['kitchen_housewares'], mod, 100)
torch.save(model.state_dict(), '../models/art_model_kitchen.pt')
