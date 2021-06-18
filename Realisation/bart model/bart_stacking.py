import torch

from sklearn.linear_model import LogisticRegression

from transformers import BartForSequenceClassification, BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

auto_domains = ('automotive.t2.train', 'automotive.t4.train', 'automotive.t5.train')

test_domains = ('books.t2.test', 'dvd.t2.test', 'electronics.t2.test', 'kitchen_housewares.t2.test',
                'books.t4.test', 'dvd.t4.test', 'electronics.t4.test', 'kitchen_housewares.t4.test',
                'books.t5.test', 'dvd.t5.test', 'electronics.t5.test', 'kitchen_housewares.t5.test')


def transform_domains(domains, dom_type):
    domains_test = {}
    dom_name = len(dom_type) + 1
    for file in test_domains:
        with open('../DiverseFewShot_Amazon/Amazon_few_shot/' + file) as f:
            f = f.readlines()
            if file[:-5] not in domains_test:
                domains_test[file[:-dom_name]] = []
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
                domains_test[file[:-dom_name]].append((tokens, label))
    return domains_test


domains_test = transform_domains(test_domains, 'test')
domains_train = transform_domains(auto_domains, 'train')

mod_books = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
ckpt = torch.load('../models/bart_model_books.pt')
mod_books.load_state_dict(ckpt)
mod_books.eval()

mod_elec = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
ckpt = torch.load('../models/bart_model_elec.pt')
mod_elec.load_state_dict(ckpt)
mod_elec.eval()

mod_dvd = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
ckpt = torch.load('../models/bart_model_dvd.pt')
mod_dvd.load_state_dict(ckpt)
mod_dvd.eval()

mod_kith = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
ckpt = torch.load('../models/bart_model_kitchen.pt')
mod_kith.load_state_dict(ckpt)
mod_kith.eval()

train_samples = sum(list(domains_train.values()), [])
predicted = []
true = []

for posneg, label in train_samples:
    pb = []
    for mod in [mod_books, mod_elec, mod_dvd, mod_kith]:
        output = mod(posneg)[0]
        posneg_output = output[:, [0, 2]]
        probs = posneg_output.softmax(dim=1)
        part_pos = probs[0][1].item()
        part_neg = probs[1][1].item()
        pb.append(part_pos)
        pb.append(part_neg)
    predicted.append(pb)
    true.append(label)

lr = LogisticRegression(max_iter=500)
lr.fit(predicted, true)

total_acc = []
for domain, examples in domains_test.items():
    cor = 0
    for posneg, label in examples:
        pb = []
        for mod in [mod_books, mod_elec, mod_dvd, mod_kith]:
            output = mod(posneg)[0]
            posneg_output = output[:, [0, 2]]
            probs = posneg_output.softmax(dim=1)
            part_pos = probs[0][1].item()
            part_neg = probs[1][1].item()
            pb.append(part_pos)
            pb.append(part_neg)
        pred = lr.predict([pb])
        if pred[0] == label:
            cor += 1
    accuracy = cor / len(examples)
    print(f'{domain} ACCURACY : {accuracy} ({round(accuracy * 100, 1)})')
    total_acc.append(accuracy)
print(f'TOTAL ACCURACY : {round(sum(total_acc) / len(total_acc) * 100, 1)} ({sum(total_acc) / len(total_acc)})')
