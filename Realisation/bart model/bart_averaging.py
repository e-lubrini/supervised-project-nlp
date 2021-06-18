import torch
from transformers import BartForSequenceClassification, BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

domains_test = {}

test_domains = ('books.t2.test', 'dvd.t2.test', 'electronics.t2.test', 'kitchen_housewares.t2.test',
                'books.t4.test', 'dvd.t4.test', 'electronics.t4.test', 'kitchen_housewares.t4.test',
                'books.t5.test', 'dvd.t5.test', 'electronics.t5.test', 'kitchen_housewares.t5.test')

for file in test_domains:
    with open('../DiverseFewShot_Amazon/Amazon_few_shot/' + file) as f:
        f = f.readlines()
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
            domains_test[file[:-5]].append((tokens, label))

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

total_acc = []
for domain, examples in domains_test.items():
    cor = 0
    for posneg, label in examples:
        pos = []
        neg = []
        for mod in [mod_books, mod_elec, mod_dvd, mod_kith]:
            output = mod(posneg)[0]
            posneg_output = output[:, [0, 2]]
            probs = posneg_output.softmax(dim=1)
            part_pos = probs[0][1].item()
            part_neg = probs[1][1].item()
            pos.append(part_pos)
            neg.append(part_neg)
        pospb = sum(pos) / len(pos)
        negpb = sum(neg) / len(neg)

        if pos >= neg and label == 1:
            cor += 1
        elif pos < neg and label == 0:
            cor += 1
    accuracy = cor / len(examples)
    print(f'{domain} ACCURACY : {accuracy} ({round(accuracy * 100, 1)})')
    total_acc.append(accuracy)
print(f'TOTAL ACCURACY : {round(sum(total_acc) / len(total_acc) * 100, 1)} ({sum(total_acc) / len(total_acc)})')
