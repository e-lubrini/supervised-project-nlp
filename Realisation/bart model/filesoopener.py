domains_train = {}
domains_test = {}

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
            domains_train[file[:-9]] = []
        elif file.endswith('test'):
            domains_test[file[:-8]] = []
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
                domains_train[file[:-9]].append((tokens, label))
            elif file.endswith('test'):
                domains_test[file[:-8]].append((tokens, label))
