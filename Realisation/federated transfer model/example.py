import torch

from simplemodel import BiLSTMDNN, train, get_num


#you need to download SNLI dataset from SentEval. For example, like this:
#git clone https://github.com/facebookresearch/SentEval.git
#cd SentEval/data/downstream
#./get_transfer_data.bash
#cd ../../..


DATA_FOLDER = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l2i = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

w2i = {}
i = 0


with open(f'{DATA_FOLDER}/s1.train') as file1, open(f'{DATA_FOLDER}/s2.train') as file2, open(f'{DATA_FOLDER}/labels.train') as file3:
  file1 = [x.split() for x in file1.readlines()]
  file2 = [x.split() for x in file2.readlines()]
  file3 = [x.split() for x in file3.readlines()]
dataset = [(s1, s2, l) for s1, s2, l in zip(file1, file2, file3)]

for sentence in file1+file2:
  for word in sentence:
    if word not in w2i:
      w2i[word] = i
      i += 1
w2i['UNK'] = len(w2i)

with open(f'{DATA_FOLDER}/s1.test') as file4, open(f'{DATA_FOLDER}/s2.test') as file5, open(f'{DATA_FOLDER}/labels.test') as file6:
  file4 = [x.split() for x in file4.readlines()]
  file5 = [x.split() for x in file5.readlines()]
  file6 = [x.split() for x in file6.readlines()]
testset = [(s1, s2, l) for s1, s2, l in zip(file4, file5, file6)]

num_devices = 500
datasets = [[] for _ in range(num_devices)]
for obj in dataset:
  num = get_num()
  datasets[num-1].append(obj)

conc_model = BiLSTMDNN(len(w2i), 'conc', device=device, hid_lstm=200, hid_dnn=400)
conc_model.train()

train(conc_model, datasets[0], testset[:200], w2i, l2i, epoches=10)