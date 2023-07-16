import os
import pickle
from tqdm import tqdm
from config import pickle_file,trans_file,wav_folder
# 获得pickle文件
def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)

def get_data(split, n_samples):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(trans_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    tran_dict = dict()
    for line in lines:
        tokens = line.split()
        key = tokens[0]
        trn = ''.join(tokens[1:])
        tran_dict[key] = trn

    samples = []

    #n_samples = 5000
    rest = n_samples 
    
    folder = os.path.join(wav_folder, split)
    ensure_folder(folder)
    dirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for dir in tqdm(dirs):
        files = [f for f in os.listdir(dir) if f.endswith('.wav')]

        rest = len(files) if n_samples <= 0 else rest

        for f in files[:rest]:

            wave = os.path.join(dir, f)

            key = f.split('.')[0]

            if key in tran_dict:
                trn = tran_dict[key]
                trn = ['<sos>']+list(trn.strip()) + ['<eos>']

                for token in trn:
                    build_vocab(token)

                trn = [VOCAB[token] for token in trn]

                samples.append({'trn': trn, 'wave': wave})
        
        rest = rest - len(files) if n_samples > 0 else rest
        if rest <= 0 :
            break  

    print('split: {}, num_files: {}'.format(split, len(samples)))
    return samples


def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token


n_samples = {"train":-1, "dev":-1,"test":-1}

VOCAB = {'<PAD>':0,'<sos>': 1, '<eos>': 2}
IVOCAB = {0:'<PAD>',1: '<sos>', 2: '<eos>'}

data = dict()
data['VOCAB'] = VOCAB
data['IVOCAB'] = IVOCAB
data['train'] = get_data('train', n_samples["train"])
data['dev'] = get_data('dev', n_samples["dev"])
data['test'] = get_data('test', n_samples["test"])

with open(pickle_file, 'wb') as file:
    pickle.dump(data, file)

print('num_train: ' + str(len(data['train'])))
print('num_dev: ' + str(len(data['dev'])))
print('num_test: ' + str(len(data['test'])))
print('vocab_size: ' + str(len(data['VOCAB'])))
# split: test, num_files: 7176
# num_train: 120098
# num_dev: 14326
# num_test: 7176
# vocab_size: 4335