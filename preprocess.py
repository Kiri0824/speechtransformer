import os
import pickle
from tqdm import tqdm
# config
DATA_DIR = '../dataset/'
aishell_folder = '../dataset/data_aishell/'
wav_folder = os.path.join(aishell_folder, 'wav')
tran_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
pickle_file = '../dataset/aishell.pickle'
# utils
import argparse
def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)
def parse_args():
    parser = argparse.ArgumentParser(description='Speech Transformer')
    # Low Frame Rate (stacking and skipping frames)
    parser.add_argument('--LFR_m', default=4, type=int,
                        help='Low Frame Rate: number of frames to stack')
    parser.add_argument('--LFR_n', default=3, type=int,
                        help='Low Frame Rate: number of frames to skip')
    # Network architecture
    # encoder
    # TODO: automatically infer input dim
    parser.add_argument('--d_input', default=80, type=int,
                        help='Dim of encoder input (before LFR)')
    parser.add_argument('--n_layers_enc', default=6, type=int,
                        help='Number of encoder stacks')
    parser.add_argument('--n_head', default=8, type=int,
                        help='Number of Multi Head Attention (MHA)')
    parser.add_argument('--d_k', default=64, type=int,
                        help='Dimension of key')
    parser.add_argument('--d_v', default=64, type=int,
                        help='Dimension of value')
    parser.add_argument('--d_model', default=512, type=int,
                        help='Dimension of model')
    parser.add_argument('--d_inner', default=2048, type=int,
                        help='Dimension of inner')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')
    parser.add_argument('--pe_maxlen', default=5000, type=int,
                        help='Positional Encoding max len')
    # decoder
    parser.add_argument('--d_word_vec', default=512, type=int,
                        help='Dim of decoder embedding')
    parser.add_argument('--n_layers_dec', default=6, type=int,
                        help='Number of decoder stacks')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int,
                        help='share decoder embedding with decoder projection')
    # Loss
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')

    # Training config
    parser.add_argument('--epochs', default=150, type=int,
                        help='Number of maximum epochs')
    # minibatch
    parser.add_argument('--shuffle', default=1, type=int,
                        help='reshuffle the data at every epoch')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--batch_frames', default=0, type=int,
                        help='Batch frames. If this is not 0, batch size will make no sense')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of workers to generate minibatch')
    # optimizer
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--k', default=0.2, type=float,
                        help='tunable scalar multiply to learning rate')
    parser.add_argument('--warmup_steps', default=4000, type=int,
                        help='warmup steps')

    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    
    parser.add_argument('--n_samples', default="train:-1,dev:-1,test:-1", type=str,
                        help='choose the number of examples to use')
    args = parser.parse_args()
    # args =parser.parse_known_args()[0]

    return args
# pre_process
def get_data(split, n_samples):
    print('getting {} data...'.format(split))

    global VOCAB

    with open(tran_file, 'r', encoding='utf-8') as file:
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
                trn = list(trn.strip()) + ['<eos>']

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
global args
args = parse_args()
tmp = args.n_samples.split(",")
tmp = [a.split(":") for a in tmp]
tmp = {a[0]:int(a[1]) for a in tmp}
args.n_samples = {"train":-1, "dev":-1,"test":-1}
args.n_samples.update(tmp)

VOCAB = {'<sos>': 0, '<eos>': 1}
IVOCAB = {0: '<sos>', 1: '<eos>'}

data = dict()
data['VOCAB'] = VOCAB
data['IVOCAB'] = IVOCAB
data['train'] = get_data('train', args.n_samples["train"])
data['dev'] = get_data('dev', args.n_samples["dev"])
data['test'] = get_data('test', args.n_samples["test"])

with open(pickle_file, 'wb') as file:
    pickle.dump(data, file)

print('num_train: ' + str(len(data['train'])))
print('num_dev: ' + str(len(data['dev'])))
print('num_test: ' + str(len(data['test'])))
print('vocab_size: ' + str(len(data['VOCAB'])))