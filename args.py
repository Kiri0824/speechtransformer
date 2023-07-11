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

    return args
