from config import IGNORE_ID
def pad_list(x, pad_value):
    n_batch = len(x)
    max_len = max(x.size(0) for x in x)
    pad = x[0].new(n_batch, max_len, *x[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :x[i].size(0)] = x[i]
    return pad

def outprocess(padded_input):
    ys = [y[y != IGNORE_ID] for y in padded_input]
    eos = ys[0].new([1])
    return pad_list(ys,eos[0])