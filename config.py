import os
import torch
# -------------------------------------------------

# 预处理添加噪声参数
seed=111
# 遮盖次数
num_mask=2
# 频率维度遮盖最大比例
freq_masking_max_percentage=0.15
# 时间维度遮盖最大比例
time_masking_max_percentage=0.3
# -------------------------------------------------

# 预处理添加跳帧和堆叠(体现在图中见jupyter)
LFR_stack=4
LFR_skip=3
d_feature=80
# -------------------------------------------------

# extract_feature
# "sr"代表采样率（sample rate）
# 采样率是指在一秒钟内对声音信号进行采样的次数。它表示每秒从连续模拟信号中获取的离散样本数
sample_rate=16000
# mask填充的数值
NEG_INF = -1e9
# -------------------------------------------------

# 训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=30
d_model=256
d_input=320
num_heads=4
drop_prob=0.1
# 句子最长50
max_sequence_length=50
# feedforward神经网络的隐藏层维度
ffn_hidden=1024
num_layers=6
shuffle=True
num_workers=0
learning_rate=1e-4
epochs=20
pin_memory=True
shuffle=True
num_workers=0
START_TOKEN='<sos>'
END_TOKEN='<eos>'
PADDING_TOKEN='<PAD>'
# -------------------------------------------------

# 文本位置参数
DATA_DIR = '../dataset/'
aishell_folder = '../dataset/data_aishell/'
wav_folder = os.path.join(aishell_folder, 'wav')
trans_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
pickle_file = '../dataset/aishell.pickle'
# -------------------------------------------------

PADDING=0