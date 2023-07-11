import os
n_samples="train:-1,dev:-1,test:-1"
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
# 'Dim of encoder input (before LFR)'
d_input=80
# -------------------------------------------------
# extract_feature
# "sr"代表采样率（sample rate）
# 采样率是指在一秒钟内对声音信号进行采样的次数。它表示每秒从连续模拟信号中获取的离散样本数
sample_rate=16000
# 填充的数值
IGNORE_ID = -1
# -------------------------------------------------
# 文本位置参数
DATA_DIR = '../dataset/'
aishell_folder = '../dataset/data_aishell/'
wav_folder = os.path.join(aishell_folder, 'wav')
trans_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
pickle_file = '../dataset/aishell.pickle'
# -------------------------------------------------