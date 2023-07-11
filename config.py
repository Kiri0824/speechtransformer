import os
n_samples="train:-1,dev:-1,test:-1"
# ---------------------------------
# 预处理添加噪声参数
seed=111
# 遮盖次数
num_mask=2
# 频率维度遮盖最大比例
freq_masking_max_percentage=0.15
# 时间维度遮盖最大比例
time_masking_max_percentage=0.3
# ---------------------------------
# 文本位置参数
DATA_DIR = '../dataset/'
aishell_folder = '../dataset/data_aishell/'
wav_folder = os.path.join(aishell_folder, 'wav')
trans_file = os.path.join(aishell_folder, 'transcript/aishell_transcript_v0.8.txt')
pickle_file = '../dataset/aishell.pickle'
# ---------------------------------