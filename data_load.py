from config import seed
import librosa
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import pickle
from config import pickle_file
from config import num_mask,freq_masking_max_percentage,time_masking_max_percentage,sample_rate,LFR_stack,LFR_skip,d_input
from config import max_sequence_length,d_feature
def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt
def extract_feature(input_file, feature='fbank', dim=80, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=sample_rate)
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=yt, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(yt, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')
random.seed(seed)
# 在时空上添加噪声
def spec_augment(spec: np.ndarray, num_mask=num_mask, 
                 freq_masking_max_percentage=freq_masking_max_percentage, time_masking_max_percentage=time_masking_max_percentage):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec

def build_LFR_features(inputs, m=LFR_stack, n=LFR_skip):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


class SpeechDataset(Dataset):
    def __init__(self,data_name):
        # data loading (data_name 是哪一个数据集)
        with open(pickle_file,'rb') as f:
            self.data=pickle.load(f)
        self.samples=self.data[data_name]
        self.n_samples=len(self.samples)
        # 抗锯齿
        self.transform = transforms.Resize((max_sequence_length, 320),antialias=True)
    def __getitem__(self,index):
        # dataset[0]
        sample=self.samples[index]
        wave=sample['wave']
        trn=sample['trn']
        feature=extract_feature(input_file=wave, feature='fbank', dim=d_feature, cmvn=True)
        # 标准化
        feature = (feature - feature.mean()) / feature.std()
        # 添加噪声
        # feature=spec_augment(feature)
        feature=build_LFR_features(feature)
        feature = torch.from_numpy(feature)
        feature = torch.unsqueeze(feature, 0)
        feature = torch.reshape(feature, (1, feature.shape[1], feature.shape[2]))
        feature = self.transform(feature)
        feature = torch.reshape(feature, (feature.shape[1], feature.shape[2]))
        merged_sentence = ''.join(trn)
        return feature,merged_sentence
    def __len__(self):
        # len(dataset)
        return self.n_samples