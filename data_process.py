import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pickle
from noise_and_feature import extract_feature,spec_augment,build_LFR_features
from config import d_input,pickle_file
class AiShellDataset(Dataset):
    def __init__(self,data_name):
        # data loading (data_name 是哪一个数据集)
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
        self.samples=data[data_name]
        self.n_samples=len(self.samples)
        
    def __getitem__(self, index): 
        # dataset[0]
        sample=self.samples[index]
        wave=sample['wave']
        trn=sample['trn']
        feature=extract_feature(input_file=wave, feature='fbank', dim=d_input, cmvn=True)
        # 标准化
        feature = (feature - feature.mean()) / feature.std()
        # 添加噪声
        feature=spec_augment(feature)
        feature=build_LFR_features(feature)
        return feature,trn
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
