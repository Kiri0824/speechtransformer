from Transformer import Transformer
import torch
from config import pickle_file
from config import device,d_model,d_input,ffn_hidden,d_feature
from config import num_heads,drop_prob,num_layers,max_sequence_length
from config import LFR_skip,LFR_stack,START_TOKEN,END_TOKEN,PADDING_TOKEN
from data_load import extract_feature,build_LFR_features
from tqdm import tqdm
import torchvision.transforms as transforms
from mask import create_masks
import pickle
import numpy as np
from wer import chinese_word_error_rate
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
char_list=data['IVOCAB']
number_list=data['VOCAB']
samples=data['test']
transformer=Transformer(d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,number_list,START_TOKEN,END_TOKEN,PADDING_TOKEN).to(device)
transformer.load_state_dict(torch.load('./results/transformer.pth'))

def test_loop(model):
    model.eval()
    num_samples = len(samples)
    total_loss = 0
    for i in tqdm(range(num_samples)):
      sample = samples[i]
      wave = sample['wave']
      trn = sample['trn']
      feature = extract_feature(input_file=wave, feature='fbank', dim=d_feature, cmvn=True)
      feature = build_LFR_features(feature, m=LFR_stack, n=LFR_skip)
      feature = torch.from_numpy(feature)
      feature = torch.unsqueeze(feature, 0)
      feature = transforms.Resize((max_sequence_length, 320),antialias=True)(feature).to(device)
      with torch.no_grad():
        predicted_sentence = ("",)
        for j in range(max_sequence_length):
            decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(predicted_sentence)
            predictions = transformer(feature,
                                    predicted_sentence,
                                    decoder_self_attention_mask.to(device), 
                                    decoder_cross_attention_mask.to(device),
                                    START_TOKEN,END_TOKEN)
            next_prob_distribution = predictions[0][j]
            next_index = torch.argmax(next_prob_distribution).item()
            next_char = char_list[next_index]
            if next_char==END_TOKEN:
                break 
            predicted_sentence=(predicted_sentence[0] + next_char, )
        if i%100==0:
          print(predicted_sentence[0])
          trn_merge= ''.join(trn)
          print(trn_merge)
        loss=chinese_word_error_rate(predicted_sentence[0],trn_merge)
        total_loss+=loss
    return total_loss / len(samples)
  

test_loop(transformer)