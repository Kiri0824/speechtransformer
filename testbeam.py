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
from torch.nn.functional import log_softmax
from wer import wer
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
char_list=data['IVOCAB']
number_list=data['VOCAB']
samples=data['test']
transformer=Transformer(d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,number_list,START_TOKEN,END_TOKEN,PADDING_TOKEN).to(device)
transformer.load_state_dict(torch.load('./log/首次基准模型/transformer.pth'))


def beam_search_decode(model, input_features, beam_width, max_output_length,decoder_self_attention_mask, start_token, stop_token):
    # 初始化束搜索的候选序列
    candidates = [("", 0)]  # (序列, 对数概率)
    
    for _ in range(max_output_length):
        next_beam = []
        all_candidates = []
        for seq, score in candidates:
            seqtu = (seq,)
            # 获取模型预测的下一个标记和对应的对数概率
            log_prob = model(input_features, seqtu,decoder_self_attention_mask.to(device),START_TOKEN,end_token=False)
            next_index = torch.argmax(log_prob).item()
            next_char = char_list[next_index]
            if next_char == stop_token:  # 如果预测到停止标记，则将当前序列作为最终候选
                all_candidates.append((seq, score))
            else:
                # 更新序列和分数
                
                new_seq = seq + next_char
                new_score = score + log_prob[0, -1, next_index]
                all_candidates.append((new_seq, new_score))
        # 根据分数排序候选序列，选择前 beam_width 个作为新的候选
        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    print(candidates[0][0][1:])
    # 返回最终的输出序列（去掉起始标记）返回了一个元组，第一个元素是序列，第二个元素是分数
    return candidates[0][0][1:]


def test_loop(model):
    model.eval()
    num_samples = len(samples)
    total_loss = []
    
    for i in tqdm(range(num_samples)):
      sample = samples[i]
      wave = sample['wave']
      trn = sample['trn']
      feature = extract_feature(input_file=wave, feature='fbank', dim=d_feature, cmvn=True)
      feature = build_LFR_features(feature, m=LFR_stack, n=LFR_skip)
      feature = torch.from_numpy(feature)
      feature = torch.unsqueeze(feature, 0).to(device)
      with torch.no_grad():
        predicted_sentence = ("",)
        decoder_self_attention_mask,decoder_cross_attention_mask= create_masks(predicted_sentence)
        sentence_beam = beam_search_decode(transformer, feature, 3, max_sequence_length,decoder_self_attention_mask,START_TOKEN, END_TOKEN)
        trn_merge= ''.join(trn)
        # loss=wer(predicted_sentence[0],trn_merge)
        # total_loss.append(loss)
        # if loss>0.2:
        #     print(predicted_sentence[0])
        #     print(trn_merge)
        #     print(i)
        #     print(loss)
        #     with open('./wrong.txt', "a") as file:
        #         file.write(str(i))
        #         file.write('\n')
        #         file.write("预测:") 
        #         file.write(predicted_sentence[0])
        #         file.write('\n')
        #         file.write("实际:") 
        #         file.write(trn_merge)
        #         file.write("损失:") 
        #         file.write(str(loss))
        #         file.write('\n')
        if i % 100 == 0:
            print(sum(total_loss)/(i+1))
    with open('./wrong.txt', "a") as file:
        file.write(str(total_loss))
    return sum(total_loss) / len(samples)
  
print(test_loop(transformer))