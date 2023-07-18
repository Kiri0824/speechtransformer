import torch
import numpy as np
from config import max_sequence_length,NEG_INF
def create_masks(batch):
    num_sentences = len(batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    
    for idx in range(len(batch)):
      sentence_length =len(batch[idx])
      chars_to_padding_mask = np.arange(sentence_length + 1, max_sequence_length)
      decoder_padding_mask_self_attention[idx, :, chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, chars_to_padding_mask, :] = True
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INF, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INF, 0)
    return decoder_self_attention_mask,decoder_cross_attention_mask