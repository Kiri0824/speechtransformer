from SpeechTransformer import Transformer
import torch.nn as nn
import torch
import pickle
from data_load import SpeechDataset,pad_collate
from config import batch_size,pickle_file,START_TOKEN,END_TOKEN,PADDING_TOKEN,LFR_skip,LFR_stack,num_layers
from config import device,learning_rate,d_model,d_input,ffn_hidden
from config import num_heads,drop_prob,max_sequence_length
from config import epochs,shuffle,num_workers,pin_memory,d_feature,en_num_layers,ou_num_layers
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from mask import create_masks
from data_load import extract_feature,build_LFR_features
import torch
from torch.utils.tensorboard import SummaryWriter



def train_loop(model, opt, loss_fn, dataloader,epoch):
    model.train()
    # losses = AverageMeter()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_batch, target_batch = batch[0].to(device), batch[1]
        decoder_self_attention_mask,decoder_cross_attention_mask=create_masks(batch[1])
        opt.zero_grad()
        predicted_batch = model(input_batch, target_batch,decoder_self_attention_mask.to(device),START_TOKEN,END_TOKEN).to(device)
        labels = transformer.decoder.embedding.batch_tokenize(target_batch, start_token=False, end_token=True)
        
        
        loss=loss_fn(predicted_batch.view(-1,len(number_list)).to(device),labels.view(-1)).to(device)
        valid_position = torch.where(labels.view(-1)== number_list[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_position.sum()
        loss.backward()
        opt.step()
        

        total_loss+=loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        if progress_bar.n % 100 == 0:
            
            print(f"Iteration {progress_bar.n} : {loss.item()}")
            
            print(f"Iteration {progress_bar.n} : {loss.item()}")
            predicted = torch.argmax(predicted_batch[0], axis=1)
            label = labels[0]
            predicted_sentence = ""
            for idx in predicted:
              if idx == number_list[PADDING_TOKEN]:
                break
              predicted_sentence += char_list[idx.item()]
            print(f"pred: {predicted_sentence}")
            label_sentence = ""
            for idx in label:
              if idx == number_list[PADDING_TOKEN]:
                break
              label_sentence += char_list[idx.item()]
            print(f"real: {label_sentence}")
        writer.add_scalar('Train batch Loss (Epoch {})'.format(epoch),total_loss / (progress_bar.n + 1), progress_bar.n + 1)
    return total_loss / len(dataloader)
    # return losses.avg
def val_loop(model, dataloader,epoch):
    model.eval()
    # losses = AverageMeter()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in progress_bar:
            input_batch, target_batch = batch[0].to(device), batch[1]
            decoder_self_attention_mask,decoder_cross_attention_mask=create_masks(batch[1])
            predicted_batch = model(input_batch, target_batch,decoder_self_attention_mask.to(device),START_TOKEN,END_TOKEN).to(device)
            labels = transformer.decoder.embedding.batch_tokenize(target_batch, start_token=False, end_token=True)
 
            loss=loss_fn(predicted_batch.view(-1,len(number_list)).to(device),labels.view(-1)).to(device)
            valid_position = torch.where(labels.view(-1)== number_list[PADDING_TOKEN], False, True)
            loss = loss.sum() / valid_position.sum()
            total_loss+=loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Validation batch Loss (Epoch {})'.format(epoch), total_loss / (progress_bar.n + 1), progress_bar.n + 1)
            if progress_bar.n % 100 == 0:
                wave = batch[0][0].to(device)
                trn = batch[1][0]
                wave = torch.unsqueeze(wave, 0)
                predicted_sentence = ("",)
                for j in range(max_sequence_length):
                    decoder_self_attention_mask,decoder_cross_attention_mask= create_masks(predicted_sentence)
                    
                    predictions = transformer(wave,
                                            predicted_sentence,
                                            decoder_self_attention_mask.to(device), 
                                            START_TOKEN,END_TOKEN)
                    next_prob_distribution = predictions[0][j]
                    next_index = torch.argmax(next_prob_distribution).item()
                    next_char = char_list[next_index]
                    if next_char==END_TOKEN or len(predicted_sentence[0])>=max_sequence_length-2:
                        break 
                    predicted_sentence=(predicted_sentence[0] + next_char, )
                
                print(predicted_sentence[0])
                print(trn)
    # return losses.avg
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []
    best_loss = float('inf')
    epochs_since_improvement = 0
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)    
        train_loss = train_loop(model, opt, loss_fn, train_dataloader,epoch)
        writer.add_scalar('Loss/train', train_loss, epoch) 
        train_loss_list += [train_loss]
        
        validation_loss = val_loop(model,val_dataloader,epoch)
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

        is_best = validation_loss < best_loss
        best_loss = min(validation_loss, best_loss)
        
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
        else:
            epochs_since_improvement = 0
            torch.save(model.state_dict(), 'results/transformer.pth')
    writer.close()  
    return train_loss_list, validation_loss_list


writer = SummaryWriter("results/logs")
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
train_dataset=SpeechDataset('train')
train_dataloader=DataLoader(train_dataset, batch_size=batch_size,collate_fn=pad_collate,pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)
val_dataset=SpeechDataset('dev')
val_dataloader=DataLoader(val_dataset, batch_size=batch_size,collate_fn=pad_collate,pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)

char_list=data['IVOCAB']
number_list=data['VOCAB']
transformer=Transformer(d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,number_list,START_TOKEN=START_TOKEN,END_TOKEN=END_TOKEN,PADDING_TOKEN=PADDING_TOKEN).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=number_list[PADDING_TOKEN],reduction='none')
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)


# state_dict = torch.load('results/transformer.pth')
# transformer.load_state_dict(state_dict)
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
train_loss_list, validation_loss_list = fit(model=transformer, opt=optimizer, loss_fn=loss_fn, 
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=epochs)