from Transformer import Transformer
import torch.nn as nn
import torch
import pickle
from data_load import SpeechDataset
from config import batch_size,pickle_file,PADDING
from config import device,learning_rate,d_model,d_input,ffn_hidden
from config import num_heads,drop_prob,num_layers,max_sequence_length
from config import epochs,shuffle,num_workers,pin_memory
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from mask import create_masks
import torch
from torch.utils.tensorboard import SummaryWriter


def train_loop(model, opt, loss_fn, dataloader,epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_batch, target_batch = batch[0].to(device), batch[1].to(device)
        labels=target_batch[:,1:]
        decoder_self_attention_mask = create_masks(target_batch)
        decoder_self_attention_mask=decoder_self_attention_mask.to(device)
        opt.zero_grad()
        predicted_batch = model(input_batch, target_batch,decoder_self_attention_mask).to(device)
        predicted_batch=predicted_batch[:,:-1,:]
        loss=loss_fn(predicted_batch.reshape(-1,len(char_list)).to(device),labels.reshape(-1).to(device).long()).to(device)
        valid_position = torch.where(labels.reshape(-1)== PADDING, False, True)
        loss=loss.sum()/valid_position.sum()
        loss.backward()
        opt.step()
        
        total_loss+=loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        if progress_bar.n % 100 == 0:
            print(f"Iteration {progress_bar.n} : {loss.item()}")
            predicted = torch.argmax(predicted_batch[0], axis=1)
            label = labels[0]
            predicted_sentence = ""
            for idx in predicted:
              if idx == PADDING:
                break
              predicted_sentence += char_list[idx.item()]
            print(f"pred_en: {predicted}")
            print(f"pred: {predicted_sentence}")
            label_sentence = ""
            for idx in label:
              if idx == PADDING:
                break
              label_sentence += char_list[idx.item()]
            print(f"real_en: {label}")
            print(f"real: {label_sentence}")
        writer.add_scalar('Train batch Loss (Epoch {})'.format(epoch),total_loss / (progress_bar.n + 1), progress_bar.n + 1)
    return total_loss / len(dataloader)
def val_loop(model, opt, loss_fn, dataloader,epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Validation')
    with torch.no_grad():
        for batch in progress_bar:
            input_batch, target_batch = batch[0].to(device), batch[1].to(device)
            labels=target_batch[:,1:]
            decoder_self_attention_mask = create_masks(target_batch)
            decoder_self_attention_mask=decoder_self_attention_mask.to(device)
            opt.zero_grad()
            predicted_batch = model(input_batch, target_batch,decoder_self_attention_mask)
            predicted_batch=predicted_batch[:,:-1,:]
            loss=loss_fn(predicted_batch.reshape(-1,len(char_list)).to(device),labels.reshape(-1).to(device).long()).to(device)
            loss=loss.sum()/valid_position.sum()
            total_loss+=loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Validation batch Loss (Epoch {})'.format(epoch), total_loss / (progress_bar.n + 1), progress_bar.n + 1)
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
        
        validation_loss = val_loop(model, loss_fn, val_dataloader,epoch)
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")

        is_best = validation_loss < best_loss
        best_loss = min(validation_loss, best_loss)
        
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
    writer.close()  
    return train_loss_list, validation_loss_list


writer = SummaryWriter("results/logs")
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)
train_dataset=SpeechDataset('train')
train_dataloader=DataLoader(train_dataset, batch_size=batch_size,pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)
val_dataset=SpeechDataset('dev')
val_dataloader=DataLoader(val_dataset, batch_size=batch_size,pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)

char_list=data['IVOCAB']
transformer=Transformer(d_model,d_input,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,char_list).to(device)
loss_fn = nn.CrossEntropyLoss()
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)

train_loss_list, validation_loss_list = fit(model=transformer, opt=optimizer, loss_fn=loss_fn, 
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=epochs)