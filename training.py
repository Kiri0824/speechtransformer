# design model(input,output size, forward pass)
# construct loss and optimizer
# training loop
#   -forward pass:compute prediction
#   -backward pass:gradients
#   -update weights
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data_process
from torch.utils.tensorboard import SummaryWriter
from padding import pad_collate
from config import betch_size,pin_memory,shuffle,num_workers,device,learning_rate
from model.Transformer import Transformer
from save_checkpoint import save_checkpoint




# training loop
def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, labels = batch[0], batch[1]
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_input = labels[:,:-1]
        y_expected = labels[:,1:]
        
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)
        # Forward pass
        outputs = model(inputs, y_input, tgt_mask)
        # loss
        loss = loss_fn(outputs, y_expected.float())

        # Backward and optimize
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):

    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_input = labels[:,:-1]
            y_expected = labels[:,1:]
            
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Forward pass
            outputs = model(inputs, y_input, tgt_mask)

            # Permute pred to have batch size first again
            outputs = outputs.permute(1, 2, 0)      
            loss = loss_fn(outputs, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    train_loss_list, validation_loss_list = [], []
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        writer.add_scalar('model/train_loss', train_loss, epoch)

        train_loss_list += [train_loss]
        
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        writer.add_scalar('model/valid_loss', validation_loss, epoch)
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
    return train_loss_list, validation_loss_list


if __name__ == '__main__':
    print('prepare data')
    # prepare data
    train_dataset=data_process.AiShellDataset('train')
    train_dataloader=DataLoader(train_dataset, batch_size=betch_size, collate_fn=pad_collate,
                                                pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)
    val_dataset=data_process.AiShellDataset('dev')
    val_dataloader=DataLoader(val_dataset, batch_size=betch_size, collate_fn=pad_collate,
                                                pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)

    # loss and optimizer
    model = Transformer().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    train_loss_list, validation_loss_list = fit(model=model, opt=optimizer, loss_fn=loss_fn, 
                        train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=10)
