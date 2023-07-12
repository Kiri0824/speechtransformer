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
from padding import pad_collate
from config import betch_size,pin_memory,shuffle,num_workers,device

# prepare data
dataset=data_process.AiShellDataset('train')
dataloader=DataLoader(dataset, batch_size=betch_size, collate_fn=pad_collate,
                                               pin_memory=pin_memory, shuffle=shuffle, num_workers=num_workers)

# model
class TransformerModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.encoder_layers.append(nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout))
            self.decoder_layers.append(nn.TransformerDecoderLayer(hidden_size, num_heads, hidden_size, dropout))
        
        self.fc = nn.Linear(hidden_size, output_size=None)
    
    def forward(self, src, tgt):
        enc_output = src
        
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output)
        
        dec_output = tgt
        
        for decoder_layer in self.decoder_layers:
            dec_output = decoder_layer(dec_output, enc_output)
        
        output = self.fc(dec_output)
        
        return output

    


# loss and optimizer
model = TransformerModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# training loop
for epoch in range(10):
    for i, (inputs, labels) in enumerate(dataset):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%10==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch+1}:w={w[0][0].item:.3f},loss={loss:8.3f}')
