# design model(input,output size, forward pass)
# construct loss and optimizer
# training loop
#   -forward pass:compute prediction
#   -backward pass:gradients
#   -update weights


# prepare data
# model
# loss and optimizer
# training loop
import torch
import torch.nn as nn

X=torch.tensor([[1],[2],[3],[4]])
Y=torch.tensor([])

n_samples,n_features=X.shape()
print(n_samples,n_features)

input_size=n_features
output_size=n_features

class LinearRegression(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        # define layers
        self.lin=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        return self.lin(x)
    
model = LinearRegression(input_size,output_size)

learning_rate=0.01

loss = nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range():
    # prediction = forward pass
    y_pred=model(X)
    
    # loss
    l=loss(Y,y_pred)
    
    # gradients
    l.backward()
    
    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()
    
    if epoch%10==0:
        [w,b]=model.parameters()
        print(f'epoch {epoch+1}:w={w[0][0].item:.3f},loss={l:8.3f}')