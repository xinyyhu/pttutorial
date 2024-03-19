import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import time




x_np,y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(x_np.dtype)

x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
print(x.shape, y.shape)
y = y.view(y.shape[0], 1)
print(x.shape, y.shape)

n_samples, n_features = x.shape
print(n_samples)
print(n_features)

# 1. model
input_size = n_features
output_size = 1

# 线性层输入的维度是1，因为数据是m*1,
model = nn.Linear(input_size, output_size)

# 2. loss and optimizer
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3. training loop
epochs = 1000

# time_begin = time.time()
for i in range(epochs):
    #forward pass and loss
    y_hat = model(x)
    # print(y_hat.shape)
    loss = criterion(y_hat, y)
    
    # backward pass
    loss.backward()
    
    # update 
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (i+1) % 100 == 0:
        print(f'epoch: {i+1}, w: {model.weight[0][0]:.3f}, loss: {loss.item():.8f}')
        
    if i+1 == 1000:
        # print("weight: ", model.weight)
        print("x: ", x[:, 0])
        print("-----------------------")
        print("y :", y_hat[:, 0])
    
# time_end = time.time()
# time = time_end - time_begin
# print(time)


predicted = y_hat.detach().numpy()
plt.plot(x_np, y_np, 'ro')
plt.plot(x_np, predicted, 'bo')
plt.show()
    
    



