import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

train_data=pd.read_csv("C:/Users/HP/Desktop/Data/train.csv")
train_images=train_data.iloc[:,1:].values.astype(np.float32)/255.0
train_labels=train_data.iloc[:,0].values.astype(np.int64)
train_images=torch.tensor(train_images).view(-1,1,28,28)
train_labels=torch.tensor(train_labels)
# print(train_images.shape,train_labels.shape)

batch_size=128
train_dataset=TensorDataset(train_images,train_labels)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
# for imgs,labels in train_loader:
#     print(imgs.shape,labels.shape)
#     break
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=5)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=5)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(16*4*4,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
    def forward(self,x):
        x=self.pool1(F.relu(self.conv1(x)))
        x=self.pool2(F.relu(self.conv2(x)))
        x=x.view(-1,16*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# model=LeNet()
# print(model)
model=LeNet()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
num_epochs=10
for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f'Epoch[{epoch+1}/{num_epochs}],Loss:{running_loss/len(train_loader):.4f}')

test_data=pd.read_csv('C:/Users/HP/Desktop/Data/test.csv')
test_images=test_data.values.astype(np.float32)/255.0
test_images=torch.tensor(test_images).view(-1,1,28,28)
test_loader=DataLoader(test_images,batch_size=batch_size,shuffle=False)

model.eval()
predictions=[]
with torch.no_grad():
    for images in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        predictions.extend(predicted.numpy())

plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(test_images[i].squeeze().numpy(),cmap='gray')
    plt.title(f'Pred:{predictions[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()