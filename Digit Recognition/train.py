import os
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from model import LeNet5
from torch.optim import SGD

import tqdm


# define
data_path = 'data'
batch_size = 128
epochs = 100
num_worker = 0
num_classes = 10
lr = 0.05
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = 'checkpoint'

if not os.path.exists(save_path):
    os.mkdir(save_path)

### Load dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # normalize data with mean = 0 and standard deviation = 1
])

train_dataset = datasets.MNIST(root = data_path, train = True, download = True, transform= transform) # train dataset
test_dataset = datasets.MNIST(root = data_path, train = False, download = True, transform= transform) # test dataset

# create dataloader from dataset
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, num_workers= num_worker)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle= False, num_workers= num_worker)

## define model, optimizer and criterion
model = LeNet5(num_classes = num_classes).to(device)
optimizer = SGD(params= model.parameters(), momentum=0.9,lr = lr)
loss_fn = nn.CrossEntropyLoss()

# calculate accuracy 
def accaracy(actual, predcition):
    return 100*torch.sum(actual == predcition.argmax(dim=1))/actual.size(0)

# train phase
def train(model,train_iter, loss_fn, optimizer):
    model.train()
    counter = 0
    running_loss = 0
    running_acc = 0
    tk = tqdm.tqdm(train_iter, total = len(train_iter))
    for x, target in tk:
        counter +=1
        x = x.to(device)
        target = target.to(device)
        prediction = model(x)
        loss = loss_fn(prediction, target.long())
        acc = accaracy(target, prediction)
        running_loss += loss.item()
        running_acc +=acc.item()
        tk.set_postfix(loss = running_loss/counter, acc = running_acc/counter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss/counter, running_acc/counter

# evaluate phase (for testdataset)
def evaluate(model, test_iter, loss_fn):
    model.eval()
    counter = 0
    running_loss = 0
    running_acc = 0
    tk = tqdm.tqdm(test_iter, total = len(test_iter))
    with torch.no_grad():
        for x, target in tk:
            counter +=1
            x = x.to(device)
            target = target.to(device)
            prediction = model(x)
            loss = loss_fn(prediction, target.long())
            acc = accaracy(target, prediction)
            running_loss += loss.item()
            running_acc +=acc.item()
            tk.set_postfix(loss = running_loss/counter, acc = running_acc/counter)
    return running_loss/counter, running_acc/counter

    ## training model
best_acc = 0
train_losses = []
train_accs = []
test_losses = []
test_accs = []
for epoch in range(epochs):
    
    train_loss, train_acc = train(model, train_loader, loss_fn, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    
    # save best model
    if test_acc > best_acc:
        best_acc = test_acc
        print('***Saving model*** acc = {}'.format(test_acc))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': best_acc
        }, '{}/model.pt'.format(save_path))
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)
    
    # save log
    df_data=np.array([train_losses, train_accs, test_losses, test_accs]).T
    df = pd.DataFrame(df_data, columns = ['train_losses', 'train_accs', 'test_losses','test_accs'])
    df.to_csv('{}/logs.csv'.format(save_path))
