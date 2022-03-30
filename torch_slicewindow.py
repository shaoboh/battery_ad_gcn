import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression

class MyDataset(Dataset):
    def __init__(self, path, input_seqlen=10, output_seqlen = 1, fea_num=3, fea_num_y = 2,train_precent=0.8, isTrain=True, interval=1):
        data_df = pd.read_csv(path)
        feature = data_df.loc[:, ['volt', 'current', 'energy']].values
        target_cap = data_df.loc[:, ['cap', 'timestamp']].values
        self.data_num = len(feature)
        self.input_seqlen = input_seqlen
        self.output_seqlen = output_seqlen
        self.fea_num = fea_num
        self.all_seqlen = self.input_seqlen + self.output_seqlen
        self.train_index = int(self.data_num * train_precent)
        
        self.data_seq = []
        self.target_seq = []
        
        for i in range(0, self.data_num - self.all_seqlen, interval):
            self.data_seq.append(list(feature[i:i + self.input_seqlen]))
            self.target_seq.append(list(target_cap[i]))
            
        self.data_seq = np.array(self.data_seq).reshape(len(self.data_seq), -1, fea_num)
        self.target_seq = np.array(self.target_seq).reshape(len(self.target_seq), -1, fea_num_y)
        
        self.data_seq = torch.from_numpy(self.data_seq).type(torch.float32)
        self.target_seq = torch.from_numpy(self.target_seq).type(torch.float32)
        
    def __getitem__ (self, index):
        return self.data_seq[index], self.target_seq[index]
    
    def __len__ (self):
        return len(self.data_seq)
        
        
class RegreNet(nn.Module):
    def __init__ (self):
        super(RegreNet, self).__init__()
        self.hidden1 = nn.LSTM(input_size=3, hidden_size=256, num_layers=2)
        self.hidden2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)
        
    def forward(self, x):
        out, (h_n, c_n) = self.hidden1(x)
        out = out[-1:]
        out = out.reshape(-1, 256)
        out = self.hidden2(out)
        out = self.out(out)
        out = out.reshape(1, -1, 1)
        return out

    
    
    
path = 'c45_charge_996.csv'
input_seqlen = 16
output_seqlen = 2
interval = 2
fea_num= 3
fea_num_y = 1
epoch = 500
lr = 0.007

mydata = MyDataset(path, input_seqlen = input_seqlen, output_seqlen = output_seqlen, interval = interval, fea_num = fea_num)

data_loader = DataLoader(dataset = mydata, 
                        batch_size=64,
                        shuffle = True)

model = RegreNet()
print(model)

optimizer = torch.optim.Adam(model.parameters(),lr = lr)
mse = torch.nn.MSELoss()

writer = SummaryWriter()

for epoch in range (epoch):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (batch_x, batch_y) in enumerate (data_loader):
        batch_x = batch_x.permute(1, 0, 2)
        batch_y = batch_y.permute(1, 0, 2)
        out = model(batch_x)
        
        loss = mse(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    writer.add_scalar('loss train:', loss.item(), epoch)
    print('epoch: {}, train loss: {:.6f}'.format(epoch, loss.item()))
    
writer.close()