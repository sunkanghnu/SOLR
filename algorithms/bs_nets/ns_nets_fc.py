# -*- coding: utf-8 -*-
"""
@ Description: 
-------------
Band selection network with Fullly Connected Nets (aka. MLP)
-------------
@ Time    : 2019/2/28 15:32
@ Author  : Yaoming Cai
@ FileName: BS_Net_FC.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.preprocessing import minmax_scale

sys.path.append('/home/caiyaom/python_codes/')
from Helper import Dataset as SklearnDataset  # 
from utility import eval_band_cv
from Preprocessing import Processor

# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BS_Net_FC(nn.Module):
    def __init__(self, lr, batch_size, epoch, n_selected_band, n_channel):
        super(BS_Net_FC, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_selected_band = n_selected_band
        self.n_channel = n_channel
        
        torch.manual_seed(133)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(133)
        

        self.att_dense1 = nn.Linear(n_channel, 64)
        self.att_bn1 = nn.BatchNorm1d(64)
        self.att_bottleneck = nn.Linear(64, 128)
        self.att_channel_weight = nn.Linear(128, n_channel)
        self.l1_lambda = 0.01  
        
    
        self.fcn1 = nn.Linear(n_channel, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fcn2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
      
        self.fcn3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcn4 = nn.Linear(256, n_channel)
        self.bn4 = nn.BatchNorm1d(n_channel)
        
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input):
        """
        :param x_input: (N, n_bands)
        :return: channel_weight, output
        """
        input_norm = nn.BatchNorm1d(self.n_channel).to(device)(x_input)
        
        dense_att_1 = self.att_dense1(input_norm)
        bn_att_1 = self.relu(self.att_bn1(dense_att_1))
        bottleneck = self.relu(self.att_bottleneck(bn_att_1))
        channel_weight = self.sigmoid(self.att_channel_weight(bottleneck))  # (N, n_channel)
        
        reweight_out = channel_weight * input_norm
        
        fcn_1 = self.fcn1(reweight_out)
        batch_norm_1 = self.relu(self.bn1(fcn_1))
        
        fcn_2 = self.fcn2(batch_norm_1)
        batch_norm_2 = self.relu(self.bn2(fcn_2))
        
        fcn_3 = self.fcn3(batch_norm_2)
        batch_norm_3 = self.relu(self.bn3(fcn_3))
        
        fcn_4 = self.fcn4(batch_norm_3)
        batch_norm_4 = self.bn4(fcn_4)
        output = self.sigmoid(batch_norm_4)
        
        return channel_weight, output

    def fit(self, X, img=None, gt=None):
        n_sam, n_channel = X.shape
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        class CustomDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.data[idx]  
        
        dataset = CustomDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        
        mse_loss = nn.MSELoss()
        
        # TensorBoard
        writer = SummaryWriter('logs/fc')
        
        loss_history = []
        score_list = []
        channel_weight_list = []
        
        self.train()
        
        for i_epoch in range(self.epoch):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                channel_weight, output = self(batch_x)
                
                loss_recons = mse_loss(output, batch_y)
                l1_reg = self.l1_lambda * torch.norm(channel_weight, p=1)  # L1正则项
                total_loss = loss_recons + l1_reg
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                channel_weight_pre, output_val = self(X_tensor)
                loss_reocns = mse_loss(output_val, X_tensor).item()
                l1_reg = self.l1_lambda * torch.norm(channel_weight_pre, p=1).item()
                total_loss_val = loss_reocns + l1_reg
                
                print(f'epoch {i_epoch} ==> loss={total_loss_val:.6f}')
                loss_history.append(total_loss_val)
                
                writer.add_scalar('loss/total', total_loss_val, i_epoch)
                writer.add_histogram('channel_weight', channel_weight_pre.cpu().numpy(), i_epoch)
                
                channel_weight_np = channel_weight_pre.cpu().numpy()
                channel_weight_list.append(channel_weight_np)
                
                if img is not None:
                    mean_weight = np.mean(channel_weight_np, axis=0)
                    band_indx = np.argsort(mean_weight)[::-1][:self.n_selected_band]
                    print('=============================')
                    print('SELECTED BAND: ', band_indx)
                    print('=============================')
                    
                    x_new = img[:, :, band_indx]
                    n_row, n_clm, n_band = x_new.shape
                    img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
                    p = Processor()
                    img_correct, gt_correct = p.get_correct(img_, gt)
                    score = eval_band_cv(img_correct, gt_correct, times=20, test_size=0.95)
                    print('acc=', score)
                    score_list.append(score)
                    writer.add_scalar('accuracy', score, i_epoch)
            
            if i_epoch % 10 == 0:
                np.savez('history-FC.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
            
            self.train()
        
        np.savez('history-FC.npz', loss=loss_history, score=score_list, channel_weight=channel_weight_list)
        torch.save(self.state_dict(), './IndianPine-model-FC.pth')
        writer.close()


'''
===================================
        Demo: train model
===================================
'''
if __name__ == '__main__':
    root = './Dataset/'
    # root = '/home/caiyaom/HSI_Files/'
    # im_, gt_ = 'SalinasA_corrected', 'SalinasA_gt'
    im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    # im_, gt_ = 'Pavia', 'Pavia_gt'
    # im_, gt_ = 'PaviaU', 'PaviaU_gt'
    # im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    # im_, gt_ = 'Botswana', 'Botswana_gt'
    # im_, gt_ = 'KSC', 'KSC_gt'

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)

    p = Processor()
    img, gt = p.prepare_data(img_path, gt_path)
    n_row, n_column, n_band = img.shape
    X_img = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    X_train = np.reshape(X_img, (n_row * n_column, n_band))
    print('training img shape: ', X_train.shape)

    LR, BATCH_SIZE, EPOCH = 0.00002, 64, 100
    N_BAND = 5
    time_start = time.time()
    
    # 初始化模型
    model = BS_Net_FC(LR, BATCH_SIZE, EPOCH, N_BAND, n_channel=n_band).to(device)
    model.fit(X_train, img=X_img, gt=gt)
    
    run_time = round(time.time() - time_start, 3)
    print('running time=', run_time)