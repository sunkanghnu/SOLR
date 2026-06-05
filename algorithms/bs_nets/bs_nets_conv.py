import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import time
from sklearn.preprocessing import minmax_scale, maxabs_scale
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC, LinearSVC
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')


class Processor:
    def prepare_data(self, img_path, gt_path):
        """read mat file"""
        try:
            img_mat = loadmat(img_path)
            gt_mat = loadmat(gt_path)
            
            #extract data
            img_keys = [k for k in img_mat.keys() if not k.startswith('_') and img_mat[k].ndim == 3]
            gt_keys = [k for k in gt_mat.keys() if not k.startswith('_') and gt_mat[k].ndim == 2]
            
            if not img_keys or not gt_keys:
                raise ValueError("valiad data")
            
            img = img_mat[img_keys[0]]
            gt = gt_mat[gt_keys[0]]
            
            # 标准化维度顺序为 (H, W, C)
            if img.shape[2] != np.min(img.shape) and img.shape[0] == np.min(img.shape):
                img = np.transpose(img, (1, 2, 0))
            
            return img.astype(np.float32), gt.astype(np.int32)
        except Exception as e:
            raise RuntimeError(f"read file failure: {str(e)}")
    
    def get_correct(self, img, gt):
        
        mask = gt > 0
        img_correct = img[mask]
        gt_correct = gt[mask]
        return img_correct, gt_correct

def eval_band_cv(X, y, times=20, n_splits=2):
    """交叉验证评估波段选择效果，优化了CV策略"""
    acc_list = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=133)
    
    for _ in range(times):
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 适配不同数据规模的分类器
            if X_train.shape[0] > 1000:
                model = LinearSVC(C=1, max_iter=10000)
            else:
                model = KNN(n_neighbors=3)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
    
    return np.mean(acc_list)

class HSIDataset(Dataset):
    
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]

class HSIFullImageDataset(Dataset):

    def __init__(self, img, ksize=16, stride=1, padding_mode='none', pad_value=0):
        self.img = img  # HSI data (H,W,C)
        self.ksize = ksize
        self.stride = stride
        self.padding_mode = padding_mode  # 'none'/'zero'/'reflect'
        self.pad_value = pad_value
        self.H, self.W, self.C = img.shape
        
        # process image border
        self.padded_img = self._pad_image()
        
        # 
        self.h_coords, self.w_coords = self._get_valid_coords()
        self.n_samples = len(self.h_coords) * len(self.w_coords)
        
        # 
        self.window_coords = [(h, w) for h in self.h_coords for w in self.w_coords]
    
    def _pad_image(self):
        
        if self.padding_mode == 'none':
            return self.img
        
        # 
        pad_h = max(0, self.ksize - self.H % self.stride) if self.H < self.ksize else 0
        pad_w = max(0, self.ksize - self.W % self.stride) if self.W < self.ksize else 0
        
        if pad_h == 0 and pad_w == 0:
            return self.img
        
        # 
        if self.padding_mode == 'zero':
            padded = np.pad(self.img, ((0, pad_h), (0, pad_w), (0, 0)), 
                           mode='constant', constant_values=self.pad_value)
        elif self.padding_mode == 'reflect':
            padded = np.pad(self.img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            raise ValueError(f"Unsupported pad mode: {self.padding_mode}")
        
        return padded
    
    def _get_valid_coords(self):
        
        H_padded, W_padded = self.padded_img.shape[:2]
        
        # compute start coord
        h_coords = np.arange(0, H_padded - self.ksize + 1, self.stride)
        w_coords = np.arange(0, W_padded - self.ksize + 1, self.stride)
        
        #
        if len(h_coords) == 0 or h_coords[-1] + self.ksize < H_padded:
            h_coords = np.append(h_coords, H_padded - self.ksize)
        if len(w_coords) == 0 or w_coords[-1] + self.ksize < W_padded:
            w_coords = np.append(w_coords, W_padded - self.ksize)
        
        return h_coords, w_coords
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        h, w = self.window_coords[idx]
        
        window = self.padded_img[h:h+self.ksize, w:w+self.ksize, :]
        
        window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).float()
        
        return window_tensor, (h, w)

class BS_Net_Conv(nn.Module):
    def __init__(self, n_channel, lr=1e-4, batch_size=32, epoch=100, n_selected_band=20):
        super(BS_Net_Conv, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.n_selected_band = n_selected_band
        self.n_channel = n_channel
        
        #random seed
        torch.manual_seed(133)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(133)
        
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.att_conv1 = nn.Conv2d(n_channel, 64, kernel_size=3, padding=0)
        self.att_bn1 = nn.BatchNorm2d(64)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_channel),
            nn.Sigmoid()
        )
        
        # backbone
        self.conv1 = nn.Conv2d(n_channel, 128, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        
        # output
        self.output_conv = nn.Conv2d(128, n_channel, kernel_size=1, padding=0)
        self.output_act = nn.Sigmoid()
        
        # L1 
        self.l1_lambda = 0.01
        
        # 
        self.to(self.device)

    def forward(self, x, is_training=True):
        # BN
        x_input = x
        x = F.batch_norm(
            x, 
            running_mean=torch.zeros(self.n_channel, device=self.device),
            running_var=torch.ones(self.n_channel, device=self.device),
            training=is_training
        )
        
        # attention
        att = F.relu(self.att_bn1(self.att_conv1(x)), inplace=True)
        # pooling
        global_pool = torch.mean(att, dim=[2, 3])  # [B, 64]
        # 
        channel_weight = self.bottleneck(global_pool)  # [B, n_channel]
        
        # L1
        l1_loss = self.l1_lambda * torch.norm(channel_weight, p=1)
        
        # 
        channel_weight_ = channel_weight.view(-1, self.n_channel, 1, 1)
        reweight_out = x_input * channel_weight_
        
        # 
        conv1_out = F.relu(self.bn1(self.conv1(reweight_out)), inplace=True)
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)), inplace=True)
        
        # 
        deconv1_out = F.relu(self.bn3(self.deconv1(conv2_out)), inplace=True)
        deconv2_out = F.relu(self.bn4(self.deconv2(deconv1_out)), inplace=True)
        
       
        output = self.output_act(self.output_conv(deconv2_out))
        
        return channel_weight, output, l1_loss

    def fit(self, X, img=None, gt=None, save_path='./'):
       
        dataset = HSIFullImageDataset(
            X, 
            ksize = 16, 
            stride = 8,
            padding_mode='reflect'  
        )
        dataloader = DataLoader(
            dataset, 
            batch_size = self.batch_size, 
            shuffle = True,
            num_workers = 0 if self.device.type == 'cpu' else 4,  
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
       
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        
        
        mse_loss = nn.MSELoss()
        
        
        loss_history = []
        score_list = []
        channel_weight_list = []
        best_acc = 0.0
        best_epoch = 0
        
        
        print(f"Begin trainning, device: {self.device}, Total Epoch: {self.epoch}")
        start_time = time.time()
        
        for epoch in range(self.epoch):
            self.train()
            train_loss = 0.0
            
            
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                
                
                _, output, l1_loss = self.forward(batch_x, is_training=True)
                
                recons_loss = mse_loss(output, batch_x)
                total_loss = recons_loss + l1_loss
                
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item() * batch_x.shape[0]
            
            
            scheduler.step()
            
            
            avg_train_loss = train_loss / len(dataset)
            
            
            self.eval()
            total_val_loss = 0.0
            weight_batch = []
            
            with torch.no_grad():
                val_dataset = HSIFullImageDataset(X, ksize=16, stride=1, padding_mode='reflect')
                val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
                
                for batch_x, _ in val_dataloader:
                    batch_x = batch_x.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    channel_weight, output, l1_loss = self.forward(batch_x, is_training=False)
                    
                    recons_loss = mse_loss(output, batch_x)
                    total_loss = recons_loss + l1_loss
                    
                    total_val_loss += total_loss.item() * batch_x.shape[0]
                    weight_batch.append(channel_weight.cpu().numpy())
                    del batch_x, output, l1_loss, recons_loss, total_loss
            
            avg_val_loss = total_val_loss / len(val_dataset)
            loss_history.append(avg_val_loss)
            
            weight_batch = np.concatenate(weight_batch, axis=0)
            channel_weight_list.append(weight_batch)
            
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{self.epoch}] | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | '
                  f'耗时: {elapsed_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}')
            mean_weight = np.mean(weight_batch, axis=0)
            band_idx = np.argsort(mean_weight)[::-1][:self.n_selected_band]
            if img is not None and gt is not None:
              
                               
                x_new = img[:, :, band_idx]
                n_row, n_clm, n_band = x_new.shape
                img_ = minmax_scale(x_new.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
                
                p = Processor()
                img_correct, gt_correct = p.get_correct(img_, gt)
                score = eval_band_cv(img_correct, gt_correct, times=10, n_splits=3)
                score_list.append(score)
                
                if score > best_acc:
                    best_acc = score
                    best_epoch = epoch
                    torch.save(self.state_dict(), f'{save_path}best-model-Conv.pth')
                
                print(f'Selected bands: {band_idx}| OA: {score:.4f} | Best OA: {best_acc:.4f}(Epoch {best_epoch})')
            
            if (epoch + 1) % 10 == 0:
                np.savez(f'{save_path}history-Conv-epoch{epoch+1}.npz', 
                         loss=loss_history, 
                         score=score_list, 
                         channel_weight=channel_weight_list,
                         selected_bands=band_idx if img is not None else None)
        
        # 最终保存
        total_time = time.time() - start_time
        np.savez(f'{save_path}history-Conv-final.npz', 
                 loss=loss_history, 
                 score=score_list, 
                 channel_weight=channel_weight_list,
                 best_acc=best_acc,
                 best_epoch=best_epoch,
                 total_training_time=total_time)
        
        # 保存最终模型
        torch.save(self.state_dict(), f'{save_path}final-model-Conv.pth')
        
        print(f"Finish trainning! Cost time: {total_time:.2f}s | Best OA: {best_acc:.4f}(Epoch {best_epoch})")
        return band_idx, loss_history, score_list

def run_bs_net_conv(X_3D, gt_img, n_selected_band = 20, epoch = 50, batch_size = 32, lr =1e-4):
    
    img = X_3D

    n_row, n_clm, n_band = img.shape
    img_normalized = minmax_scale(img.reshape((n_row * n_clm, n_band))).reshape((n_row, n_clm, n_band))
    
    model = BS_Net_Conv(
        n_channel=n_band,
        lr=lr,
        batch_size=batch_size,
        epoch=epoch,
        n_selected_band=n_selected_band
    )
    
    selected_bands, loss_history, score_list = model.fit(img_normalized, img=img, gt=gt_img)
    
    print(f"\nSelected bands: {selected_bands}")
    return selected_bands, loss_history, score_list