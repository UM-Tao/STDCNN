import os
import random
import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import datetime, time
from layers import Conv2dWithConstraint

# Set GPU configuration
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import torch.backends.cudnn as cudnn

cudnn.benchmark = False
cudnn.deterministic = True

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# SSM State-Space Model (Temporal Dynamics)
class StateSpaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop=0.1):
        super(StateSpaceModel, self).__init__()

        # Initialize SSM state-space components
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim))    # learnable state transition matrix A
        self.B = nn.Parameter(torch.zeros(input_dim, hidden_dim))   # learnable input matrix B
        self.C = nn.Parameter(torch.eye(hidden_dim, output_dim))    # learnable output matrix C

        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        # Apply state-space model (Kalman filtering)
        b, l, d = x.shape  # b: batch, l: sequence length, d: input dimension

        # Process inputs and compute state updates
        state = torch.zeros(b, self.hidden_dim).to(x.device)  # Initial state
        states_list = []

        for t in range(l):
            # Compute the predicted state
            state = torch.matmul(state, self.A) + torch.matmul(x[:, t, :], self.B.T)
            # Compute the output using the observation matrix
            output = torch.matmul(state, self.C.T)  # Observation
            states_list.append(output)

        # Stack all states and outputs
        output_tensor = torch.stack(states_list, dim=1)
        output_tensor = self.dropout(output_tensor)
        return output_tensor


# STDCNN Backbone with Innovations
class STDCNN(nn.Module):
    def __init__(self,
                 num_channels=62,
                 sampling_rate=250,
                 F1=8, D=2, F2='auto', P1=4, P2=8, pool_mode='mean',
                 drop_out=0.25, layer_scale_init_value=1e-6, nums=4):
        super(STDCNN, self).__init__()

        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D


        # Spectral
        self.spectral_1 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 125], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )
        self.spectral_2 = nn.Sequential(
            Conv2dWithConstraint(1, F1, kernel_size=[1, 30], padding='same', max_norm=2.),
            nn.BatchNorm2d(F1),
        )

        # Spatial 1
        self.spatial_1 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, (num_channels, 1), padding=0, groups=F1, bias=False, max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.Dropout(drop_out),
            Conv2dWithConstraint(F2, F2, kernel_size=[1, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, 8), stride=8),
            nn.Dropout(drop_out),
        )

        # Spatial 2
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, kernel_size=[num_channels, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 25), stride=10),
            ActLog(),
            nn.Dropout(drop_out),
        )

        # SSM State-Space Model (Temporal Dynamics)
        self.SSM = StateSpaceModel(input_dim=F2, hidden_dim=F2, output_dim=F2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=F2, nhead=8, dim_feedforward=64),
            num_layers=2
        )
        # Final layers
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_out)

        self.w_q = nn.Linear(F2, F2)
        self.w_k = nn.Linear(F2, F2)
        self.w_v = nn.Linear(F2, F2)

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)  # (B, F1, 62, 200)
        # Extract spatial features
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)

        # Combine the spatial features
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape

        # Flatten for attention mechanism
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)

        # Apply SSM (Temporal Modeling)
        x_attention_SSM = self.SSM(x_attention)  # (B, 12, F2)
        x_attention = self.transformer(x_attention_SSM)
        x_attention = x_attention.reshape(B2, W2, 1, C2).permute(0, 3, 2, 1)
        return x_attention


# Classifier Layer
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(16, num_classes, (1, 43)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x


# Main Network with all Innovations
class NetInnovative(nn.Module):
    def __init__(self, num_classes=3, num_channels=62, sampling_rate=250):
        super(NetInnovative, self).__init__()

        self.backbone = STDCNN(num_channels=num_channels, sampling_rate=sampling_rate)

        self.classifier = classifier(num_classes)


    def forward(self, x):

        # Apply STDCNN backbone and temporal modeling
        x = self.backbone(x)

        # Classification
        x = self.classifier(x)

        return x

# Activation functions
class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ExP():
    def __init__(self, nsub, fold):
        super(ExP, self).__init__()
        self.batch_size = 200
        self.n_epochs = 2
        self.img_height = 22
        self.img_width = 600
        self.channels = 1
        self.lr = 0.001
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.start_epoch = 0

        self.root = '/icto/user/mc25011/SSM/Data/SEED/data_cv5fold/'
        # 如果不存在该路径，则选择下一个路径
        if not os.path.exists(self.root):
            self.root = 'D:\A_SSM\EEG-Conformer-main\Data\SEED\data_cv5fold/'
        log_dir = f"./results/seed/STDCNN/"
        os.makedirs(log_dir, exist_ok=True)
        self.mod_dir = f"./results/seed/STDCNN/save_model/"
        os.makedirs(self.mod_dir, exist_ok=True)

        self.log_write = open(f"{log_dir}/log_subject{self.nSub}_fold{fold + 1}.txt", "w")
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = NetInnovative().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def get_source_data(self, fold):
        # Assuming self.all_data and self.all_label contain the entire dataset and corresponding labels
        self.all_data = np.load(self.root + 'S%d_session1.npy' % self.nSub, allow_pickle=True)
        self.all_label = np.load(self.root + 'S%d_session1_label.npy' % self.nSub, allow_pickle=True)

        # Divide the dataset into 5 folds
        num_samples = self.all_data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        fold_size = num_samples // 5

        test_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = np.delete(indices, np.arange(fold * fold_size, (fold + 1) * fold_size))

        self.train_data = self.all_data[train_indices]
        self.train_label = self.all_label[train_indices]
        self.test_data = self.all_data[test_indices]
        self.test_label = self.all_label[test_indices]

        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.test_data = np.expand_dims(self.test_data, axis=1)

        # Standardize
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std


        return self.train_data, self.train_label, self.test_data, self.test_label


    def interaug(self, timg, label):
        """Inter-class data augmentation."""
        aug_data = []
        aug_label = []
        for cls4aug in range(3):  # Assuming you have 3 classes
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            # Use the minimum of available samples and the desired number of augmented samples
            num_aug_samples = min(len(tmp_label), int(self.batch_size / 4))
            tmp_aug_data = np.zeros((num_aug_samples, 1, 60, 500))  # 60 channels, 500 time points after resampling

            for ri in range(num_aug_samples):
                for rj in range(10):
                    rand_idx = np.random.randint(0, tmp_data.shape[0])
                    tmp_aug_data[ri, :, :, rj * 50:(rj + 1) * 50] = tmp_data[rand_idx, :, :, rj * 50:(rj + 1) * 50]

            aug_data.append(tmp_aug_data)
            # Ensure the label array matches the size of the augmented data
            aug_label.append(tmp_label[:num_aug_samples])

        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label).cuda().long()

        return aug_data, aug_label

    def train(self, fold):
        img, label, test_data, test_label = self.get_source_data(fold)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        dataset = TensorDataset(img, label)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = TensorDataset(test_data, test_label)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        total_time = 0  # 用于记录总训练时间

        for e in range(self.n_epochs):
            start_time = time.time()  # 记录每个 epoch 开始时间
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            end_time = time.time()  # 记录每个 epoch 结束时间
            epoch_time = end_time - start_time  # 计算每个 epoch 的训练时间
            total_time += epoch_time  # 累积总的 epoch 时间

            if (e + 1) % 1 == 0:
                self.model.eval()
                Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.4f' % loss_test.detach().cpu().numpy(),
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc,
                      '  Epoch time: %.4f seconds' % epoch_time)  # 打印每个 epoch 的时长

                self.log_write.write(str(e) + "    " + str(acc) + "  Epoch time: %.4f seconds\n" % epoch_time)
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc

        # save model
        torch.save(self.model.state_dict(), self.mod_dir+f'subject{self.nSub}_fold{fold + 1}.pth')
        averAcc = averAcc / num
        avg_epoch_time = total_time / self.n_epochs  # 计算平均每个 epoch 的时长
        print(f'The average accuracy of fold {fold + 1} is:', averAcc)
        print(f'The best accuracy of fold {fold + 1} is:', bestAcc)
        print(f'The average epoch time is: {avg_epoch_time:.4f} seconds')  # 打印平均 epoch 时长

        self.log_write.write(f'The average accuracy of fold {fold + 1} is: ' + str(averAcc) + "\n")
        self.log_write.write(f'The best accuracy fold {fold + 1} is: ' + str(bestAcc) + "\n")
        self.log_write.write('The average epoch time is: %.3f seconds\n' % avg_epoch_time)
        return bestAcc, averAcc

def main():
    best = 0
    aver = 0
    log_dir = f"./results/seed/STDCNN/"
    os.makedirs(log_dir, exist_ok=True)
    result_write = open(log_dir + "/sub_result.txt", "w")

    for i in range(2):
        starttime = datetime.datetime.now()
        seed_n = 42

        result_write.write('--------------------------------------------------\n')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print(f'Subject {i + 1}')

        result_write.write(f'Subject {i + 1} : Seed is: ' + str(seed_n) + "\n")

        bestAcc = 0
        averAcc = 0

        for fold in range(5):
            exp = ExP(i + 1, fold)
            ba, aa = exp.train(fold)
            result_write.write(f'Best acc of fold {fold + 1} is: ' + str(ba) + "\n")
            result_write.write(f'Aver acc of fold {fold + 1} is: ' + str(aa) + "\n")
            bestAcc += ba
            averAcc += aa

        bestAcc /= 5
        averAcc /= 5
        result_write.write('5-fold Best acc is: ' + str(bestAcc) + "\n")
        result_write.write('5-fold Aver acc is: ' + str(averAcc) + "\n")
        best = best + bestAcc
        aver = aver + averAcc
        endtime = datetime.datetime.now()
        print(f'subject {i + 1} duration: ' + str(endtime - starttime))

    best /= 15
    aver /= 15

    result_write.write('--------------------------------------------------\n')
    result_write.write('All subject Best accuracy is: ' + str(best) + "\n")
    result_write.write('All subject Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
