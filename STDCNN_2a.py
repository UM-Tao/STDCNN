import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from typing import TypeAlias
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import datetime, time
from layers import Conv2dWithConstraint


import sys
import importlib.metadata
from typing import Dict, List


# Set GPU configuration
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

# Define Device TypeAlias
Device: TypeAlias = str | torch.device | None


def get_imported_packages_with_versions() -> Dict[str, str]:
    """获取当前导入的所有第三方包及其版本"""
    imported_packages = {}

    # 获取所有已导入的模块
    for name, module in sys.modules.items():
        # 过滤掉 Python 标准库、内置模块和子模块
        if (
                name in sys.stdlib_module_names  # Python 3.10+
                or name.startswith('_')
                or '.' in name  # 避免子模块重复（如 `numpy.core`）
                or not hasattr(module, '__file__')  # 内置模块（如 `sys`）
        ):
            continue

        # 尝试获取包名（`numpy` 而不是 `numpy.core`）
        package_name = name.split('.')[0]

        # 获取包版本
        try:
            version = importlib.metadata.version(package_name)
            imported_packages[package_name] = version
        except importlib.metadata.PackageNotFoundError:
            # 如果找不到版本，可能是内置模块或本地文件
            continue

    return imported_packages


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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # learnable state transition matrix A
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim))
        # learnable input matrix B
        self.B = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        # learnable output matrix C
        self.C = nn.Parameter(torch.eye(hidden_dim, output_dim))

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
                 num_channels=22,
                 F1=8, D=2, F2='auto', pool_mode='mean',
                 drop_out=0.25):
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
            pooling_layer((1, 32), stride=32),
            nn.Dropout(drop_out),
        )

        # Spatial 2
        self.spatial_2 = nn.Sequential(
            Conv2dWithConstraint(F1, F2, kernel_size=[num_channels, 1], padding='valid', max_norm=2.),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, 75), stride=25),
            ActLog(),
            nn.Dropout(drop_out),
        )

        # SSM State-Space Model (Temporal Dynamics)
        self.SSM = StateSpaceModel(input_dim=F2, hidden_dim=F2, output_dim=F2)

        # Transformer Encoder Layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=F2, nhead=8, dim_feedforward=64),
            num_layers=2
        )

    def forward(self, x):
        x_1 = self.spectral_1(x)
        x_2 = self.spectral_2(x)
        # Extract spatial features
        x_filter_1 = self.spatial_1(x_1)
        x_filter_2 = self.spatial_2(x_2)

        # Combine the spatial features
        x_noattention = torch.cat((x_filter_1, x_filter_2), 3)
        B2, C2, H2, W2 = x_noattention.shape

        # Flatten for attention mechanism
        x_attention = x_noattention.reshape(B2, C2, H2 * W2).permute(0, 2, 1)

        # Apply SSM (Temporal Modeling)
        x_attention_SSM = self.SSM(x_attention)
        # Transformer Encoder Layer
        x_attention = self.transformer(x_attention_SSM)
        x_attention = x_attention.reshape(B2, W2, 1, C2).permute(0, 3, 2, 1)
        return x_attention


# Classifier Layer
class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(16, num_classes, (1, 69)),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)
        return x


# Main Network with all Innovations
class NetInnovative(nn.Module):
    def __init__(self, num_classes=4, num_channels=22):
        super(NetInnovative, self).__init__()

        self.backbone = STDCNN(num_channels=num_channels)

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


# Function to log model information
def get_model_size(model):
    """Calculate the size of the model in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def measure_inference_time(model, input_data, num_runs=100):
    """Measure the average inference time in milliseconds."""
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_data)
        end_time = time.time()
    avg_inference_time = (end_time - start_time) * 1000 / num_runs
    return avg_inference_time


def measure_inference_time_cuda(model, input_data, num_runs=100):
    """Measure the average inference time on CUDA."""
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        # Warm-up
        _ = model(input_data)

        # Measure
        torch.cuda.synchronize()
        starter.record()
        for _ in range(num_runs):
            _ = model(input_data)
        ender.record()
        torch.cuda.synchronize()

    elapsed_time_ms = starter.elapsed_time(ender) / num_runs
    return elapsed_time_ms


def log_model_info(model, input_data, log_file, num_runs=100):
    """Log the model size and inference time to the given log file."""
    # Calculate model size
    model_size = get_model_size(model)

    # Measure inference time on CPU
    avg_inference_time = measure_inference_time(model, input_data, num_runs)

    # Measure inference time on CUDA
    avg_inference_time_cuda = measure_inference_time_cuda(model, input_data, num_runs)

    # Write to log file
    log_file.write(f"Model size: {model_size:.3f} MB\n")
    log_file.write(f"Average inference time: {avg_inference_time:.3f} ms\n")
    log_file.write(f"Average CUDA inference time: {avg_inference_time_cuda:.3f} ms\n")

    print(f"Model size: {model_size:.3f} MB")
    print(f"Average inference time: {avg_inference_time:.3f} ms")
    print(f"Average CUDA inference time: {avg_inference_time_cuda:.3f} ms")


# Experiment class for training and testing
class ExP():
    def __init__(self, nsub):
        self.batch_size = 72
        self.n_epochs = 2
        self.lr = 0.001
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.start_epoch = 0

        # Create directories if they don't exist
        log_dir = f"results/2a/STDCNN"
        self.mod_dir = f"./results/2a/STDCNN/save_model/"
        os.makedirs(self.mod_dir, exist_ok=True)

        os.makedirs(log_dir, exist_ok=True)

        self.log_write = open(f"{log_dir}/log_subject{self.nSub}.txt", "w")
        self.result_write = open(f"{log_dir}/sub_result_subject{self.nSub}.txt", "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = NetInnovative().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])

    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for i in range(int(self.batch_size / 4)):
                for j in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[i, :, :, j * 125:(j + 1) * 125] = tmp_data[rand_idx[j], :, :,
                                                                      j * 125:(j + 1) * 125]
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda().float()
        aug_label = torch.from_numpy(aug_label).cuda().long()
        return aug_data, aug_label

    def get_source_data(self):
        dataset = BNCI2014_001()
        paradigm = MotorImagery(n_classes=4, fmin=0, fmax=40)
        self.total_data, self.total_label, meta = paradigm.get_data(dataset, subjects=[self.nSub])
        session_T_mask = meta['session'] == '0train'
        self.train_data = self.total_data[session_T_mask]
        self.train_label = self.total_label[session_T_mask]
        label_mapping = {"left_hand": 0, "right_hand": 1, "tongue": 2, "feet": 3}
        self.train_label = np.array([label_mapping[label] for label in self.train_label])

        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data[:, :, :, :1000]
        self.allLabel = self.train_label

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        session_E_mask = meta['session'] == '1test'
        self.test_data = self.total_data[session_E_mask]
        self.test_label = self.total_label[session_E_mask]
        self.test_label = np.array([label_mapping[label] for label in self.test_label])
        self.test_data = np.expand_dims(self.test_data, axis=1)

        self.testData = self.test_data[:, :, :, :1000]
        self.testLabel = self.test_label

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

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
        Y_true = 0
        Y_pred = 0

        epoch_times = []

        for e in range(self.n_epochs):
            epoch_start_time = time.time()  # Start timing the epoch
            self.model.train()
            for img, label in self.dataloader:
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

            epoch_end_time = time.time()  # End timing the epoch
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            if (e + 1) % 1 == 0:
                self.model.eval()
                Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(
                    f'Epoch: {e}  Train loss: {loss.detach().cpu().numpy():.6f}  Test loss: {loss_test.detach().cpu().numpy():.6f}  Train accuracy: {train_acc:.6f}  Test accuracy: {acc:.6f}\n')

                num += 1
                averAcc += acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        torch.save(self.model.module.state_dict(), self.mod_dir + f'STDCNN_subj{self.nSub}.pth')
        averAcc /= num
        avg_epoch_time = np.mean(epoch_times)  # Calculate average epoch time
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        print('The average epoch time is: %.3f seconds' % avg_epoch_time)

        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.write('The average epoch time is: %.3f seconds\n' % avg_epoch_time)

        # Log model information
        input_data = torch.randn(1, 1, 22, 1000).cuda()  # Example input tensor
        log_model_info(self.model.module, input_data, self.result_write)

        return bestAcc, averAcc, Y_true, Y_pred


def main():
    best = 0
    aver = 0
    log_dir = f"results/2a/STDCNN"
    os.makedirs(log_dir, exist_ok=True)
    result_write = open(log_dir + "/sub_result.txt", "w")

    for i in range(2):
        starttime = datetime.datetime.now()
        seed_n = 42  # Fixed random seed
        print('Seed is ' + str(seed_n))
        set_seed(seed_n)

        print('Subject %d' % (i + 1))
        exp = ExP(i + 1)
        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('Subject %d duration: ' % (i + 1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc

    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
    packages = get_imported_packages_with_versions()
    print("本次运行导入的第三方包及版本：")
    for pkg, version in packages.items():
        print(f"{pkg}=={version}")