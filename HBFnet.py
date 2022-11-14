import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from numpy.core.fromnumeric import mean
import requests

from dataloader import LidarDataset2D
from loss_functions import SumRate

N = 24
M = 10
N_RF = 12
#L = N_RF ** N
sendProgressUpdate = False      # Enables/Disables notifications

args = {
    'lidar_training_data': 'lidar_train.npz',
    'lidar_validation_data': 'lidar_validation.npz',
    'beam_training_data': 'beams_output_train.npz',
    'beam_validation_data': 'beams_output_validation.npz',
    'model_path': 'test_model',
    'channel_training_data': 'myChannelInfo_train.npz',
    'channel_validation_data': 'myChannelInfo_valid.npz',
    'codebook_real_data': 'reduced_codebook_real.mat',
    'codebook_imag_data': 'reduced_codebook_imag.mat'
}

def initialize_codebook(path_real, path_imag):
    data_real = torch.tensor(loadmat(path_real)['cb_real'])
    data_imag = torch.tensor(loadmat(path_imag)['cb_imag'])

    # When transferring from matlab to python, values that are zero have a small round_off error
    # Fix this by replacing all these values with actual zero
    data_real[data_real < 0.1] = 0
    data_imag[data_imag < 0.1] = 0

    codebook = data_real + 1j*data_imag
    return codebook

def sendNotification(text, silent=True):
    if sendProgressUpdate:
        token = 'xxxxxxxxxx:your_token'
        url = f'https://api.telegram.org/bot{token}'
        params = {'chat_id':1388173517, 'text':text, 'disable_notification':silent}
        r = requests.get(url + '/sendMessage', params=params)

class HBFnet(nn.Module):
    def __init__(self, M, N_RF, L):
        super().__init__()
        self.channels = 5
        self.conv1 = nn.Conv2d(1, self.channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.PReLU(num_parameters=self.channels)
        self.conv2 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.PReLU(num_parameters=self.channels)
        self.conv3 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu3 = nn.PReLU(num_parameters=self.channels)
        self.conv4 = nn.Conv2d(self.channels, self.channels, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(self.channels)
        self.relu4 = nn.PReLU(num_parameters=self.channels)
        self.conv5 = nn.Conv2d(self.channels, self.channels, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.channels)
        self.relu5 = nn.PReLU(num_parameters=self.channels)
        self.conv6 = nn.Conv2d(self.channels, self.channels, 3, (1, 2), 1)
        self.bn6 = nn.BatchNorm2d(self.channels)
        self.relu6 = nn.PReLU(num_parameters=self.channels)
        self.linear7 = nn.Linear(125 * self.channels, 16)
        self.relu7 = nn.ReLU()

        self.linear_DP_R = nn.Linear(16, M * N_RF)  # Real part of digital precoder
        self.linear_DP_I = nn.Linear(16, M * N_RF)  # Imaginary part of digital precoder
        self.linear_AP = nn.Linear(16, L)

        self.M = M
        self.N_RF = N_RF
        self.L = L

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        #
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = x.view(-1, 125 * self.channels)

        x = self.linear7(x)
        x = self.relu7(x)

        DP_R = self.linear_DP_R(x).double()
        DP_I = self.linear_DP_I(x).double()
        DP_out = DP_R + 1j*DP_I
        DP_out = DP_out.view(-1, self.N_RF, self.M)
        AP_out = self.linear_AP(x)

        return DP_out, AP_out

def criterion(probs, codebook, DP, channel):
    bs = probs.size(0)
    L = len(codebook)   # Length of the codebook (Number of codewords)
    
    # Compute the rates for each codeword in the codebook
    prob_rates = torch.zeros(L)     # Placeholder for probs*rates values
    for i in range(L):
        AP = codebook[i].unsqueeze(0).repeat(bs, 1, 1)
        HBF = torch.matmul(AP, DP)
        rates = SumRate(HBF, channel)
        prob_rates[i] = torch.mean(probs[:,i] * rates, dim=0)   # Check if doing the mean here is correct
    
    # Compute the loss function
    return -torch.sum(prob_rates)

def evaluate(net, test_dataloader, codebook):
    with torch.no_grad():
        net.eval()
        avg_sumRate = 0
        test_dataloader_length = 0
        for i, data in enumerate(test_dataloader):
            lidar, beams, channel = data
            lidar = lidar.cuda()
            channel = channel.cuda()
            DP, AP_output = net(lidar)
            AP_probs = F.softmax(AP_output, dim=1)
            _, AP_index = torch.max(AP_probs, dim=1)
            AP = codebook[AP_index]
            HBF = torch.matmul(AP, DP)
            sumRate = -SumRate(HBF, channel)   
            avg_sumRate += sumRate.item()   # Check if this is correct with batch size > 1
            test_dataloader_length += 1
        avg_sumRate = avg_sumRate / test_dataloader_length
        print(f'Validation Sum Rate: {avg_sumRate}')
        sendNotification(f'Validation Sum Rate: {avg_sumRate}')
    net.train()
    return avg_sumRate

def ZF_FDP(loader, batch_size):
    rates = []
    for i, data in enumerate(loader):
        _, _, H = data
        H = H.cuda()
        H_transpose = torch.transpose(H, 2, 1)
        ZF = torch.matmul(
            H_transpose,
            torch.linalg.pinv(
                torch.matmul(torch.conj(H), H_transpose)
            )
        )
        ZF = ZF / torch.linalg.norm(ZF, dim=1, keepdim=True)
        ZF = torch.nan_to_num(ZF, nan=0)    # Replace nan with 0 (Necessary for normalization of 0 channels)
        W = torch.abs(torch.matmul(torch.conj(H), ZF)) ** 2
    
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + 1e-12)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        sum_sumRate = torch.sum(sumRate) # Sum of Sum rates over one batch
        rates.append(sum_sumRate)
    return sum(rates) / (batch_size * len(rates))  # Get average sum rate

if __name__ == '__main__':

    #train_dataset = LidarDataset2D(args['lidar_training_data'], args['beam_training_data'])
    train_dataset = LidarDataset2D(args['lidar_training_data'], args['beam_training_data'], args['channel_training_data'])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    if args['lidar_validation_data'] is None and args['beam_validation_data'] is None:
        args['lidar_validation_data'] = args['lidar_training_data']
        args['beam_validation_data'] = args['beam_training_data']

    #validation_dataset = LidarDataset2D(args['lidar_validation_data'], args['beam_validation_data'])
    validation_dataset = LidarDataset2D(args['lidar_validation_data'], args['beam_validation_data'], args['channel_validation_data'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    ################ Compute ZF_FDP ###################
    print(f'Sum Rate for Train Dataset = {ZF_FDP(train_dataloader, batch_size=16)}')
    print(f'Sum Rate for Validation Dataset = {ZF_FDP(validation_dataloader, batch_size=1)}')

    # Initialize reduced codebook
    codebook = initialize_codebook(args['codebook_real_data'], args['codebook_imag_data']).cuda()
    # Get length of codebook
    codebook_length = codebook.size(0)

    net = HBFnet(M, N_RF, codebook_length).cuda()
    optimizer = optim.Adam(net.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)

    running_loss = 0.0
    sendNotification('----------------------------------------------------------------------------')
    sendNotification('Training Start')
    print('Training Start:\n--------------')
    train_accumulated_loss_per_epoch = []
    val_accumulated_loss_per_epoch = []
    for epoch in range(20):
        accumulated_loss = []
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            lidar, beams, channel = data
            lidar = lidar.cuda()
            channel = channel.cuda()
            
            DP, AP_output = net(lidar)
            AP_probs = F.softmax(AP_output, dim=1)

            loss = criterion(AP_probs, codebook, DP, channel)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            accumulated_loss.append(loss.item())
            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_dataloader), running_loss / 10))
                sendNotification('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_dataloader), running_loss / 10))
                running_loss = 0.0

        train_accumulated_loss_per_epoch.append(mean(accumulated_loss))
        val_accumulated_loss_per_epoch.append(evaluate(net, validation_dataloader, codebook))
        scheduler.step()

        # # Save checkpoints of models
        # FDP_model_path = os.path.join(args['checkpoints_path'], 'FDP_checkpoint_' + str(epoch))
        # AP_model_path = os.path.join(args['checkpoints_path'], 'AP_checkpoint_' + str(epoch))
        # torch.save(FDP_model.state_dict(), FDP_model_path)
        # torch.save(AP_model.state_dict(), AP_model_path)

    from numpy import savez
    file_path = 'results.npz'
    savez(file_path, train=train_accumulated_loss_per_epoch, val=val_accumulated_loss_per_epoch)

    sendNotification('Done Training', silent=False)