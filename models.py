from torch import nn


class Lidar2D(nn.Module):
    def __init__(self, N, M, AP_or_FDP):
        super(Lidar2D, self).__init__()
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
        #self.linear7 = nn.Linear(125 * (1 + self.channels), 16) # My Code
        self.relu7 = nn.ReLU()
        #self.linear8 = nn.Linear(16, 256)
        
        self.linear_FDP = nn.Linear(16, 2 * N * M)
        self.linear_AP = nn.Linear(16, M)   # In AP, M represents the codebook length

        self.N = N
        self.M = M
        self.AP_or_FDP = AP_or_FDP


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

        if self.AP_or_FDP == 'FDP':
            x = self.linear_FDP(x)
            x = x.view(2, -1, self.N, self.M).double()
            x = x[0] + 1j*x[1]
        elif self.AP_or_FDP == 'AP':
            # Here, M represents the codebook length, and N is not used
            x = self.linear_AP(x)
        else:
            raise ValueError('"AP_or_FDP" value can be either "AP" or "FDP".')

        return x

# My Code
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2 * N * M)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        #self.logsoftmax = nn.LogSoftmax(dim=1)

        self.N = N
        self.M = M

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        #return self.logsoftmax(output), matrix3x3, matrix64x64
        output = output.view(2, -1, self.M, self.N)
        output = output[0] + 1j*output[1]
        return output, matrix3x3, matrix64x64


