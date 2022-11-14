import torch
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt
import os


def beams_log_scale(y, thresholdBelowMax):
    y_shape = y.shape

    for i in range(0, y_shape[0]):
        thisOutputs = y[i, :]
        logOut = 20 * np.log10(thisOutputs + 1e-30)
        minValue = np.amax(logOut) - thresholdBelowMax
        zeroedValueIndices = logOut < minValue
        thisOutputs[zeroedValueIndices] = 0
        thisOutputs = thisOutputs / sum(thisOutputs)
        y[i, :] = thisOutputs

    return y 


def get_beam_output(output_file):
    thresholdBelowMax = 6

    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    # new ordering of the beams, provided by the Organizers
    y = np.zeros((yMatrix.shape[0], num_classes))
    for i in range(0, yMatrix.shape[0], 1):  # go over all examples
        codebook = np.absolute(yMatrix[i, :])  # read matrix
        Rx_size = codebook.shape[0]  # 8 antenna elements
        Tx_size = codebook.shape[1]  # 32 antenna elements
        for tx in range(0, Tx_size, 1):
            for rx in range(0, Rx_size, 1):  # inner loop goes over receiver
                y[i, tx * Rx_size + rx] = codebook[rx, tx]  # impose ordering

    yy = beams_log_scale(y, thresholdBelowMax)

    return yy, num_classes


def lidar_to_2d(lidar_data_path):

    lidar_data = np.load(lidar_data_path)['input']
    print(lidar_data.shape)

    lidar_data1 = np.zeros_like(lidar_data)[:, :, :, 1]

    lidar_data1[np.max(lidar_data == 1, axis=-1)] = 1
    lidar_data1[np.max(lidar_data == -2, axis=-1)] = -2
    lidar_data1[np.max(lidar_data == -1, axis=-1)] = -1

    return lidar_data1
    
    #return lidar_data   # My Code

def removeNan(channel_data, lidar_data):
    # Indices to be removed
    rm_idx = np.array([], dtype=int)
    for i in range(channel_data.shape[0]):
        if np.isnan(channel_data[i]).any():
            rm_idx = np.append(rm_idx, int(i))
        
    channel_data = np.delete(channel_data, rm_idx, axis=0)
    lidar_data = np.delete(lidar_data, rm_idx, axis=0)

    return channel_data, lidar_data

class LidarDataset2D(Dataset):
    def __init__(self, lidar_data_path, beam_data_path, channel_data_path):
        # this allows us to merge multiple dsets into one
        if isinstance(lidar_data_path, list) and isinstance(beam_data_path, list):
            lidar_data = None
            beam_output = None
            for lidar_path, beam_path in zip(lidar_data_path, beam_data_path):
                if lidar_data is not None and beam_output is not None:
                    lidar_data = np.concatenate([lidar_data, lidar_to_2d(lidar_path)], axis=0)
                    beam_output = np.concatenate([beam_output, get_beam_output(beam_path)[0]], axis=0)
                else:
                    lidar_data = lidar_to_2d(lidar_path)
                    beam_output = get_beam_output(beam_path)[0]

        else:
            lidar_data = lidar_to_2d(lidar_data_path)
            if beam_data_path is None:
                beam_output = np.zeros((lidar_data.shape[0], 256))
            else:
                beam_output = get_beam_output(beam_data_path)[0]

        self.lidar_data = lidar_data
        self.beam_output = beam_output
        self.channel = np.load(channel_data_path)['channel_data']

        self.channel, self.lidar_data = removeNan(self.channel, self.lidar_data)

        self.lidar_data = torch.from_numpy(self.lidar_data).float()
        self.beam_output = torch.from_numpy(self.beam_output).float()
        self.channel = torch.from_numpy(self.channel)

    def __len__(self):
        return self.lidar_data.shape[0]

    def __getitem__(self, idx):
        return self.lidar_data[idx], self.beam_output[idx], self.channel[idx]

if __name__ == '__main__':
    lidar_data_path = 'lidar_train.npz'
    beam_data_path = 'beams_output_train.npz'
    channel_data_path = 'myChannelInfo_train.npz'
    dataset = LidarDataset2D(lidar_data_path, beam_data_path, channel_data_path)
    lidar, beam, channel = dataset[100]
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    params = {
    'ytick.color': 'w',
    'xtick.color': 'w',
    'axes.labelcolor': 'w',
    'axes.edgecolor': 'w'
    }
    plt.rcParams.update(params)
    plt.imshow(lidar)
    plt.savefig(r'2d_LiDAR_transparent.png', transparent=True)
    plt.show()