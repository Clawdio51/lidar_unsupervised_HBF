import torch
import numpy as np
import requests

sendProgressUpdate = False      # Disables notifications

def removeNan(channel_data):
    # Indices to be removed
    rm_idx = np.array([], dtype=int)
    for i in range(channel_data.shape[0]):
        if np.isnan(channel_data[i]).any():
            rm_idx = np.append(rm_idx, int(i))
        
    channel_data = np.delete(channel_data, rm_idx, axis=0)

    return channel_data

def sendNotification(text, silent=True):
    if sendProgressUpdate:
        token = '2009509203:AAHfCImvziYXf1R3nHTO5nBVxHUXKz0KDj4'
        url = f'https://api.telegram.org/bot{token}'
        params = {'chat_id':1388173517, 'text':text, 'disable_notification':silent}
        r = requests.get(url + '/sendMessage', params=params)

def SumRate(FDP_func, noise_pwr=1e-12):

    def getSumRate(FDP, channel):
        FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
        W = torch.abs(torch.matmul(torch.conj(channel), FDP)) ** 2
        SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + noise_pwr)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)
        return sumRate
    
    def wrapper(channel):
        FDP = FDP_func(channel)
        sumRate = getSumRate(FDP, channel)
        mean_sumRate = torch.mean(sumRate, dim=0)
        return mean_sumRate

    return wrapper

@SumRate
def ZF_FDP(channel):
    for i in range(channel.size(0)):
        try:
            H_inv = torch.linalg.pinv(channel[i])
            print(f'Computing FDP for channel[{i}]')
        except:
            continue
    return H_inv

if __name__ == '__main__':
    channel_train_data_path = 'N:\\Claudio\\Raymobtime\\Raymobtime Dataset and Scenarios\\Raymobtime_s008\\processed_raw_data\\myChannelInfo_train.npz'
    channel_train = np.load(channel_train_data_path)['channel_data']
    channel_test_data_path = 'N:\\Claudio\\Raymobtime\\Raymobtime Dataset and Scenarios\\Raymobtime_s008\\processed_raw_data\\myChannelInfo_valid.npz'
    channel_test = np.load(channel_test_data_path)['channel_data']

    channel_train = removeNan(channel_train)
    channel_test = removeNan(channel_test)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    channel_train = torch.from_numpy(channel_train).to(device)
    channel_test = torch.from_numpy(channel_test).to(device)

    train_sum_rate = ZF_FDP(channel_train)
    test_sum_rate = ZF_FDP(channel_test)

    print(f'Train Sum Rate = {train_sum_rate}')
    sendNotification(f'Train Sum Rate = {train_sum_rate}')
    print(f'Test Sum Rate = {test_sum_rate}')
    sendNotification(f'Test Sum Rate = {test_sum_rate}')