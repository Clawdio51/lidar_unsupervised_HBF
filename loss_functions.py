import torch

def SumRate(FDP, channel, noise_pwr=1e-12):
    #FDP = torch.unsqueeze(FDP, axis=0)
    FDP = FDP / torch.linalg.norm(FDP, dim=1, keepdim=True)
    #FDP = torch.unsqueeze(FDP, axis=1) + 0j
    #channel = torch.unsqueeze(channel, axis=2)
    #print(FDP.size())
    #print(channel.size()) 
    num_nan = torch.sum(torch.isnan(FDP))
    #print(f'Number of nan in FDP = {num_nan}')
    # Do not transpose channel since it is already transposed when created
    #W = torch.abs(torch.matmul(torch.transpose(torch.conj(channel), 2, 1), FDP)) ** 2
    W = torch.abs(torch.matmul(torch.conj(channel), FDP)) ** 2
    #print(W.size())
    SINR = torch.diagonal(W, dim1=1, dim2=2) / (torch.sum(W, 2) - torch.diagonal(W, dim1=1, dim2=2) + noise_pwr)
    userRates = torch.log2(1 + SINR)
    sumRate = userRates.sum(1)
    avgRate = userRates.mean(1)
    #print(sumRate)
    return sumRate#, avgRate

#criterion = lambda A, W, H: -SumRate(A, W, H)
criterionFDP = lambda U, H: torch.mean(-SumRate(U, H), axis=0)

def criterionAP(probs, codebook, FDP, channel):
    bs = probs.size(0)
    L = len(codebook)   # Length of the codebook (Number of codewords)
    
    # Compute the rates for each codeword in the codebook
    prob_rates = torch.zeros(L)     # Placeholder for probs*rates values
    for i in range(L):
        AP = codebook[i].unsqueeze(0).repeat(bs, 1, 1)
        DP = torch.bmm(torch.linalg.pinv(AP), FDP)
        HBF = torch.matmul(AP, DP)
        rates = SumRate(HBF, channel)
        prob_rates[i] = torch.mean(probs[:,i] * rates, dim=0)
    
    # Compute the loss function
    return -torch.sum(prob_rates)


# def criterion(preds, channel):
#     bs = preds.size(0)
#     codewords = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.complex128).cuda()
#     # Try to parallelize + optimize
#     idx = torch.zeros(bs, N, dtype=int)
#     for b in range(bs):
#         outputs_str = '0b'
#         for i in range(preds.size(1)):
#             outputs_str += str(int(preds[b,i].item()))
#         idx[b] = Decimal2Kary(int(outputs_str, 2), codewords.size(0), N) 
    
#     A = codewords[idx]
#     l = torch.mean(-SumRate(A, channel), axis=0) + 1e-4 * torch.sum(preds)
#     return l
#     return torch.mean(-SumRate(A, channel), axis=0)
# def criterion(preds, channel):
#     rates = torch.zeros(L).cuda()
#     bs = preds.size(0)
#     for i in range(L):
#         # Fix this: All the batch size has the same codeword at each iteration!!
#         rates[i] = torch.mean(-SumRate(codebook[i].repeat(bs, 1), channel), axis=0)
#     return torch.sum(preds * rates)


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterionFDP(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)