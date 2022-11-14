from itertools import product
from numpy import kron  # Kronecker Product
import numpy as np
from torch import tensor

def Decimal2Kary(x, K, Nb):
    '''
        x: number to be converted
        K: K-ary
        Nb: number of bits
    '''
    # x should fit on Nb bits
    assert x < (K ** Nb), f'{x} does not fit on {Nb} bits for {K}-base'
    # Initialize output
    y = np.zeros(Nb, dtype=int)
    for i in range(Nb):
        _temp = K ** (Nb - i - 1)
        y[i] = x / _temp
        x %= _temp

    return tensor(y)


'''
    Try to generate codebook from N_RF directly
    Normalize codebooks!! As per equation (9) from "Autonomous Codebook Generation" Paper
'''
def DFT_Codebook(Nt, Nr, M):
    '''
        Returns codebook with Nt Tx antennas, Nr Rx antennas, and M users.
    '''

    codebook_Tx = [p for p in product(
        [1+1j, 1-1j, -1+1j, -1-1j],
        repeat=Nt
    )]
    # codebook_Rx = [p for p in product(
    #     [1+1j, 1-1j, -1+1j, -1-1j],
    #     repeat=Nr
    # )]

    # DFTcodebook = kron(codebook_Tx, codebook_Rx)

    codebook = [p for p in product(codebook_Tx, repeat=M)]
    codebook = tensor([list(item) for item in codebook])

    return codebook

if __name__ == '__main__':
    codebook = DFT_Codebook(4, 8, 2)
    print(codebook.size())
    #print(Decimal2Kary(32,4,4))

    