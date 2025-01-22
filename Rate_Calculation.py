import numpy as np
from scipy.special import erfcinv

def rate_calculation(pu, hu, uf):
    B = 15e3 # sub-carrier bandwidth
    Tf = 1e-3 # time-length of one mini-slot
    epsilon_D = 1e-3 # decoding error probability
    sigma = 1e-12 # variance of the additive white Gaussian noise

    if uf == 1 or uf == 2 or uf == 3:
        Ru_k = B*np.log2(1 + (pu * np.real(np.conj(hu).T @ hu)[0][0] / sigma**2)) # Ru_T_k_n_l: channel 2 and 3 throughput
    else:
        Ru_k = B/np.log(2) * (np.log(1 + (pu * np.real(np.conj(hu).T @ hu)[0][0] / (sigma**2 * B)))
                              - np.sqrt(1 / (Tf * B)) * erfcinv(2*epsilon_D)) # Ru_D_k_n_l channel 2 and 3 delay

    return Ru_k
