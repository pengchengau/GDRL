import numpy as np


def LEO_status():
    C_l = np.random.randint(150, 200) # The total number of VM instances
    LEO_place = np.random.randint(300e3, 500e3 + 1)  # LEO location from 300km to 500km
    return C_l, LEO_place


def HAPS_status():
    C_n = np.random.randint(100, 150) # The total number of VM instances
    HAPS_place = np.random.randint(20e3, 30e3+1) # HPAS location from 20km to 30km
    return C_n, HAPS_place


def all_LEO_status(L):
    C_l, LEO, C_l_ori = [], [], []
    for l in range(L):
        cl, leo = LEO_status()
        C_l.append(cl)
        LEO.append(leo)
        C_l_ori.append(cl)
    return np.array(C_l), np.array(LEO), np.array(C_l_ori)


def all_HAPS_status(L):
    C_n, HAPS, C_n_ori = [], [], []
    for l in range(L):
        cn, haps = HAPS_status()
        C_n.append(cn)
        HAPS.append(haps)
        C_n_ori.append(cn)
    return np.array(C_n), np.array(HAPS), np.array(C_n_ori)