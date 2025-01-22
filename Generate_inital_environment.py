import numpy as np

from HAPSLEO_Status import all_HAPS_status, all_LEO_status
from UserStatus import all_user_status
import torch

def ResetFunction(N, L, T, U):
    # Generate LEO Status with each vector of size (L, 1)
    C_l, LEO_place, C_l_ori = all_LEO_status(L)
    LEO_status = {
        'C_l': C_l,
        'LEO_place': LEO_place,
        'C_l_ori': C_l_ori
    }

    # Generate HAPS Status with each vector of size (N, 1)
    C_n, HAPS_place, C_n_ori = all_HAPS_status(N)
    HAPS_status = {
        'C_n': C_n,
        'HAPS_place': HAPS_place,
        'C_n_ori': C_n_ori
    }

    # Create a matrix to store the information of changing resources with the size (L, T), (N, T) and (T, 1)
    LEO_resource_status = {
        'C_resource_l': np.zeros((L, T))
    }

    HAPS_resource_status = {
        'C_resource_n': np.zeros((N, T))
    }
    A_resource = np.zeros((T))

    # Generate user status with each of the size (U, 1)
    C_u_ori, U_place = all_user_status(U)
    user_status = {
        'C_u_ori': C_u_ori,
        'U_place': U_place
    }

    # Generate the original subchannel status
    A_u = 30
    A_u_ori = 30


    return LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
           user_status, A_u, A_u_ori, A_resource





