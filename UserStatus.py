import numpy as np
import torch

def user_status():
    VM_user = 0 # The total number of VM instances
    user_place = np.random.randint(50, 100) # user location from 50 to 100
    #subchan_user = 0 # The number of allocated sunchannels
    return VM_user, user_place

# user_place, VM_user, subchan_user

def all_user_status(U):
    VM_user, user_place = [], []
    for u in range(U):
        vm, place = user_status()
        VM_user.append(vm)
        user_place.append(place)
    return torch.Tensor(VM_user), torch.Tensor(user_place)
