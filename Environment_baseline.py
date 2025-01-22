import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from Channel_Model import ChannelModel
from Rate_Calculation import rate_calculation
from Generate_inital_environment import ResetFunction


class NetworkEnvironment(gym.Env):
    def __init__(self, U, L, N, T, user_requests, user_lists, save_var, load_var, encoder_dis, encoder_con):
        super(NetworkEnvironment, self).__init__()
        # Define action space: 32 from the policy network of TRPO directly
        lower_bounds = np.ones(32) * -10
        upper_bounds = np.ones(32) * 10
        self.action_space = spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float32)

        # Define the state space
        # self.observation_space = spaces.Dict({
        ###### Var state
        #     'Uf': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       #User request type
        #     'Pu': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       # User transmission power
        #     'Su': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       # Size of the task(bit)
        #     'Ou': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       # Number of CPU cycles (cycle/bit)
        #     'Vu': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       # Size of the computation result (bit)
        #     'Lu': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),       # Delay constraint of the task
        #     'C_l_new': spaces.Box(low=-np.inf, high=np.inf, shape=(L, )),  # The remaining number of VM instances
        #     'C_n_new': spaces.Box(low=-np.inf, high=np.inf, shape=(N, )),
        #     'A_u': spaces.Box(low=-np.inf, high=np.inf, shape=(1, )),   # The remaining number of subchannels
        #     'A_u_ori': spaces.Box(low=-np.inf, high=np.inf, shape=(1, )),   # The original number of subchannels
        ######## Con state
        #     'C_u_ori': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),  # The initial remaining number of VM instances for user
        #     'C_l_ori': spaces.Box(low=-np.inf, high=np.inf, shape=(L,)),  # The initial remaining number of VM instances
        #     'C_n_ori': spaces.Box(low=-np.inf, high=np.inf, shape=(N,)),
        #     'U_place': spaces.Box(low=-np.inf, high=np.inf, shape=(U,)),  # User location
        #     'LEO_place': spaces.Box(low=-np.inf, high=np.inf, shape=(L,)),  # LEO location from 300km to 500km
        #     'HAPS_place': spaces.Box(low=-np.inf, high=np.inf, shape=(N,)),
        # })

        lower_bounds_state = np.ones(8*U+3*L+3*N+2)*-np.inf
        upper_bounds_state = np.ones(8*U+3*L+3*N+2)*np.inf
        self.observation_space = spaces.Box(low=lower_bounds_state, high=upper_bounds_state, dtype=np.float32)
        # Define other variables
        self.U = U
        self.L = L
        self.N = N
        self.T = T
        self.gamma1 = 0.8
        self.gamma2 = 1
        self.c = 3e8
        self.user_requests = user_requests
        self.save_var = save_var
        self.load_var = load_var
        self.user_lists = user_lists
        self.encoder_dis = encoder_dis
        self.encoder_con = encoder_con

    def step(self, action):
        tua_max = 10e9
        ## load all variables
        LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,\
        user_status, A_u, A_u_ori, A_resource, counter = self.load_var()

        Uf = self.user_requests['Uf'][:, counter]
        Pu = self.user_requests['Pu'][:, counter]
        Su = self.user_requests['Su'][:, counter]
        Ou = self.user_requests['Ou'][:, counter]
        Vu = self.user_requests['Vu'][:, counter]
        Lu = self.user_requests['Lu'][:, counter]
        C_l = LEO_status['C_l']
        LEO_place =  LEO_status['LEO_place']
        C_l_ori = LEO_status['C_l_ori']
        C_n = HAPS_status['C_n']
        HAPS_place =  HAPS_status['HAPS_place']
        C_n_ori = HAPS_status['C_n_ori']
        # C_u = user_status['C_u']
        U_place = user_status['U_place']
        C_u_ori = user_status['C_u_ori']
        C_resource_l = LEO_resource_status['C_resource_l']
        C_resource_n = HAPS_resource_status['C_resource_n']
        # C_resource_u = user_resource_status['C_resource_u']

        # Change the action from the policy network into exact discrete and continous action
        action = torch.tensor(action).type(torch.float32)
        dis_action = self.encoder_dis(action[0:16].unsqueeze(0), self.user_lists).squeeze() #action的前半段用来生成dis action
        con_action = self.encoder_con(action[16:32].unsqueeze(0)) #action的后半段用来生成con action
        # con_action: [u1 portion of the computational resources, u1 portion of the subcarriers, u2 .......]
        dis_action = [format(num, '07b') for num in dis_action] #从二进制变成十进制，注意最后都是string ['000100'....]
        s = torch.tensor([[int(char) for char in s] for s in dis_action]) #从 string变成torch tensor
        # 分割字符串
        part_length = len(s) // self.U
        parts = [s[i:i + part_length] for i in range(0, len(s), part_length)]
        C_l_new = C_l
        C_n_new = C_n
        A_u_new = A_u
        reward = 0
        latency_sum = torch.tensor(0.0)
        u = 0
        for part in parts:
            part = part.reshape(-1).tolist()
            last_four_part = part[-4:]
            index = ''.join(str(digit) for digit in last_four_part)
            index = int(index, 2)
            if part[2] == 1:
                reward += 0
            else:
                if part[0] == 0:
                    if C_l_new[index] == 0:
                        C_l_new[index] = C_l_new[index] + C_resource_l[index, counter]
                        reward = reward - 5
                        break
                    if A_u_new == 0:
                        A_u_new = A_u_new + A_resource[counter]
                        reward = reward - 5
                        break
                    process_time = (Su[u] * Ou[u]) / (con_action[u, 0] * C_l[index] * tua_max)  # second
                    number_mini_slot = torch.ceil(process_time * 1e4)  # number of minislot till this package is processed
                    index_time = int(min(counter+number_mini_slot, C_resource_l.shape[1]-1))
                    C_resource_l[index, index_time] += torch.ceil(con_action[u, 0] * C_l[index])
                    A_resource[int(min(counter+number_mini_slot, A_resource.shape[0]-1))] += torch.ceil(con_action[u, 1] * A_u)
                    C_l_new[index] = min(C_l_new[index] - torch.ceil(con_action[u, 0] * C_l[index]) + C_resource_l[index, counter], C_l_ori[index])
                    A_u_new = min(A_u_new - torch.ceil(con_action[u, 1] * A_u) + A_resource[counter], A_u_ori)
                    C_l_new[index] = max(C_l_new[index], 0)
                    A_u_new = max(A_u_new, 0)
                    hu = ChannelModel(Uf[u], LEO_place[index])
                else:
                    if C_n_new[index] == 0:
                        C_n_new[index] = C_n_ori[index]
                        reward = reward - 5
                        break
                    if A_u_new == 0:
                        A_u_new = A_u_new + A_resource[counter]
                        reward = reward - 5
                        break
                    process_time = (Su[u] * Ou[u]) / (con_action[u, 0] * C_n[index] * tua_max)  # second
                    number_mini_slot = torch.ceil(process_time * 1e4)  # number of minislot till this package is processed
                    C_resource_n[index, int(min(counter+number_mini_slot, C_resource_n.shape[1]-1))] += torch.ceil(con_action[u, 0] * C_n[index])
                    A_resource[int(min(counter+number_mini_slot, A_resource.shape[0]-1))] += torch.ceil(con_action[u, 1] * A_u)
                    C_n_new[index] = min(C_n_new[index] - torch.ceil(con_action[u, 0] * C_n[index]) + C_resource_n[index, counter], C_n_ori[index])
                    A_u_new = min(A_u_new - torch.ceil(con_action[u, 1] * A_u) + A_resource[counter], A_u_ori)
                    C_n_new[index] = max(C_n_new[index], 0)
                    A_u_new = max(A_u_new, 0)
                    hu = ChannelModel(Uf[u], HAPS_place[index])
                Ru_k = con_action[u, 1] * A_u * rate_calculation(Pu[u], hu, Uf[u])
                if part[1] == 0:
                    latency = process_time + Su[u]/Ru_k
                else:
                    if part[0] == 0:
                        d = min(np.abs(LEO_place[index] - LEO_place[u]), np.abs(LEO_place[index] - LEO_place[u+1]))
                        latency = process_time + Su[u]/Ru_k + d/self.c
                    else:
                        d = min(np.abs(HAPS_place[index] - HAPS_place[u]), np.abs(HAPS_place[index] - HAPS_place[u + 1]))
                        latency = process_time + Su[u] / Ru_k + d/self.c
                if Uf[u] == 4 or Uf[u] == 5 or Uf[u] == 6:
                    Lu_new = Lu[u]
                else:
                    Lu_new = Lu[u]

                reward += self.gamma1 * (Lu_new - latency * 1e4) - self.gamma2 * latency * 1e3
                latency_sum += latency

            u += 1
        LEO_status = {
            'C_l': C_l_new,
            'LEO_place': LEO_place,
            'C_l_ori': C_l_ori
        }
        HAPS_status = {
            'C_n': C_n_new,
            'HAPS_place': HAPS_place,
            'C_n_ori': C_n_ori
        }
        user_status = {
            'U_place': U_place,
            'C_u_ori': C_u_ori
        }
        LEO_resource_status = {
            'C_resource_l': C_resource_l
        }

        HAPS_resource_status = {
            'C_resource_n': C_resource_n
        }

        A_u_new = torch.tensor(A_u_new).unsqueeze(0)
        A_u_ori = torch.tensor(A_u_ori).unsqueeze(0)
        state_list = [Uf, Pu, Su, Ou, Vu, Lu, C_l_new, C_n_new, A_u_new, A_u_ori, C_u_ori, C_l_ori, C_n_ori,
                 U_place, LEO_place, HAPS_place]
        state = np.concatenate(state_list, axis=0)
        state = state.astype(np.float32)
        counter += 1
        A_u_new = A_u_new.squeeze(0)
        A_u_ori = A_u_ori.squeeze(0)
        self.save_var(LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                      user_status, A_u_new, A_u_ori, A_resource, counter)
        # Compute done and counter
        if counter == self.T:
            done = True
        else:
            done = False
        #print('counter', counter)
        #print('reward', reward)
        #print('latencys', latency_sum)
        reward = float(reward)
        info = {'latency': latency_sum.detach().clone(),
                'dis_action': dis_action,
                'con_action': con_action.detach().clone()}
        return state, reward, done, False, info

    def reset(self, seed=None, options=None):
        counter = 0
        LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
        user_status, A_u, A_u_ori, A_resource = ResetFunction(self.N, self.L, self.T, self.U)

        self.save_var(LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
                      user_status, A_u, A_u_ori, A_resource, counter)

        user_requests = np.load('user_requests.npz')
        Uf = user_requests['Uf'][:, counter]
        Pu = user_requests['Pu'][:, counter]
        Su = user_requests['Su'][:, counter]
        Ou = user_requests['Ou'][:, counter]
        Vu = user_requests['Vu'][:, counter]
        Lu = user_requests['Lu'][:, counter]
        C_l = LEO_status['C_l']
        C_l_ori = LEO_status['C_l_ori']
        LEO_place = LEO_status['LEO_place']
        C_n = HAPS_status['C_n']
        C_n_ori = HAPS_status['C_n_ori']
        HAPS_place = HAPS_status['HAPS_place']
        U_place = user_status['U_place']
        C_u_ori = user_status['C_u_ori']

        A_u = torch.tensor(A_u).unsqueeze(0)
        A_u_ori = torch.tensor(A_u_ori).unsqueeze(0)
        state_list = [Uf, Pu, Su, Ou, Vu, Lu, C_l, C_n, A_u, A_u_ori, C_u_ori, C_l_ori, C_n_ori,
                      U_place, LEO_place, HAPS_place]
        state = np.concatenate(state_list, axis=0)
        state = state.astype(np.float32)
        return state, {}
