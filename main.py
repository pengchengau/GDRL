import os

import torch.nn

from Environment_baseline import NetworkEnvironment
from Generate_adj_matrix import *
from UserRequest import all_user_feature
from UserStatus import *
from torch_geometric.utils import to_edge_index
from arg_parser import get_args
from UpdateVariable import updatevalue
from amp import *
from sb3_contrib import TRPO
from CallBack import CustomCallback
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Feature import CustomFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    args = get_args()

    Uf, Pu, Su, Ou, Vu, Lu = all_user_feature(args.U, args.T)
    user_requests = {
        'Uf': Uf,
        'Pu': Pu,
        'Su': Su,
        'Ou': Ou,
        'Vu': Vu,
        'Lu': Lu,
    }
    np.save('user_requests.npz', user_requests)
    action_space_len, Adj_Matrix, user_lists = GenerateAdjacency(args.U, args.L, args.N)
    save_var, load_var = updatevalue()
    #encoder_dis = M1EncoderDis(16, action_space_len)  # The output dim is the total number of possible actions foe all users
    #encoder_con = M1EncoderCon(16, 2*args.U)  # The output dim is the number of continious actions
    custom_callback = CustomCallback()  # Define the callback function for recording the reward and action
    # Defination related to autoencoder
    autoencoder_dis = AutoencoderDis(16, action_space_len, args.U, user_lists, M1EncoderDis, M1DecoderDis)
    autoencoder_con = AutoencoderCon(16, 2*args.U, M1EncoderCon, M1DecoderCon)
    encoder_dis = autoencoder_dis.encoder
    encoder_con = autoencoder_con.encoder
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer1 = torch.optim.Adam(autoencoder_dis.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(autoencoder_con.parameters(), lr=0.001)

    sparse_Adj_Matrix = torch.Tensor(Adj_Matrix).to_sparse()
    edge_index, _ = to_edge_index(sparse_Adj_Matrix)
    torch.save(edge_index, 'edge_index.pt')
    train_model_num = 1
    # Load the pretrained encoder decoder if you have one
    for i in range(train_model_num):
        try:
            state_dict_dis = torch.load('./model_save/autoencoder_dis.pth')
            autoencoder_dis.load_state_dict(state_dict_dis)
            state_dict_con = torch.load('./model_save/autoencoder_con.pth')
            autoencoder_con.load_state_dict(state_dict_con)
            encoder_dis = autoencoder_dis.encoder
            encoder_con = autoencoder_con.encoder
        except FileNotFoundError:
            pass
        env = NetworkEnvironment(args.U, args.L, args.N, args.T, user_requests, user_lists, save_var,
                                 load_var, encoder_dis, encoder_con)
        env = Monitor(env, filename="monitor_logs", allow_early_resets=True)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        #policy_kwargs = dict(activation_fn=torch.nn.Sigmoid,
                             #net_arch=dict(pi=[32, 32], vf=[16, 4]))
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=32
            ),
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4])
        )
        model_file = "final_trained.zip"
        # Check if the model file exists
        if os.path.isfile(model_file):
            # Load the existing model
            model = TRPO.load(model_file)
            model.set_env(env)
        else:
            # Initialize a new model because the saved model does not exist
            model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T, batch_size=args.T,
                         verbose=1, device="cuda", tensorboard_log="./trpo_tensorboard/")
        model.learn(total_timesteps=args.total_step, tb_log_name="first_run", callback=custom_callback)
        model.save("./model_save/trpo_trained")
        actions, rewards, latency = custom_callback.get_training_data()
        np.save('latency'+str(i), latency)
        np.save('action'+str(i), actions)
        # The shape of action is [total_timesteps, 1, 32], reward: [total_timesteps, 1]
        actions = torch.from_numpy(actions).squeeze()
        actions_dis = actions[:, 0:16]
        actions_con = actions[:, 16:32]
        # For autoencoders, input data is also the target. So, both inputs and targets are X_tensor.
        dataset_dis = TensorDataset(actions_dis, actions_dis)
        dataloader_dis = DataLoader(dataset_dis, batch_size=args.T, shuffle=False)
        dataset_con = TensorDataset(actions_con, actions_con)
        dataloader_con = DataLoader(dataset_con, batch_size=args.T, shuffle=False)
        writer = SummaryWriter('./autoencoder_status')

        for epoch in range(100):
            for input, target in dataloader_dis:
                # Assume data is your input
                # Training Autoencoder con
                optimizer1.zero_grad()
                output1 = autoencoder_dis(input)
                loss1 = criterion1(output1, target)
                loss1.backward()
                optimizer1.step()
                writer.add_scalar('Loss/Autoencoder_DIS', loss1.item(), epoch * len(dataloader_dis) + i)
            for input, target in dataloader_con:
                # Training Autoencoder 2
                optimizer2.zero_grad()
                output2 = autoencoder_con(input)
                loss2 = criterion2(output2, target)
                loss2.backward()
                optimizer2.step()
                writer.add_scalar('Loss/Autoencoder_CON', loss2.item(), epoch * len(dataloader_con) + i)

        print('Finish Training autoencoder')
        torch.save(autoencoder_dis.state_dict(), './model_save/autoencoder_dis.pth')
        torch.save(autoencoder_con.state_dict(), './model_save/autoencoder_con.pth')
        writer.close()



