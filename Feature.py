import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
from stable_baselines3.common.policies import ActorCriticPolicy
from arg_parser import get_args
import torch.nn.functional as F

args = get_args()


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 32):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)  # features_dim should match the output of the last layer

        self.output_dim = args.U + args.L + args.N
        self.edge_index = torch.load('edge_index.pt')

        # Graph Neural Network part
        self.gnn1 = GCNConv(2, 16)
        self.gnn2 = GCNConv(16, 1)

        # Feedforward Neural Network part
        self.fnn1 = nn.Sequential(
            nn.Linear(87, 128),  # Adjusted for observation_space
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fnn_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Assuming observations is a dict with keys 'x' and 'edge_index'
        x = observations

        batch_size = x.size(0)
        var_state = x[:, :-2 * self.output_dim]
        con_state = x[:, -2 * self.output_dim:].reshape(batch_size, self.output_dim, 2)
        con_output = self.gnn1(con_state, self.edge_index)
        con_output = F.relu(con_output)
        con_output = F.dropout(con_output, training=self.training)
        con_output = self.gnn2(con_output, self.edge_index).squeeze(2)
        var_input = torch.cat((var_state, con_output), dim=1)
        fnn_input = self.fnn1(var_input)
        output_final = self.fnn_out(fnn_input)

        return output_final


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Custom feature extractor
        U = kwargs['U']
        L = kwargs.get('L')
        N = kwargs.get('N')
        edge_index = kwargs.get('edge_index')
        net_arch = kwargs.get('net_arch')

        self.features_extractor = CustomFeaturesExtractor(observation_space, U, L, N, edge_index)

        # Network architectures for policy (pi) and value function (vf)
        pi_arch = net_arch['pi'] if net_arch is not None else []
        vf_arch = net_arch['vf'] if net_arch is not None else []

        # Create the actor (policy) network
        self.actor = self.create_mlp(
            input_dim=self.features_extractor.features_dim,
            output_dim=action_space.shape[0],
            net_arch=pi_arch,
            activation_fn=nn.ReLU
        )

        # Create the critic (value function) network
        self.critic = self.create_mlp(
            input_dim=self.features_extractor.features_dim,
            output_dim=1,  # Value function outputs a single scalar
            net_arch=vf_arch,
            activation_fn=nn.ReLU
        )

    def _build_mlp_extractor(self):
        # Override this method to avoid creating the default MLP extractor
        pass


