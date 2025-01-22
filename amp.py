from torch import nn
from torch.distributions import Normal
import torch
import torch.nn.functional as F


class M1EncoderDis(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(M1EncoderDis, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, 128, batch_first=True)
        self.fnn_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, latent_action, user_lists):

        x = latent_action
        batch_size = x.size(0)
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.view(batch_size, -1)
        x = x[:, -128:]
        x = self.fnn_net(x)
        x_out = []
        index = 0
        for u in range(len(user_lists)):
            x1 = F.softmax(x[:, index:index + len(user_lists[u])], dim=1)
            index += len(user_lists[u])
            x_out.append(x1)
        x_out = torch.cat(x_out, dim=1)
        index = 0
        all_action = []
        for u in range(len(user_lists)):
            max_idx = torch.argmax(x_out[:, index:index + len(user_lists[u])], dim=1)
            repeated_list = torch.tile(torch.tensor(user_lists[u]), (x_out.size(0), 1))
            final_action = repeated_list[torch.arange(x_out.size(0)), max_idx]
            index += len(user_lists[u])
            all_action.append(final_action)
        all_action = torch.stack(all_action).t()
        return all_action


class M1DecoderDis(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(M1DecoderDis, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fnn_net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.lstm_out = nn.LSTM(16, self.output_dim, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.float32)
        x = self.fnn_net(x).unsqueeze(1) #(batch,1,16)
        x, _ = self.lstm_out(x)
        x = x.view(batch_size, -1)
        return x


class AutoencoderDis(nn.Module):
    def __init__(self, input_dim, latent_dim, U, user_lists, Encoder, Decoder):
        super(AutoencoderDis, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(U, input_dim)
        self.user_lists = user_lists

    def forward(self, x):
        # Encode input data
        encoded = self.encoder(x, self.user_lists)

        # Decode the encoded representation
        decoded_x = self.decoder(encoded)

        return decoded_x


class M1EncoderCon(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(M1EncoderCon, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, 128, batch_first=True)
        self.fnn_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, latent_action):
        x = latent_action
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.fnn_net(x)
        x_out = []
        for i in range(x.size(0)):
            x_one = x[i, :].reshape(-1, 2)
            softmax_results = F.softmax(x_one[:, 1], dim=0)
            other = x_one[:, 0]

            # Concatenate the three parts to form the new x_one without in-place modification
            x_one = torch.cat([other.unsqueeze(1), softmax_results.unsqueeze(1)], dim=1)

            x_out.append(x_one)
        x_out = torch.stack(x_out).squeeze()
        return x_out


class M1DecoderCon(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(M1DecoderCon, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fnn_net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(16, self.output_dim, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.fnn_net(x).unsqueeze(1) #(batch,1,16)
        x, _ = self.lstm(x) #
        x_out = x.view(batch_size, -1)
        return x_out


class AutoencoderCon(nn.Module):
    def __init__(self, input_dim, latent_dim, Encoder, Decoder):
        super(AutoencoderCon, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        # Encode input data
        encoded = self.encoder(x)

        # Decode the encoded representation
        decoded_x = self.decoder(encoded)

        return decoded_x
