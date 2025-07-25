import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, widths, act_type='relu'):
        super().__init__()
        layers = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                if act_type == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif act_type == 'elu':
                    layers.append(nn.ELU())
                elif act_type == 'relu':
                    layers.append(nn.ReLU())
                elif act_type == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class KoopmanNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        widths = params['widths']
        depth = (len(widths) - 4) // 2
        encoder_widths = widths[:depth + 2]
        decoder_widths = widths[depth + 2:]
        act_type = params.get('act_type', 'relu')
        self.encoder = MLP(encoder_widths, act_type)
        self.decoder = MLP(decoder_widths, act_type)
        self.num_complex_pairs = params['num_complex_pairs']
        self.num_real = params['num_real']
        self.delta_t = params['delta_t']
        hidden_omega = params['hidden_widths_omega']
        self.omega_nets = nn.ModuleList()
        for _ in range(self.num_complex_pairs):
            widths_omega = [1] + hidden_omega + [2]
            self.omega_nets.append(MLP(widths_omega, act_type))
        for _ in range(self.num_real):
            widths_omega = [1] + hidden_omega + [1]
            self.omega_nets.append(MLP(widths_omega, act_type))
        self.shifts = [int(s) for s in params['shifts']]
        self.shifts_middle = [int(s) for s in params['shifts_middle']]

    def forward(self, x):
        batch, _, _ = x.shape
        g_list = []
        for shift in [0] + self.shifts_middle:
            g_list.append(self.encoder(x[:, shift, :]))
        y_preds = [self.decoder(g_list[0])]
        y_latent = g_list[0]
        max_shift = max(self.shifts) if self.shifts else 0
        for step in range(1, max_shift + 1):
            omegas = self.get_omegas(y_latent)
            y_latent = self._advance(y_latent, omegas)
            if step in self.shifts:
                y_preds.append(self.decoder(y_latent))
        return y_preds, g_list

    def get_omegas(self, y_latent):
        omegas = []
        idx = 0
        for i in range(self.num_complex_pairs):
            pair = y_latent[:, 2*i:2*i+2]
            radius = torch.sum(pair * pair, dim=1, keepdim=True)
            omegas.append(self.omega_nets[idx](radius))
            idx += 1
        for i in range(self.num_real):
            val = y_latent[:, 2*self.num_complex_pairs + i].unsqueeze(1)
            omegas.append(self.omega_nets[idx](val))
            idx += 1
        return omegas

    def _advance(self, y, omegas):
        complex_parts = []
        for j in range(self.num_complex_pairs):
            omega = omegas[j]
            scale = torch.exp(torch.clamp(omega[:, 1], -5.0, 5.0) * self.delta_t)
            entry11 = scale * torch.cos(omega[:, 0] * self.delta_t)
            entry12 = scale * torch.sin(omega[:, 0] * self.delta_t)
            y_pair = y[:, 2*j:2*j+2]
            mat = torch.stack([entry11, entry12, -entry12, entry11], dim=1).view(-1, 2, 2)
            complex_parts.append(torch.bmm(y_pair.unsqueeze(1), mat).squeeze(1))
        real_parts = []
        for j in range(self.num_real):
            omega = omegas[self.num_complex_pairs + j]
            scale = torch.exp(torch.clamp(omega[:, 0], -5.0, 5.0) * self.delta_t).unsqueeze(1)
            val = y[:, 2*self.num_complex_pairs + j].unsqueeze(1)
            real_parts.append(scale * val)
        return torch.cat(complex_parts + real_parts, dim=1)

class KoopmanNetControl_v2(KoopmanNet):
    def __init__(self, params):
        super().__init__(params)
        act_type = params.get('act_type', 'relu')
        control_dim = params.get('control_dim', 1)
        hidden_control = params.get('hidden_widths_control', [64, 64])
        latent_dim = 2 * self.num_complex_pairs + self.num_real
        #self.control_net = MLP(
        #    [latent_dim + params['control_dim']] + params.get('hidden_widths_control', [64, 64]) + [latent_dim],
        #    act_type
        #)
        self.control_net = MLP(
            [params['control_dim']] + params.get('hidden_widths_control', [64, 64]) + [latent_dim],
            act_type
        )
        self.control_gain = params.get('control_gain', 0.01)  # Safe initial gain to stabilize early training

    def forward(self, x, u):
        batch, T, _ = x.shape
        g_list = []
        for shift in [0] + self.shifts_middle:
            g_list.append(self.encoder(x[:, shift, :]))
        y_preds = [self.decoder(g_list[0])]
        y_latent = g_list[0]
        max_shift = max(self.shifts) if self.shifts else 0
        for step in range(1, max_shift + 1):
            omegas = self.get_omegas(y_latent)
            u_step = u[:, step - 1, :]  # u aligned with y_{t+1}
            y_latent = self._advance(y_latent, omegas, u_step)
            if step in self.shifts:
                y_preds.append(self.decoder(y_latent))
        return y_preds, g_list

    def _advance(self, y, omegas, u_t):
        complex_parts = []
        for j in range(self.num_complex_pairs):
            omega = omegas[j]
            scale = torch.exp(torch.clamp(omega[:, 1], -5.0, 5.0) * self.delta_t)
            entry11 = scale * torch.cos(omega[:, 0] * self.delta_t)
            entry12 = scale * torch.sin(omega[:, 0] * self.delta_t)
            y_pair = y[:, 2 * j:2 * j + 2]
            mat = torch.stack([entry11, entry12, -entry12, entry11], dim=1).view(-1, 2, 2)
            complex_parts.append(torch.bmm(y_pair.unsqueeze(1), mat).squeeze(1))
        
        real_parts = []
        for j in range(self.num_real):
            omega = omegas[self.num_complex_pairs + j]
            scale = torch.exp(torch.clamp(omega[:, 0], -5.0, 5.0) * self.delta_t).unsqueeze(1)
            val = y[:, 2 * self.num_complex_pairs + j].unsqueeze(1)
            real_parts.append(scale * val)

        y_next = torch.cat(complex_parts + real_parts, dim=1)

        control_input = u_t  # Instead of torch.cat([y, u_t], dim=1)
        B_u = self.control_gain * self.control_net(control_input)

        return y_next + B_u

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class KoopmanNetControl_RNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.input_dim = params['input_dim']
        self.num_complex_pairs = params['num_complex_pairs']
        self.num_real = params['num_real']
        self.latent_dim = 2 * self.num_complex_pairs + self.num_real
        self.delta_t = params['delta_t']
        self.shifts = [int(s) for s in params.get('shifts', [1, 2, 3])]
        self.shifts_middle = [int(s) for s in params.get('shifts_middle', [])]

        act_type = params.get('act_type', 'relu')

        # MLP → LSTM encoder
        self.encoder_funnel = MLP([self.input_dim] + params['encoder_funnel_widths'], act_type)
        proj_dim = params['encoder_funnel_widths'][-1]
        self.encoder_lstm = nn.LSTM(proj_dim, self.latent_dim, batch_first=True)

        # Decoder MLP
        self.decoder = MLP([self.latent_dim] + params['decoder_widths'] + [self.input_dim], act_type)

        # Omega networks
        hidden_omega = params['hidden_widths_omega']
        self.omega_nets = nn.ModuleList()
        for _ in range(self.num_complex_pairs):
            self.omega_nets.append(MLP([1] + hidden_omega + [2], act_type))
        for _ in range(self.num_real):
            self.omega_nets.append(MLP([1] + hidden_omega + [1], act_type))

        # Control network B(u)
        self.control_gain = params.get('control_gain', 0.01)
        control_dim = params.get('control_dim', 1)
        hidden_control = params.get('hidden_widths_control', [64, 64])
        self.control_net = MLP(
            [control_dim] + hidden_control + [self.latent_dim * self.latent_dim],
            act_type
        )

    def encoder(self, x_seq):
        x_proj = self.encoder_funnel(x_seq)  # (batch, time, proj_dim)
        _, (h_n, _) = self.encoder_lstm(x_proj)
        return h_n.squeeze(0)

    def get_omegas(self, y_latent):
        omegas = []
        idx = 0
        for i in range(self.num_complex_pairs):
            pair = y_latent[:, 2*i:2*i+2]
            radius = torch.sum(pair * pair, dim=1, keepdim=True)
            omegas.append(self.omega_nets[idx](radius))
            idx += 1
        for i in range(self.num_real):
            val = y_latent[:, 2*self.num_complex_pairs + i].unsqueeze(1)
            omegas.append(self.omega_nets[idx](val))
            idx += 1
        return omegas

    def forward(self, x, u):
        batch, T, _ = x.shape
        g_list = []

        # Encode inputs at shifts
        for shift in [0] + self.shifts_middle:
            g_list.append(self.encoder(x[:, :shift + 1, :]))  # shape (batch, latent_dim)

        y_preds = [self.decoder(g_list[0])]
        y_latent = g_list[0]
        max_shift = max(self.shifts) if self.shifts else 0

        for step in range(1, max_shift + 1):
            omegas = self.get_omegas(y_latent)
            u_step = u[:, step - 1, :]  # (batch, control_dim)
            y_latent = self._advance(y_latent, omegas, u_step)

            if step in self.shifts:
                g_list.append(y_latent)
                y_preds.append(self.decoder(y_latent))

        return y_preds, g_list

    def _advance(self, y, omegas, u_t):
        complex_parts = []
        for j in range(self.num_complex_pairs):
            omega = omegas[j]
            scale = torch.exp(torch.clamp(omega[:, 1], -5.0, 5.0) * self.delta_t)
            entry11 = scale * torch.cos(omega[:, 0] * self.delta_t)
            entry12 = scale * torch.sin(omega[:, 0] * self.delta_t)
            y_pair = y[:, 2*j:2*j+2]
            mat = torch.stack([entry11, entry12, -entry12, entry11], dim=1).view(-1, 2, 2)
            complex_parts.append(torch.bmm(y_pair.unsqueeze(1), mat).squeeze(1))

        real_parts = []
        for j in range(self.num_real):
            omega = omegas[self.num_complex_pairs + j]
            scale = torch.exp(torch.clamp(omega[:, 0], -5.0, 5.0) * self.delta_t).unsqueeze(1)
            val = y[:, 2*self.num_complex_pairs + j].unsqueeze(1)
            real_parts.append(scale * val)

        y_next = torch.cat(complex_parts + real_parts, dim=1)

        # Control influence
        B_flat = self.control_net(u_t)  # (batch, latent_dim * latent_dim)
        B = B_flat.view(-1, self.latent_dim, self.latent_dim)
        Bu = torch.bmm(B, y.unsqueeze(-1)).squeeze(-1)

        return y_next + self.control_gain * Bu
