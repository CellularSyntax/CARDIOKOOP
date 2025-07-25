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
    
class KoopmanNetControl(KoopmanNet):
    def __init__(self, params):
        super().__init__(params)
        act_type = params.get('act_type', 'relu')
        control_dim = params.get('control_dim', 1)
        hidden_control = params.get('hidden_widths_control', [64, 64])
        latent_dim = 2 * self.num_complex_pairs + self.num_real
        self.control_net = MLP([control_dim] + hidden_control + [latent_dim], act_type)
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

        # Optional: Clamp to avoid instability from omega explosions
        #y_next = torch.clamp(y_next, -10.0, 10.0)

        # Control influence with scaled, bounded effect
        B_u = self.control_gain * torch.tanh(self.control_net(u_t))

        # Optional debug logging
        # if torch.isnan(B_u).any() or torch.isinf(B_u).any():
        #     print("[ERROR] B_u contains NaNs or Infs")
        # if torch.isnan(y_next).any() or torch.isinf(y_next).any():
        #     print("[ERROR] y_next contains NaNs or Infs")
        # print(f"[DEBUG] B_u stats: min={B_u.min().item():.4f}, max={B_u.max().item():.4f}, mean={B_u.mean().item():.4f}")
        # print(f"[DEBUG] y_next stats before control: min={y_next.min().item():.4f}, max={y_next.max().item():.4f}")

        return y_next + B_u


class KoopmanNetControl_v2(KoopmanNet):
    def __init__(self, params):
        super().__init__(params)
        act_type = params.get('act_type', 'relu')
        control_dim = params.get('control_dim', 1)
        hidden_control = params.get('hidden_widths_control', [64, 64])
        latent_dim = 2 * self.num_complex_pairs + self.num_real
        self.control_net = MLP(
            [latent_dim + control_dim] + hidden_control + [latent_dim],
            act_type
        )
        self.control_gain = params.get('control_gain', 0.01)

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

    def _advance_old(self, y, omegas, u_t):
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

        control_input = torch.cat([y, u_t], dim=1)
        # support legacy checkpoints where control_net only took u as input
        first_layer = self.control_net.net[0]
        in0 = getattr(first_layer, 'in_features', None)
        if in0 == control_input.shape[1]:
            u_in = control_input
        elif in0 == u_t.shape[1]:
            u_in = u_t
        else:
            raise RuntimeError(
                f"Control net first-layer expects {in0} inputs, but got control_input.size(1)="
                f"{control_input.shape[1]} or u_t.size(1)={u_t.shape[1]}"
            )
        B_u = self.control_gain * self.control_net(u_in)

        return y_next + B_u
    
    def _advance(self, y, omegas, u_t):
        batch = y.size(0)
        num_cp = self.num_complex_pairs
        num_real = self.num_real
        delta_t = self.delta_t

        y_complex = y[:, :2*num_cp].view(batch, num_cp, 2)
        omegas_complex = torch.stack(omegas[:num_cp], dim=1)

        scale = torch.exp(torch.clamp(omegas_complex[:, :, 1], -5.0, 5.0) * delta_t)
        angle = omegas_complex[:, :, 0] * delta_t
        cos_a = scale * torch.cos(angle)
        sin_a = scale * torch.sin(angle)

        rot_mat = torch.stack([
            torch.stack([cos_a, sin_a], dim=-1),
            torch.stack([-sin_a, cos_a], dim=-1)
        ], dim=-2)

        y_complex_rot = torch.matmul(y_complex.unsqueeze(-2), rot_mat).squeeze(-2).reshape(batch, 2*num_cp)

        y_real = y[:, 2*num_cp:]
        omegas_real = torch.stack(omegas[num_cp:], dim=1).squeeze(-1)
        scale_real = torch.exp(torch.clamp(omegas_real, -5.0, 5.0) * delta_t)
        y_real_scaled = y_real * scale_real

        B_u = self.control_gain * torch.tanh(self.control_net(u_t))
        return torch.cat([y_complex_rot, y_real_scaled], dim=1) + B_u
