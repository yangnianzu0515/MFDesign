import copy
import numpy as np
import torch
import boltz.data.const as const
from torch.nn.functional import one_hot
class Masker:
    """Token masker"""

    def __init__(
        self, 
        noise_token_id,
        timesteps=200, 
        noise_schedule="linear",
        noise_type="discrete_absorb"
    ):
        self.timesteps = timesteps
        self.noise_type = noise_type
        if noise_type == "discrete_absorb":
            self.noise_token_id = noise_token_id
            self.mask_rates = torch.linspace(
                0, 1, timesteps
            )
        elif noise_type == "discrete_uniform":
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
            f_t = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
            self.alpha_bar = f_t / f_t[0]
            uniform_protein_dist = torch.zeros(const.num_tokens)
            uniform_protein_dist[2:22] = 1.0 / 20.0
            self.uniform_protein = uniform_protein_dist

    def convert_noise_level(c_skip):
        if self.noise_type == "discrete_absorb":
            return 1 - c_skip
        elif self.noise_type == "discrete_uniform":
            return torch.minimum(torch.tensor(1.0), (1 - c_skip) * 20.0 / 19.0)

    def corrupt(self, seq, noise, seq_mask=None):
        if self.noise_type == "discrete_absorb":
            return self.absorb_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "discrete_uniform":
            return self.uniform_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "continuous":
            return self.continuous_corrupt(seq, noise, seq_mask)
        else:
            raise ValueError(f"No implementations for {self.noise_type} noise type")

    def absorb_corrupt(self, seq, noise_level, seq_mask=None):
        device = seq.device
        self.mask_rates = self.mask_rates.to(device)
        
        batch_mask_rates = self.mask_rates[noise_level].unsqueeze(-1).to(device)
        mask = torch.rand(seq.shape, device=device) < batch_mask_rates
        
        if seq_mask is not None:
            mask = mask & seq_mask.to(torch.bool)

        res = torch.where(mask, self.noise_token_id, seq)
        return res, mask
    
    def uniform_corrupt(self, seq, timesteps, seq_mask=None):
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)
        
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            seqs = seq.float()
        alpha_bar = self.alpha_bar[timesteps].view(seq.size(0), 1, 1)
        res = alpha_bar * seqs + (1.0 - alpha_bar) * self.uniform_protein
        if seq_mask is not None:
            res = torch.where(seq_mask.unsqueeze(-1), res, seqs)
        return res
    
    def uniform_posterior(self, seq_t, seq, timesteps, seq_mask=None):
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)
        if len(seq_t.shape) == 2:
            x_t = one_hot(seq_t, num_classes=const.num_tokens).float()
        else:
            x_t = seq_t.float()
        if len(seq.shape) == 2:
            x_0 = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            x_0 = seq.float()
            
        alpha = self.alpha_bar[timesteps] / (self.alpha_bar[timesteps - 1] + 1e-8)
        alpha_bar = self.alpha_bar[timesteps - 1]
        
        alpha = alpha.view(seq.size(0), 1, 1)
        alpha_bar = alpha_bar.view(seq.size(0), 1, 1)
        
        q_x_t_from_x_t_minus_1 = alpha * x_t + (1.0 - alpha) * self.uniform_protein
        q_x_t_minus_1_from_x_0 = alpha_bar * x_0 + (1.0 - alpha_bar) * self.uniform_protein
        res = q_x_t_from_x_t_minus_1 * q_x_t_minus_1_from_x_0
        res = res / (res.sum(dim=-1, keepdim=True) + 1e-8)
        if seq_mask is not None:
            res = torch.where(seq_mask.unsqueeze(-1), res, x_t)
        return res    

    def continuous_corrupt(self, seq, sigmas, seq_mask, omega=0.25, clamp_v=3.0):
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            seqs = seq.float()
        seq_noise = torch.randn_like(seqs) * seq_mask.unsqueeze(-1)
        res = seqs + omega * sigmas * seq_noise
        res = torch.clamp(res, min=-clamp_v, max=clamp_v)
        return res