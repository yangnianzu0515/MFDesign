# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
import random
from typing import Any, Optional
from einops import rearrange
import torch
from torch import nn
from torch.nn import Module
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.nn.functional import one_hot
from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.data.mask.masker import Masker
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    default,
    log,
)
from boltz.model.modules.trunk import InputEmbedder

class SequenceD3PM(Module):
    def __init__(
        self, 
        hidden_dim, 
        vocab_size,
        dropout
    ):
        super().__init__()
        self.type_embed = nn.Embedding(4, hidden_dim, padding_idx=0) # 1: Heavy, 2: Light, 3: Ag
        self.region_embed = nn.Embedding(10, hidden_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )
                
          
    def forward(self, res_feat, cond = None):
        """Denoise the sequence feature.

        Args:
            res_feat: The sequence feature. 

            cond: The condition feature.

        Returns:
            res (batch_size, max_tokens, vocab_size): The denoised sequence one-hot code.
        """
        res = self.encoder(res_feat)
        type_embed = self.type_embed(cond["type"])
        region_embed = self.region_embed(cond["region"])
        res = torch.cat([res, type_embed, region_embed], dim=-1)
        res = self.dropout(self.LayerNorm(self.proj(res)))
        res = self.decoder(res)
        return res
    
class DiffusionModule(Module):
    """Diffusion module"""  

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        sequence_train: bool = False,
        sequence_model_args: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)
        
        self.start_restype = token_s
        self.end_restype = self.start_restype + const.num_tokens

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )
        
        self.sequence_train = sequence_train
        if sequence_train:
            if sequence_model_args is None:
                raise ValueError("sequence model args must be provided when training sequence model")
            self.sequence_model = SequenceD3PM(
                **sequence_model_args
            )
            
    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):

        if self.sequence_train:
            if len(feats["masked_seq"].shape) == 2:
                new_restype = one_hot(feats["masked_seq"], num_classes=const.num_tokens)
                new_restype = new_restype * feats["attn_mask"].unsqueeze(-1)
            else: # for continuous noise
                new_restype = feats["masked_seq"] * feats["attn_mask"].unsqueeze(-1)
            s_inputs = torch.cat([
                s_inputs[..., :self.start_restype],     
                new_restype,            
                s_inputs[..., self.end_restype:],
            ], dim=-1) # Update s_inputs

        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Denoise the sequence feature
        if self.sequence_train:
            cond = {}
            cond["type"] = feats["chain_type"].repeat_interleave(multiplicity, 0)
            cond["region"] = feats["region_type"].repeat_interleave(multiplicity, 0)
            res = self.sequence_model(a, cond)  
        else:
            res = None

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a, "seq": res}


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    """Atom diffusion module"""

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=200,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        noise_type="discrete_absorb",
        temperature=1.0,
        accumulate_token_repr=False,
        **kwargs,
    ):
        """Initialize the atom diffusion module.

        Parameters
        ----------
        score_model_args : dict
            The arguments for the score model.
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        compile_score : bool, optional
            Whether to compile the score model, by default False.
        alignment_reverse_diff : bool, optional
            Whether to use alignment reverse diff, by default False.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        use_inference_model_cache : bool, optional
            Whether to use the inference model cache, by default False.
        accumulate_token_repr : bool, optional
            Whether to accumulate the token representation, by default False.

        """
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        self.sequence_train = score_model_args["sequence_train"]
        self.structure_train = score_model_args["structure_train"]
        self.noise_type = noise_type
        self.temperature = temperature
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        self.accumulate_token_repr = accumulate_token_repr
        self.token_s = score_model_args["token_s"]
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        self.masker = Masker(noise_token_id=const.unk_token_ids["PROTEIN"], 
                             timesteps=num_sampling_steps,
                             noise_type=noise_type)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"], net_out["seq"]

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        inpaint=False,
        **network_condition_kwargs,
    ):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        resolved_atom_mask = network_condition_kwargs["feats"]["atom_resolved_mask"].repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        seq_timesteps = list(range(num_sampling_steps))[::-1]

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        seq_logits_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None
        atom_coords_gt = None

        if inpaint:
            coords_gt = network_condition_kwargs["feats"]["coords_gt"][0].repeat_interleave(multiplicity, 0)
            coords_mask = network_condition_kwargs["feats"]["coord_mask"].repeat_interleave(multiplicity, 0)
        else:
            coords_gt = None
            coords_mask = None

        network_condition_kwargs["s_inputs"] = network_condition_kwargs["s_inputs"].repeat_interleave(multiplicity, 0)
        network_condition_kwargs["feats"]["masked_seq"] = network_condition_kwargs["feats"]["masked_seq"].repeat_interleave(multiplicity, 0)
        if self.sequence_train:
            gt_vals = network_condition_kwargs["feats"]["masked_seq"]
            seq_masks = network_condition_kwargs["feats"]["seq_mask"]
            cdr_masks = network_condition_kwargs["feats"]["cdr_mask"]
            # Mask all the CDRs in input sequence
            if self.noise_type == "continuous":
                seqs = one_hot(gt_vals, num_classes=const.num_tokens)
                seqs = torch.where(cdr_masks.unsqueeze(-1).bool(), torch.zeros_like(seqs), seqs)
                network_condition_kwargs["feats"]["masked_seq"] = self.masker.corrupt(
                    seqs,
                    init_sigma, 
                    cdr_masks
                )
            else:
                assert self.masker.timesteps == num_sampling_steps # We don't allow to change inference timesteps if using discrete diffusion
                noise_t = torch.tensor([num_sampling_steps - 1] * gt_vals.shape[0], device=gt_vals.device)
                if self.noise_type == "discrete_absorb":
                    network_condition_kwargs["feats"]["masked_seq"] = self.masker.corrupt(
                        gt_vals, 
                        noise_t, 
                        cdr_masks
                    )[0]
                else:   
                    res = self.masker.corrupt(
                        gt_vals, 
                        noise_t, 
                        cdr_masks
                    )
                    network_condition_kwargs["feats"]["masked_seq"] = Categorical(probs=res).sample() + 2

        else:
            gt_vals = seq_masks = cdr_masks = None

        seq_noisy = network_condition_kwargs["feats"]["masked_seq"]

        # gradually denoise
        for i, ((sigma_tm, sigma_t, gamma), t) in enumerate(zip(sigmas_and_gammas, seq_timesteps)):
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps
            with torch.no_grad():
                atom_coords_denoised, token_a, seq_logits_denoised = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            if self.sequence_train:
                if t != seq_timesteps[-1]:
                    if self.noise_type == "continuous":
                        seqs_denoised = torch.zeros((seq_logits_denoised.shape[0], 
                                                     seq_logits_denoised.shape[1], 
                                                     const.num_tokens), 
                                                    device=seq_logits_denoised.device)
                        # 2~22 is protein token id in all tokens
                        seqs_denoised[..., 2:22] = torch.softmax(seq_logits_denoised * self.temperature, dim=-1)
                        sigma_now = sigma_t * (1 + sigmas_and_gammas[i+1][2])
                        seq_noisy = self.masker.corrupt(seqs_denoised, sigma_now, seq_masks)
                        noise_gt = self.masker.corrupt(gt_vals, sigma_now, cdr_masks)
                        seq_noisy = torch.where(seq_masks[...,None].bool(), seq_noisy, noise_gt)
                    elif self.noise_type == "discrete_absorb":
                        seqs_denoised = Categorical(logits=seq_logits_denoised * self.temperature).sample() + 2
                        noise_t = torch.tensor([t - 1] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        seq_noisy = self.masker.corrupt(seqs_denoised, noise_t, seq_masks)[0]
                        noise_gt = self.masker.corrupt(gt_vals, noise_t, cdr_masks)[0]
                        seq_noisy = torch.where(seq_masks.bool(), seq_noisy, noise_gt)
                    else: # discrete_uniform:
                        seqs_denoised = torch.zeros((seq_logits_denoised.shape[0], 
                                                     seq_logits_denoised.shape[1], 
                                                     const.num_tokens), 
                                                    device=seq_logits_denoised.device)
                        # 2~22 is protein token id in all tokens
                        seqs_denoised[..., 2:22] = torch.softmax(seq_logits_denoised * self.temperature, dim=-1)
                        noise_t = torch.tensor([t] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        # For D3PM-uniform we cannot directly use corrupt to get x_t-1. t -> t-1 posterior
                        seq_noisy = self.masker.uniform_posterior(seq_noisy, seqs_denoised, noise_t, seq_masks)
                        noise_t = torch.tensor([t - 1] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        noise_gt = self.masker.corrupt(gt_vals, noise_t, cdr_masks)
                        seq_noisy = torch.where(seq_masks[...,None].bool(), seq_noisy, noise_gt)
                        seq_noisy = Categorical(probs=seq_noisy).sample() + 2
                    
                    network_condition_kwargs["feats"]["masked_seq"] = seq_noisy

            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            if inpaint:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_gt = weighted_rigid_align(
                        coords_gt.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        mask=resolved_atom_mask.float(),
                    )
                
                atom_coords_gt = atom_coords_gt.to(atom_coords_denoised)
                atom_coords_denoised = torch.where(coords_mask[..., None].bool(), atom_coords_denoised, atom_coords_gt)

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        if self.sequence_train:
            seqs_denoised = Categorical(logits=seq_logits_denoised * self.temperature).sample() + 2
            seqs_denoised = torch.where(seq_masks.bool(), seqs_denoised, gt_vals)
        else:
            seqs_denoised = None
        
        if inpaint:
            with torch.autocast("cuda", enabled=False):
                atom_coords_gt = weighted_rigid_align(
                    coords_gt.float(),
                    atom_coords.float(),
                    atom_mask.float(),
                    mask=resolved_atom_mask.float(),
                )
                
            atom_coords = torch.where(coords_mask[..., None].bool(), atom_coords, atom_coords_gt)

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr, sample_seqs=seqs_denoised)

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        num_sampling_steps=None,
        multiplicity=1,
    ):
        
        if self.sequence_train and self.noise_type != "continuous":
            sigmas = self.sample_schedule(num_sampling_steps)
            gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
            sigmas = sigmas[:-1]
            gammas = gammas[1:]
            sigmas = sigmas * (1 + gammas)
            t = torch.randint(self.masker.timesteps, size=(feats["seq"].size(0),), device=feats["seq"].device)
            feats["time"] = t
            sigmas = sigmas[num_sampling_steps - 1 - t]
            padded_sigmas = rearrange(sigmas, "b -> b 1 1")
            res = self.masker.corrupt(
                feats["seq"], 
                t,
                feats["cdr_mask"]
            )
            if self.noise_type == "discrete_absorb":
                feats["masked_seq"], feats["seq_mask"] = res
            elif self.noise_type =="discrete_uniform":
                feats["masked_seq"] = Categorical(probs=res).sample() + 2
            
        else:
            batch_size = feats["coords"].shape[0]
            if self.synchronize_sigmas:
                sigmas = self.noise_distribution(batch_size).repeat_interleave(
                    multiplicity, 0
                )
            else:
                sigmas = self.noise_distribution(batch_size * multiplicity)
            padded_sigmas = rearrange(sigmas, "b -> b 1 1")
            if self.sequence_train:
                feats["masked_seq"] = self.masker.corrupt(
                    feats["seq"], 
                    padded_sigmas,
                    feats["cdr_mask"]
                )   

        atom_coords = feats["coords"]
        
        B, N, L = atom_coords.shape[0:3]   
        atom_coords = atom_coords.reshape(B * N, L, 3)
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)
        feats["coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )
        
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise     
                                    
        denoised_atom_coords, _, denoised_seqs = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs=dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
        )

        return dict(
            noised_atom_coords=noised_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            denoised_seqs=denoised_seqs,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]
        
        total_loss = 0
        loss_breakdown = {}

        if self.sequence_train:
            denoised_seqs = out_dict["denoised_seqs"]
            seqs_ground_truth = feats["seq"]
            seq_masks = feats["seq_mask"] if self.noise_type == "discrete_absorb" else feats["cdr_mask"]
            valid_mask = (seqs_ground_truth >= 2) & (seqs_ground_truth <= 21) & (seq_masks.bool())

            denoised_seqs_filtered = denoised_seqs[valid_mask]
            seqs_ground_truth_filtered = seqs_ground_truth[valid_mask] - 2

            loss_fct = nn.CrossEntropyLoss(reduction="mean")

            if denoised_seqs_filtered.numel() > 0:
                seq_loss = loss_fct(denoised_seqs_filtered, seqs_ground_truth_filtered)
            else:
                seq_loss = 0.0 * denoised_seqs.sum()  

            seq_acc = (denoised_seqs_filtered.argmax(dim=-1) == seqs_ground_truth_filtered).float().mean()
            
            total_loss += seq_loss
            loss_breakdown["seq_loss"] = seq_loss
            loss_breakdown["seq_acc"] = seq_acc

        if self.structure_train:
            resolved_atom_mask = feats["atom_resolved_mask"]
            resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = align_weights * (
                1
                + nucleotide_loss_weight
                * (
                    torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                )
                + ligand_loss_weight
                * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
            )

            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                atom_coords = out_dict["aligned_true_atom_coords"]
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach().float(),
                    denoised_atom_coords.detach().float(),
                    align_weights.detach().float(),
                    mask=resolved_atom_mask.detach().float(),
                )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
                dim=-1
            )
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss += mse_loss
            loss_breakdown["mse_loss"] = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=feats["atom_resolved_mask"],
                    multiplicity=multiplicity,
                )

                total_loss += lddt_loss
                loss_breakdown["smooth_lddt_loss"] = lddt_loss

        loss_breakdown["total_loss"] = total_loss
        print(loss_breakdown)
        return dict(loss=total_loss, loss_breakdown=loss_breakdown)
