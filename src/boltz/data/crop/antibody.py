from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from boltz.data import const
from boltz.data.crop.cropper import Cropper
from boltz.data.types import Tokenized

def pick_chain_token(
    tokens: np.ndarray,
    token_mask: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    cdr_tokens = tokens[(tokens["asym_id"] == chain_id) & token_mask]
    return cdr_tokens[random.randint(len(cdr_tokens))]

class AntibodyCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(self, add_antigen: Optional[bool] = False, 
                 min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        self.add_antigen = add_antigen
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        token_mask: np.ndarray,
        token_region: np.ndarray,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        h_chain_id: Optional[int] = None,
        l_chain_id: Optional[int] = None,
    ):
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]
        valid_masks = token_mask[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # Select cropped indices
        cropped: set[int] = set()
        total_atoms = 0
        
        # Get the antibody tokens
        if h_chain_id is not None:
            h_chain_tokens = token_data[(token_data["asym_id"] == h_chain_id) & 
                                        token_data["resolved_mask"]]
            cropped.update(h_chain_tokens["token_idx"])
            total_atoms += np.sum(h_chain_tokens["atom_num"])
        if l_chain_id is not None:
            l_chain_tokens = token_data[(token_data["asym_id"] == l_chain_id) & 
                                        token_data["resolved_mask"]]
            cropped.update(l_chain_tokens["token_idx"])
            total_atoms += np.sum(h_chain_tokens["atom_num"])
        if self.add_antigen:
            epitope_tokens = token_data[(token_data["asym_id"] != h_chain_id) & 
                                        (token_data["asym_id"] != l_chain_id) &
                                        token_mask & token_data["resolved_mask"]]
            # Similar to Boltz Cropper
            neighborhood_size = random.choice(self.neighborhood_sizes)
            if chain_id is None:
                chain_id = random.choice([x for x in [h_chain_id, l_chain_id] if x is not None])
            query = pick_chain_token(valid_tokens, valid_masks, chain_id, random)
            dists = epitope_tokens["center_coords"] - query["center_coords"]
            indices = np.argsort(np.linalg.norm(dists, axis=1))

            for idx in indices:
                token = epitope_tokens[idx]
                chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]
                if len(chain_tokens) <= neighborhood_size:
                    new_tokens = chain_tokens
                else:
                    min_idx = token["res_idx"] - neighborhood_size
                    max_idx = token["res_idx"] + neighborhood_size

                    max_token_set = chain_tokens
                    max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                    max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                    new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]
                    min_idx = max_idx = token["res_idx"]
                    while new_tokens.size < neighborhood_size:
                        min_idx = min_idx - 1
                        max_idx = max_idx + 1
                        new_tokens = max_token_set
                        new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                        new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]
                
                new_indices = set(new_tokens["token_idx"]) - cropped
                new_tokens = token_data[list(new_indices)]
                new_atoms = np.sum(new_tokens["atom_num"])

                if (len(new_indices) > (max_tokens - len(cropped))) or (
                    (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)
                ):
                    break

                cropped.update(new_indices)
                total_atoms += new_atoms

        if max_tokens is not None and len(cropped) > max_tokens:
            raise ValueError("Cropped tokens exceed maximum")
        if max_atoms is not None and total_atoms > max_atoms:
            raise ValueError("Cropped atoms exceed maximum")

        # Get the cropped tokens sorted by index
        token_data = token_data[sorted(cropped)]
        token_mask = token_mask[sorted(cropped)]
        token_region = token_region[sorted(cropped)]

        # Only keep bonds within the cropped tokens
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        # Return the cropped tokens
        return replace(data, tokens=token_data, bonds=token_bonds), token_mask, token_region
