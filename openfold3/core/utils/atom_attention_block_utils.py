# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


def get_subset_center_padding(
    n_atom: int, n_query: int, n_key: int
) -> tuple[int, int, int]:
    """
    Calculate padding for a structure with n_atoms such that the block centers
    match the subset centers in Alg. 7 and the q/k dimensions are divisible by
    n_query and n_key respectively.

    Args:
        n_atom:
            Number of atoms
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)

    Returns:
        pad_len_right_q:
            Padding for the query seqlen dim so that it is divisible by n_query.
            No left padding is needed since the first block center is at
            n_query // 2.
        pad_len_left_k:
            Left padding for the key seqlen dim. Because the subset centers start
            at n_query // 2, padding is needed for even block sizes of length n_key.
            This is an issue for the first two blocks, which would have lengths 80
            and 112 if the default block sizes are used.
        pad_len_right_k:
            Right padding for the key seqlen dim. Because the subset centers are
            shifted by n_query, padding is needed for even block sizes of
            length n_key. This addresses uneven block sizes in the ending blocks.
    """
    offset = n_query // 2
    num_blocks = math.ceil(n_atom / n_query)

    subset_centers = offset + torch.arange(num_blocks) * n_query

    # Calculate padding for rows of plm to be divisible by n_query
    pad_len_right_q = (n_query - n_atom % n_query) % n_query

    # Calculate padding for columns of plm to be divisible by n_key
    # Pad left and right to ensure that the block centers match the
    # subset_centers in Alg. 7
    pad_len_right_k = subset_centers[-1] + n_key // 2 - n_atom
    pad_len_left_k = n_key // 2 - subset_centers[0]

    return pad_len_right_q, pad_len_left_k, pad_len_right_k


def get_block_indices(n_atom: int, n_query: int, n_key: int, device: torch.device):
    """
    Calculate padding for a structure with n_atoms such that the block centers
    match the subset centers in Alg. 7 and the q/k dimensions are divisible by
    n_query and n_key respectively.

    Args:
        n_atom:
            Number of atoms
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        device:
            Device to create the tensors on
    """
    offset = n_query // 2
    num_blocks = math.ceil(n_atom / n_query)

    subset_centers = offset + torch.arange(num_blocks, device=device) * n_query

    initial_gathers = (
        subset_centers[:, None]
        + torch.arange(-n_key // 2, n_key // 2, device=device)[None, :]
    )

    initial_gathers = initial_gathers.int()

    if n_key <= n_atom:
        # For normal cases, shift windows to be fully in-bounds.
        # For each row, calculate how much its start index is below 0.
        underflow = torch.relu(-initial_gathers[:, 0])

        # For each row, calculate how much its end index is above the max valid index.
        overflow = torch.relu(initial_gathers[:, -1] - (n_atom - 1))

        # The total shift required for each row is the underflow
        # (shifting right, positive) minus the overflow (shifting left, negative).
        total_shift = underflow - overflow

        # Apply the calculated shift to each row.
        # We add `[:, None]` to the shift tensor to broadcast the per-row shift
        # value across all columns of that row in `initial_gathers`.
        final_gathers = initial_gathers + total_shift[:, None]
    else:
        # If n_key > n_atom, apply a more nuanced shift.
        # Always correct underflow by shifting windows to start at 0.
        shift_right = torch.relu(-initial_gathers[:, 0])
        gathers_no_underflow = initial_gathers + shift_right[:, None]

        # Conditionally correct overflow.
        # Calculate potential overflow shift for the already right-shifted gathers.
        overflow = torch.relu(gathers_no_underflow[:, -1] - (n_atom - 1))
        shift_left = overflow

        # Determine if applying this left shift would create a new underflow.
        # A window should not be shifted left if doing so makes its start
        # index negative.
        would_become_negative = (gathers_no_underflow[:, 0] - shift_left) < 0

        # Only apply the left shift to rows where it wouldn't create a negative start.
        final_shift_left = torch.where(would_become_negative, 0, shift_left)

        # Apply the final conditional left shift.
        final_gathers = gathers_no_underflow - final_shift_left[:, None]

    # Create a boolean mask to identify all indices that are invalid.
    invalid_mask = (final_gathers < 0) | (final_gathers >= n_atom)

    # Create "safe" indices by clamping the generated ones to the valid range.
    # This prevents torch.gather from throwing an error.
    safe_indices = torch.clamp(final_gathers, 0, n_atom - 1)

    return safe_indices.flatten(), invalid_mask.flatten()


def convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Convert single atom representation to q/k blocks for attention.
    Optionally convert the atom mask to a 2D mask to account for the padding on the
    first and last blocks.

    Args:
        ql:
            [*, N_atom, c_atom] Atom single representation
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        atom_mask:
            [*, N_atom] Mask for token or atom-level embedding (Optional)

    Returns:
        ql_query:
            [*, N_blocks, N_query, c_atom] Atom single representation
        ql_key:
            [*, N_blocks, N_key, c_atom] Atom single representation
        mask:
            [*, N_blocks, N_query, N_key] 2D mask for atom-level embedding
    """
    batch_dims = ql.shape[:-2]
    n_atom, n_dim = ql.shape[-2:]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q, _, _ = get_subset_center_padding(
        n_atom=n_atom, n_query=n_query, n_key=n_key
    )

    # Pad and convert ql to blocks of width n_query
    # [*, N_atom, c_atom] -> [*, N_blocks, N_query, c_atom]
    ql_query = torch.nn.functional.pad(ql, (0, 0, 0, pad_len_right_q), value=0.0)
    ql_query = ql_query.reshape((*batch_dims, num_blocks, n_query, n_dim))

    key_block_idxs, invalid_mask = get_block_indices(
        n_atom=n_atom, n_query=n_query, n_key=n_key, device=ql.device
    )

    key_block_idxs = key_block_idxs.reshape(
        *((1,) * len(batch_dims) + key_block_idxs.shape)
    )
    invalid_mask = invalid_mask.reshape(*((1,) * len(batch_dims) + invalid_mask.shape))

    ql_key = torch.gather(
        ql,
        dim=-2,
        index=key_block_idxs[..., None].expand((*batch_dims, -1, ql.shape[-1])).long(),
    )

    ql_key.masked_fill_(
        invalid_mask[..., None].expand((*batch_dims, -1, ql.shape[-1])), 0
    )

    ql_key = ql_key.reshape((*batch_dims, num_blocks, n_key, ql.shape[-1]))

    atom_pair_mask = None
    if atom_mask is not None:
        # Pad and convert atom mask to blocks of width n_query
        # [*, N_atom] -> [*, N_blocks, N_query]
        atom_mask_q = torch.nn.functional.pad(
            atom_mask, (0, pad_len_right_q), value=0.0
        )
        atom_mask_q = atom_mask_q.reshape((*atom_mask.shape[:-1], num_blocks, n_query))

        atom_mask_k = torch.gather(
            atom_mask,
            dim=-1,
            index=key_block_idxs.expand((*atom_mask.shape[:-1], -1)).long(),
        )

        atom_mask_k.masked_fill_(invalid_mask.expand((*atom_mask.shape[:-1], -1)), 0)

        atom_mask_k = atom_mask_k.reshape((*atom_mask.shape[:-1], num_blocks, n_key))

        # Create pair mask
        # [*, N_blocks, N_query, N_key]
        atom_pair_mask = atom_mask_q[..., None] * atom_mask_k[..., None, :]

    return ql_query, ql_key, atom_pair_mask


def convert_trunk_pair_rep_to_blocks(
    batch: dict,
    zij_trunk: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair atom representation to blocks for attention.

    Args:
        batch:
            Feature dictionary
        zij_trunk:
            [*, N_token, N_token, c_atom_pair] Pair trunk embedding
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)

    Returns:
        plm:
            [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
    """
    # Get atom_to_token_index to map each token to the corresponding
    # number of atoms for broadcasting
    atom_to_token_index = batch["atom_to_token_index"]

    batch_dims = zij_trunk.shape[:-3]
    n_atom = atom_to_token_index.shape[-1]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q, _, _ = get_subset_center_padding(
        n_atom=n_atom, n_query=n_query, n_key=n_key
    )

    # Pad and convert atom_to_token_index to blocks of width n_query
    atom_to_token_index_q = torch.nn.functional.pad(
        atom_to_token_index, (0, pad_len_right_q), value=0.0
    )

    # [*, N_atom] -> [*, N_blocks, N_query]
    atom_to_token_index_q = atom_to_token_index_q.reshape(
        (*batch_dims, num_blocks, n_query)
    )

    # Expand zij to the number of blocks needed for indexing without allocating mem
    # [*, N_blocks, N_token, N_token, c_atom_pair]
    zij_trunk = zij_trunk.unsqueeze(-4).expand(
        (*batch_dims, num_blocks, *zij_trunk.shape[-3:])
    )

    # Aggregate blocked atom query dimension from tokens
    # [*, N_blocks, N_query, N_token, c_atom_pair]
    zij_trunk = torch.gather(
        zij_trunk,
        dim=-3,
        index=atom_to_token_index_q[..., None, None]
        .expand((*batch_dims, num_blocks, n_query, *zij_trunk.shape[-2:]))
        .long(),
    )

    key_block_idxs, invalid_mask = get_block_indices(
        n_atom=n_atom, n_query=n_query, n_key=n_key, device=zij_trunk.device
    )

    key_block_idxs = key_block_idxs.reshape(
        *((1,) * len(batch_dims) + key_block_idxs.shape)
    )
    invalid_mask = invalid_mask.reshape(*((1,) * len(batch_dims) + (num_blocks, n_key)))

    # [*, N_atom] -> [*, N_blocks, N_key]
    atom_to_token_index_k = torch.gather(
        atom_to_token_index,
        dim=-1,
        index=key_block_idxs.expand((*atom_to_token_index.shape[:-1], -1)).long(),
    ).reshape((*batch_dims, num_blocks, n_key))

    # Aggregate blocked atom key dimension from tokens
    # [*, N_blocks, N_query, N_key, c_atom_pair]
    zij_trunk = torch.gather(
        zij_trunk,
        dim=-2,
        index=atom_to_token_index_k[..., None, :, None]
        .expand((*batch_dims, num_blocks, n_query, n_key, zij_trunk.shape[-1]))
        .long(),
    )

    zij_trunk.masked_fill_(
        invalid_mask[..., None, :, None].expand(
            (*batch_dims, num_blocks, n_query, n_key, zij_trunk.shape[-1])
        ),
        0,
    )

    # Compute atom pair mask for masking out padding
    # Gather() will set the token at index 0 for all padding, we need to reset this
    atom_mask = batch["atom_mask"]

    # Pad and convert atom mask to blocks of width n_query
    # [*, N_atom] -> [*, N_blocks, N_query]
    # Pad and convert atom mask to blocks of width n_query
    # [*, N_atom] -> [*, N_blocks, N_query]
    atom_mask_q = torch.nn.functional.pad(atom_mask, (0, pad_len_right_q), value=0.0)
    atom_mask_q = atom_mask_q.reshape((*atom_mask.shape[:-1], num_blocks, n_query))

    atom_mask_k = torch.gather(
        atom_mask,
        dim=-1,
        index=key_block_idxs.expand((*atom_mask.shape[:-1], -1)).long(),
    )

    atom_mask_k.masked_fill_(
        invalid_mask.reshape(*invalid_mask.shape[:-2], -1).expand(
            (*atom_mask.shape[:-1], -1)
        ),
        0,
    )

    atom_mask_k = atom_mask_k.reshape((*atom_mask.shape[:-1], num_blocks, n_key))

    # Create pair mask
    # [*, N_blocks, N_query, N_key]
    atom_pair_mask = atom_mask_q[..., None] * atom_mask_k[..., None, :]

    # Mask out padding
    zij_trunk = zij_trunk * atom_pair_mask.unsqueeze(-1)

    return zij_trunk
