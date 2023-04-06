import math

import torch
import torch.nn as nn
from configs.hyp_params import SCALES_MAX, SCALES_MIN, SCALES_LEVELS


def get_scale_table(min_val=SCALES_MIN, max_val=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x) - x.detach() + x


def find_name_module(module, query):
    """
    Helper function to find a named module.
    Return a "nn.Module" or None
    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    return torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_register_buffer(
        module,
        buffer_name,
        state_dict_key,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int
):
    new_size = state_dict[state_dict_key].size()
    registered_buffer = find_name_module(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buffer is None:
            raise RuntimeError(f'Buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buffer.numel() == 0:
            registered_buffer.resize_(new_size)

    elif policy == "register":
        if registered_buffer is not None:
            raise RuntimeError(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffer(
        module,
        module_name,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    """

    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name: "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_register_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )

