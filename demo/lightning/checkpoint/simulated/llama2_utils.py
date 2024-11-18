from typing import Dict, Optional

import torch


class ModelConfig:

    def __init__(self, model_layers, intermediate_size, hidden_size,
                 attention_head, kv_heads):
        self.model_layers = model_layers
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.kv_heads = kv_heads
        self.vocab_size = 32000

    def __repr__(self):
        return (f"ModelConfig(layers={self.model_layers}, "
                f"intermediate_size={self.intermediate_size}, "
                f"hidden_size={self.hidden_size}, "
                f"attention_head={self.attention_head}, "
                f"kv_heads={self.kv_heads})")


model_7b = ModelConfig(32, 11008, 4096, 32, 32)
model_13b = ModelConfig(40, 13824, 5120, 40, 40)
model_70b = ModelConfig(80, 28672, 8192, 64, 8)

models = {"7b": model_7b, "13b": model_13b, "70b": model_70b}


def create_tensor(shape: tuple,
                  empty: bool = False,
                  dtype=torch.float32) -> torch.Tensor:
    """Helper function to create either empty or random tensors."""
    if empty:
        return torch.empty(*shape, dtype=dtype).normal_()
    return torch.randn(*shape, dtype=dtype)


def add_attention_weights(state_dict: Dict[str, torch.Tensor], prefix: str,
                          hidden_dim: int, empty: bool) -> None:
    """Adds attention layer weights directly to state dict."""
    state_dict[f'{prefix}attn.wq.weight'] = create_tensor(
        (hidden_dim, hidden_dim), empty)
    state_dict[f'{prefix}attn.wk.weight'] = create_tensor(
        (hidden_dim, hidden_dim), empty)
    state_dict[f'{prefix}attn.wv.weight'] = create_tensor(
        (hidden_dim, hidden_dim), empty)
    state_dict[f'{prefix}attn.wo.weight'] = create_tensor(
        (hidden_dim, hidden_dim), empty)


def add_ffn_weights(state_dict: Dict[str, torch.Tensor], prefix: str,
                    hidden_dim: int, intermediate_dim: int,
                    empty: bool) -> None:
    """Adds feed-forward network weights directly to state dict."""
    state_dict[f'{prefix}ff.w1.weight'] = create_tensor(
        (intermediate_dim, hidden_dim), empty)
    state_dict[f'{prefix}ff.w2.weight'] = create_tensor(
        (hidden_dim, intermediate_dim), empty)
    state_dict[f'{prefix}ff.w3.weight'] = create_tensor(
        (intermediate_dim, hidden_dim), empty)


def add_adamw_states(state_dict: Dict[str, torch.Tensor], prefix: str,
                     param_prefix: str, shape: tuple, empty: bool) -> None:
    """Adds AdamW optimizer states directly to state dict."""
    state_dict[f'{prefix}{param_prefix}.exp_avg'] = create_tensor(shape, empty)
    state_dict[f'{prefix}{param_prefix}.exp_avg_sq'] = create_tensor(
        shape, empty)


def add_layer_weights(state_dict: Dict[str, torch.Tensor], prefix: str,
                      hidden_dim: int, intermediate_dim: int, empty: bool,
                      use_adamw: bool) -> None:
    """Adds all weights for a single transformer layer directly to state dict."""
    # Attention weights
    add_attention_weights(state_dict, prefix, hidden_dim, empty)

    # Attention norm
    state_dict[f'{prefix}attn_norm.weight'] = create_tensor((hidden_dim, ),
                                                            empty)

    # FFN weights
    add_ffn_weights(state_dict, prefix, hidden_dim, intermediate_dim, empty)

    # FFN norm
    state_dict[f'{prefix}ffn_norm.weight'] = create_tensor((hidden_dim, ),
                                                           empty)

    # Add AdamW states if needed
    if use_adamw:
        # Attention AdamW states
        for weight_name in ['wq', 'wk', 'wv', 'wo']:
            add_adamw_states(state_dict, prefix, f'attn.{weight_name}',
                             (hidden_dim, hidden_dim), empty)

        # Attention norm AdamW states
        add_adamw_states(state_dict, prefix, 'attn_norm', (hidden_dim, ),
                         empty)

        # FFN AdamW states
        add_adamw_states(state_dict, prefix, 'ff.w1',
                         (intermediate_dim, hidden_dim), empty)
        add_adamw_states(state_dict, prefix, 'ff.w2',
                         (hidden_dim, intermediate_dim), empty)
        add_adamw_states(state_dict, prefix, 'ff.w3',
                         (intermediate_dim, hidden_dim), empty)

        # FFN norm AdamW states
        add_adamw_states(state_dict, prefix, 'ffn_norm', (hidden_dim, ), empty)


def create_llama2_state_dict(world_size: int,
                             rank: int,
                             parameters: str,
                             optimizer: str,
                             empty: bool = False) -> Dict[str, torch.Tensor]:
    """
    Creates a state dictionary matching LLAMA2 architecture dimensions.
    Optimized version using direct assignments.

    Args:
        world_size: Total number of processes/devices
        rank: Current process/device index
        parameters: Model size ('7b', '13b', or '70b')
        optimizer: Optimizer type ('sgd' or 'adamw')
        empty: If True, creates empty state dictionary

    Returns:
        State dictionary with model weights and optimizer states
    """
    if parameters not in models:
        raise ValueError(
            "Invalid parameter size. Must be one of: 7b, 13b, 70b")

    model_config = models[parameters]
    if optimizer.lower() not in ['sgd', 'adamw']:
        return ValueError("Invalid optmizer. Must be either sgd or adamW")
    use_adamw = optimizer.lower() == 'adamw'

    # Pre-allocate dictionary with estimated size
    state_dict = {}

    # Global weights (embeddings, norm, output)
    state_dict['tok_embeddings.weight'] = create_tensor(
        (model_config.vocab_size, model_config.hidden_size), empty)
    state_dict['norm.weight'] = create_tensor((model_config.hidden_size, ),
                                              empty)
    state_dict['output.weight'] = create_tensor(
        (model_config.vocab_size, model_config.hidden_size), empty)

    # Layer weights
    for layer in range(model_config.model_layers):
        if layer % world_size == rank:
            prefix = f'layers.{layer}.'
            add_layer_weights(state_dict, prefix, model_config.hidden_size,
                              model_config.intermediate_size, empty, use_adamw)

    return state_dict
