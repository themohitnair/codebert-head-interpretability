from __future__ import annotations

from typing import Any

import torch


class HeadAblationIntervention:
    def __init__(self, layer_idx: int, head_idx: int, debug_shapes: bool = False):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.debug_shapes = debug_shapes
        self._handle: Any = None

    def _hook(self, module: torch.nn.Module, inputs: tuple[Any, ...], outputs: Any):
        if not isinstance(outputs, tuple) or len(outputs) == 0:
            return outputs

        context_layer = outputs[0]
        if not isinstance(context_layer, torch.Tensor) or context_layer.ndim != 3:
            return outputs

        batch_size, seq_len, hidden_size = context_layer.shape
        num_heads = module.num_attention_heads
        head_dim = module.attention_head_size

        if hidden_size != num_heads * head_dim:
            return outputs

        if self.head_idx < 0 or self.head_idx >= num_heads:
            raise ValueError(
                f"head_idx={self.head_idx} out of bounds for num_heads={num_heads}"
            )

        if self.debug_shapes:
            print(
                "HeadAblationIntervention hook: "
                f"context={tuple(context_layer.shape)}, heads={num_heads}, head_dim={head_dim}"
            )

        per_head = context_layer.view(batch_size, seq_len, num_heads, head_dim)
        per_head = per_head.permute(0, 2, 1, 3).contiguous()
        per_head[:, self.head_idx, :, :] = 0.0

        merged = per_head.permute(0, 2, 1, 3).contiguous()
        merged = merged.view(batch_size, seq_len, hidden_size)

        output_items = list(outputs)
        output_items[0] = merged
        return tuple(output_items)

    def register(self, model: torch.nn.Module) -> None:
        if self._handle is not None:
            self.remove()

        layers = model.encoder.layer
        if self.layer_idx < 0 or self.layer_idx >= len(layers):
            raise ValueError(
                f"layer_idx={self.layer_idx} out of bounds for num_layers={len(layers)}"
            )

        attention_self = layers[self.layer_idx].attention.self
        self._handle = attention_self.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
