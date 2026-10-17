from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from src.data.classes.enums import Architecture
from src.data.classes.settings import BlockSettings

FEEDFORWARD_EXPANSION = 2


class MlpBody(nn.Module):
    def __init__(
        self,
        *,
        settings: BlockSettings,
        num_layers: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(normalized_shape=settings.hidden_size),
                nn.Linear(
                    in_features=settings.hidden_size,
                    out_features=settings.hidden_size,
                ),
                nn.GELU(),
                nn.Dropout(p=settings.dropout),
            )
            for _ in range(num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = tokens + layer(tokens)
        return tokens


class LstmBody(nn.Module):
    def __init__(
        self,
        *,
        settings: BlockSettings,
        num_layers: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=settings.hidden_size,
            hidden_size=settings.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=settings.dropout if num_layers > 1 else 0.0,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(tokens)
        return outputs


class AttentionBody(nn.Module):
    def __init__(
        self,
        *,
        settings: BlockSettings,
        num_layers: int,
        num_tokens: int,
    ) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, num_tokens, settings.hidden_size))
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=settings.hidden_size,
                nhead=settings.attention_heads,
                dim_feedforward=FEEDFORWARD_EXPANSION * settings.hidden_size,
                dropout=settings.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens + self.position)


BodyBuilder = Callable[..., nn.Module]

BODIES: dict[Architecture, BodyBuilder] = {
    Architecture.MLP: MlpBody,
    Architecture.LSTM: LstmBody,
    Architecture.ATTENTION: AttentionBody,
}


def build_body(
    *,
    architecture: Architecture,
    settings: BlockSettings,
    num_layers: int,
    num_tokens: int,
) -> nn.Module:
    return BODIES[architecture](
        settings=settings,
        num_layers=num_layers,
        num_tokens=num_tokens,
    )
