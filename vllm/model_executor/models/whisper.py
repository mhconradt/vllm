# Derived from Whisper implementation posted on HuggingFace; license below:
# coding=utf-8
# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Whisper model."""

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import WhisperConfig
from transformers.utils import logging

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

logger = logging.get_logger(__name__)


class WhisperLearnedPositionalEmbedding(VocabParallelEmbedding):
    def forward(
            self,
            positions: torch.Tensor,
    ) -> torch.Tensor:
        # Whisper uses learned positional embedding in decoder only
        # Encoder uses two convolutional layers followed by sinusoidal positional embedding.
        return super().forward(positions)


class WhisperEncoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: Optional[WhisperConfig] = None,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.ENCODER)

        output, _ = self.out_proj(attn_output)
        return output


class WhisperDecoderSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: Optional[WhisperConfig] = None,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.DECODER)

        output, _ = self.out_proj(attn_output)
        return output


class WhisperDecoderCrossAttention(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            config: Optional[WhisperConfig] = None,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads "
                             f"(got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_world_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv_dec, _ = self.qkv_proj(decoder_hidden_states)
        q, _, _ = qkv_dec.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        if encoder_hidden_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
            _, k, v = qkv_enc.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)

        attn_output = self.attn(q,
                                k,
                                v,
                                kv_cache,
                                attn_metadata,
                                attn_type=AttentionType.ENCODER_DECODER)

        output, _ = self.out_proj(attn_output)
        return output


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Whisper, MBART->WHISPER
class WhisperEncoderLayer(nn.Module):
    def __init__(
            self,
            config: WhisperConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperEncoderAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            # bias=True,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(
            config.activation_function,
            quant_config
        )
        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = nn.Linear(
            self.embed_dim,
            config.encoder_ffn_dim
        )
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states
                torch.Tensor of *encoder* input embeddings.
            kv_cache:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Encoder layer output torch.Tensor
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class WhisperDecoderLayer(nn.Module):

    def __init__(
            self,
            config: WhisperConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = WhisperDecoderSelfAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            bias=True,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(
            config.activation_function,
            quant_config,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = WhisperDecoderCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            bias=True,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_hidden_states
                torch.Tensor of *decoder* input embeddings.
            kv_cache:
                KV cache tensor
            attn_metadata:
                vLLM Attention metadata structure
            encoder_hidden_states
                torch.Tensor of *encoder* input embeddings.
        Returns:
            Decoder layer output torch.Tensor
        """
        residual = decoder_hidden_states
        hidden_states = self.self_attn_layer_norm(decoder_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states
        # add cross-attn to positions 1 of present_key_value tuple

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(
            self,
            config: WhisperConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            input_features: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            positions
                Positions of *encoder* input sequence tokens.
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Encoder output torch.Tensor
        """

        # TODO: Consistently use WhisperConfig's activation_fn.
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        # This is a sinusoidal positional embedding.
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(
            self,
            config: WhisperConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = nn.Embedding(self.max_target_positions, config.d_model)

        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            decoder_input_ids: torch.Tensor,
            decoder_positions: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor],
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            decoder_positions
                Positions of *decoder* input sequence tokens.
            encoder_hidden_states:
                Tensor of encoder output embeddings
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Decoder output torch.Tensor
        """
        inputs_embeds = self.embed_tokens(decoder_input_ids)
        embed_pos = self.embed_positions(
            decoder_positions,
            AttentionType.DECODER,
        )
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperModel(nn.Module):
    def __init__(self,
                 config: WhisperConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(
            config,
            cache_config, quant_config, lora_config
        )
        self.decoder = WhisperDecoder(
            config,
            cache_config, quant_config, lora_config
        )

    def forward(
            self,
            decoder_input_ids: torch.Tensor,
            positions: torch.Tensor,
            encoder_input_features: torch.Tensor,
            encoder_positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        encoder_hidden_states = None
        if encoder_input_features.numel():
            encoder_hidden_states = self.encoder(
                input_features=encoder_input_features,
                positions=encoder_positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return decoder_outputs


class WhisperForConditionalGeneration(nn.Module):
    base_model_prefix = "model"

    def __init__(
            self,
            config: WhisperConfig,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None
    ):
        super().__init__()
        self.config = config
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.model = WhisperModel(
            config,
            cache_config,
            quant_config,
            lora_config,
        )
        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            bias=False,
            # TODO: QuantConfig?
        )
        self.max_target_positions = config.max_target_positions
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size
        )
        self.sampler = Sampler()

    def forward(
            self,
            decoder_input_ids: torch.Tensor,
            decoder_positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            encoder_input_features: torch.Tensor,
            encoder_positions: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_input_ids
                torch.Tensor of *decoder* input token ids.
            decoder_positions
                torch.Tensor of *decoder* position indices.
            encoder_input_features
                torch.Tensor of *encoder* input token ids.
            encoder_positions
                torch.Tensor of *encoder* position indices
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Output torch.Tensor
        """
        outputs = self.model(
            decoder_input_ids=decoder_input_ids,
            decoder_positions=decoder_positions,
            encoder_input_features=encoder_input_features,
            encoder_positions=encoder_positions,
            attn_metadata=attn_metadata,
            kv_caches=kv_caches
        )
        return outputs

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    stacked_params_mapping = {
        "q_proj": {
            "param_name": "qkv_proj",
            "shard_id": "q",
        },
        "k_proj": {
            "param_name": "qkv_proj",
            "shard_id": "k",
        },
        "v_proj": {
            "param_name": "qkv_proj",
            "shard_id": "v",
        },
    }
    params_mapping = {}

    def _rename_key(self, key: str):
        prefix = f"{self.base_model_prefix}."
        key = key[len(prefix):] if key.startswith(prefix) else key

        for src, dst in self.params_mapping.items():
            key = key.replace(src, dst)

        return key

    def _rename_stacked_param(
            self,
            name: str,
    ) -> Tuple[str, Optional[str]]:
        for key, mapping in self.stacked_params_mapping.items():
            if key in name:
                name = name.replace(key, mapping["param_name"])
                return name, mapping["shard_id"]
        return name, None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        model_params_dict = dict(self.model.named_parameters())
        weights_tuple_list = list(weights)
        for name, loaded_weight in weights_tuple_list:
            name = self._rename_key(name)
            name, shard_id = self._rename_stacked_param(name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in model_params_dict:
                continue
            param = model_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            if shard_id:
                weight_loader(param, loaded_weight, shard_id)
            else:
                weight_loader(param, loaded_weight)
