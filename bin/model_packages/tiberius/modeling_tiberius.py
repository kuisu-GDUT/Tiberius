import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, logging, DynamicCache, Cache, ROPE_INIT_FUNCTIONS
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput
# from src.models.utils import tiberius_f1_loss
from .configuration_tiberius import TiberiusConfig

logger = logging.get_logger(__name__)


class TiberiusPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TiberiusConfig
    base_model_prefix = "tiberius"
    _no_split_modules = ["TiberiusModel", "TiberiusEmbeddings"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TransposeLayer(nn.Module):
    """A layer that transposes the input."""

    def __init__(
            self,
    ):
        super().__init__()

    def forward(self, x):
        """
        Transpose the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed tensor.
        """
        x = torch.transpose(x, 1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, padding="same", layer_norm=True, relu=True, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            TransposeLayer(),
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=kernel_size,
                padding=padding,
            ),
            TransposeLayer(),
        )
        self.layer_norm = nn.LayerNorm(output_size) if layer_norm else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[TiberiusConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: TiberiusConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: TiberiusConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if position_ids is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
            position_ids = cache_position.unsqueeze(0)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TiberiusModel(TiberiusPreTrainedModel):
    def __init__(self, config: TiberiusConfig):
        super(TiberiusModel, self).__init__(config)
        self.input_size = config.input_size
        self.units = config.units
        self.filter_size = config.filter_size
        self.kernel_size = config.kernel_size
        self.numb_conv = config.numb_conv
        self.numb_lstm = config.numb_lstm
        self.pool_size = config.pool_size
        self.num_labels = config.num_labels
        self.output_dense_size = config.output_dense_size

        # Convolutional layers
        # self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=self.filter_size, kernel_size=3, padding="same"
        #                        )
        self.initial_conv = ConvLayer(
            input_size=self.input_size,
            output_size=self.filter_size,
            kernel_size=3,
            padding="same",
            layer_norm=False,
        )
        self.conv_layers = nn.Sequential(
            *[
                ConvLayer(
                    input_size=self.filter_size,
                    output_size=self.filter_size,
                    kernel_size=self.kernel_size,
                    padding="same",
                )
                for _ in range(self.numb_conv - 1)
            ]
        )

        # Dense layers
        self.pre_lstm_dense = nn.Linear(in_features=self.pool_size * (self.filter_size + self.input_size),
                                        out_features=2 * self.units)

        # LSTM layers
        if config.decoder_model == "lstm":
            self.decoder_layers = nn.ModuleList([
                nn.LSTM(
                    input_size=self.units * 2,
                    hidden_size=self.units,
                    bidirectional=True,
                    batch_first=True,
                    num_layers=self.numb_lstm, )
            ])
        elif config.decoder_model == "transformer":
            self.decoder_layers = nn.ModuleList(
                [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.numb_lstm)]
            )
        else:
            raise ValueError(
                f"Decoder model {config.decoder_model} not recognized. only support 'lstm' and 'transformer'")
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.out_dense = nn.Linear(in_features=self.units * 2, out_features=self.pool_size * self.output_dense_size)
        self.out_dense = nn.Sequential(
            nn.Linear(
                in_features=self.units * 2,
                out_features=self.pool_size * self.output_dense_size
            ),
            nn.GELU()
        )
        self.classifier = nn.Linear(
            in_features=self.filter_size + self.output_dense_size,
            out_features=self.num_labels
        )

    def forward(self, x):
        # Convolutional layers
        input_x = x
        x = self.initial_conv(x)
        x = self.conv_layers(x)
        cnn_out = x  # TODO: whether the cnn_out is origin x or the last x
        # Concatenate input with convolutional output, B*L*H
        x = torch.cat([input_x, cnn_out], dim=-1)

        # Reshape layer
        if self.pool_size > 1:
            x = x.view(x.size(0), -1, self.pool_size * (self.filter_size + self.input_size))

        # Dense layer to match unit size of LSTM
        x = self.pre_lstm_dense(x)

        # Bidirectional LSTM layers
        for decoder_layer in self.decoder_layers:
            layer_outputs = decoder_layer(x)
            x = layer_outputs[0]
        x = self.norm(x)

        # Dense layer
        x = self.out_dense(x)
        # Reshape if necessary
        x = x.view(x.size(0), -1, self.output_dense_size)
        x = torch.cat([x, cnn_out], dim=-1)
        x = self.classifier(x)

        # Activation
        # x = F.softmax(x, dim=-1)

        return x


class TiberiusEmbeddings(nn.Module):
    def __init__(self, config: TiberiusConfig):
        super(TiberiusEmbeddings, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size

    def forward(self, x):
        if x.dim() > 2:  # already onehot embedded
            return x
        else:  # if categorically encoded
            return F.one_hot(x.long(), num_classes=self.vocab_size).float()


class TiberiusMaskedLM(TiberiusPreTrainedModel):
    def __init__(self, config: TiberiusConfig):
        super(TiberiusMaskedLM, self).__init__(config)
        self.embed = TiberiusEmbeddings(config)
        self.model = TiberiusModel(config)

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = True,
            **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_ids = self.embed(input_ids)
        logits = self.model(
            input_ids,
        )

        loss = None
        if labels is not None:
            if self.config.loss_weights is not None and len(self.config.loss_weights) == self.config.num_labels:
                loss_weights = torch.Tensor(self.config.loss_weights).to(logits.device)
            else:
                loss_weights = None
            loss_fct = CrossEntropyLoss(weight=loss_weights)
            loss = loss_fct(logits.permute(0, 2, 1), labels)
            f1_loss = self.f1_loss(
                labels,
                logits,
                num_labels=self.config.num_labels,
                f1_factor=self.config.f1_factor,
                # from_logits=True
            )
            loss += f1_loss
            # if labels.max() > 1000:
            #     print(f"cce loss: {loss.item()}, f1 loss: {f1_loss.item()}")
            #     self.show_label(torch.argmax(logits, dim=-1), labels)

        if not return_dict:
            output = (logits,)
            return (
                ((loss,) + output) if loss is not None else output
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
        )

    def show_label(self, logits, labels):
        predict = logits[0].cpu().detach().numpy()
        labels = labels[0].cpu().detach().numpy()

        import numpy as np
        index_to_label = np.array(['#', ",", ".", "_", '=', '+', "^", "1", "2", "3", "4", "5", "6", "7", "8"])
        decoded_label = index_to_label[labels]
        label_str = ''.join(decoded_label)
        print(f"annotation:\n{label_str}")
        print(f"annotation shape: {labels.shape}")

        predict_decoded_label = index_to_label[predict]
        predict_label_str = ''.join(predict_decoded_label)
        print(f"predict:\n{predict_label_str}")

    @staticmethod
    def f1_loss(y_true, y_pred, f1_factor=2, num_labels=9):
        # softmax
        mask = y_true != -100
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        y_pred = F.softmax(y_pred, dim=-1)
        y_true = torch.eye(num_labels, device=y_true.device)[y_true.to(torch.int64)].contiguous()
        batch_size = y_pred.shape[0]
        # Compute the f1 loss
        if y_pred.shape[-1] == 9:  # FOR BEND task
            cds_pred = y_pred[..., [0, 1, 3, 4, 5, 7]].contiguous()
            cds_true = y_true[:, [0, 1, 3, 4, 5, 7]].contiguous()
        elif y_pred.shape[-1] == 7:  # FOR BEND task
            cds_pred = y_pred[..., 4:].contiguous()
            cds_true = y_true[..., 4:].contiguous()
        else:
            cds_pred = y_pred[..., 4:].contiguous()
            cds_true = y_true[..., 4:].contiguous()

        # Compute precision and recall for the specified class
        true_positives = (cds_pred * cds_true).sum(dim=1)
        predicted_positives = cds_pred.sum(dim=1)
        possible_positives = cds_true.sum(dim=1)
        any_positives = (possible_positives > 0).float()

        precision = true_positives / (predicted_positives + 1e-7)
        recall = true_positives / (possible_positives + 1e-7)

        # For the examples with positive class, maximize the F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)  # f1 score per sequence
        f1_loss = (1 - f1_score) * any_positives
        f1_loss = f1_loss.sum() / batch_size  # mean over batch with global batch size

        # For the examples with no positive class, minimize the false positive rate
        L = y_pred.shape[1]
        fpr = (cds_pred * (1 - any_positives).unsqueeze(-1)).sum() / (L * batch_size)

        # Combine CCE loss and F1 score
        combined_loss = f1_factor * (f1_loss + fpr)
        return combined_loss


if __name__ == '__main__':
    # Example usage
    config = TiberiusConfig()
    model = TiberiusModel(config=config)
    print(model)

    input_ids = torch.randn(2, 9999, 6)
    output = model(input_ids)
    print(f"input shape: {input_ids.shape}, output shape: {output.shape}")
