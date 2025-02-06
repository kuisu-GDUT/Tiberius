from transformers import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class TiberiusConfig(PretrainedConfig):
    model_type = "tiberius"

    def __init__(
            self,
            vocab_size=None,
            units=384,  # hidden_size
            filter_size=128,
            kernel_size=9,
            numb_conv=3,
            numb_lstm=2,
            pool_size=9,
            num_labels=7,
            output_dense_size=30,
            initializer_range=0.02,
            loss_weights=None,
            repeat_mask=False,
            f1_factor=2.0,
            pretraining_tp=1,
            num_attention_heads=8,
            max_position_embeddings=2048,
            intermediate_size=None,
            rope_theta=10000.0,
            rope_scaling=None,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            head_dim=None,
            num_key_value_heads=None,
            hidden_act="silu",
            decoder_model="lstm",  # transformer
            rms_norm_eps=1e-6,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.numb_conv = numb_conv
        self.numb_lstm = numb_lstm
        self.pool_size = pool_size
        self.input_size = vocab_size
        self.vocab_size = vocab_size
        self.output_dense_size = output_dense_size
        self.initializer_range = initializer_range
        self.loss_weights = loss_weights
        self.num_labels = num_labels
        self.repeat_mask = repeat_mask
        self.f1_factor = f1_factor
        # FOR ATTN
        self.decoder_model = decoder_model
        self.pretraining_tp = pretraining_tp
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.num_attention_heads = num_attention_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.hidden_size = units * 2
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        if intermediate_size is None:
            intermediate_size = 4 * self.hidden_size
        self.intermediate_size = intermediate_size
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__}
