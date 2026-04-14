"""
vec2text CorrectorEncoderModel hard-codes embedder_dim=768 for non-OpenAI embedders.
sentence-transformers/all-MiniLM-L6-v2 uses 384 — causes assert in get_encoder_embedding.

Also, CorrectorExperiment.load_model only copies embedder_dim for inversion_from_logits,
not for standard inversion. We set config from the loaded inversion model before building
the corrector.
"""
from __future__ import annotations


def apply_corrector_embedder_dim_compat() -> None:
    import vec2text.experiments as vexp
    import vec2text.models.corrector_encoder as ce
    from vec2text.models.corrector_encoder_from_logits import CorrectorEncoderFromLogitsModel

    if getattr(vexp.CorrectorExperiment, "_embedder_dim_patched", False):
        return

    _orig_load = vexp.CorrectorExperiment.load_model

    def load_model(self, inversion_trainer):
        # Experiment.config is a *property* that builds a fresh InversionConfig each
        # access. vec2text's upstream pattern (self.config.foo = ... then
        # Model(config=self.config)) drops mutations because the constructor sees a
        # second, new config. Use one cfg object and pass it through.
        exp = inversion_trainer.args.experiment
        m = inversion_trainer.model
        cfg = self.config
        if exp == "inversion_from_logits":
            cfg.embedder_dim = m.embedder_dim
            cfg.num_repeat_tokens = m.num_repeat_tokens
            return CorrectorEncoderFromLogitsModel(config=cfg)
        cfg.embedder_dim = m.embedder_dim
        cfg.num_repeat_tokens = m.num_repeat_tokens
        return ce.CorrectorEncoderModel(config=cfg)

    vexp.CorrectorExperiment.load_model = load_model  # type: ignore[assignment]
    vexp.CorrectorExperiment._embedder_dim_patched = True  # type: ignore[attr-defined]

    def __init__(self, config):
        import torch.nn as nn
        import transformers

        transformers.PreTrainedModel.__init__(self, config=config)
        if getattr(config, "embedder_model_api", None):
            embedder_dim = 1536
        elif getattr(config, "embedder_dim", None) is not None:
            embedder_dim = int(config.embedder_dim)
        else:
            embedder_dim = 768
        bottleneck_dim = embedder_dim

        num_repeat_tokens = config.num_repeat_tokens
        ignore_hypothesis_embedding = getattr(
            config, "corrector_ignore_hypothesis_embedding", False
        )
        self.use_ff_dropout = False

        encoder_decoder = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path
        )
        self.encoder_decoder = encoder_decoder
        self.embedder_dim = embedder_dim
        self.num_repeat_tokens = num_repeat_tokens
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedding_transform_1 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_2 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.embedding_transform_3 = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(
                self.encoder_decoder.config.dropout_rate if self.use_ff_dropout else 0.0
            ),
            nn.GELU(),
            nn.Linear(bottleneck_dim, self.encoder_hidden_dim * num_repeat_tokens),
        )
        self.ignore_hypothesis_embedding = ignore_hypothesis_embedding
        self.training_embedding_noise_level = 0
        self.use_ln = True
        if self.use_ln:
            self.layernorm = nn.LayerNorm(self.encoder_hidden_dim)

    ce.CorrectorEncoderModel.__init__ = __init__  # type: ignore[assignment]
