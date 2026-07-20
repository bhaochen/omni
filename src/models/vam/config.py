from models.lm.config import LMConfig


class VAMConfig(LMConfig):
    model_type = "minimind-o"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 4)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)
        self.audio_ids = kwargs.get("audio_ids", [16])
        self.audio_special_token = kwargs.get("audio_special_token", "<|audio_pad|>")
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2112)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)
        self.think_end_ids = kwargs.get("think_end_ids", [26, 234, 234])
        self.image_ids = kwargs.get("image_ids", [12])
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.bridge_layer = kwargs.get("bridge_layer", self.num_hidden_layers // 2 - 1)
