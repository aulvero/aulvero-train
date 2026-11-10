import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from aulvero.chat import CHAT_SPECIAL_TOKENS
from aulvero.config import ModelConfig


@pytest.fixture
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        name="test",
        vocab_size=300,
        max_seq_len=64,
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        qk_norm=True,
    )


@pytest.fixture(scope="session")
def tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"] + CHAT_SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(["Bonjour le monde. Hello world. user assistant system"] * 50, trainer=trainer)
    return tok
