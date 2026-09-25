"""Tokenizers for Wave-Field LLM.

Five backends are provided, sharing a duck-typed interface:

    enc.encode(text: str) -> list[int]
    enc.decode(ids: list[int]) -> str
    enc.vocab_size  -> int
    enc.eos_id      -> int | None
    enc.name        -> str   # used by hub/sample loaders
    enc.save(path)  / .load(path) classmethod

Zero-dep:
    * ``ByteTokenizer``  — UTF-8 byte-level. 256 vocab. Lossless on any text.
    * ``CharTokenizer``  — corpus-derived character vocab. Toy training.

Optional (lazy-imported):
    * ``BPETokenizer``         — HF ``tokenizers`` BPE. Train ~8-32K vocab
                                  on the actual corpus. Best param ratio.
    * ``TiktokenTokenizer``    — ``tiktoken`` (OpenAI). ~100K vocab.
                                  No training, OpenAI-ecosystem interop.
    * ``SentencePieceTokenizer`` — ``sentencepiece`` (Llama/Mistral style).

Use ``load_tokenizer(path)`` to load any saved tokenizer without knowing
its type up front — the loader inspects the ``type`` field in the saved
JSON metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


# ── Byte-level ───────────────────────────────────────────────────────


class ByteTokenizer:
    """UTF-8 byte tokenizer. Vocab is fixed at 256."""

    name = "byte"
    vocab_size = 256
    eos_id: int | None = None  # No reserved EOS in raw bytes.

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte"}))

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "byte"
        return cls()


# ── Character-level ──────────────────────────────────────────────────


class CharTokenizer:
    """Corpus-derived character vocab + special tokens (PAD, EOS)."""

    name = "char"

    def __init__(self, chars: list[str], specials: list[str] | None = None):
        specials = specials or ["<pad>", "<eos>"]
        # Specials get the lowest ids so they're easy to spot in logits.
        self.itos: list[str] = list(specials) + chars
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.eos_id = self.stoi["<eos>"]
        self.vocab_size = len(self.itos)

    @classmethod
    def fit(cls, corpus: str, specials: list[str] | None = None) -> "CharTokenizer":
        chars = sorted(set(corpus))
        return cls(chars, specials=specials)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        unk_id = self.stoi.get("?", 0)  # no real UNK; fall back to '?' or pad.
        for ch in text:
            out.append(self.stoi.get(ch, unk_id))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(
            self.itos[i] if 0 <= i < len(self.itos) and not self.itos[i].startswith("<")
            else ""
            for i in ids
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "itos": self.itos}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "char"
        itos = meta["itos"]
        # Reconstruct: split specials from regular chars.
        specials = [c for c in itos if c.startswith("<")]
        chars = [c for c in itos if not c.startswith("<")]
        return cls(chars, specials=specials)


# ── Defaults shared by BPE-style tokenizers ──────────────────────────


DEFAULT_SPECIAL_TOKENS = [
    "<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>",
    # Chat-template specials (ChatML + CRUMB-shaped).
    "<|im_start|>", "<|im_end|>",
    "<|crumb_begin|>", "<|crumb_end|>",
    "<|goal|>", "<|context|>", "<|response|>",
]


def _missing_dep(pkg: str, install_hint: str) -> ImportError:
    return ImportError(
        f"{pkg} is required for this tokenizer. Install with:\n    {install_hint}"
    )


# ── HF tokenizers BPE ────────────────────────────────────────────────


class BPETokenizer:
    """Byte-level BPE trained with HF ``tokenizers``.

    Saves/loads to a single JSON file via ``Tokenizer.save``/``from_file``,
    with a small metadata sidecar (`tokenizer.json` contains the HF blob
    plus our ``{"type": "bpe", ...}`` header).
    """

    name = "bpe"

    def __init__(self, hf_tokenizer, eos_id: int | None = None, pad_id: int | None = None):
        self._tok = hf_tokenizer
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.vocab_size = hf_tokenizer.get_vocab_size()

    @classmethod
    def fit(
        cls,
        corpus: str,
        vocab_size: int = 8192,
        specials: list[str] | None = None,
    ) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        except ImportError as e:  # pragma: no cover
            raise _missing_dep("tokenizers", "pip install tokenizers") from e
        specials = specials or DEFAULT_SPECIAL_TOKENS
        tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator([corpus], trainer=trainer)
        eos = tok.token_to_id("<|eos|>")
        pad = tok.token_to_id("<|pad|>")
        return cls(tok, eos_id=eos, pad_id=pad)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        blob = self._tok.to_str()
        p.write_text(json.dumps({
            "type": "bpe",
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "hf_tokenizer": json.loads(blob),
        }, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as e:  # pragma: no cover
            raise _missing_dep("tokenizers", "pip install tokenizers") from e
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "bpe"
        tok = Tokenizer.from_str(json.dumps(meta["hf_tokenizer"]))
        return cls(tok, eos_id=meta.get("eos_id"), pad_id=meta.get("pad_id"))


# ── tiktoken (OpenAI BPE) ────────────────────────────────────────────


class TiktokenTokenizer:
    """Thin wrapper around an OpenAI ``tiktoken`` encoding.

    Default encoding is ``cl100k_base`` (~100K vocab, GPT-4 family).
    No training is performed; the encoding name is the only saved state.
    """

    name = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise _missing_dep("tiktoken", "pip install tiktoken") from e
        self.encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab
        # tiktoken doesn't reserve a canonical EOS for base encodings.
        self.eos_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, ids) -> str:
        return self._enc.decode(list(ids))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "type": "tiktoken",
            "encoding_name": self.encoding_name,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "TiktokenTokenizer":
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "tiktoken"
        return cls(encoding_name=meta.get("encoding_name", "cl100k_base"))


# ── SentencePiece ────────────────────────────────────────────────────


class SentencePieceTokenizer:
    """SentencePiece BPE (Llama/Mistral style).

    Trained model bytes are embedded in the JSON sidecar so save/load
    is a single-file operation, matching the other tokenizers.
    """

    name = "sentencepiece"

    def __init__(self, model_bytes: bytes, eos_id: int | None = None, pad_id: int | None = None):
        try:
            import sentencepiece as spm
        except ImportError as e:  # pragma: no cover
            raise _missing_dep("sentencepiece", "pip install sentencepiece") from e
        self._model_bytes = model_bytes
        self._sp = spm.SentencePieceProcessor()
        self._sp.LoadFromSerializedProto(model_bytes)
        self.vocab_size = self._sp.vocab_size()
        self.eos_id = eos_id if eos_id is not None else self._sp.eos_id()
        self.pad_id = pad_id if pad_id is not None else self._sp.pad_id()

    @classmethod
    def fit(
        cls,
        corpus: str,
        vocab_size: int = 16384,
        specials: list[str] | None = None,
        model_type: str = "bpe",
    ) -> "SentencePieceTokenizer":
        try:
            import sentencepiece as spm
        except ImportError as e:  # pragma: no cover
            raise _missing_dep("sentencepiece", "pip install sentencepiece") from e
        import io
        import tempfile
        specials = specials or DEFAULT_SPECIAL_TOKENS
        # SP needs corpus on disk; use a temp file.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write(corpus)
            corpus_path = tf.name
        out_buf = io.BytesIO()
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_writer=out_buf,
            vocab_size=vocab_size,
            model_type=model_type,
            user_defined_symbols=specials,
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        return cls(out_buf.getvalue())

    def encode(self, text: str) -> list[int]:
        return self._sp.EncodeAsIds(text)

    def decode(self, ids) -> str:
        return self._sp.DecodeIds(list(ids))

    def save(self, path: str | Path) -> None:
        import base64
        Path(path).write_text(json.dumps({
            "type": "sentencepiece",
            "model_b64": base64.b64encode(self._model_bytes).decode("ascii"),
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "SentencePieceTokenizer":
        import base64
        meta = json.loads(Path(path).read_text())
        assert meta.get("type") == "sentencepiece"
        model_bytes = base64.b64decode(meta["model_b64"])
        return cls(model_bytes, eos_id=meta.get("eos_id"), pad_id=meta.get("pad_id"))


# ── Factory ──────────────────────────────────────────────────────────


_REGISTRY: dict[str, type] = {
    "byte": ByteTokenizer,
    "char": CharTokenizer,
    "bpe": BPETokenizer,
    "tiktoken": TiktokenTokenizer,
    "sentencepiece": SentencePieceTokenizer,
}


def load_tokenizer(path: str | Path):
    """Load any saved tokenizer by inspecting its ``type`` metadata."""
    meta = json.loads(Path(path).read_text())
    t = meta.get("type", "byte")
    cls = _REGISTRY.get(t)
    if cls is None:
        raise ValueError(f"unknown tokenizer type: {t!r}")
    return cls.load(path)


def build_tokenizer(name: str, corpus: str | None = None, **kwargs):
    """Build a fresh tokenizer by name. ``corpus`` is required for trainable types."""
    name = name.lower()
    if name == "byte":
        return ByteTokenizer()
    if name == "char":
        if corpus is None:
            raise ValueError("char tokenizer requires a corpus to fit")
        return CharTokenizer.fit(corpus)
    if name == "bpe":
        if corpus is None:
            raise ValueError("bpe tokenizer requires a corpus to fit")
        return BPETokenizer.fit(corpus, **kwargs)
    if name == "tiktoken":
        return TiktokenTokenizer(**kwargs)
    if name == "sentencepiece":
        if corpus is None:
            raise ValueError("sentencepiece tokenizer requires a corpus to fit")
        return SentencePieceTokenizer.fit(corpus, **kwargs)
    raise ValueError(f"unknown tokenizer: {name!r}")
