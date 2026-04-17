from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset as TorchDataset

from utils import (
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_ID,
    UNK_TOKEN,
    cpu_count_for_workers,
    ensure_dir,
)


_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def simple_word_tokenize(text: str) -> list[str]:
    return normalize_text(text).lower().split()


class WordTokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id = token_to_id or {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_freq: int = 1) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(simple_word_tokenize(text))
        token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for token, freq in counter.most_common():
            if freq < min_freq or token in token_to_id:
                continue
            token_to_id[token] = len(token_to_id)
            if len(token_to_id) >= vocab_size:
                break
        return cls(token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(tok, UNK_ID) for tok in simple_word_tokenize(text)]
        if add_special_tokens:
            return [BOS_ID, *ids, EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), UNK_TOKEN)
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        torch.save({"token_to_id": self.token_to_id}, path)

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = torch.load(path, map_location="cpu")
        return cls(data["token_to_id"])


class BPETokenizer:
    EOW_SUFFIX = "</w>"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPETokenizer":
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN, end_of_word_suffix=cls.EOW_SUFFIX))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = BPEDecoder(suffix=cls.EOW_SUFFIX)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            end_of_word_suffix=cls.EOW_SUFFIX,
        )
        tokenizer.train_from_iterator((normalize_text(t).lower() for t in texts), trainer=trainer)
        return cls(tokenizer)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self.tokenizer.encode(normalize_text(text).lower()).ids
        if add_special_tokens:
            return [BOS_ID, *ids, EOS_ID]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.decoder is None:
            tokenizer.decoder = BPEDecoder(suffix=cls.EOW_SUFFIX)
        return cls(tokenizer)


TokenizerLike = WordTokenizer | BPETokenizer


@dataclass
class ParallelExample:
    src: str
    tgt: str


def _select_splits(dataset_name: str, dataset_config: str | None) -> DatasetDict:
    if dataset_name == "multi30k":
        return load_dataset("bentrevett/multi30k")
    if dataset_name == "opus100":
        if not dataset_config:
            dataset_config = "en-fr"
        return load_dataset("opus100", dataset_config)
    if dataset_name == "opus_books":
        if not dataset_config:
            dataset_config = "en-fr"
        raw_train = load_dataset("Helsinki-NLP/opus_books", dataset_config, split="train")
        split = raw_train.train_test_split(test_size=0.05, seed=42)
        tmp = split["test"].train_test_split(test_size=0.5, seed=42)
        return DatasetDict(train=split["train"], validation=tmp["train"], test=tmp["test"])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _extract_pair(row: dict, src_lang: str, tgt_lang: str) -> ParallelExample:
    if "translation" in row:
        trans = row["translation"]
        return ParallelExample(normalize_text(trans[src_lang]), normalize_text(trans[tgt_lang]))
    return ParallelExample(normalize_text(row[src_lang]), normalize_text(row[tgt_lang]))


def load_parallel_splits(
    dataset_name: str,
    src_lang: str,
    tgt_lang: str,
    dataset_config: str | None = None,
    train_size: int | None = None,
    valid_size: int | None = None,
    test_size: int | None = None,
    seed: int = 42,
) -> tuple[list[ParallelExample], list[ParallelExample], list[ParallelExample]]:
    raw = _select_splits(dataset_name, dataset_config)
    train_split = raw["train"].shuffle(seed=seed)
    valid_key = "validation" if "validation" in raw else "valid"
    valid_split = raw[valid_key].shuffle(seed=seed + 1)
    test_split = (raw["test"] if "test" in raw else valid_split).shuffle(seed=seed + 2)

    if train_size:
        train_split = train_split.select(range(min(train_size, len(train_split))))
    if valid_size:
        valid_split = valid_split.select(range(min(valid_size, len(valid_split))))
    if test_size:
        test_split = test_split.select(range(min(test_size, len(test_split))))

    train = [_extract_pair(row, src_lang, tgt_lang) for row in train_split]
    valid = [_extract_pair(row, src_lang, tgt_lang) for row in valid_split]
    test = [_extract_pair(row, src_lang, tgt_lang) for row in test_split]
    return train, valid, test


def train_tokenizers(
    train_examples: list[ParallelExample],
    tokenizer_type: str,
    vocab_size: int,
    shared_vocab: bool = False,
) -> tuple[TokenizerLike, TokenizerLike]:
    if shared_vocab:
        texts = [ex.src for ex in train_examples] + [ex.tgt for ex in train_examples]
        if tokenizer_type == "bpe":
            tok = BPETokenizer.train(texts, vocab_size=vocab_size)
        else:
            tok = WordTokenizer.train(texts, vocab_size=vocab_size)
        return tok, tok

    if tokenizer_type == "bpe":
        src_tok = BPETokenizer.train((ex.src for ex in train_examples), vocab_size=vocab_size)
        tgt_tok = BPETokenizer.train((ex.tgt for ex in train_examples), vocab_size=vocab_size)
    elif tokenizer_type == "word":
        src_tok = WordTokenizer.train((ex.src for ex in train_examples), vocab_size=vocab_size)
        tgt_tok = WordTokenizer.train((ex.tgt for ex in train_examples), vocab_size=vocab_size)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_type}")
    return src_tok, tgt_tok


class TranslationDataset(TorchDataset):
    def __init__(
        self,
        examples: list[ParallelExample],
        src_tokenizer: TokenizerLike,
        tgt_tokenizer: TokenizerLike,
        max_len: int,
    ) -> None:
        self.items: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            src_ids = src_tokenizer.encode(ex.src)[:max_len]
            tgt_ids = tgt_tokenizer.encode(ex.tgt)[:max_len]
            if src_ids[-1] != EOS_ID:
                src_ids[-1] = EOS_ID
            if tgt_ids[-1] != EOS_ID:
                tgt_ids[-1] = EOS_ID
            if len(src_ids) >= 3 and len(tgt_ids) >= 3:
                self.items.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.items[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_ID)
    return src, tgt


def make_dataloaders(
    train_examples: list[ParallelExample],
    valid_examples: list[ParallelExample],
    test_examples: list[ParallelExample],
    src_tokenizer: TokenizerLike,
    tgt_tokenizer: TokenizerLike,
    max_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TranslationDataset(train_examples, src_tokenizer, tgt_tokenizer, max_len)
    valid_ds = TranslationDataset(valid_examples, src_tokenizer, tgt_tokenizer, max_len)
    test_ds = TranslationDataset(test_examples, src_tokenizer, tgt_tokenizer, max_len)
    workers = cpu_count_for_workers()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=workers)
    return train_loader, valid_loader, test_loader


def save_tokenizers(src_tok: TokenizerLike, tgt_tok: TokenizerLike, output_dir: str | Path, tokenizer_type: str) -> None:
    suffix = "json" if tokenizer_type == "bpe" else "pt"
    src_tok.save(Path(output_dir) / f"src_tokenizer.{suffix}")
    tgt_tok.save(Path(output_dir) / f"tgt_tokenizer.{suffix}")
