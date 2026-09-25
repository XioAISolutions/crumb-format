"""Enhanced data loading and preprocessing for Wave Field LLM training.

Provides:
- TextDataset for plain text
- CRUMBDataset for CRUMB-structured documents
- Dynamic batching by sequence length
- Data augmentation options
- Support for multiple data formats (text, JSONL, CRUMB)
- Efficient DataLoader integration
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .crumb_adapter import iter_crumb_files, parse_crumb_structure
from .tokenizer import ByteTokenizer, CharTokenizer


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Data source
    data_path: Optional[str] = None
    data_format: str = "auto"  # auto | text | jsonl | crumb
    
    # Tokenization
    tokenizer_type: str = "byte"  # byte | char
    vocab_size: Optional[int] = None
    
    # Sequence settings
    block_size: int = 512
    stride: Optional[int] = None  # for sliding window, default = block_size
    
    # Batching
    batch_size: int = 32
    dynamic_batching: bool = False
    max_tokens_per_batch: Optional[int] = None
    
    # Data augmentation
    random_crop: bool = True
    shuffle: bool = True
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Performance
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2


class TextDataset(Dataset):
    """Dataset for tokenized text with sliding window."""
    
    def __init__(
        self,
        text: str,
        tokenizer,
        block_size: int = 512,
        stride: Optional[int] = None,
        random_crop: bool = False,
    ):
        """Initialize text dataset.
        
        Args:
            text: Raw text string
            tokenizer: Tokenizer instance
            block_size: Sequence length
            stride: Stride for sliding window (default: block_size for no overlap)
            random_crop: If True, randomly crop sequences instead of sliding window
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        self.random_crop = random_crop
        
        # Tokenize entire text
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        # Compute number of samples
        if random_crop:
            self.num_samples = max(1, (len(self.tokens) - block_size) // stride)
        else:
            self.num_samples = max(1, (len(self.tokens) - block_size) // stride)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        if self.random_crop:
            # Random starting position
            max_start = len(self.tokens) - self.block_size - 1
            start = random.randint(0, max(0, max_start))
        else:
            # Sliding window
            start = idx * self.stride
        
        # Extract sequence
        end = start + self.block_size
        input_ids = self.tokens[start:end]
        labels = self.tokens[start + 1:end + 1]
        
        # Pad if necessary
        if len(input_ids) < self.block_size:
            pad_len = self.block_size - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class CRUMBDataset(Dataset):
    """Dataset for CRUMB-structured documents with structure-aware sampling."""
    
    def __init__(
        self,
        crumb_files: list[Path],
        tokenizer,
        block_size: int = 512,
        preserve_structure: bool = True,
    ):
        """Initialize CRUMB dataset.
        
        Args:
            crumb_files: List of CRUMB file paths
            tokenizer: Tokenizer instance
            block_size: Sequence length
            preserve_structure: If True, don't split across section boundaries
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.preserve_structure = preserve_structure
        
        # Load and tokenize all CRUMB files
        self.documents = []
        self.section_boundaries = []
        
        for crumb_file in crumb_files:
            text = crumb_file.read_text(encoding="utf-8", errors="replace")
            tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
            
            # Parse structure if needed
            if preserve_structure:
                structure = parse_crumb_structure(text)
                boundaries = self._find_section_boundaries(text, tokenizer)
            else:
                boundaries = []
            
            self.documents.append(tokens)
            self.section_boundaries.append(boundaries)
        
        # Compute samples per document
        self.samples_per_doc = []
        for tokens in self.documents:
            num_samples = max(1, (len(tokens) - block_size) // block_size)
            self.samples_per_doc.append(num_samples)
        
        self.total_samples = sum(self.samples_per_doc)
    
    def _find_section_boundaries(self, text: str, tokenizer) -> list[int]:
        """Find token positions of section boundaries."""
        boundaries = []
        lines = text.split('\n')
        pos = 0
        
        for line in lines:
            if line.startswith('[') and line.endswith(']'):
                # This is a section header
                token_pos = len(tokenizer.encode(text[:pos]))
                boundaries.append(token_pos)
            pos += len(line) + 1  # +1 for newline
        
        return boundaries
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get a training sample.
        
        Returns:
            Dictionary with 'input_ids', 'labels', and optional 'section_mask'
        """
        # Find which document this sample belongs to
        doc_idx = 0
        sample_idx = idx
        
        for i, num_samples in enumerate(self.samples_per_doc):
            if sample_idx < num_samples:
                doc_idx = i
                break
            sample_idx -= num_samples
        
        tokens = self.documents[doc_idx]
        boundaries = self.section_boundaries[doc_idx]
        
        # Compute start position
        if self.preserve_structure and boundaries:
            # Try to align with section boundaries
            start = self._get_structure_aware_start(sample_idx, tokens, boundaries)
        else:
            start = sample_idx * self.block_size
        
        # Extract sequence
        end = start + self.block_size
        input_ids = tokens[start:end]
        labels = tokens[start + 1:end + 1]
        
        # Pad if necessary
        if len(input_ids) < self.block_size:
            pad_len = self.block_size - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        
        # Create section boundary mask
        section_mask = torch.zeros(self.block_size, dtype=torch.bool)
        for boundary in boundaries:
            if start <= boundary < end:
                section_mask[boundary - start] = True
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "section_mask": section_mask,
        }
    
    def _get_structure_aware_start(
        self,
        sample_idx: int,
        tokens: Tensor,
        boundaries: list[int],
    ) -> int:
        """Get start position that respects section boundaries."""
        # Simple strategy: try to start at or near a boundary
        ideal_start = sample_idx * self.block_size
        
        # Find nearest boundary
        if not boundaries:
            return ideal_start
        
        nearest = min(boundaries, key=lambda b: abs(b - ideal_start))
        
        # Use boundary if it's close enough
        if abs(nearest - ideal_start) < self.block_size // 4:
            return nearest
        
        return ideal_start


class DynamicBatchSampler:
    """Batch sampler that groups sequences by length for efficiency."""
    
    def __init__(
        self,
        dataset: Dataset,
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        """Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            max_tokens: Maximum tokens per batch
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get sequence lengths
        self.lengths = self._get_lengths()
        
        # Create batches
        self.batches = self._create_batches()
    
    def _get_lengths(self) -> list[int]:
        """Get length of each sequence in dataset."""
        lengths = []
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            lengths.append(len(sample["input_ids"]))
        return lengths
    
    def _create_batches(self) -> list[list[int]]:
        """Create batches grouped by length."""
        # Sort indices by length
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        else:
            indices.sort(key=lambda i: self.lengths[i])
        
        # Group into batches
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in indices:
            seq_len = self.lengths[idx]
            
            # Check if adding this sequence would exceed max_tokens
            if current_tokens + seq_len > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += seq_len
        
        # Add final batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)
    
    def __len__(self) -> int:
        return len(self.batches)


def create_dataloader(
    dataset: Dataset,
    config: DataConfig,
    is_train: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate settings.
    
    Args:
        dataset: Dataset to load from
        config: Data configuration
        is_train: Whether this is for training (affects shuffling)
        
    Returns:
        Configured DataLoader
    """
    # Sampler for distributed training
    sampler = None
    shuffle = config.shuffle and is_train
    
    if config.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    
    # Dynamic batching
    batch_sampler = None
    batch_size = config.batch_size
    
    if config.dynamic_batching and config.max_tokens_per_batch:
        batch_sampler = DynamicBatchSampler(
            dataset,
            max_tokens=config.max_tokens_per_batch,
            shuffle=shuffle,
        )
        batch_size = 1  # batch_sampler handles batching
        shuffle = False
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    
    return loader


def collate_fn(batch: list[dict]) -> dict[str, Tensor]:
    """Collate function for batching samples.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors
    """
    # Stack all tensors
    result = {}
    
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        result[key] = torch.stack(values)
    
    return result


def load_dataset(
    config: DataConfig,
    split: str = "train",
) -> Dataset:
    """Load dataset based on configuration.
    
    Args:
        config: Data configuration
        split: 'train' or 'eval'
        
    Returns:
        Dataset instance
    """
    # Load tokenizer
    if config.tokenizer_type == "byte":
        tokenizer = ByteTokenizer()
    elif config.tokenizer_type == "char":
        # Need to fit on data first
        text = _load_text(config.data_path)
        tokenizer = CharTokenizer.fit(text)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer_type}")
    
    # Detect data format
    data_format = config.data_format
    if data_format == "auto":
        data_format = _detect_format(config.data_path)
    
    # Load dataset
    if data_format == "crumb":
        crumb_files = list(iter_crumb_files(Path(config.data_path)))
        dataset = CRUMBDataset(
            crumb_files,
            tokenizer,
            block_size=config.block_size,
        )
    else:
        # Load as text
        text = _load_text(config.data_path)
        
        # Train/eval split
        if split == "eval":
            split_point = int(len(text) * 0.95)
            text = text[split_point:]
        else:
            split_point = int(len(text) * 0.95)
            text = text[:split_point]
        
        dataset = TextDataset(
            text,
            tokenizer,
            block_size=config.block_size,
            stride=config.stride,
            random_crop=config.random_crop,
        )
    
    return dataset


def _load_text(path: Optional[str]) -> str:
    """Load text from path."""
    if path is None:
        # Use synthetic corpus
        from .data import SYNTHETIC_CORPUS
        return SYNTHETIC_CORPUS
    
    path = Path(path)
    
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    
    if path.is_dir():
        chunks = []
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.suffix in (".txt", ".md", ".crumb"):
                chunks.append(f.read_text(encoding="utf-8", errors="replace"))
        return "\n\n".join(chunks)
    
    raise FileNotFoundError(f"Data path not found: {path}")


def _detect_format(path: Optional[str]) -> str:
    """Detect data format from path."""
    if path is None:
        return "text"
    
    path = Path(path)
    
    if path.is_file():
        if path.suffix == ".crumb":
            return "crumb"
        elif path.suffix == ".jsonl":
            return "jsonl"
        else:
            return "text"
    
    if path.is_dir():
        # Check for CRUMB files
        if any(path.glob("*.crumb")):
            return "crumb"
        return "text"
    
    return "text"

# Made with Bob
