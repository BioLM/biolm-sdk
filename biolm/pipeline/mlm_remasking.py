"""
Masked Language Model (MLM) remasking utilities.

Provides functionality for iterative remasking and prediction using
masked language models like ESM, ESM-1v, ESM-2.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RemaskingConfig:
    """
    Configuration for MLM remasking.

    Args:
        model_name: MLM model to use (e.g., 'esm-150m', 'esm-650m', 'esm-3b', 'esm3', 'esmc')
        mask_fraction: Fraction of positions to mask (default: 0.15)
        mask_positions: Specific positions to mask, or 'auto' for random
        num_iterations: Number of remasking iterations (default: 1)
        temperature: Sampling temperature (default: 1.0)
        top_k: Top-k sampling (default: None)
        top_p: Nucleus sampling (default: None)
        mask_token: Token to use for masking (default: '<mask>')
        conserved_positions: Positions that should never be masked
        mask_strategy: Strategy for selecting positions ('random', 'low_confidence', 'blocks')
        block_size: Size of blocks for block masking (default: 3)
        confidence_threshold: Threshold for low-confidence masking (default: 0.8)

    Example:
        >>> # ESM2 150M remasking
        >>> config = RemaskingConfig(
        ...     model_name='esm-150m',
        ...     mask_fraction=0.15,
        ...     num_iterations=5,
        ...     temperature=1.0
        ... )
        >>>
        >>> # ESM3 with higher temperature
        >>> config = RemaskingConfig(
        ...     model_name='esm3',
        ...     mask_fraction=0.20,
        ...     temperature=1.5
        ... )
    """

    model_name: str = "esm-150m"  # Default to ESM2 150M
    action: str = "predict"  # API action: 'predict' for ESM2 (logits), 'generate' for DSM (filled seq)
    mask_fraction: float = 0.15
    mask_positions: Union[str, list[int]] = "auto"
    num_iterations: int = 1
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    mask_token: str = "<mask>"
    conserved_positions: Optional[list[int]] = None
    mask_strategy: str = "random"
    block_size: int = 3
    confidence_threshold: float = 0.8
    # GEN-06: proper dataclass fields for parent_sequence and num_variants.
    # Previously set as dynamic attributes after construction, which caused
    # to_spec() to omit them and from_db() reconstruction to always fail.
    parent_sequence: Optional[str] = None
    num_variants: int = 100

    def __post_init__(self):
        if not (0 < self.mask_fraction <= 1.0):
            raise ValueError(
                f"mask_fraction must be in (0, 1], got {self.mask_fraction}"
            )
        if self.num_iterations < 1:
            raise ValueError(
                f"num_iterations must be >= 1, got {self.num_iterations}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0. For greedy decoding, use temperature=1e-6 "
                f"(very low). Got {self.temperature}"
            )

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        return {
            "type": "RemaskingConfig",
            "model_name": self.model_name,
            "action": self.action,
            "mask_fraction": self.mask_fraction,
            "mask_positions": self.mask_positions,
            "num_iterations": self.num_iterations,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "mask_token": self.mask_token,
            "conserved_positions": self.conserved_positions,
            "mask_strategy": self.mask_strategy,
            "block_size": self.block_size,
            "confidence_threshold": self.confidence_threshold,
            "parent_sequence": self.parent_sequence,
            "num_variants": self.num_variants,
        }


class MLMRemasker:
    """
    Masked Language Model remasking utility.

    Handles iterative masking and prediction for generating sequence variants
    using masked language models.

    Args:
        config: RemaskingConfig instance
        api_client: BioLM API client (optional, for actual predictions)
        model_name: Model name (e.g., 'esm2', 'esm1v')

    Example:
        >>> config = RemaskingConfig(mask_fraction=0.15, num_iterations=5)
        >>> remasker = MLMRemasker(config, model_name='esm2')
        >>> variants = remasker.generate_variants('MKTAYIAKQRQ', num_variants=10)
    """

    def __init__(
        self, config: RemaskingConfig, api_client=None, model_name: Optional[str] = None
    ):
        self.config = config
        self.api_client = api_client
        # Use model_name from config, or override if provided
        self.model_name = model_name or config.model_name
        self.random = random.Random(42)  # For reproducibility

    def select_mask_positions(
        self, sequence: str, confidences: Optional[np.ndarray] = None
    ) -> list[int]:
        """
        Select positions to mask based on strategy.

        Args:
            sequence: Input sequence
            confidences: Optional confidence scores per position (for low_confidence strategy)

        Returns:
            List of positions to mask (0-indexed)
        """
        seq_len = len(sequence)

        # Explicit positions
        if isinstance(self.config.mask_positions, list):
            return [p for p in self.config.mask_positions if 0 <= p < seq_len]

        # Calculate number to mask
        num_to_mask = max(1, int(seq_len * self.config.mask_fraction))

        # Get conserved positions
        conserved = set(self.config.conserved_positions or [])
        available_positions = [i for i in range(seq_len) if i not in conserved]

        if len(available_positions) < num_to_mask:
            num_to_mask = len(available_positions)

        # guard against empty available_positions before any strategy branch
        if not available_positions:
            warnings.warn(
                f"All positions are conserved; cannot mask any positions. "
                f"Returning empty mask selection.",
                UserWarning,
                stacklevel=2,
            )
            return []

        # Select based on strategy
        if self.config.mask_strategy == "random":
            return self.random.sample(available_positions, num_to_mask)

        elif self.config.mask_strategy == "low_confidence":
            if confidences is None:
                # Fallback to random
                return self.random.sample(available_positions, num_to_mask)

            # Select positions with lowest confidence
            available_confidences = [
                (i, confidences[i]) for i in available_positions if i < len(confidences)
            ]
            if len(available_confidences) == 0:
                # Fall back to random if no valid confidence scores
                return self.random.sample(available_positions, num_to_mask)
            available_confidences.sort(key=lambda x: x[1])

            return [i for i, _ in available_confidences[:num_to_mask]]

        elif self.config.mask_strategy == "blocks":
            # Mask contiguous blocks
            positions = []
            block_size = self.config.block_size
            max_block_iters = max(len(available_positions) * 10, 100)
            block_iter_count = 0

            # guard against empty available_positions before entering loop
            if not available_positions:
                return []

            while len(positions) < num_to_mask:
                block_iter_count += 1
                if block_iter_count > max_block_iters:
                    break  # Return whatever positions we have
                # Pick random start position
                start = self.random.choice(available_positions)

                # Add block
                for i in range(start, min(start + block_size, seq_len)):
                    if (
                        i not in conserved
                        and i not in positions
                        and i in available_positions
                    ):
                        positions.append(i)
                        if len(positions) >= num_to_mask:
                            break

            return positions[:num_to_mask]

        else:
            raise ValueError(f"Unknown mask_strategy: {self.config.mask_strategy}")

    def create_masked_sequence(self, sequence: str, positions: list[int]) -> str:
        """
        Create masked sequence with mask token at specified positions.

        Args:
            sequence: Original sequence
            positions: Positions to mask

        Returns:
            Masked sequence
        """
        seq_list = list(sequence)

        for pos in positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = self.config.mask_token

        return "".join(seq_list)

    async def predict_masked_positions(
        self, sequence: str, mask_positions: list[int]
    ) -> tuple[str, dict[int, float]]:
        """
        Predict amino acids at masked positions.

        Builds the masked sequence client-side (inserting ``config.mask_token``
        at each position), sends it to the model's predict endpoint, and
        decodes the returned logits with temperature/top-k/top-p sampling.

        Falls back to reading a ``"sequence"`` key from the response if the
        model returns a filled sequence directly instead of logits.

        Args:
            sequence: Original (unmasked) sequence.
            mask_positions: 0-indexed positions to replace.

        Returns:
            Tuple of (predicted_sequence, confidences_dict)
        """
        if self.api_client is None:
            # Mock: replace positions in-place with random amino acids
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            confidences = {}
            seq_list = list(sequence)
            for pos in mask_positions:
                replacement = self.random.choice(amino_acids)
                seq_list[pos] = replacement
                confidences[pos] = self.random.uniform(0.5, 1.0)
            return "".join(seq_list), confidences

        else:
            # Build masked sequence client-side, send to API.
            masked_seq = self.create_masked_sequence(sequence, mask_positions)

            # Call the configured action (predict for ESM2, generate for DSM)
            action_fn = getattr(self.api_client, self.config.action)
            if self.config.action == "generate":
                result = await action_fn(
                    items=[{"sequence": masked_seq}],
                    params={"num_sequences": 1, "temperature": self.config.temperature},
                )
            else:
                result = await action_fn(
                    items=[{"sequence": masked_seq}],
                )

            # Unwrap response — formats differ by model/action:
            #   ESMC predict  → dict directly
            #   ESM2 predict  → [{...}]
            #   DSM generate  → [[{...}]]
            if isinstance(result, dict):
                pred_result = result
            elif isinstance(result, list) and len(result) > 0:
                pred_result = result[0]
                if isinstance(pred_result, list):
                    if not pred_result:
                        raise ValueError("API returned empty nested list")
                    pred_result = pred_result[0]
            else:
                raise ValueError(
                    f"API returned empty or invalid result: {result}"
                )

            if not isinstance(pred_result, dict):
                raise ValueError(
                    f"API returned unexpected type: {type(pred_result)}"
                )

            # API error dict — surface clearly rather than falling through to decode
            if "error" in pred_result and "status_code" in pred_result:
                raise ValueError(
                    f"API error {pred_result['status_code']}: {pred_result['error']}"
                )

            # Path 1: Model returns filled sequence directly (DSM generate)
            predicted_seq = pred_result.get("sequence")
            if predicted_seq is not None and self.config.mask_token not in predicted_seq:
                # Build confidences from log_prob if available
                confidences: dict[int, float] = {}
                log_prob = pred_result.get("log_prob")
                if log_prob is not None:
                    # Convert from nats/position to probability-space [0, 1]
                    per_pos = min(
                        1.0,
                        math.exp(-abs(float(log_prob)) / max(len(mask_positions), 1)),
                    )
                    for pos in mask_positions:
                        confidences[pos] = per_pos
                return predicted_seq, confidences

            # MLM-06: explicit error when mask token survives in the returned sequence
            if predicted_seq is not None and self.config.mask_token in predicted_seq:
                raise ValueError(
                    f"Model returned sequence still containing mask token: "
                    f"{predicted_seq!r}. Expected all mask positions to be filled."
                )

            # Path 2: Model returns logits (ESM2/ESMC predict)
            logits = pred_result.get("logits")
            seq_tokens = pred_result.get("sequence_tokens")
            vocab_tokens = pred_result.get("vocab_tokens")

            if logits is not None:
                if seq_tokens is None:
                    raise ValueError(
                        "API response returned 'logits' but is missing 'sequence_tokens'. "
                        "Cannot decode logits without knowing the token-to-position mapping. "
                        f"Got keys: {list(pred_result.keys())}"
                    )
                if vocab_tokens is None:
                    raise ValueError(
                        "API response returned 'logits' but is missing 'vocab_tokens'. "
                        "Cannot decode logits without knowing the vocabulary ordering — "
                        "vocab column order differs between models (e.g. ESM2 vs ESMC). "
                        f"Got keys: {list(pred_result.keys())}"
                    )
                return self._decode_logits(
                    sequence, mask_positions, logits, seq_tokens, vocab_tokens
                )

            raise ValueError(
                f"API result missing both 'logits' and 'sequence'. "
                f"Got keys: {list(pred_result.keys())}"
            )

    def _decode_logits(
        self,
        original_sequence: str,
        mask_positions: list[int],
        logits: list,
        seq_tokens: list[str],
        vocab_tokens: list[str],
    ) -> tuple[str, dict[int, float]]:
        """Decode logits at mask positions into amino acids.

        Applies temperature scaling and optional top-k/top-p sampling.
        """
        logits_arr = np.array(logits, dtype=np.float64)
        seq_list = list(original_sequence)
        confidences: dict[int, float] = {}

        if len(logits_arr) == len(seq_tokens) + 2:
            # ESMC/ESM3 style: logits include BOS and EOS special tokens — strip them
            logits_arr = logits_arr[1:-1]
        elif len(logits_arr) != len(seq_tokens):
            raise ValueError(
                f"Logits array length ({len(logits_arr)}) != sequence token length "
                f"({len(seq_tokens)}). Got {len(logits_arr)} logit rows for "
                f"{len(seq_tokens)} tokens."
            )

        # Map mask_positions (0-indexed in sequence) to token indices
        # seq_tokens mirrors the sequence: ['M', 'K', '<mask>', ...]
        for seq_pos in mask_positions:
            if seq_pos >= len(seq_tokens):
                raise ValueError(
                    f"mask position {seq_pos} is out of bounds for seq_tokens length "
                    f"{len(seq_tokens)} — this indicates the mask was built from a different "
                    f"sequence length than the API returned tokens for"
                )

            pos_logits = logits_arr[seq_pos]

            # Temperature scaling
            temp = self.config.temperature or 1.0
            if temp != 1.0 and temp > 0:
                pos_logits = pos_logits / temp

            # Softmax
            pos_logits = pos_logits - np.max(pos_logits)
            probs = np.exp(pos_logits) / np.sum(np.exp(pos_logits))

            # Top-k filtering
            if self.config.top_k is not None and self.config.top_k > 0:
                top_k_idx = np.argsort(probs)[::-1][: self.config.top_k]
                mask = np.zeros_like(probs)
                mask[top_k_idx] = 1.0
                probs = probs * mask
                probs = probs / (probs.sum() or 1.0)

            # Top-p (nucleus) filtering
            if self.config.top_p is not None and 0 < self.config.top_p < 1.0:
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, self.config.top_p) + 1
                keep = sorted_idx[:cutoff]
                mask = np.zeros_like(probs)
                mask[keep] = 1.0
                probs = probs * mask
                probs = probs / (probs.sum() or 1.0)

            # Guard: if all probs were zeroed by top-k/top-p, fall back to uniform
            if probs.sum() <= 0:
                probs = np.ones(len(probs)) / len(probs)

            # Sample
            chosen_idx = self.random.choices(range(len(vocab_tokens)), weights=probs, k=1)[0]
            chosen_aa = vocab_tokens[chosen_idx]
            confidence = float(probs[chosen_idx])

            if seq_pos >= len(seq_list):
                raise ValueError(
                    f"internal inconsistency: seq_pos {seq_pos} is in-range for "
                    f"seq_tokens (len={len(seq_tokens)}) but out-of-range for "
                    f"seq_list (len={len(seq_list)}) — both should derive from the same sequence"
                )
            seq_list[seq_pos] = chosen_aa
            confidences[seq_pos] = confidence

        return "".join(seq_list), confidences

    async def generate_variant(
        self, parent_sequence: str, iteration: int = 0
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a single variant through remasking (async).

        Args:
            parent_sequence: Starting sequence
            iteration: Iteration number (for seeding)

        Returns:
            Tuple of (variant_sequence, metadata_dict)
        """
        current_sequence = parent_sequence
        all_mask_positions = []
        all_confidences = {}

        # Perform remasking iterations
        for _iter_num in range(self.config.num_iterations):
            # Select positions to mask
            mask_positions = self.select_mask_positions(current_sequence)

            # Predict: pass original sequence + positions; API handles masking internally
            predicted_seq, confidences = await self.predict_masked_positions(
                current_sequence, mask_positions
            )

            # Update
            current_sequence = predicted_seq
            all_mask_positions.extend(mask_positions)
            all_confidences.update(confidences)

        # Calculate statistics
        num_mutations = sum(
            1
            for i, (a, b) in enumerate(zip(parent_sequence, current_sequence))
            if a != b
        )

        mutation_rate = (
            num_mutations / len(parent_sequence) if len(parent_sequence) > 0 else 0
        )

        metadata = {
            "parent_sequence": parent_sequence,
            "num_iterations": self.config.num_iterations,
            "mask_positions": list(set(all_mask_positions)),
            "num_mutations": num_mutations,
            "mutation_rate": mutation_rate,
            "avg_confidence": (
                np.mean(list(all_confidences.values())) if all_confidences else 0.0
            ),
            "mask_strategy": self.config.mask_strategy,
            "mask_fraction": self.config.mask_fraction,
        }

        return current_sequence, metadata

    async def generate_variants(
        self, parent_sequence: str, num_variants: int = 100, deduplicate: bool = True
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Generate multiple variants through remasking (async, concurrent).

        Uses asyncio.gather with a semaphore to generate variants concurrently
        rather than sequentially, significantly reducing wall-clock time when
        the API client supports concurrent calls.

        Args:
            parent_sequence: Starting sequence
            num_variants: Number of variants to generate
            deduplicate: Remove duplicate sequences (parent excluded when True)

        Returns:
            List of (variant_sequence, metadata) tuples
        """
        max_concurrent = min(num_variants, 16)
        semaphore = asyncio.Semaphore(max_concurrent)
        failure_count = 0

        async def _generate_one(attempt_idx: int) -> Optional[tuple[str, dict[str, Any]]]:
            nonlocal failure_count
            async with semaphore:
                try:
                    return await self.generate_variant(parent_sequence, attempt_idx)
                except Exception as e:
                    failure_count += 1
                    logger.warning(
                        "Variant generation attempt %d failed: %s", attempt_idx, e
                    )
                    return None

        # progressive batching — fire exactly what we need and add more
        # only if duplicates or failures force extra attempts.  The old approach
        # always launched num_variants * 3 tasks regardless of how many unique
        # results the first batch produced, wasting up to 2/3 of all API calls.
        #
        # Strategy: each round requests exactly (num_variants - len(variants))
        # tasks so we never over-provision by more than one batch.  A hard cap
        # of num_variants * 3 prevents infinite loops on tiny sequence spaces.
        seen = {parent_sequence} if deduplicate else set()
        variants: list[tuple[str, dict[str, Any]]] = []
        attempt_offset = 0
        hard_cap = num_variants * 3

        while len(variants) < num_variants and attempt_offset < hard_cap:
            needed = num_variants - len(variants)
            this_batch = min(needed, hard_cap - attempt_offset)
            raw_results = await asyncio.gather(
                *[_generate_one(attempt_offset + i) for i in range(this_batch)]
            )
            attempt_offset += this_batch

            for item in raw_results:
                if item is None:
                    continue
                seq, metadata = item
                if deduplicate and seq in seen:
                    continue
                seen.add(seq)
                variants.append((seq, metadata))
                if len(variants) >= num_variants:
                    break

        if len(variants) == 0 and failure_count > 0:
            raise RuntimeError(
                f"All {failure_count} variant generation attempt(s) failed — "
                "check logged warnings above for per-attempt errors. "
                "This is likely a transient API error, not a sequence space problem."
            )

        if len(variants) < num_variants:
            warnings.warn(
                f"Only generated {len(variants)} unique variants (requested {num_variants}). "
                "Sequence space may be too small for the requested count.",
                UserWarning,
                stacklevel=2,
            )

        return variants[:num_variants]

    async def iterative_refinement(
        self,
        sequence: str,
        fitness_function: callable,
        num_iterations: int = 10,
        population_size: int = 20,
        keep_top_k: int = 5,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Perform iterative refinement using remasking and a fitness function (async).

        Args:
            sequence: Starting sequence
            fitness_function: Function that scores sequences (higher is better)
            num_iterations: Number of refinement iterations
            population_size: Number of variants per iteration
            keep_top_k: Number of top sequences to keep per iteration

        Returns:
            List of (sequence, fitness, metadata) tuples for final population
        """
        # validate keep_top_k < population_size before starting
        if keep_top_k is not None and keep_top_k >= population_size:
            raise ValueError(
                f"keep_top_k ({keep_top_k}) must be < population_size ({population_size}). "
                f"A keep_top_k >= population_size would immediately collapse the population to a single sequence."
            )

        current_population = [(sequence, fitness_function(sequence), {})]

        for iteration in range(num_iterations):
            new_variants = []

            for parent_seq, _parent_fitness, _ in current_population:
                variants = await self.generate_variants(
                    parent_seq,
                    num_variants=population_size // len(current_population),
                    deduplicate=True,
                )

                for variant_seq, metadata in variants:
                    fitness = fitness_function(variant_seq)
                    new_variants.append((variant_seq, fitness, metadata))

            all_sequences = current_population + new_variants
            all_sequences.sort(key=lambda x: x[1], reverse=True)
            current_population = all_sequences[:keep_top_k]

            if current_population:
                print(
                    f"Iteration {iteration + 1}: Best fitness = {current_population[0][1]:.3f}"
                )

        return current_population


def create_remasker_from_dict(config_dict: dict[str, Any], **kwargs) -> MLMRemasker:
    """
    Create MLMRemasker from a configuration dictionary.

    Args:
        config_dict: Dictionary with configuration parameters
        **kwargs: Additional arguments for MLMRemasker

    Returns:
        MLMRemasker instance

    Example:
        >>> config = {
        ...     'mask_fraction': 0.2,
        ...     'num_iterations': 5,
        ...     'temperature': 1.0
        ... }
        >>> remasker = create_remasker_from_dict(config, model_name='esm2')
    """
    config = RemaskingConfig(**config_dict)
    return MLMRemasker(config, **kwargs)


# Predefined configurations for common use cases

CONSERVATIVE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.10,
    num_iterations=3,
    temperature=0.5,
    mask_strategy="random",
)

MODERATE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.15,
    num_iterations=5,
    temperature=1.0,
    mask_strategy="random",
)

AGGRESSIVE_CONFIG = RemaskingConfig(
    model_name="esm-150m",
    mask_fraction=0.25,
    num_iterations=10,
    temperature=1.5,
    mask_strategy="random",
)

BLOCK_MASKING_CONFIG = RemaskingConfig(
    mask_fraction=0.15,
    num_iterations=5,
    temperature=1.0,
    mask_strategy="blocks",
    block_size=3,
)
