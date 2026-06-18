"""
Generative pipeline for sequence generation using language models.

Supports:
- Masked language models (ESM, ESM-1v) with remasking
- Inherently generative models (ProteinMPNN, ProGen2, etc.)
- Temperature and sampling parameter scanning
- Multi-model generation in parallel
- DMS-style scanning: SaturationMutagenesisConfig (single-mutant library + scoring)
  and IterativeMaskingDMSConfig (greedy MLM argmax 2-point DMS)
- Structured config hierarchy: ScoringProtocolConfig / GenerativeProtocolConfig base classes
"""

from __future__ import annotations

import asyncio
import itertools as _itertools
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Module-level counter for unique temporary DuckDB table names
_GENERATIVE_TEMP_COUNTER = _itertools.count()

# Allowlists for getattr() dispatch — tightly scoped per config type so that
# user-supplied action strings cannot resolve to arbitrary BioLMApiClient attrs.
_ALLOWED_SCORING_ACTIONS: frozenset = frozenset({"predict", "score"})
_ALLOWED_MLM_ACTIONS: frozenset = frozenset({"predict"})  # IterativeMaskingDMS needs logits

# Reserved column names that must not be used as a score_field key, because they
# would silently overwrite core sequence/provenance data in the output row dict.
# Includes both core pipeline columns and the provenance columns written by
# _run_saturation_mutagenesis itself (sat_position, sat_wt_aa, sat_mut_aa).
_RESERVED_SCORE_COLUMNS: frozenset = frozenset(
    {
        "sequence", "model_name", "temperature", "sampling_params",
        "generation_method", "parent_sequence", "source_label",
        "sequence_id", "length", "hash",
        "sat_position", "sat_wt_aa", "sat_mut_aa",
        "dms_round", "dms_pos1", "dms_aa1", "dms_pos2", "dms_aa2",
    }
)

import pandas as pd

from biolm.client import BioLMApiClient  # Use async client
from biolm.pipeline.base import BasePipeline, Stage, StageResult, WorkingSet
from biolm.pipeline.datastore_duckdb import DuckDBDataStore as DataStore
from biolm.pipeline.mlm_remasking import MLMRemasker, RemaskingConfig


class ScoringProtocolConfig:
    """Marker base class for stages that score sequences and return a ranked subset.

    Subclasses call a BioLM prediction model to assign a numeric score to each
    variant sequence enumerated by the config (e.g. all single-point mutants),
    then sort and return the top-N results.  They do **not** produce wholly new
    sequences — the output is a scored, ranked subset of the candidate library
    built internally by the config (no separate generation step is needed).

    The ``scoring_action`` field on concrete subclasses selects the client
    method: ``'predict'`` calls :py:meth:`BioLMApiClient.predict` and ``'score'``
    calls :py:meth:`BioLMApiClient.score`.  Both are allowlisted
    (``'predict'``, ``'score'``) to prevent arbitrary method dispatch.

    Use :func:`isinstance(config, ScoringProtocolConfig)` to branch on config
    type in pipeline dispatch functions.

    Example::

        from biolmai.pipeline.generative import (
            ScoringProtocolConfig,
            SaturationMutagenesisConfig,
        )

        def dispatch(config):
            if isinstance(config, ScoringProtocolConfig):
                return run_scoring_stage(config)  # returns ranked variant library
            return run_generative_stage(config)   # returns novel sequences

        config = SaturationMutagenesisConfig(
            parent_sequence="MKTAYIAKQRQ",
            scoring_model="thermompnn-d",
            score_field="ddg",
        )
        assert isinstance(config, ScoringProtocolConfig)  # True
    """


class GenerativeProtocolConfig:
    """Marker base class for stages that produce new amino-acid sequences.

    Subclasses drive generative or masked-language models to emit novel
    sequences — either via autoregressive sampling (ProteinMPNN, DSM,
    AntiFold) or by greedy-argmax masking over an MLM (ESM2, ESMC).
    Unlike :class:`ScoringProtocolConfig` subclasses, these stages expand the
    candidate pool rather than filtering it; they are the *source* stage in a
    generative design pipeline.

    Use :func:`isinstance(config, GenerativeProtocolConfig)` to branch on
    config type in pipeline dispatch functions.

    Example::

        from biolmai.pipeline.generative import (
            GenerativeProtocolConfig,
            DirectGenerationConfig,
            IterativeMaskingDMSConfig,
        )

        def dispatch(config):
            if isinstance(config, GenerativeProtocolConfig):
                return run_generative_stage(config)  # emits new sequences
            return run_scoring_stage(config)          # returns ranked variants

        cfg_direct = DirectGenerationConfig(
            "dsm-150m-base", sequence="MKTAYIAKQRQ", num_sequences=50
        )
        cfg_dms = IterativeMaskingDMSConfig(
            parent_sequence="MKTAYIAKQRQ", model_name="esm2-650m"
        )
        assert isinstance(cfg_direct, GenerativeProtocolConfig)  # True
        assert isinstance(cfg_dms, GenerativeProtocolConfig)     # True
    """


@dataclass
class DirectGenerationConfig(GenerativeProtocolConfig):
    """
    Configuration for structure- or sequence-conditioned generation.

    Use with models such as ProteinMPNN, AntiFold, HyperMPNN, LigandMPNN, DSM.

    The caller is responsible for providing the correct ``item_field`` and
    ``params`` for the target model — these vary per model and are documented in
    the BioLM API schema (``/schema/<model>/generate/``).  Common values:

    +------------------------+-------------+-------------------------------------------+
    | Model                  | item_field  | Key params                                |
    +========================+=============+===========================================+
    | protein-mpnn, hyper-   | ``'pdb'``   | ``batch_size``, ``temperature``           |
    | mpnn, soluble-mpnn,    |             |                                           |
    | ligand-mpnn,           |             |                                           |
    | global-label-membrane- |             |                                           |
    | mpnn                   |             |                                           |
    +------------------------+-------------+-------------------------------------------+
    | antifold               | ``'pdb'``   | ``heavy_chain``, ``light_chain``,         |
    |                        |             | ``num_seq_per_target``, ``sampling_temp`` |
    +------------------------+-------------+-------------------------------------------+
    | dsm-150m-base, dsm-    | ``'sequence'``| ``num_sequences``, ``temperature``,     |
    | 650m-base              |             | ``remasking`` (``'random'``/``'low_confidence'``/``'high_confidence'``), |
    |                        |             | ``step_divisor`` (int; lower = more masks per step = higher edit distance from WT) |
    +------------------------+-------------+-------------------------------------------+

    Args:
        model_name: BioLM model slug (e.g. ``'protein-mpnn'``, ``'antifold'``,
            ``'dsm-150m-base'``).
        structure_path: Path to a PDB or CIF file.
        structure_column: DataFrame column holding PDB strings (for chained
            pipelines where structure was predicted upstream).
        sequence: Parent sequence string for sequence-conditioned models (DSM).
        item_field: The item dict key expected by the model API — ``'pdb'`` for
            structure-conditioned models, ``'sequence'`` for DSM.  Defaults to
            ``'pdb'``.
        params: **Model-specific params dict (required for non-trivial calls).**
            Keys must exactly match the model's API param names.  When empty the
            stage sends ``{'num_sequences': num_sequences, 'temperature': temperature}``
            as a simple fallback — useful only for models that accept these exact
            param names.
        num_sequences: Fallback when ``params`` is empty (default 100).
        temperature: Fallback when ``params`` is empty (default 1.0).

    Example::

        from biolmai.pipeline.generative import DirectGenerationConfig, GenerativePipeline

        # Structure-conditioned design with ProteinMPNN
        cfg_mpnn = DirectGenerationConfig(
            model_name="protein-mpnn",
            structure_path="/path/to/protein.pdb",
            item_field="pdb",
            params={"num_sequences": 50, "temperature": 0.1},
        )

        # Sequence-conditioned design with DSM
        cfg_dsm = DirectGenerationConfig(
            model_name="dsm-150m-base",
            sequence="MKTAYIAKQRQ",
            item_field="sequence",
            params={
                "num_sequences": 100,
                "temperature": 1.0,
                "remasking": "low_confidence",
                "step_divisor": 8,
            },
        )

        pipeline = GenerativePipeline(configs=[cfg_mpnn])
        results = pipeline.run()
    """

    model_name: str
    structure_path: Optional[str] = None
    structure_column: Optional[str] = None
    sequence: Optional[str] = None
    item_field: str = "pdb"
    params: dict[str, Any] = field(default_factory=dict)
    # Simple fallbacks used only when params is empty
    num_sequences: int = 100
    temperature: float = 1.0
    # Read structures from the DuckDB structures table (set by an upstream stage)
    structure_from_stage: Optional[str] = None
    structure_from_model: Optional[str] = None
    # Run the same generation call n_runs times in parallel.
    # Total output = (sequences per call) × n_runs, then deduped.
    n_runs: int = 1
    # Optional human-readable label stored in generation_metadata and surfaced
    # as the ``source_label`` column in results().  Useful when running many
    # configs in one pipeline and needing to distinguish e.g. region or method.
    label: Optional[str] = None


    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        return {
            "type": "DirectGenerationConfig",
            "model_name": self.model_name,
            "structure_path": self.structure_path,
            "structure_column": self.structure_column,
            "sequence": self.sequence,
            "item_field": self.item_field,
            "params": self.params,
            "num_sequences": self.num_sequences,
            "temperature": self.temperature,
            "structure_from_stage": self.structure_from_stage,
            "structure_from_model": self.structure_from_model,
            "n_runs": self.n_runs,
            "label": self.label,
        }


@dataclass
class SequenceSourceConfig:
    """Use existing sequences as the generation-slot source — no API calls made.

    Plug into ``set_generation()`` (or use ``pipeline.use_sequences()``) to
    feed existing data through prediction/filter stages without generating new
    sequences.  The provided sequences are added to the DuckDB via the normal
    dedup path, so sequences already present just return their existing IDs.

    Args:
        sequences: Source of sequences — one of:

            * ``list[str]``: plain amino-acid strings
            * ``pd.DataFrame``: must contain *column* (default ``"sequence"``)
            * ``str`` / ``Path``: path to a CSV or FASTA (``.fasta``/``.fa``) file
            * ``None``: reload all sequences already in the DuckDB (requires
              ``from_db=True`` OR leaving *sequences* as None)

        column: Column name when *sequences* is a DataFrame or CSV
            (default ``"sequence"``).
        from_db: Pull all sequences already present in the pipeline's DuckDB
            instead of loading new ones.  Equivalent to ``sequences=None``.

    Example::

        # Inject a list
        pipeline.use_sequences(["MKTAY", "MKLLIV"]).run()

        # Use all sequences already in the DB (e.g. after from_db() recovery)
        pipeline.use_sequences(from_db=True).run()

        # Load from CSV
        pipeline.use_sequences("candidates.csv").run()
    """

    sequences: Optional[Union[list, "pd.DataFrame", str, Path]] = None
    column: str = "sequence"
    from_db: bool = False

    def to_spec(self) -> dict:
        """Serialize for pipeline definition persistence.

        Live DataFrames are not serializable — on reconstruct we fall back to
        ``from_db=True`` so the existing DB sequences are reused.  Plain
        ``list[str]`` sequences are included directly so from_db() reconstruction
        can replay the same input without requiring an existing DB (GEN-09 fix).
        """
        # GEN-09: include list[str] sequences when serializable
        if isinstance(self.sequences, list):
            return {
                "type": "SequenceSourceConfig",
                "sequences": self.sequences,
                "column": self.column,
                "from_db": False,
            }
        # DataFrames, Path objects, and None all fall back to from_db=True
        return {
            "type": "SequenceSourceConfig",
            "from_db": True,
            "column": self.column,
        }


@dataclass(frozen=True)
class SaturationMutagenesisConfig(ScoringProtocolConfig):
    """Source config that generates a single-mutant library and filters by a prediction model.

    Enumerates every single amino-acid substitution at the specified positions,
    scores each variant with *scoring_model* using *scoring_action*, then returns
    the top-*top_n* variants ranked by *score_field*.

    Typical use: ThermoMPNN-D or ESM2StabP-guided design — predict ΔΔG for all
    single-point mutants, keep the most stabilising ones.

    Args:
        parent_sequence: Wild-type sequence to mutate.
        scoring_model: BioLM model slug used to score variants (e.g. ``'thermompnn-d'``).
        positions: 0-indexed positions to enumerate.  If ``None``, all positions
            in the sequence are enumerated.
        alphabet: Amino acids to substitute.  Defaults to the 20 canonical AAs.
        scoring_action: API action for the scoring model (default ``'predict'``).
        scoring_params: Extra params forwarded to the scoring model API.
        score_field: Key inside each model response that holds the numeric score
            (default ``'ddg'``).  Supports nested access with ``'.'`` separator,
            e.g. ``'result.ddg'``.
        top_n: Number of top-scoring variants to retain (default 50).  ``None``
            keeps all variants that receive a valid score.
        ascending: If ``True``, lower scores are better (e.g. negative ΔΔG means
            stabilising).  Defaults to ``True``.
        exclude_synonymous: If ``True`` (default), skip substitutions that are
            identical to the wild-type residue.
        batch_size: Sequences per API request when scoring (default 8).
        label: Optional label stored as ``source_label`` in results.
        pdb_str: Raw PDB file contents as a string (not a file path).  When
            provided, each scoring item is built as
            ``{"pdb": pdb_str, "mutations": ["<WT><pos+1><MUT>"], "chain": chain}``
            instead of ``{"sequence": mutant_sequence}``.  Required by
            structure-aware models such as ThermoMPNN-D.  Pass ``None`` (default)
            for sequence-only models like ESM2StabP.
        chain: Chain identifier forwarded in structure-aware scoring items
            (default ``'A'``).

    Example::

        from biolmai.pipeline.generative import SaturationMutagenesisConfig, GenerativePipeline

        # Score all single-mutant variants at three positions with ThermoMPNN-D
        config = SaturationMutagenesisConfig(
            parent_sequence="MKTAYIAKQRQ",
            scoring_model="thermompnn-d",
            positions=[3, 7, 10],          # 0-indexed; None → all positions
            score_field="ddg",             # key in API response holding ΔΔG
            top_n=25,                      # keep top-25 most stabilising variants
            ascending=True,                # lower ΔΔG = more stabilising
            pdb_str=open("protein.pdb").read(),  # required by structure-aware models
        )

        pipeline = GenerativePipeline(configs=[config])
        pipeline.add_prediction("esmfold", extractions="mean_plddt", columns="plddt")
        results = pipeline.run()
    """

    parent_sequence: str
    scoring_model: str
    positions: Optional[list[int]] = None
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    scoring_action: str = "predict"
    scoring_params: dict[str, Any] = field(default_factory=dict)
    score_field: str = "ddg"
    top_n: Optional[int] = 50
    ascending: bool = True
    exclude_synonymous: bool = True
    batch_size: int = 8
    label: Optional[str] = None
    pdb_str: Optional[str] = None
    chain: str = "A"

    def __post_init__(self):
        import re as _re
        if not self.parent_sequence:
            raise ValueError(
                "SaturationMutagenesisConfig.parent_sequence cannot be empty"
            )
        if self.scoring_action not in _ALLOWED_SCORING_ACTIONS:
            raise ValueError(
                f"SaturationMutagenesisConfig.scoring_action must be one of "
                f"'predict' or 'score' (got {self.scoring_action!r}). "
                f"'predict' calls api.predict() and expects a response dict; "
                f"'score' calls api.score() and also expects a response dict."
            )
        _score_field_pattern = _re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*){0,2}$")
        if not _score_field_pattern.match(self.score_field):
            raise ValueError(
                f"SaturationMutagenesisConfig.score_field must match "
                f"'identifier' or 'identifier.identifier' (max 3 components), "
                f"got {self.score_field!r}"
            )
        _leaf = self.score_field.replace(".", "_")
        if _leaf in _RESERVED_SCORE_COLUMNS:
            raise ValueError(
                f"SaturationMutagenesisConfig.score_field resolves to reserved "
                f"column name {_leaf!r}. Choose a different field name."
            )
        if self.positions is not None:
            if len(self.positions) == 0:
                raise ValueError(
                    "SaturationMutagenesisConfig.positions must be None (scan all "
                    "positions) or a non-empty list of position indices"
                )
            _bad = [p for p in self.positions if not (0 <= p < len(self.parent_sequence))]
            if _bad:
                raise ValueError(
                    f"SaturationMutagenesisConfig.positions contains out-of-range "
                    f"indices {_bad} for parent_sequence of length "
                    f"{len(self.parent_sequence)}. All positions must be in "
                    f"[0, {len(self.parent_sequence) - 1}]."
                )

    def to_spec(self) -> dict:
        return {
            "type": "SaturationMutagenesisConfig",
            "parent_sequence": self.parent_sequence,
            "scoring_model": self.scoring_model,
            "positions": self.positions,
            "alphabet": self.alphabet,
            "scoring_action": self.scoring_action,
            "scoring_params": self.scoring_params,
            "score_field": self.score_field,
            "top_n": self.top_n,
            "ascending": self.ascending,
            "exclude_synonymous": self.exclude_synonymous,
            "batch_size": self.batch_size,
            "label": self.label,
            "pdb_str": self.pdb_str,
            "chain": self.chain,
        }


@dataclass(frozen=True)
class IterativeMaskingDMSConfig(GenerativeProtocolConfig):
    """Source config that builds multi-point variants via sequential greedy masking.

    Implements an iterative argmax masking procedure using a masked language model:

    1. For each target position, mask it in the parent sequence and query the model
       for the highest-probability residue (greedy argmax — not sampled).
    2. If ``rounds > 1``, apply the round-1 preferred substitution at each position
       and repeat for round 2: mask each *other* target position in the round-1
       sequence and collect the argmax residue.
    3. Yield all resulting sequences as pipeline outputs.

    This matches the ESM2 two-round DMS design pattern in the EGF generation notebook.

    Args:
        parent_sequence: Starting sequence.
        model_name: MLM model slug (e.g. ``'esm2-650m'``, ``'esmc-300m'``).
        positions: 0-indexed positions to probe.  Defaults to all positions.
        rounds: Number of sequential masking rounds (default 2).  Round N uses
            the variant produced by round N-1 as its starting sequence.
        mask_token: Token inserted at masked positions (default ``'<mask>'``).
        alphabet: Vocabulary used to identify valid AA positions in logits.
            Defaults to the 20 canonical amino acids.
        exclude_synonymous: Skip round-1 variants where the argmax matches WT
            (default ``True``).
        batch_size: Sequences per API request (default 32).
        label: Optional label stored as ``source_label`` in results.
        action: API action for the model (default ``'predict'``; the model must
            return logits).  Only ``'predict'`` is allowed because the
            greedy-argmax procedure requires per-token logit arrays.

    Example::

        from biolmai.pipeline.generative import IterativeMaskingDMSConfig, GenerativePipeline

        # Two-round greedy DMS at three positions using ESM2-650M
        config = IterativeMaskingDMSConfig(
            parent_sequence="MKTAYIAKQRQ",
            model_name="esm2-650m",
            positions=[2, 5, 8],   # 0-indexed; None → all positions
            rounds=2,              # round 1: per-position; round 2: 2-point combos
            exclude_synonymous=True,
        )

        pipeline = GenerativePipeline(configs=[config])
        results = pipeline.run()
        # results DataFrame has columns:
        #   sequence, dms_round, dms_pos1, dms_aa1, dms_pos2, dms_aa2
    """

    parent_sequence: str
    model_name: str
    positions: Optional[list[int]] = None
    rounds: int = 2
    mask_token: str = "<mask>"
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY"
    exclude_synonymous: bool = True
    batch_size: int = 32
    label: Optional[str] = None
    action: str = "predict"

    def __post_init__(self):
        if not self.parent_sequence:
            raise ValueError(
                "IterativeMaskingDMSConfig.parent_sequence cannot be empty"
            )
        if self.action not in _ALLOWED_MLM_ACTIONS:
            raise ValueError(
                f"IterativeMaskingDMSConfig.action must be one of: 'predict' "
                f"(got {self.action!r}). Only 'predict' is supported because "
                f"the greedy-argmax procedure requires per-token logit arrays "
                f"returned by the model's predict endpoint."
            )
        if self.rounds < 1:
            raise ValueError(f"IterativeMaskingDMSConfig.rounds must be >= 1, got {self.rounds}")
        if self.rounds > 2:
            raise ValueError(
                f"IterativeMaskingDMSConfig.rounds > 2 is not yet supported "
                f"(got {self.rounds}). Only rounds=1 (single-point) and "
                f"rounds=2 (two-point DMS) are implemented."
            )
        if self.positions is not None:
            _bad = [p for p in self.positions if not (0 <= p < len(self.parent_sequence))]
            if _bad:
                raise ValueError(
                    f"IterativeMaskingDMSConfig.positions contains out-of-range "
                    f"indices {_bad} for parent_sequence of length "
                    f"{len(self.parent_sequence)}."
                )

    def to_spec(self) -> dict:
        return {
            "type": "IterativeMaskingDMSConfig",
            "parent_sequence": self.parent_sequence,
            "model_name": self.model_name,
            "positions": self.positions,
            "rounds": self.rounds,
            "mask_token": self.mask_token,
            "alphabet": self.alphabet,
            "exclude_synonymous": self.exclude_synonymous,
            "batch_size": self.batch_size,
            "label": self.label,
            "action": self.action,
        }


@dataclass
class FoldingEntity:
    """
    A molecular entity for co-folding models (Boltz2, Chai-1).

    Used with :meth:`DataPipeline.add_cofolding_prediction` and
    :meth:`GenerativePipeline.add_cofolding_prediction` to inject static
    entities — ligands, cofactors, DNA/RNA strands, or additional protein
    chains — alongside the pipeline's primary sequences.

    Args:
        id: Chain identifier (Boltz uses single letter(s); Chai-1 uses a name).
        entity_type: Molecule type — ``'protein'``, ``'dna'``, ``'rna'``, or
            ``'ligand'``.
        sequence: Amino-acid / nucleotide sequence for protein / DNA / RNA entities.
        smiles: SMILES string for small-molecule ligands.
        ccd: CCD code for ligands defined in the Chemical Component Dictionary
            (e.g. ``'ATP'``, ``'HEM'``).
    """

    id: str
    entity_type: str  # 'protein' | 'dna' | 'rna' | 'ligand'
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd: Optional[str] = None


@dataclass
class GenerationConfig:
    """
    Configuration for sequence generation.

    .. deprecated::
        Use :class:`RemaskingConfig` for MLM-based generation or
        :class:`DirectGenerationConfig` for structure-conditioned models.

    Args:
        model_name: BioLM model name
        num_sequences: Number of sequences to generate
        temperature: Temperature or list of temperatures to scan
        sampling_params: Additional sampling parameters
        generation_method: 'generate' or 'remask' (for MLMs)
        parent_sequence: Parent sequence (for remasking or conditioning)
        mask_positions: Positions to mask (for remasking), or 'auto' for automatic
        mask_fraction: Fraction of positions to mask (if mask_positions='auto')
        batch_size: Batch size for generation
    """

    model_name: str
    num_sequences: int = 100
    temperature: Union[float, list[float]] = 1.0
    sampling_params: dict[str, Any] = field(default_factory=dict)
    generation_method: str = "generate"  # 'generate' or 'remask'
    parent_sequence: Optional[str] = None
    mask_positions: Union[str, list[int]] = "auto"
    mask_fraction: float = 0.15
    batch_size: int = 32

    def __post_init__(self):
        warnings.warn(
            "GenerationConfig is deprecated. Use RemaskingConfig for MLM remasking "
            "or DirectGenerationConfig for structure-conditioned generation.",
            DeprecationWarning,
            stacklevel=2,
        )


class GenerationStage(Stage):
    """
    Stage for generating sequences using generative models.

    Accepts either the new typed configs (RemaskingConfig / DirectGenerationConfig)
    or the legacy GenerationConfig list.

    For ``DirectGenerationConfig`` the caller supplies the correct ``item_field``
    (e.g. ``'pdb'`` or ``'sequence'``) and ``params`` dict with model-specific param
    names — no auto-detection is performed.

    The stage handles the three main response formats returned by BioLM generation
    models:

    * **Flat list** (MPNN): ``[{sequence, pdb, ...}, ...]``
    * **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
    * **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``

    Args:
        name: Stage name.
        config: Single RemaskingConfig or DirectGenerationConfig (new API).
        configs: List of config objects — GenerationConfig, RemaskingConfig, or
                 DirectGenerationConfig (old / multi-model API).
        deduplicate: Whether to deduplicate generated sequences.
    """

    def __init__(
        self,
        name: str = "generation",
        config: Optional[Union[RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"]] = None,
        configs: Optional[
            list[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"]]
        ] = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if config is not None:
            self.configs = [config]
        elif configs is not None:
            self.configs = configs
        else:
            self.configs = []
        self.deduplicate = deduplicate

    def to_spec(self) -> dict:
        """Return a serializable dict for pipeline definition persistence."""
        configs_specs = []
        for cfg in self.configs:
            if hasattr(cfg, "to_spec"):
                configs_specs.append(cfg.to_spec())
            else:
                raise NotImplementedError(
                    f"GenerationStage '{self.name}': config type "
                    f"'{type(cfg).__name__}' does not implement to_spec(). "
                    "Only RemaskingConfig and DirectGenerationConfig are serializable. "
                    "Migrate away from GenerationConfig (deprecated) to enable from_db() recovery."
                )
        return {
            "type": "GenerationStage",
            "name": self.name,
            "configs": configs_specs,
            "deduplicate": self.deduplicate,
            "depends_on": self.depends_on,
        }

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sequences(raw: Any) -> list[dict]:
        """
        Extract sequence dicts from any BioLM generation response format.

        Handles:
        - **Single dict** (MPNN with unwrap_single=True): ``{sequence, pdb, ...}``
        - **Flat list** (MPNN, multiple results): ``[{sequence, pdb, ...}, ...]``
        - **Nested samples** (AntiFold): ``[{sequences: [{heavy, light, ...}]}]``
        - **Double-nested** (DSM): ``[[{sequence, log_prob, ...}, ...]]``
        """
        # Normalize: BioLMApiClient with unwrap_single=True returns a bare dict
        # for single-item inputs. Wrap so the rest of the logic sees a uniform list.
        if isinstance(raw, dict):
            raw = [raw]
        if not raw:
            return []
        sequences: list[dict] = []
        for item in raw:
            if isinstance(item, list):
                # DSM-style: inner list = num_sequences per input item
                for sub in item:
                    if isinstance(sub, dict):
                        seq = sub.get("sequence", "")
                        if seq:
                            sequences.append(
                                {
                                    "sequence": seq,
                                    "log_prob": sub.get("log_prob"),
                                    "perplexity": sub.get("perplexity"),
                                }
                            )
                        else:
                            # log when a DSM sub-item has no sequence
                            logger.warning(
                                "Skipping generation result item with missing or empty "
                                "'sequence': %s",
                                str(sub)[:200],
                            )
            elif isinstance(item, dict):
                sequences_list = item.get("sequences", [])
                # GEN2-03: narrow AntiFold detection — require the first element
                # to be a dict with 'heavy' or 'light' key so that any other model
                # using a "sequences" key is not accidentally routed here.
                is_antifold = (
                    isinstance(sequences_list, list)
                    and len(sequences_list) > 0
                    and isinstance(sequences_list[0], dict)
                    and ("heavy" in sequences_list[0] or "light" in sequences_list[0])
                )
                if is_antifold:
                    # AntiFold-style: each item has a 'sequences' list of samples
                    for sample in sequences_list:
                        if isinstance(sample, dict):
                            heavy = sample.get("heavy")
                            light = sample.get("light")
                            if heavy and light:
                                seq = f"{heavy}:{light}"
                            elif heavy:
                                seq = heavy
                            else:
                                seq = sample.get("sequence", "")
                            if seq:
                                sequences.append(
                                    {
                                        "sequence": seq,
                                        "heavy_chain": heavy,
                                        "light_chain": light,
                                        "score": sample.get("score"),
                                        "global_score": sample.get("global_score"),
                                        "seq_recovery": sample.get("seq_recovery"),
                                    }
                                )
                            else:
                                # log when an AntiFold sample has no sequence
                                logger.warning(
                                    "Skipping generation result item with missing or empty "
                                    "'sequence': %s",
                                    str(sample)[:200],
                                )
                else:
                    # MPNN / flat dict
                    seq = item.get("sequence", "")
                    if seq:
                        sequences.append({"sequence": seq})
                    else:
                        # log when a flat-dict item has no sequence
                        logger.warning(
                            "Skipping generation result item with missing or empty "
                            "'sequence': %s",
                            str(item)[:200],
                        )
        return sequences

    @staticmethod
    def _create_masked_sequence(
        sequence: str,
        mask_positions: Union[str, list[int]],
        mask_fraction: float = 0.15,
        mask_token: str = "<mask>",
    ) -> tuple[str, list[int]]:
        """
        Create a masked version of a sequence.

        Args:
            sequence: Original sequence
            mask_positions: List of positions to mask, or 'auto' for random masking
            mask_fraction: Fraction of positions to mask (if mask_positions='auto')
            mask_token: Token to use for masking

        Returns:
            Tuple of (masked_sequence, masked_positions)
        """
        seq_list = list(sequence)

        if mask_positions == "auto":
            # Randomly select positions to mask
            num_to_mask = max(1, int(len(sequence) * mask_fraction))
            positions = random.sample(range(len(sequence)), num_to_mask)
        else:
            positions = mask_positions

        # Apply masking
        for pos in positions:
            if 0 <= pos < len(seq_list):
                seq_list[pos] = mask_token

        masked_seq = "".join(seq_list)
        return masked_seq, sorted(positions)

    async def _run_remasking(
        self,
        config: RemaskingConfig,
        parent_sequence: str,
        num_variants: int,
    ) -> list[dict]:
        """Run MLM remasking generation using RemaskingConfig directly."""
        print(
            f"  {config.model_name} (remasking): generating {num_variants} variants..."
        )
        api = None
        try:
            api = BioLMApiClient(config.model_name)
            remasker = MLMRemasker(config, api_client=api, model_name=config.model_name)
            variants = await remasker.generate_variants(
                parent_sequence,
                num_variants=num_variants,
                deduplicate=True,
            )
            return [
                {
                    "sequence": seq,
                    "model_name": config.model_name,
                    "temperature": config.temperature,
                    "sampling_params": {},
                    "generation_method": "remask",
                    "parent_sequence": parent_sequence,
                    "num_mutations": meta.get("num_mutations"),
                    "mutation_rate": meta.get("mutation_rate"),
                }
                for seq, meta in variants
            ]
        finally:
            if api:
                await api.shutdown()

    async def _run_direct_generation(
        self,
        config: DirectGenerationConfig,
        datastore: DataStore,
        df_input: pd.DataFrame,
        context=None,
        ws_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Run generation using DirectGenerationConfig.

        The caller is responsible for setting the correct ``item_field`` and
        ``params`` for the target model.  No auto-detection is performed.

        Args:
            ws_ids: Optional list of sequence IDs from the current working set.
                When provided, structure lookups are scoped to only these IDs to
                prevent cross-run contamination (GEN-05 fix).
        """
        # ---- Resolve input values ----------------------------------------
        # warn clearly when structure_from_stage is set without
        # structure_from_model so the user knows all structures will be used.
        if config.structure_from_stage and not config.structure_from_model:
            warnings.warn(
                f"DirectGenerationConfig has 'structure_from_stage' set but "
                f"'structure_from_model' is not set. All structures from all upstream "
                f"models will be used, which may cause incorrect protein designs if "
                f"multiple structure-prediction stages have run. Set "
                f"structure_from_model='esmfold' (or your model name) to select the "
                f"correct structures.",
                UserWarning,
                stacklevel=3,
            )

        if config.structure_from_stage or config.structure_from_model:
            # GEN-05 fix: scope structure lookup to current working set sequence_ids.
            # Collect candidate IDs: prefer ws_ids, fall back to df_input sequence_ids,
            # then fall back to all sequences (unscoped).
            if ws_ids is not None:
                candidate_ws_ids = ws_ids
            elif not df_input.empty and "sequence_id" in df_input.columns:
                candidate_ws_ids = df_input["sequence_id"].tolist()
            else:
                candidate_ws_ids = None  # no scope available — use unscoped query

            # Read structures from DuckDB (set by an upstream prediction stage)
            model_filter = config.structure_from_model
            if candidate_ws_ids is not None:
                # Scoped query: only structures for sequences in the current WS
                ws_ids_df = pd.DataFrame({"sequence_id": candidate_ws_ids})
                _tmp_ws = f"_direct_gen_ws_{next(_GENERATIVE_TEMP_COUNTER)}"
                datastore.conn.register(_tmp_ws, ws_ids_df)
                try:
                    if model_filter:
                        seq_ids_rows = datastore.conn.execute(
                            f"SELECT st.sequence_id FROM structures st "
                            f"INNER JOIN {_tmp_ws} w ON st.sequence_id = w.sequence_id "
                            f"WHERE st.model_name = ?",
                            [model_filter],
                        ).fetchall()
                    else:
                        seq_ids_rows = datastore.conn.execute(
                            f"SELECT st.sequence_id FROM structures st "
                            f"INNER JOIN {_tmp_ws} w ON st.sequence_id = w.sequence_id"
                        ).fetchall()
                finally:
                    datastore.conn.unregister(_tmp_ws)
            elif model_filter:
                seq_ids_rows = datastore.conn.execute(
                    "SELECT sequence_id FROM structures WHERE model_name = ?",
                    [model_filter],
                ).fetchall()
            else:
                seq_ids_rows = datastore.conn.execute(
                    "SELECT sequence_id FROM structures"
                ).fetchall()
            struct_seq_ids = [r[0] for r in seq_ids_rows]
            if not struct_seq_ids:
                raise ValueError(
                    f"No structures found in datastore"
                    f"{' for model ' + model_filter if model_filter else ''}"
                )
            structs_df = datastore.get_structures_bulk(struct_seq_ids)
            if structs_df.empty:
                raise ValueError(
                    f"No structures found in datastore"
                    f"{' for model ' + model_filter if model_filter else ''}"
                )
            input_values = structs_df["structure_str"].dropna().tolist()
        elif config.structure_column and config.structure_column in df_input.columns:
            input_values = df_input[config.structure_column].dropna().tolist()
        elif config.structure_path:
            import tempfile

            from biolm.pipeline.utils import cif_to_pdb, load_structure_string

            fmt, structure_str = load_structure_string(config.structure_path)
            if fmt == "cif":
                import os

                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                    pdb_path = f.name
                try:
                    cif_to_pdb(config.structure_path, pdb_path)
                    _, structure_str = load_structure_string(pdb_path)
                finally:
                    if os.path.exists(pdb_path):
                        os.remove(pdb_path)
            input_values = [structure_str]
        elif config.sequence is not None:
            input_values = [config.sequence]
        else:
            raise ValueError(
                "DirectGenerationConfig requires structure_path, structure_column, or sequence"
            )

        # ---- Params: user dict takes priority; fallback to convenience fields
        params: dict[str, Any] = (
            config.params
            if config.params
            else {
                "num_sequences": config.num_sequences,
                "temperature": config.temperature,
            }
        )

        # ---- Generate ----------------------------------------------------
        n_runs = getattr(config, "n_runs", 1) or 1
        results: list[dict] = []
        api = None
        try:
            api = BioLMApiClient(config.model_name)
            for input_value in input_values:
                if n_runs > 1:
                    print(
                        f"  {config.model_name} (direct): generating sequences "
                        f"({n_runs} parallel runs)..."
                    )
                else:
                    print(f"  {config.model_name} (direct): generating sequences...")
                items = [{config.item_field: input_value}]

                # Run generation n_runs times in parallel
                # GEN2-02: each concurrent run gets its own client to avoid
                # sharing a single api instance across concurrent coroutines.
                if n_runs > 1:
                    _model_name = config.model_name

                    async def _single_run(items=items, params=params):
                        _api = BioLMApiClient(_model_name)
                        try:
                            return await _api.generate(items=items, params=params)
                        finally:
                            await _api.shutdown()

                    _all_results = await asyncio.gather(
                        *[_single_run() for _ in range(n_runs)],
                        return_exceptions=True,
                    )
                    # filter out exceptions so one failed run does
                    # not cancel all parallel runs.
                    raw_list = []
                    for _i, _r in enumerate(_all_results):
                        if isinstance(_r, Exception):
                            logger.warning(
                                "Generation run %d/%d failed: %s", _i + 1, n_runs, _r
                            )
                        else:
                            raw_list.append(_r)
                    # All runs failed — surface the first exception rather than
                    # silently returning an empty list with no indication of why.
                    # Mirrors the multi-config all-failures guard in process().
                    if not raw_list:
                        _first_exc = next(
                            r for r in _all_results if isinstance(r, Exception)
                        )
                        raise _first_exc
                else:
                    raw_list = [await api.generate(items=items, params=params)]

                for raw in raw_list:
                    if isinstance(raw, dict) and "error" in raw:
                        raise ValueError(
                            f"API error from {config.model_name}: {raw.get('error')}"
                        )
                    extracted = self._extract_sequences(raw)
                    # GEN2-09: if extraction yielded nothing but raw is non-empty,
                    # check whether the first element is an error dict.
                    if not extracted and isinstance(raw, list) and raw and isinstance(raw[0], dict) and "error" in raw[0]:
                        raise ValueError(f"API returned error: {raw[0]}")
                    for seq_data in extracted:
                        results.append(
                            {
                                "model_name": config.model_name,
                                "temperature": params.get(
                                    "temperature", config.temperature
                                ),
                                "sampling_params": params,
                                "generation_method": "direct",
                                "parent_sequence": config.sequence,
                                "source_label": config.label,
                                **seq_data,
                            }
                        )
        finally:
            if api:
                await api.shutdown()

        return results

    async def _generate_with_config(
        self, config: GenerationConfig, datastore: DataStore
    ) -> list[dict]:
        """
        Generate sequences with a single configuration.

        Returns:
            List of dicts with keys: sequence, model_name, temperature, sampling_params
        """
        results = []

        # Determine if we need temperature scanning
        temps = (
            [config.temperature]
            if isinstance(config.temperature, (int, float))
            else config.temperature
        )

        for temp in temps:
            print(
                f"  {config.model_name} @ T={temp}: generating {config.num_sequences} sequences..."
            )

            # Create async API client
            api = None
            try:
                api = BioLMApiClient(config.model_name)
                if config.generation_method == "remask":
                    # Remasking approach for MLMs
                    if config.parent_sequence is None:
                        raise ValueError("parent_sequence required for remasking")

                    # Create RemaskingConfig from GenerationConfig
                    remask_config = RemaskingConfig(
                        model_name=config.model_name,  # Use model from GenerationConfig
                        mask_fraction=config.mask_fraction,
                        mask_positions=config.mask_positions,
                        num_iterations=config.sampling_params.get("num_iterations", 1),
                        temperature=temp,
                        top_k=config.sampling_params.get("top_k"),
                        top_p=config.sampling_params.get("top_p"),
                        conserved_positions=config.sampling_params.get(
                            "conserved_positions"
                        ),
                        mask_strategy=config.sampling_params.get(
                            "mask_strategy", "random"
                        ),
                    )

                    # Create remasker
                    remasker = MLMRemasker(
                        remask_config, api_client=api, model_name=config.model_name
                    )

                    # Generate variants (now async)
                    variants = await remasker.generate_variants(
                        config.parent_sequence,
                        num_variants=config.num_sequences,
                        deduplicate=True,
                    )

                    # Store results
                    for seq, metadata in variants:
                        results.append(
                            {
                                "sequence": seq,
                                "model_name": config.model_name,
                                "temperature": temp,
                                "sampling_params": config.sampling_params,
                                "generation_method": "remask",
                                "parent_sequence": config.parent_sequence,
                                "num_mutations": metadata.get("num_mutations"),
                                "mutation_rate": metadata.get("mutation_rate"),
                            }
                        )

                else:
                    # Direct generation (for inherently generative models)
                    params = {
                        **config.sampling_params,
                        "temperature": temp,
                        "num_sequences": config.num_sequences,
                    }

                    # Prepare items
                    if config.parent_sequence:
                        items = [{"sequence": config.parent_sequence}]
                    else:
                        items = [{}]  # Unconditional generation

                    # Generate (now async)
                    result = await api.generate(items=items, params=params)

                    # Check for API errors
                    if isinstance(result, dict) and "error" in result:
                        raise ValueError(
                            f"API error from {config.model_name}: {result.get('error')}"
                        )

                    # Extract sequences using the same format-aware extractor
                    # that handles flat/MPNN, AntiFold-nested, and DSM double-nested.
                    for extracted in GenerationStage._extract_sequences(result):
                        results.append(
                            {
                                **extracted,
                                "model_name": config.model_name,
                                "temperature": temp,
                                "sampling_params": config.sampling_params,
                                "generation_method": "generate",
                                "parent_sequence": config.parent_sequence,
                            }
                        )

            finally:
                if api:
                    await api.shutdown()

        return results

    @staticmethod
    async def _run_sequence_source(
        config: SequenceSourceConfig,
        datastore: DataStore,
        run_id: Optional[str] = None,
    ) -> list[dict]:
        """Load sequences from an existing source without calling any API.

        Handles list[str], pd.DataFrame, CSV/FASTA paths, and DB reload.

        When ``config.from_db=True`` and a ``run_id`` is provided, the query is
        scoped to sequences that have a ``generation_metadata`` row for that run
        (i.e. sequences generated in that run), preventing cross-run contamination
        on resume.  If no such rows exist (e.g. sequences were added via
        ``DataPipeline`` rather than generated), all DB sequences are returned.
        """
        sequences: list[str] = []

        if config.from_db or config.sequences is None:
            # GEN-02: scope to current run_id when resuming a GenerativePipeline run.
            # Only scope when this run has generation_metadata rows (i.e. sequences
            # were generated in this run).  Otherwise return all DB sequences to
            # preserve the use case of reloading externally-added sequences (e.g. from
            # a prior DataPipeline run loaded via use_sequences(from_db=True)).
            scoped = False
            if run_id:
                try:
                    count_row = datastore.conn.execute(
                        "SELECT COUNT(*) FROM generation_metadata WHERE run_id = ?",
                        [run_id],
                    ).fetchone()
                    if count_row and count_row[0] > 0:
                        rows = datastore.conn.execute(
                            """
                            SELECT s.sequence
                            FROM sequences s
                            INNER JOIN generation_metadata gm ON s.sequence_id = gm.sequence_id
                            WHERE gm.run_id = ?
                            """,
                            [run_id],
                        ).fetchall()
                        sequences = [r[0] for r in rows if r[0]]
                        scoped = True
                except Exception as _scope_err:
                    logger.warning(
                        "SequenceSourceConfig: failed to scope sequences to run_id %s, "
                        "falling back to all sequences in DB: %s",
                        run_id,
                        _scope_err,
                    )
            if not scoped:
                df_db = datastore.get_all_sequences()
                sequences = df_db["sequence"].dropna().tolist() if not df_db.empty else []

        elif isinstance(config.sequences, list):
            sequences = [str(s) for s in config.sequences if s]

        elif isinstance(config.sequences, pd.DataFrame):
            col = config.column
            if col not in config.sequences.columns:
                raise ValueError(
                    f"SequenceSourceConfig: column '{col}' not found in DataFrame. "
                    f"Available: {list(config.sequences.columns)}"
                )
            sequences = config.sequences[col].dropna().tolist()

        elif isinstance(config.sequences, (str, Path)):
            path = Path(config.sequences)
            if path.suffix.lower() in (".fasta", ".fa", ".faa"):
                # Minimal FASTA parser — no external deps
                seqs: list[str] = []
                current: list[str] = []
                with open(path) as fh:
                    for line in fh:
                        line = line.rstrip()
                        if line.startswith(">"):
                            if current:
                                seqs.append("".join(current))
                                current = []
                        else:
                            current.append(line)
                    if current:
                        seqs.append("".join(current))
                sequences = [s for s in seqs if s]
            else:
                df_csv = pd.read_csv(path)
                col = config.column
                if col not in df_csv.columns:
                    raise ValueError(
                        f"SequenceSourceConfig: column '{col}' not found in '{path}'. "
                        f"Available: {list(df_csv.columns)}"
                    )
                sequences = df_csv[col].dropna().tolist()

        else:
            raise TypeError(
                f"SequenceSourceConfig.sequences must be list, DataFrame, file path, or None. "
                f"Got {type(config.sequences)}"
            )

        return [
            {
                "sequence": s,
                "model_name": "data_source",
                "temperature": None,
                "sampling_params": {},
                "generation_method": "data_source",
                "parent_sequence": None,
            }
            for s in sequences
            if s
        ]

    @staticmethod
    def _get_nested(obj: dict, dotted_key: str):
        """Retrieve a value from a dict using a dotted key path, e.g. 'result.ddg'."""
        parts = dotted_key.split(".")
        cur = obj
        for part in parts:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    async def _run_saturation_mutagenesis(
        self, config: "SaturationMutagenesisConfig"
    ) -> list[dict]:
        """Enumerate single-mutant library, score with a prediction model, keep top-N."""
        import numpy as np

        parent = config.parent_sequence
        positions = config.positions if config.positions is not None else list(range(len(parent)))
        alphabet = list(config.alphabet)

        # Build all single-point mutant sequences
        mutants: list[tuple[str, int, str, str]] = []  # (seq, pos, wt_aa, mut_aa)
        for pos in positions:
            wt_aa = parent[pos] if pos < len(parent) else None
            for aa in alphabet:
                if config.exclude_synonymous and aa == wt_aa:
                    continue
                seq_list = list(parent)
                seq_list[pos] = aa
                mutants.append(("".join(seq_list), pos, wt_aa or "", aa))

        if not mutants:
            logger.warning("SaturationMutagenesisConfig: no mutants generated (empty library)")
            return []

        print(f"  SaturationMutagenesis ({config.scoring_model}): scoring {len(mutants)} variants …")

        seqs, positions_list, wt_aas, mut_aas = zip(*mutants)
        seqs = list(seqs)

        # Build items — structure-aware models (e.g. ThermoMPNN-D) need pdb + mutations list
        def _build_item(seq: str, pos: int, wt_aa: str, mut_aa: str) -> dict:
            item: dict[str, Any] = {}
            if config.pdb_str:
                item["pdb"] = config.pdb_str
                item["mutations"] = [f"{wt_aa}{pos + 1}{mut_aa}"]
            else:
                item["sequence"] = seq
            if config.chain:
                item["chain"] = config.chain
            if config.scoring_params:
                item.update({k: v for k, v in config.scoring_params.items()
                              if k not in ("pdb", "sequence", "mutations", "chain")})
            return item

        items = [
            _build_item(seq, pos, wt, mut)
            for seq, pos, wt, mut in zip(seqs, positions_list, wt_aas, mut_aas)
        ]

        # Score in batches
        scores: list[Optional[float]] = []
        _example_bad_response: Optional[dict] = None
        api = BioLMApiClient(config.scoring_model)
        try:
            action_fn = getattr(api, config.scoring_action)
            _score_field_validated = False
            _preflight_sample: Optional[dict] = None  # first response, captured for validation
            for i in range(0, len(items), config.batch_size):
                batch = items[i:i + config.batch_size]
                try:
                    raw = await action_fn(items=batch, params=config.scoring_params or {})
                    # Normalize to list
                    if isinstance(raw, dict):
                        raw = [raw]
                    if len(raw) != len(batch):
                        raise ValueError(
                            f"SaturationMutagenesis batch {i // config.batch_size}: "
                            f"API returned {len(raw)} results for {len(batch)} items — "
                            "length mismatch would cause positional drift; treating batch as failed"
                        )
                    # Capture only after confirming correct batch length, so a
                    # malformed-batch response never poisons the preflight sample.
                    if not _score_field_validated and raw and isinstance(raw[0], dict):
                        _preflight_sample = raw[0]
                    batch_scores: list = []
                    for r in raw:
                        val = self._get_nested(r, config.score_field) if isinstance(r, dict) else None
                        try:
                            batch_scores.append(float(val) if val is not None else None)
                        except (TypeError, ValueError):
                            batch_scores.append(None)
                        if val is None and _example_bad_response is None and isinstance(r, dict):
                            _example_bad_response = r
                    scores.extend(batch_scores)
                except Exception as e:
                    logger.warning("SaturationMutagenesis scoring batch %d failed: %s", i // config.batch_size, e)
                    scores.extend([None] * len(batch))
                # Preflight validation OUTSIDE per-batch except — propagates to caller.
                # Runs once on the first successful response only.
                if not _score_field_validated and _preflight_sample is not None:
                    _probe_val = self._get_nested(_preflight_sample, config.score_field)
                    if _probe_val is None:
                        raise ValueError(
                            f"SaturationMutagenesisConfig: score_field {config.score_field!r} not found "
                            f"in API response for model {config.scoring_model!r}. "
                            f"Available top-level keys: {list(_preflight_sample.keys())}. "
                            f"Full response sample: {_preflight_sample}"
                        )
                    _score_field_validated = True
        finally:
            await api.shutdown()

        # Build result rows
        rows = []
        for seq, pos, wt_aa, mut_aa, score in zip(seqs, positions_list, wt_aas, mut_aas, scores):
            if score is None:
                continue
            rows.append({
                "sequence": seq,
                "model_name": config.scoring_model,
                "temperature": None,
                "sampling_params": {},
                "generation_method": "saturation_mutagenesis",
                "parent_sequence": parent,
                "source_label": config.label,
                "sat_position": pos,
                "sat_wt_aa": wt_aa,
                "sat_mut_aa": mut_aa,
                config.score_field.replace(".", "_"): score,
            })

        if not rows:
            logger.warning(
                "SaturationMutagenesisConfig: no variants received valid scores "
                "(score_field=%r). Example API response where score extraction returned "
                "None: %r",
                config.score_field,
                _example_bad_response,
            )
            return []

        # Rank and keep top-N
        import operator
        rows.sort(key=lambda r: r.get(config.score_field.replace(".", "_"), 0), reverse=not config.ascending)
        if config.top_n is not None:
            rows = rows[: config.top_n]

        print(f"  SaturationMutagenesis: kept {len(rows)} top-{config.top_n} variants")
        return rows

    async def _run_iterative_masking_dms(
        self, config: "IterativeMaskingDMSConfig"
    ) -> list[dict]:
        """Greedy argmax masking across positions for N sequential rounds."""
        import numpy as np

        parent = config.parent_sequence
        positions = config.positions if config.positions is not None else list(range(len(parent)))
        alphabet = list(config.alphabet)

        def _argmax_from_response(result: Any, pos: int) -> Optional[str]:
            """Extract the argmax amino acid at *pos* from a model response dict."""
            if not isinstance(result, dict):
                return None
            logits = result.get("logits")
            vocab = result.get("vocab_tokens")
            if logits is None:
                return None
            try:
                logits_arr = np.array(logits, dtype=np.float64)
                # Strip BOS/EOS if present (ESMC/ESM3 style)
                if logits_arr.ndim == 2 and logits_arr.shape[0] == len(parent) + 2:
                    logits_arr = logits_arr[1:-1]
                row = logits_arr[pos]
                if vocab:
                    # Map to alphabet indices
                    aa_indices = [vocab.index(aa) for aa in alphabet if aa in vocab]
                    best_idx = aa_indices[int(np.argmax(row[aa_indices]))]
                    return vocab[best_idx]
                else:
                    # Assume first 20 positions correspond to canonical AAs
                    return alphabet[int(np.argmax(row[:20]))]
            except Exception as e:
                logger.warning("DMS argmax extraction failed at pos %d: %s", pos, e)
                return None

        print(f"  IterativeMaskingDMS ({config.model_name}): {config.rounds} round(s) × {len(positions)} positions …")

        # Round-1: for each position, mask it in parent, get argmax AA
        round1_preferred: dict[int, str] = {}  # pos → preferred AA
        api = BioLMApiClient(config.model_name)
        try:
            action_fn = getattr(api, config.action)
            # Batch all round-1 queries together
            items_r1 = []
            for pos in positions:
                seq_list = list(parent)
                seq_list[pos] = config.mask_token
                items_r1.append({"sequence": "".join(seq_list)})

            results_r1: list[Any] = []
            _logit_format_validated = False
            _logit_preflight_sample: Optional[dict] = None
            for i in range(0, len(items_r1), config.batch_size):
                batch = items_r1[i:i + config.batch_size]
                try:
                    raw = await action_fn(items=batch)
                    if isinstance(raw, dict):
                        raw = [raw]
                    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
                        raw = [r[0] if r else {} for r in raw]
                    if not _logit_format_validated and raw and isinstance(raw[0], dict):
                        _logit_preflight_sample = raw[0]  # validated OUTSIDE this try block
                    if len(raw) != len(batch):
                        raise ValueError(
                            f"IterativeMaskingDMS round-1 batch {i // config.batch_size}: "
                            f"API returned {len(raw)} results for {len(batch)} items — "
                            "length mismatch would cause positional drift; treating batch as failed"
                        )
                except Exception as e:
                    logger.warning("IterativeMaskingDMS round-1 batch %d failed: %s; skipping batch", i // config.batch_size, e)
                    raw = [{}] * len(batch)
                results_r1.extend(raw)
                # Preflight OUTSIDE per-batch except — propagates to caller.
                # Only runs once on the first response that contains logits.
                if not _logit_format_validated and _logit_preflight_sample is not None:
                    if "logits" in _logit_preflight_sample and "vocab_tokens" not in _logit_preflight_sample:
                        raise ValueError(
                            f"IterativeMaskingDMSConfig: model {config.model_name!r} returned "
                            f"'logits' but not 'vocab_tokens' — cannot decode without vocabulary "
                            f"ordering (differs between models). "
                            f"Available keys: {list(_logit_preflight_sample.keys())}"
                        )
                    _logit_format_validated = True

            for pos, result in zip(positions, results_r1):
                aa = _argmax_from_response(result, pos)
                if aa is not None:
                    if not config.exclude_synonymous or aa != parent[pos]:
                        round1_preferred[pos] = aa

            print(f"  Round 1: {len(round1_preferred)}/{len(positions)} positions have a preferred substitution")

            if config.rounds == 1:
                # Each position yields one single-mutant sequence
                output_rows = []
                for pos, aa in round1_preferred.items():
                    seq_list = list(parent)
                    seq_list[pos] = aa
                    output_rows.append({
                        "sequence": "".join(seq_list),
                        "model_name": config.model_name,
                        "temperature": None,
                        "sampling_params": {},
                        "generation_method": "iterative_masking_dms",
                        "parent_sequence": parent,
                        "source_label": config.label,
                        "dms_round": 1,
                        "dms_pos1": pos,
                        "dms_aa1": aa,
                    })
                return output_rows

            # Round-2: for each round-1 variant, mask each other position and get argmax
            output_rows = []
            r2_items: list[tuple[int, int, str, str]] = []  # (pos1, pos2, aa1, starting_seq)
            for pos1, aa1 in round1_preferred.items():
                seq1 = list(parent)
                seq1[pos1] = aa1
                seq1_str = "".join(seq1)
                for pos2 in positions:
                    if pos2 == pos1:
                        continue
                    seq2 = list(seq1_str)
                    seq2[pos2] = config.mask_token
                    r2_items.append((pos1, pos2, aa1, "".join(seq2)))

            # Batch round-2 API calls
            r2_sequences = [item[3] for item in r2_items]
            r2_results: list[Any] = []
            for i in range(0, len(r2_sequences), config.batch_size):
                batch = [{"sequence": s} for s in r2_sequences[i:i + config.batch_size]]
                try:
                    raw = await action_fn(items=batch)
                    if isinstance(raw, dict):
                        raw = [raw]
                    elif isinstance(raw, list) and raw and isinstance(raw[0], list):
                        raw = [r[0] if r else {} for r in raw]
                    if len(raw) != len(batch):
                        raise ValueError(
                            f"IterativeMaskingDMS round-2 batch {i // config.batch_size}: "
                            f"API returned {len(raw)} results for {len(batch)} items — "
                            "length mismatch would cause positional drift; treating batch as failed"
                        )
                except Exception as e:
                    logger.warning("IterativeMaskingDMS round-2 batch %d failed: %s; skipping batch", i // config.batch_size, e)
                    raw = [{}] * len(batch)
                r2_results.extend(raw)

            seen: set[str] = set()
            for (pos1, pos2, aa1, _), result in zip(r2_items, r2_results):
                aa2 = _argmax_from_response(result, pos2)
                if aa2 is None:
                    continue
                seq_out = list(parent)
                seq_out[pos1] = aa1
                seq_out[pos2] = aa2
                seq_str = "".join(seq_out)
                if seq_str in seen:
                    continue
                seen.add(seq_str)
                output_rows.append({
                    "sequence": seq_str,
                    "model_name": config.model_name,
                    "temperature": None,
                    "sampling_params": {},
                    "generation_method": "iterative_masking_dms",
                    "parent_sequence": parent,
                    "source_label": config.label,
                    "dms_round": 2,
                    "dms_pos1": pos1,
                    "dms_aa1": aa1,
                    "dms_pos2": pos2,
                    "dms_aa2": aa2,
                })

            if not output_rows:
                logger.warning(
                    "IterativeMaskingDMS: round-2 produced 0 unique 2-point variants "
                    "(all positions may be synonymous or all round-2 batches failed)"
                )
            else:
                logger.debug("  Round 2: %d unique 2-point variants", len(output_rows))
            return output_rows

        finally:
            await api.shutdown()

    async def _dispatch_config(
        self,
        cfg: Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig, SequenceSourceConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"],
        datastore: DataStore,
        df_input: pd.DataFrame,
        context=None,
        run_id: Optional[str] = None,
        ws_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Dispatch a single config to the appropriate generation method."""
        if isinstance(cfg, SequenceSourceConfig):
            return await self._run_sequence_source(cfg, datastore, run_id=run_id)
        elif isinstance(cfg, SaturationMutagenesisConfig):
            return await self._run_saturation_mutagenesis(cfg)
        elif isinstance(cfg, IterativeMaskingDMSConfig):
            return await self._run_iterative_masking_dms(cfg)
        elif isinstance(cfg, RemaskingConfig):
            # RemaskingConfig takes one parent → N variants.  The parent_sequence must
            # be supplied by the caller (set as a dynamic attribute on the config), because
            # accepting multiple parents from df_input would explode into ambiguous provenance:
            # which variant came from which parent?  Use one RemaskingConfig per parent.
            num = cfg.num_variants
            parent = cfg.parent_sequence
            if parent is None:
                raise ValueError(
                    "RemaskingConfig requires a parent_sequence. Set it before dispatch:\n"
                    "    config = RemaskingConfig(...)\n"
                    "    config.parent_sequence = 'MKTAYIAKQRQ'\n"
                    "    config.num_variants = 50\n"
                    "Or use GenerationConfig(remasking_config=..., parent_sequence=...) "
                    "for the legacy API."
                )
            return await self._run_remasking(cfg, parent_sequence=parent, num_variants=num)
        elif isinstance(cfg, DirectGenerationConfig):
            return await self._run_direct_generation(
                cfg, datastore, df_input, context=context, ws_ids=ws_ids
            )
        elif isinstance(cfg, GenerationConfig):
            return await self._generate_with_config(cfg, datastore)
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")

    async def process(
        self, df: pd.DataFrame, datastore: DataStore, **kwargs
    ) -> tuple[pd.DataFrame, StageResult]:
        """Generate sequences using configured models."""
        context = kwargs.get("context")
        run_id = kwargs.get("run_id")
        # GEN-05: extract WS IDs for structure scoping.
        # Prefer ws_ids passed explicitly from process_ws() so that structure
        # lookups are scoped to the current pipeline working set even when
        # process() is called with an empty DataFrame (the WorkingSet path).
        ws_ids: Optional[list[int]] = kwargs.get("ws_ids") or (
            df["sequence_id"].tolist()
            if not df.empty and "sequence_id" in df.columns
            else None
        )

        print(f"  Generating with {len(self.configs)} model configuration(s)...")

        if len(self.configs) == 1:
            results = await self._dispatch_config(
                self.configs[0], datastore, df, context=context, run_id=run_id,
                ws_ids=ws_ids,
            )
        else:
            tasks = [
                self._dispatch_config(
                    cfg, datastore, df, context=context, run_id=run_id, ws_ids=ws_ids,
                )
                for cfg in self.configs
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            failures: list[BaseException] = []
            for i, r in enumerate(results_list):
                if isinstance(r, BaseException):
                    failures.append(r)
                    logger.warning(
                        "Generation config %d/%d failed: %s",
                        i + 1, len(self.configs), r,
                    )
                else:
                    results.extend(r)
            # If every config failed, surface the first exception rather than
            # returning an empty success — the stage produced nothing usable
            # and silently swallowing all errors hides hard breakage.
            if failures and not results:
                raise failures[0]

        df_generated = (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                columns=[
                    "sequence",
                    "model_name",
                    "temperature",
                    "sampling_params",
                    "generation_method",
                    "parent_sequence",
                ]
            )
        )
        initial_count = len(df_generated)

        # warn when generation produces 0 sequences so the user
        # gets an actionable message rather than silently passing nothing downstream.
        if df_generated.empty:
            warnings.warn(
                f"GenerationStage '{self.name}' produced 0 sequences. "
                "Check API response and generation config.",
                UserWarning,
                stacklevel=2,
            )

        if self.deduplicate and initial_count > 0:
            # GEN2-06: normalize to uppercase before dedup so case variants
            # (e.g. 'mktay' vs 'MKTAY') are treated as identical.
            df_generated["sequence"] = df_generated["sequence"].str.upper()
            df_generated = df_generated.drop_duplicates(
                subset=["sequence"]
            ).reset_index(drop=True)
            deduplicated_count = initial_count - len(df_generated)
            if deduplicated_count > 0:
                print(
                    f"  Deduplicated: {deduplicated_count} sequences ({len(df_generated)} unique)"
                )

        if len(df_generated) > 0:
            print(f"  Adding {len(df_generated)} sequences to datastore...")

            # Detect extra columns from generation (e.g. heavy_chain, light_chain
            # from AntiFold) and store them on the sequences table so they survive
            # WorkingSet materialization.
            _gen_extra_cols = [
                c for c in ("heavy_chain", "light_chain")
                if c in df_generated.columns
                and df_generated[c].notna().any()
            ]
            if _gen_extra_cols:
                datastore.ensure_input_columns(_gen_extra_cols)
                _all_cols = ["sequence"] + _gen_extra_cols
                seq_ids = datastore.add_sequences_batch(
                    input_df=df_generated[_all_cols],
                    input_columns=_all_cols,
                )
            else:
                # Batch-insert all sequences in one vectorized call
                seq_ids = datastore.add_sequences_batch(df_generated["sequence"].tolist())
            df_generated["sequence_id"] = seq_ids

            # Store numeric generation scores (global_score, score, seq_recovery)
            # as predictions so they survive WorkingSet materialization and are
            # available for downstream filters.
            _score_cols = [
                c for c in ("global_score", "score", "seq_recovery")
                if c in df_generated.columns
                and df_generated[c].notna().any()
            ]
            if _score_cols:
                pred_rows = []
                for sid, row in zip(seq_ids, df_generated.itertuples(index=False)):
                    for col in _score_cols:
                        val = getattr(row, col, None)
                        if val is not None:
                            pred_rows.append({
                                "sequence_id": sid,
                                "prediction_type": col,
                                "model_name": row.model_name,
                                "value": float(val),
                            })
                if pred_rows:
                    datastore.add_predictions_batch(pred_rows)

            # Store generation metadata in one batched INSERT
            run_id = kwargs.get("run_id", "")

            # Detect the dynamic score column added by SaturationMutagenesisConfig
            # (e.g. 'ddg', 'result_ddg').  It is the only column in df_generated
            # that is not part of the standard or DMS-provenance column sets.
            _KNOWN_GEN_COLS = frozenset({
                "sequence", "model_name", "temperature", "sampling_params",
                "generation_method", "parent_sequence", "source_label",
                "sat_position", "sat_wt_aa", "sat_mut_aa",
                "dms_round", "dms_pos1", "dms_aa1", "dms_pos2", "dms_aa2",
                "global_score", "score", "seq_recovery",
                "sequence_id", "heavy_chain", "light_chain",
            })
            _sat_score_cols = [
                c for c in df_generated.columns if c not in _KNOWN_GEN_COLS
            ]
            _sat_score_col = _sat_score_cols[0] if _sat_score_cols else None

            meta_rows = []
            for seq_id, row in zip(seq_ids, df_generated.itertuples(index=False)):
                sampling_params = getattr(row, "sampling_params", None) or {}

                # Pack DMS provenance into the metadata JSON field so it survives
                # session end and pipeline resume.
                dms_meta: dict = {}

                # SaturationMutagenesisConfig provenance
                sat_pos = getattr(row, "sat_position", None)
                if sat_pos is not None and not (
                    isinstance(sat_pos, float) and pd.isna(sat_pos)
                ):
                    dms_meta["sat_position"] = int(sat_pos)
                    sat_wt = getattr(row, "sat_wt_aa", None)
                    sat_mut = getattr(row, "sat_mut_aa", None)
                    if sat_wt is not None:
                        dms_meta["sat_wt_aa"] = sat_wt
                    if sat_mut is not None:
                        dms_meta["sat_mut_aa"] = sat_mut
                    # Capture the dynamic score column value (e.g. ddg, result_ddg)
                    if _sat_score_col is not None:
                        score_val = getattr(row, _sat_score_col, None)
                        if score_val is not None and not (
                            isinstance(score_val, float) and pd.isna(score_val)
                        ):
                            dms_meta[_sat_score_col] = float(score_val)

                # IterativeMaskingDMSConfig provenance
                dms_round = getattr(row, "dms_round", None)
                if dms_round is not None and not (
                    isinstance(dms_round, float) and pd.isna(dms_round)
                ):
                    dms_meta["dms_round"] = int(dms_round)
                    dms_pos1 = getattr(row, "dms_pos1", None)
                    dms_aa1 = getattr(row, "dms_aa1", None)
                    if dms_pos1 is not None:
                        dms_meta["dms_pos1"] = int(dms_pos1)
                    if dms_aa1 is not None:
                        dms_meta["dms_aa1"] = dms_aa1
                    dms_pos2 = getattr(row, "dms_pos2", None)
                    dms_aa2 = getattr(row, "dms_aa2", None)
                    if dms_pos2 is not None and not (
                        isinstance(dms_pos2, float) and pd.isna(dms_pos2)
                    ):
                        dms_meta["dms_pos2"] = int(dms_pos2)
                    if dms_aa2 is not None and not (
                        isinstance(dms_aa2, float) and pd.isna(dms_aa2)
                    ):
                        dms_meta["dms_aa2"] = dms_aa2

                meta_rows.append({
                    "sequence_id": seq_id,
                    "run_id": run_id,
                    "model_name": row.model_name,
                    "temperature": getattr(row, "temperature", None),
                    "sampling_params": sampling_params,
                    "label": getattr(row, "source_label", None),
                    "metadata": dms_meta if dms_meta else None,
                })
            datastore.add_generation_metadata_batch(meta_rows)

        return df_generated, StageResult(
            stage_name=self.name,
            input_count=0,
            output_count=len(df_generated),
            computed_count=initial_count,
            metadata={
                "deduplicated": (
                    initial_count - len(df_generated) if self.deduplicate else 0
                )
            },
        )

    async def process_ws(
        self, ws: WorkingSet, datastore: DataStore, **kwargs
    ) -> tuple[WorkingSet, StageResult]:
        """Generate sequences and return a WorkingSet of the new IDs.

        The input WorkingSet's IDs are forwarded as ``ws_ids`` so that
        ``DirectGenerationConfig`` can scope structure lookups to sequences
        already in the current pipeline run (GEN-05 consistency: no cross-run
        contamination when ``structure_from_stage`` is set).
        """
        # Forward input WS IDs so structure lookups stay scoped to this run.
        if ws and "ws_ids" not in kwargs:
            kwargs["ws_ids"] = ws.to_list()
        df_generated, result = await self.process(pd.DataFrame(), datastore, **kwargs)
        if "sequence_id" in df_generated.columns:
            return WorkingSet.from_ids(df_generated["sequence_id"].tolist()), result
        return WorkingSet(frozenset()), result


class GenerativePipeline(BasePipeline):
    """
    Pipeline for generating sequences and running predictions.

    Supports:
    - Multiple generative models in parallel
    - Temperature scanning
    - Masked language model remasking
    - Downstream predictions and filtering

    Args:
        generation_configs: List of GenerationConfig objects
        deduplicate: Whether to deduplicate generated sequences
        datastore: DataStore instance or path
        run_id: Unique run ID
        output_dir: Output directory
        resume: Resume from previous run
        verbose: Enable verbose output

    Example:
        >>> # Generate with MPNN at multiple temperatures
        >>> config1 = GenerationConfig(
        ...     model_name='proteinmpnn',
        ...     num_sequences=1000,
        ...     temperature=[0.5, 1.0, 1.5],
        ...     parent_sequence='MKTAYIAKQRQ'
        ... )
        >>>
        >>> # Also generate with ESM using remasking
        >>> config2 = GenerationConfig(
        ...     model_name='esm2',
        ...     num_sequences=500,
        ...     generation_method='remask',
        ...     parent_sequence='MKTAYIAKQRQ',
        ...     mask_fraction=0.15
        ... )
        >>>
        >>> pipeline = GenerativePipeline(
        ...     generation_configs=[config1, config2]
        ... )
        >>> pipeline.add_filter(ThresholdFilter('length', min_value=50))
        >>> pipeline.add_prediction('esmfold', extractions='mean_plddt', columns='plddt')
        >>> results = pipeline.run()
    """

    def __init__(
        self,
        generation_configs: Optional[
            list[Union[GenerationConfig, RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"]]
        ] = None,
        deduplicate: bool = True,
        # ---- convenience aliases ----
        configs: Optional[
            list[Union[RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"]]
        ] = None,
        filters=None,
        data_store=None,
        **kwargs,
    ):
        # data_store= is an alias for datastore= (BasePipeline kwarg)
        if data_store is not None and "datastore" not in kwargs:
            kwargs["datastore"] = data_store

        super().__init__(**kwargs)

        # configs= is a shorthand alias for generation_configs=.
        # Use `is not None` so an explicit generation_configs=[] is respected
        # rather than silently falling through to the alias.
        if generation_configs is not None:
            resolved_configs = generation_configs
        elif configs is not None:
            resolved_configs = configs
        else:
            resolved_configs = []
        self.generation_configs = resolved_configs
        self.deduplicate = deduplicate
        self._generated_ws = None  # set by run_async() after generation stage

        # Add generation stage automatically
        if self.generation_configs:
            gen_stage = GenerationStage(
                name="generation",
                configs=self.generation_configs,
                deduplicate=self.deduplicate,
            )
            self.stages.insert(0, gen_stage)

        # filters= is a shorthand for calling add_filter() at construction
        if filters is not None:
            self.add_filter(filters)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.datastore:
            self.datastore.close()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generation_slot(self) -> Optional[GenerationStage]:
        """Return the single GenerationStage at slot-0, or None."""
        if self.stages and isinstance(self.stages[0], GenerationStage):
            return self.stages[0]
        return None

    def _set_generation_slot(self, stage: GenerationStage) -> None:
        """Replace slot-0 with *stage*, preserving all downstream stages."""
        self.stages = [s for s in self.stages if not isinstance(s, GenerationStage)]
        self.stages.insert(0, stage)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_stage(self, stage: Stage) -> None:
        """Add a stage to the pipeline.

        If *stage* is a :class:`GenerationStage` it **replaces** the current
        generation slot (there is always exactly one, at position 0) rather
        than appending.  All other stage types are appended normally.
        """
        if isinstance(stage, GenerationStage):
            self._set_generation_slot(stage)
        else:
            super().add_stage(stage)

    def set_generation(
        self,
        *configs: Union[RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"],
        stage_name: str = "generation",
        deduplicate: bool = True,
    ) -> "GenerativePipeline":
        """Set (or replace) the generation slot with one or more configs.

        Multiple configs run in parallel — use this for multi-model generation
        or temperature scanning.  Every call to ``.run()`` re-runs generation
        from scratch; downstream stages use their prediction cache so only
        truly new sequences are computed.

        Args:
            *configs: One or more :class:`RemaskingConfig` or
                :class:`DirectGenerationConfig` objects.
            stage_name: Name for the generation stage (default ``"generation"``).
            deduplicate: Deduplicate across all configs (default True).

        Returns:
            Self, for method chaining.

        Example::

            # Single model
            pipeline.set_generation(
                DirectGenerationConfig("dsm-150m-base", sequence=parent, num_sequences=200)
            ).run()

            # Two models in parallel — sequences from both trickle through the funnel
            pipeline.set_generation(
                DirectGenerationConfig("dsm-150m-base", sequence=parent, num_sequences=100),
                RemaskingConfig("esm-150m", mask_fraction=0.15),
            ).run()
        """
        if not configs:
            raise ValueError("set_generation() requires at least one config")
        new_stage = GenerationStage(
            name=stage_name,
            configs=list(configs),
            deduplicate=deduplicate,
        )
        self._set_generation_slot(new_stage)
        return self

    # Keep replace_generation as a convenience alias for single-config swaps
    def replace_generation(
        self,
        config: Union[RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"],
        stage_name: Optional[str] = None,
        deduplicate: bool = True,
    ) -> "GenerativePipeline":
        """Swap the generation slot with a single new config.

        Equivalent to ``set_generation(config, stage_name=stage_name)``.
        Kept for backwards compatibility and single-config convenience.
        """
        existing = self._generation_slot()
        name = stage_name or (existing.name if existing else "generation")
        return self.set_generation(config, stage_name=name, deduplicate=deduplicate)

    def use_sequences(
        self,
        sequences=None,
        column: str = "sequence",
        stage_name: str = "data_source",
        from_db: bool = False,
    ) -> "GenerativePipeline":
        """Use existing sequences as the pipeline source instead of generating.

        Replaces the generation slot with a :class:`SequenceSourceConfig`.
        Downstream prediction and filter stages run on the provided sequences,
        using the normal DuckDB prediction cache for anything already computed.

        Args:
            sequences: One of:

                * ``list[str]`` — plain amino-acid strings
                * ``pd.DataFrame`` — must contain *column* (default ``"sequence"``)
                * ``str`` / ``Path`` — CSV or FASTA file
                * ``None`` — reload all sequences already in the DuckDB

            column: Column name when *sequences* is a DataFrame or CSV.
            stage_name: Name for the source stage (default ``"data_source"``).
            from_db: Pull all sequences already present in this pipeline's DuckDB.

        Returns:
            Self, for method chaining.

        Example::

            # Inject a list
            pipeline.use_sequences(["MKTAY", "MKLLIV"]).run()

            # Use all sequences already in the DB (e.g. recover + rerun)
            pipeline.use_sequences(from_db=True).run()

            # Load from CSV and run through the existing filter/predict stages
            pipeline.use_sequences("candidates.csv").run()
        """
        return self.set_generation(
            SequenceSourceConfig(sequences=sequences, column=column, from_db=from_db),
            stage_name=stage_name,
        )

    def add_generation_config(
        self, config: Union[RemaskingConfig, DirectGenerationConfig, "SaturationMutagenesisConfig", "IterativeMaskingDMSConfig"]
    ) -> "GenerativePipeline":
        """Append a config to the existing generation slot.

        Use this to add a second model or temperature variant *alongside*
        the current generation config rather than replacing it.  If there is
        no generation slot yet, one is created.

        Args:
            config: Config to add.

        Returns:
            Self, for method chaining.
        """
        slot = self._generation_slot()
        if slot is not None:
            slot.configs = list(slot.configs)
            slot.configs.append(config)
        else:
            self._set_generation_slot(
                GenerationStage(
                    name="generation",
                    configs=[config],
                    deduplicate=self.deduplicate,
                )
            )
        return self

    async def _get_initial_data(self, **kwargs) -> pd.DataFrame:
        """Not used — GenerativePipeline overrides _get_initial_data_ws directly."""
        return pd.DataFrame(columns=["sequence", "sequence_id"])

    async def _get_initial_data_ws(self, **kwargs) -> WorkingSet:
        """Return the WorkingSet produced by the generation slot.

        ``_generated_ws`` is populated by ``run_async()`` after all
        GenerationStages have run; it is the union of every generated sequence ID.
        """
        if self._generated_ws is not None:
            return self._generated_ws
        return WorkingSet(frozenset())

    def add_prediction(
        self,
        model_name: str,
        action: str = "predict",
        extractions=None,
        columns=None,
        params: Optional[dict] = None,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """Add a prediction stage (same as DataPipeline)."""
        from biolm.pipeline.data import PredictionStage

        if stage_name is None:
            if columns and isinstance(columns, str):
                stage_name = f"predict_{columns}"
            elif extractions and isinstance(extractions, str):
                stage_name = f"predict_{extractions}"
            else:
                stage_name = f"{model_name}_{action}"

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = PredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            extractions=extractions,
            columns=columns,
            params=params,
            depends_on=depends_on,
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_predictions(
        self,
        models: list[Union[str, dict]],
        action: str = "predict",
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Add multiple prediction stages at the same level (run in parallel).

        Args:
            models: List of model names or dicts with model configs
            action: Default action if not specified in dict
            depends_on: List of stage names all these depend on
            **kwargs: Default kwargs for all stages

        Returns:
            self for chaining

        Example:
            >>> pipeline.add_predictions(['temberture-regression', 'proteinmpnn', 'esm2'])
        """

        for model in models:
            if isinstance(model, str):
                # Simple model name
                self.add_prediction(
                    model_name=model, action=action, depends_on=depends_on, **kwargs
                )
            elif isinstance(model, dict):
                # Dict with config
                model_config = {**kwargs, **model}
                model_config["depends_on"] = model_config.get("depends_on", depends_on)

                model_name = model_config.pop("model_name")
                self.add_prediction(model_name=model_name, **model_config)
            else:
                raise TypeError(f"Model must be str or dict, got {type(model)}")

        return self

    def add_filter(
        self,
        filter_func,
        stage_name: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
        **kwargs,
    ):
        """Add a filter stage (same as DataPipeline)."""
        from biolm.pipeline.data import FilterStage

        stage_name = stage_name or f"filter_{len(self.stages)}"

        if depends_on is None:
            depends_on = [self.stages[-1].name] if self.stages else []

        stage = FilterStage(
            name=stage_name,
            filter_func=filter_func,
            depends_on=depends_on,
            **kwargs,
        )

        self.add_stage(stage)
        return self

    def add_cofolding_prediction(
        self,
        model_name: str,
        action: str = "predict",
        stage_name: Optional[str] = None,
        prediction_type: str = "structure",
        sequence_chain_id: str = "A",
        sequence_entity_type: str = "protein",
        static_entities: Optional[list["FoldingEntity"]] = None,
        depends_on: Optional[list[str]] = None,
        params: Optional[dict] = None,
        batch_size: int = 1,
    ):
        """
        Add a co-folding prediction stage (Boltz2, Chai-1).

        Each sequence in the pipeline is treated as the primary protein chain.
        ``static_entities`` lets you inject additional molecules — ligands,
        cofactors, DNA/RNA strands, or extra protein chains — that are held
        constant across all designs.

        Args:
            model_name: BioLM model slug, e.g. ``'boltz2'`` or ``'chai1'``.
            action: API action (default ``'predict'``).
            stage_name: Stage name (defaults to ``model_name``).
            prediction_type: Column name for the confidence score (default
                ``'structure'``).
            sequence_chain_id: Chain ID assigned to the pipeline's primary
                sequence in the multi-molecule request (e.g. ``'A'``).
            sequence_entity_type: Molecule type for the primary sequence:
                ``'protein'``, ``'dna'``, or ``'rna'`` (default ``'protein'``).
            static_entities: List of :class:`FoldingEntity` objects — ligands,
                cofactors, extra proteins/DNA/RNA — included in every request.
            depends_on: Upstream stage names this stage waits for.
            params: Model-specific params dict (e.g.
                ``{'recycling_steps': 3, 'sampling_steps': 20}`` for Boltz).
            batch_size: Sequences per API request (default 1; co-folding models
                are typically batch-size-1).

        Example::

            pipeline.add_cofolding_prediction(
                model_name='boltz2',
                static_entities=[
                    FoldingEntity(id='L', entity_type='ligand', smiles='c1ccccc1'),
                    FoldingEntity(id='B', entity_type='protein',
                                  sequence='MKTAYIAKQRQ'),
                ],
                depends_on=['filter_top50'],
            )
        """
        from biolm.pipeline.data import CofoldingPredictionStage

        stage_name = stage_name or model_name
        stage = CofoldingPredictionStage(
            name=stage_name,
            model_name=model_name,
            action=action,
            prediction_type=prediction_type,
            sequence_chain_id=sequence_chain_id,
            sequence_entity_type=sequence_entity_type,
            static_entities=static_entities or [],
            params=params or {},
            batch_size=batch_size,
            depends_on=depends_on or [],
        )
        self.add_stage(stage)
        return self

    async def run_async(self, **kwargs) -> dict[str, StageResult]:
        """
        Run the generative pipeline.

        All GenerationStages (regardless of position) are extracted and run first
        as sources — their outputs are unioned into the initial WorkingSet so
        generated sequences trickle through every downstream prediction/filter
        stage (the "funnel").  Idempotent: self.stages is always restored after
        execution.
        """
        import copy as _copy

        resume = kwargs.get("resume", False)

        # Reset generated_ws at start so previous run's data doesn't bleed through
        self._generated_ws = None

        # Save original stages so the pipeline is idempotent (can be called again)
        original_stages = self.stages

        # create the pipeline_runs record BEFORE any generation stage runs,
        # so that mark_stage_complete() never writes orphan stage_completions rows.
        # create_pipeline_run() is idempotent (ON CONFLICT DO UPDATE), so the later
        # call inside super().run_async() is harmless.
        self.datastore.create_pipeline_run(
            run_id=self.run_id,
            pipeline_type=self.pipeline_type,
            config=self._get_config(),
            status="running",
        )

        try:
            # Separate generation stages from processing stages (order-independent)
            gen_stages = [s for s in self.stages if isinstance(s, GenerationStage)]
            non_gen_stages = [s for s in self.stages if not isinstance(s, GenerationStage)]
            gen_names = {s.name for s in gen_stages}

            # Run every generation stage first, collect all produced sequence IDs
            all_gen_ids: set[int] = set()
            for gen_stage in gen_stages:
                stage_id = f"{self.run_id}_{gen_stage.name}"

                # Resume: if this generation stage already completed, reload its
                # working set from generation_metadata rather than re-running.
                # generation_metadata.run_id scopes to this run, so the reload
                # is exact and won't pick up sequences from other runs.
                if resume and self.datastore:
                    sc = self.datastore.conn.execute(
                        "SELECT status, output_count FROM stage_completions WHERE stage_id = ?",
                        [stage_id],
                    ).fetchone()
                    if sc and sc[0] == "completed":
                        expected_count = sc[1]
                        ids = self.datastore.conn.execute(
                            "SELECT DISTINCT sequence_id FROM generation_metadata "
                            "WHERE run_id = ?",
                            [self.run_id],
                        ).df()["sequence_id"].tolist()
                        if ids:
                            if expected_count is not None and len(ids) != expected_count:
                                import warnings
                                warnings.warn(
                                    f"Stage '{gen_stage.name}': stage_completions recorded "
                                    f"{expected_count} sequences but only {len(ids)} found in "
                                    f"generation_metadata. Some sequences may be missing downstream "
                                    f"predictions and will be silently skipped. Consider re-running "
                                    f"without resume=True to regenerate.",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                            ws = WorkingSet.from_ids(ids)
                            self._working_sets[gen_stage.name] = ws
                            all_gen_ids |= set(ids)
                            if self.verbose:
                                print(f"\n{'='*60}")
                                print(f"✓ Stage '{gen_stage.name}' already complete "
                                      f"— reloaded {len(ids)} sequences from DB")
                                print(f"{'='*60}")
                            continue

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Stage: {gen_stage.name} (Generation)")
                    print(f"Configs: {len(gen_stage.configs)}")
                    print(f"{'='*60}")

                # pass run_id and context so generation stages have
                # access to PipelineContext and resume bookkeeping.
                df_generated, result = await gen_stage.process(
                    pd.DataFrame(), self.datastore,
                    run_id=self.run_id, context=self.context,
                )
                self.stage_results[gen_stage.name] = result
                self._stage_data[gen_stage.name] = df_generated

                if "sequence_id" in df_generated.columns:
                    ids = df_generated["sequence_id"].tolist()
                    ws = WorkingSet.from_ids(ids)
                    self._working_sets[gen_stage.name] = ws
                    all_gen_ids |= set(ids)

                # Mark the generation stage complete so resume logic works
                if self.datastore:
                    self.datastore.mark_stage_complete(
                        run_id=self.run_id,
                        stage_name=gen_stage.name,
                        stage_id=stage_id,
                        input_count=0,
                        output_count=len(df_generated),
                        status="completed",
                    )

                if self.verbose:
                    print(f"\n{result}")
                    print(f"{'='*60}")

            # warn when generation produced no sequences so the user
            # gets an actionable message rather than silently processing 0 inputs.
            if not all_gen_ids:
                import warnings
                warnings.warn(
                    "GenerativePipeline: no sequences were produced by any generation "
                    "stage. Downstream prediction/filter stages will receive 0 inputs. "
                    "Check your generation configs, API key, and model availability.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Union of all generated IDs becomes the initial WS for downstream stages
            self._generated_ws = WorkingSet.from_ids(list(all_gen_ids)) if all_gen_ids else None

            # capture what THIS run generated before super().run_async()
            # resets internal dicts.  We restore it afterward so a second call to run()
            # always uses the fresh generation result, not a stale backup.
            _fresh_gen_ws = self._generated_ws

            # save the full pipeline definition (including GenerationStage)
            # BEFORE stripping gen stages from self.stages.  super().run_async() will
            # call _save_definition_and_register_columns() again with remaining_stages
            # only — we capture the full definition_id here and restore it afterward
            # so that pipeline_runs.definition_id points to the complete definition.
            self._save_definition_and_register_columns()
            from biolm.pipeline.pipeline_def import _pipeline_def_hash
            import json as _json
            _full_stages_specs = []
            for s in self.stages:
                try:
                    _full_stages_specs.append(s.to_spec())
                except NotImplementedError:
                    _full_stages_specs.append({"type": s.__class__.__name__, "name": s.name})
            _input_cols = self.input_schema.columns if self.input_schema else None
            _full_def_id = _pipeline_def_hash(self.pipeline_type, _input_cols, _full_stages_specs)

            # GEN2-01 fix: capture full_config_dict BEFORE the stage swap so
            # self._get_config() sees all stages (including generation stages).
            # This eliminates the concurrency hazard from the previous approach
            # of temporarily restoring original_stages inside the UPDATE block.
            full_config_dict = self._get_config()

            # Build the execution plan without the generation stages.
            # Strip gen stage names from depends_on so dependency resolution works.
            # use deepcopy so _api_client and other mutable attrs are
            # not shared between the copy and the original across repeated run() calls.
            remaining_stages = []
            for s in non_gen_stages:
                s_copy = _copy.deepcopy(s)
                s_copy.depends_on = [d for d in s.depends_on if d not in gen_names]
                remaining_stages.append(s_copy)
            self.stages = remaining_stages

            # GEN-01 fix: backup gen stage dicts before super().run_async() wipes them.
            # BasePipeline.run_async() resets stage_results, _working_sets, _stage_data
            # at the top; we restore the generation stage entries afterward so callers
            # can inspect both generation and downstream results via stage_results.
            _gen_ws_backup = {
                s.name: self._working_sets[s.name]
                for s in gen_stages
                if s.name in self._working_sets
            }
            _gen_stage_backup = {
                s.name: self._stage_data[s.name]
                for s in gen_stages
                if s.name in self._stage_data
            }
            _gen_results_backup = {
                s.name: self.stage_results[s.name]
                for s in gen_stages
                if s.name in self.stage_results
            }

            # Run remaining stages using base class (starts from _generated_ws)
            run_result = await super().run_async(**kwargs)

            # restore this run's fresh generation WS.
            # super().run_async() resets _generated_ws to None at the top of every
            # call, so on a second run() the old backup would overwrite the new
            # generation result.  Always prefer what this run actually generated.
            if _fresh_gen_ws is not None:
                self._generated_ws = _fresh_gen_ws

            # Restore generation stage entries wiped by super().run_async() reset
            for name, ws in _gen_ws_backup.items():
                if name not in self._working_sets:
                    self._working_sets[name] = ws
            for name, df_s in _gen_stage_backup.items():
                if name not in self._stage_data:
                    self._stage_data[name] = df_s
            for name, res in _gen_results_backup.items():
                if name not in self.stage_results:
                    self.stage_results[name] = res

            # Restore pipeline_runs.definition_id to the full definition
            # and update .config to include generation stages (GEN2-01 fix).
            _conn_alive = (
                self.datastore is not None
                and getattr(self.datastore, "conn", None) is not None
            )
            if _conn_alive:
                # GEN2-01: full_config was captured BEFORE the stage swap (see above),
                # so we use it directly here instead of doing a fragile stages swap.
                self.datastore.conn.execute(
                    "UPDATE pipeline_runs SET definition_id = ?, config = ? WHERE run_id = ?",
                    [_full_def_id, _json.dumps(full_config_dict), self.run_id],
                )

            return run_result
        finally:
            # Always restore original stages so the pipeline remains idempotent
            self.stages = original_stages


# Convenience function for quick generation
def Generate(
    model_name: str,
    num_sequences: int = 100,
    temperature: Union[float, list[float]] = 1.0,
    parent_sequence: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function for quick sequence generation.

    Args:
        model_name: BioLM model name
        num_sequences: Number of sequences to generate
        temperature: Temperature (float) or list of temperatures for a temperature scan.
            When a list is given, one DirectGenerationConfig is created per temperature.
        parent_sequence: Optional parent sequence for sequence-conditioned models
        **kwargs: Additional arguments forwarded to GenerativePipeline

    Returns:
        DataFrame with generated sequences

    Example:
        >>> df = Generate('dsm-150m-base', num_sequences=100, parent_sequence='MKTAY')
    """
    # GEN2-07: validate that at least one of parent_sequence or structure_path is given
    if parent_sequence is None and "structure_path" not in kwargs:
        raise ValueError(
            "Generate() requires either parent_sequence= or structure_path= to be provided"
        )

    # support list of temperatures — create one config per temperature
    temps = temperature if isinstance(temperature, list) else [float(temperature)]
    item_field = "sequence" if parent_sequence is not None else "pdb"
    configs = [
        DirectGenerationConfig(
            model_name=model_name,
            sequence=parent_sequence,
            item_field=item_field,
            num_sequences=num_sequences,
            temperature=float(t),
        )
        for t in temps
    ]

    pipeline = GenerativePipeline(generation_configs=configs, **kwargs)
    # ensure DuckDB connection is closed even if run() raises
    try:
        pipeline.run()
        return pipeline.get_final_data()
    finally:
        if getattr(pipeline, "_auto_created_datastore", False):
            try:
                pipeline.datastore.close()
            except Exception:
                pass
