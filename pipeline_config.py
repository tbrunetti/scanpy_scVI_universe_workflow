"""
pipeline_config_explained.py

This module is a documentation-heavy version of the configuration layer.

What this file is for
---------------------
PipelineConfig is the object that holds the run parameters a user selected
on the command line. It is not the pipeline state itself, and it is not a
filesystem path helper. It is the fixed "what the user asked for" object.

Why this exists
---------------
The CLI layer is the source of truth for defaults and help text. argparse
decides what values a user gets when a flag is omitted. This class then
receives those resolved values and stores them in a typed, structured form.

That separation matters because it gives you:

- one place for user-facing defaults and help text
- one object to pass into all pipeline steps
- a saved snapshot that can be reloaded for resume runs
- a clear distinction between configuration and computed pipeline state

Design choices
--------------
- slots=True: keeps the object lightweight and prevents accidental new
  attributes from being attached later.
- frozen=True: makes the config immutable after creation, which supports
  reproducibility.
- tuples instead of lists for collection-like fields: frozen dataclasses
  should not hold mutable defaults if the goal is a stable config object.
- working_dir_effective: allows a moved project directory to be handled by
  overriding the active working directory without rewriting the rest of the
  configuration.
"""

from __future__ import annotations

import pickle
from argparse import Namespace
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, TypeAlias
from enum import Enum

from gene_sets import CELL_CYCLE_GENE_SETS


# ---------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------
#
# These helpers are not pipeline logic. They only standardize values
# before they are stored in the config. Keeping them here prevents every
# pipeline step from having to worry about whether a value arrived from
# the CLI as a string, a list, or a Path-like object.


# Create a stable date stamp for filenames and saved artifacts.
#
# Why store this in the config?
# The pipeline often writes multiple outputs that should share the same
# timestamp. Generating the date once avoids subtle inconsistencies where
# one file is stamped differently from the others.
# eading underscore means “this is intended for internal/private use.” 
# Python does not actually prevent other code from calling it; it is mainly a naming convention.
def _today_stamp() -> str:
    return date.today().strftime("%m%d%Y")


# Convert CLI-provided path values into real Path objects.
#
# Why do this early?
# Because the rest of the pipeline should not have to remember whether a
# value was entered as a string or a Path. Normalizing paths at config
# creation time makes downstream code much simpler and less error-prone.
def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value.expanduser().resolve()
    return Path(value).expanduser().resolve()


# ---------------------------------------------------------------------
# Workflow identifiers
# ---------------------------------------------------------------------

# sets up pipeline config depending on which worflow is implemented
# Identifies which workflow configuration is stored in PipelineConfig.
class WorkflowType(str, Enum):
    """Supported project workflow types.

    Only PER_SAMPLE is implemented at this stage. The remaining values are
    included now so the configuration architecture has a stable place for
    future workflows without exposing their CLI arguments yet.
    """

    PER_SAMPLE = "per_sample"
    MULTI_SAMPLE = "multi_sample"
    INTEGRATION = "integration"
    ANNOTATION = "annotation"


# ---------------------------------------------------------------------
# Workflow-specific configuration
# ---------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PerSampleConfig:
    """Configuration specific to the per-sample preprocessing workflow."""

    # Input / sample identity
    filtered_feature_bc_matrix: Path | None   # Input matrix location for new project runs. Resume runs do not need to start from raw input again.
    sample_name: str | None   # Sample label used in metadata and figures.
    platform: str   # Technology/platform name, such as parse, 10x, or bdrhapsody.
    convert_ensembl: bool    # Whether Ensembl IDs should be collapsed to gene symbols.
    metadata: tuple[tuple[str, str], ...] | None  #  Optional user metadata supplied as key/value pairs. Why tuples instead of lists? The config is frozen, so the stored value should also be immutable.

    # Cell filtering and QC
    min_cells_expressed: int
    min_unique_genes: int
    min_umi_counts: int
    mito_contam: float
    max_unique_genes: int | None
    max_umi_counts: int | None
    min_complexity: float
    mito_regex: str
    ribo_regex: tuple[str, ...]
    dbl_rate: float
    remove_doublets: bool

    # Cell cycle
    s_genes: tuple[str, ...]
    g2m_genes: tuple[str, ...]

    # Normalization / transformation
    size_factor: int | None
    save_memory: bool
    data_chunk_size: int | None
    n_hvgs: int
    regress_vars: tuple[str, ...]
    hvg_ignore: str

    # Dimensionality reduction / clustering
    pca_var_change: float
    neighbors: int
    resolutions: tuple[float, ...]
    dist_metric: str
    top_genes_per_cluster_to_plot: int
    core_genes_to_plot: tuple[str, ...]

    @classmethod
    def from_namespace(cls, ns: Namespace) -> Self:
        """Build the per-sample configuration from the per-sample CLI namespace."""
        raw = vars(ns)
        config_fields = {field.name for field in fields(cls)}

        missing_fields = config_fields - raw.keys()
        if missing_fields:
            raise ValueError(
                "The per-sample argparse Namespace is missing required "
                "PerSampleConfig fields: "
                + ", ".join(sorted(missing_fields))
            )

        data = {name: raw[name] for name in config_fields}

        data["filtered_feature_bc_matrix"] = _as_path(
            data["filtered_feature_bc_matrix"]
        )

        data["metadata"] = (
            tuple(tuple(pair) for pair in data["metadata"])
            if data["metadata"] is not None
            else None
        )

        data["ribo_regex"] = tuple(data["ribo_regex"])
        data["regress_vars"] = tuple(data["regress_vars"])
        data["resolutions"] = tuple(data["resolutions"])
        data["core_genes_to_plot"] = tuple(data["core_genes_to_plot"])

        # Select the resolved cell-cycle gene sets. The model itself remains
        # a CLI concern; the persisted workflow config stores the actual gene
        # lists used by the run.
        if ns.cell_cycle_model in CELL_CYCLE_GENE_SETS:
            gene_set = CELL_CYCLE_GENE_SETS[ns.cell_cycle_model]
            data["s_genes"] = tuple(gene_set["s_genes"])
            data["g2m_genes"] = tuple(gene_set["g2m_genes"])
        elif ns.cell_cycle_model == "custom":
            if ns.s_genes is None or ns.g2m_genes is None:
                raise ValueError(
                    "--s_genes and --g2m_genes are required "
                    "when --cell_cycle_model custom"
                )
            data["s_genes"] = tuple(ns.s_genes)
            data["g2m_genes"] = tuple(ns.g2m_genes)

        return cls(**data)


@dataclass(slots=True, frozen=True)
class MultiSampleConfig:
    """Configuration specific to the multi-sample processing workflow."""

    # Input AnnData objects
    anndata_file: Path
    anndata_paths: tuple[Path, ...]

    @classmethod
    def from_namespace(cls, ns: Namespace) -> Self:
        """Build the multi-sample configuration from the multi-sample CLI namespace."""
        raw = vars(ns)
        config_fields = {field.name for field in fields(cls)}

        missing_fields = config_fields - raw.keys()
        if missing_fields:
            raise ValueError(
                "The multi-sample argparse Namespace is missing required "
                "MultiSampleConfig fields: "
                + ", ".join(sorted(missing_fields))
            )

        data = {name: raw[name] for name in config_fields}

        # Path to the text file containing one H5AD path per line.
        anndata_file = _as_path(data["anndata_file"])

        if anndata_file is None or not anndata_file.is_file():
            raise FileNotFoundError(
                f"AnnData input list was not found: {anndata_file}"
            )

        # Read one H5AD path from each non-empty line in the manifest.
        with anndata_file.open("r") as f:
            anndata_paths = tuple(
                _as_path(line.strip())
                for line in f
                if line.strip()
            )

        if not anndata_paths:
            raise ValueError(
                f"AnnData input list contains no H5AD paths: {anndata_file}"
            )

        # Make sure every listed H5AD file exists before the workflow starts.
        missing_h5ads = [
            path for path in anndata_paths
            if path is None or not path.is_file()
        ]

        if missing_h5ads:
            raise FileNotFoundError(
                "The following H5AD files listed in the AnnData input list "
                "were not found:\n"
                + "\n".join(str(path) for path in missing_h5ads)
            )

        data["anndata_file"] = anndata_file
        data["anndata_paths"] = anndata_paths

        return cls(**data)

# ---------------------------------------------------------------------
# Workflow configuration type
# ---------------------------------------------------------------------

# At this stage only the per-sample workflow configuration exists.
# As additional workflow configs are implemented, this becomes a union:
#
# WorkflowConfig: TypeAlias = (
#     PerSampleConfig
#     | MultiSampleConfig
#     | IntegrationConfig
#     | AnnotationConfig
# )
WorkflowConfig: TypeAlias = PerSampleConfig | MultiSampleConfig


# ---------------------------------------------------------------------
# PipelineConfig for Project-level configuration
# ---------------------------------------------------------------------
#
# PipelineConfig is the immutable record of the analysis run.
#
# Conceptually:
#   argparse Namespace  ->  PipelineConfig  ->  pipeline steps
#
# It is intentionally not doing filesystem work. That job belongs to
# PathConfig. It is also not storing computed outputs such as QC metrics,
# PCA thresholds, cluster labels, or gene lists. Those belong to pipeline
# state, not to configuration.
#
# Advantages of this layout:
# - clear separation between user input and derived analysis results
# - simple saved snapshot for reproducibility
# - easy to pass around without mutating shared state
#
# Disadvantages:
# - values cannot be edited in place because the object is frozen
# - any update requires creating a new config instance
# - it introduces one more object type to understand
# 
# slots=True
#   Advantages:
#     - Reduces memory usage by avoiding a per-instance __dict__.
#     - Slightly faster attribute lookup.
#     - Prevents accidental creation of new attributes (e.g. config.neigbors),
#       catching typos immediately with an AttributeError.
#
#   Disadvantages:
#     - Cannot dynamically add new attributes at runtime.
#     - Slightly less flexible for highly dynamic classes.
#
# frozen=True
#   Advantages:
#     - Makes the configuration immutable after construction.
#     - Prevents accidental modification of analysis parameters during the run.
#     - Clearly separates user-defined configuration from values computed by
#       the pipeline (which should live in a separate PipelineState object).
#
#   Disadvantages:
#     - Fields cannot be modified after creation.
#     - Changes require creating a new PipelineConfig instance.
@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """
    The project-level fields are shared by all workflows. Workflow-specific
    settings are stored in the ``workflow`` object. The concrete workflow
    configuration is selected from ``workflow_type`` when the project is
    created.
    """

    workflow_type: WorkflowType

    # Root directory for the analysis project.
    #
    # Why this is the key filesystem input:
    # Every project subfolder is derived from this one directory. If the
    # project is moved later, updating this value lets the path layer
    # rebuild everything else.
    working_dir: Path

    # User-chosen prefix used in output filenames.
    #
    # Why store it here?
    # It is part of the analysis identity and should remain fixed once the
    # run begins.
    save_prefix: str

    # Reproducibility controls.
    seed: int
    threads: int

    workflow: WorkflowConfig

    # One timestamp attached to the run so saved artifacts stay consistent.
    run_date: str = field(default_factory=_today_stamp)


    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------
    #
    # from_namespace() is the bridge from argparse to the config class.
    #
    # How it works:
    # - argparse has already filled in defaults for omitted CLI flags.
    # - vars(ns) converts the Namespace into a dictionary.
    # - the dataclass is created from those values.
    #
    # Why this is useful:
    # - You do not have to duplicate CLI defaults here.
    # - The config is built from exactly what argparse resolved.
    # - The rest of the pipeline can ignore argparse entirely.
    #
    # Tradeoff:
    # - Field names must stay synchronized with argparse destination names.
    #   If those names drift, this method needs to be updated.
    @classmethod
    def from_namespace(cls, ns: Namespace) -> Self:
        """Build a PipelineConfig from the selected workflow namespace."""
        if ns.run_mode != "newProject":
            raise ValueError(
                "PipelineConfig.from_namespace() expects a newProject namespace."
            )

        workflow_type = WorkflowType(ns.workflow_type)

        if workflow_type is WorkflowType.PER_SAMPLE:
            workflow = PerSampleConfig.from_namespace(ns)

        elif workflow_type is WorkflowType.MULTI_SAMPLE:
            workflow = MultiSampleConfig.from_namespace(ns)

        else:
            raise NotImplementedError(
                f"Workflow '{workflow_type.value}' is not implemented yet.")

        workflow = PerSampleConfig.from_namespace(ns)

        return cls(
            workflow_type=workflow_type,
            working_dir=_as_path(ns.working_dir),
            save_prefix=ns.save_prefix,
            seed=ns.seed,
            threads=ns.threads,
            workflow=workflow,
        )
    
    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    #
    # load() and save() are what make resume workflows practical.
    #
    # Why not just rebuild the config from CLI every time?
    # Because a saved configuration snapshot preserves exactly what was used
    # for a specific run, even if CLI defaults change later.
    #
    # Tradeoff of pickle:
    # - Very convenient in Python.
    # - Not a language-neutral storage format.
    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, cls):
            return obj

        raise TypeError(
            f"Unsupported config payload in {path!s}: {type(obj)!r}. "
            "Expected a PipelineConfig instance."
        )
    
    def save(self, path: str | Path) -> None:
        """Persist the complete project configuration for resume/reproducibility."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    # -----------------------------------------------------------------
    # Controlled updates
    # -----------------------------------------------------------------
    #
    # Because the class is frozen, updating a field means creating a new
    # object. This is deliberate.
    #
    # Why this is good:
    # - You avoid accidental in-place mutation.
    # - A changed configuration is explicit.
    # - It fits the idea that config is a snapshot, not working state.
    #
    # Why this can be inconvenient:
    # - Slightly more verbose than assigning a new attribute.
    def with_updates(self, **changes: Any) -> Self:
        """Create an updated config without mutating the frozen instance."""
        normalized = dict(changes)

        if "working_dir" in normalized:
            normalized["working_dir"] = _as_path(normalized["working_dir"])

        return replace(self, **normalized)


    def __getattr__(self, name: str) -> Any:
        """Temporary compatibility bridge for the existing per-sample framework.

        Once the per-sample framework is refactored, calls such as
        ``config.n_hvgs`` can become ``config.workflow.n_hvgs`` and this bridge
        can be removed.
        """
        workflow = object.__getattribute__(self, "workflow")
        try:
            return getattr(workflow, name)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from exc

    # -----------------------------------------------------------------
    # Compatibility bridge
    # -----------------------------------------------------------------
    #
    # This is useful during migration from older code that still expects
    # an args-like object.
    #
    # Advantage:
    # - Lets you adopt the new config object without rewriting everything at once.
    #
    # Disadvantage:
    # - It should only be temporary; the long-term goal is for the pipeline
    #   to consume PipelineConfig directly.
    def as_namespace(self) -> SimpleNamespace:
        """Compatibility view of the project config for older callers."""
        payload = {
            "workflow_type": self.workflow_type.value,
            "working_dir": self.working_dir,
            "save_prefix": self.save_prefix,
            "seed": self.seed,
            "threads": self.threads,
            "run_date": self.run_date,
        }
        payload.update(
            {
                field.name: getattr(self.workflow, field.name)
                for field in fields(self.workflow)
            }
        )
        return SimpleNamespace(**payload)