"""
path_config_explained.py

This module documents the filesystem side of the project.

What this file is for
---------------------
PathConfig turns one root working directory into all of the project
subfolders and standard filenames used by the pipeline.

Why this exists
---------------
The pipeline writes many different outputs:
- AnnData checkpoints
- QC images
- saved config snapshots
- doublet-detection objects
- clustering outputs

If each pipeline step reconstructed those paths by hand, the code would
become repetitive and easy to drift out of sync. PathConfig centralizes
that logic so the working directory remains the single source of truth.

Design choices
--------------
- slots=True: keeps the object compact and prevents arbitrary attributes.
- frozen=True: filesystem layout is derived and should not be mutated in
  place.
- computed properties: all subfolders are built from working_dir every
  time, which makes moving a project folder much easier to support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self

from pipeline_config import PipelineConfig


# ---------------------------------------------------------------------
# PathConfig
# ---------------------------------------------------------------------
#
# PathConfig is the derived filesystem view of the pipeline.
#
# It does not decide what analysis to run. It only decides where the
# pipeline should read and write things. This makes it a good companion to
# PipelineConfig, which decides what the run should do.
#
# Advantages of this design:
# - one place for all standard path construction
# - easy relocation of a project by changing working_dir
# - no repeated hard-coded folder names scattered through the pipeline
#
# Disadvantages:
# - the path layout is intentionally standardized, so there is less freedom
#   to vary folder names per step
# - if the project structure changes, this one class must be updated
@dataclass(slots=True, frozen=True)
class PathConfig:
    """
    Derived project paths and standard filenames for one analysis run.
    """

    # Root folder for the analysis project.
    #
    # Why this is the only directory input:
    # Everything else under the project is derived from this root. That
    # makes the project easier to move, archive, and resume.
    working_dir: Path

    # Prefix used in the standard output filenames.
    save_prefix: str

    # Date stamp attached to saved artifacts.
    #
    # Why keep the date here rather than recompute it repeatedly?
    # Because a stable run date helps ensure filenames stay consistent
    # across all outputs written during a single analysis.
    run_date: str

    # Build a PathConfig from PipelineConfig.
    #
    # How this fits into the architecture:
    # - PipelineConfig stores the user’s requested run settings.
    # - PathConfig derives the folder structure from those settings.
    #
    # Why this is useful:
    # If the project folder moves later, only the working directory changes.
    # The rest of the path tree is rebuilt automatically.
    @classmethod
    def from_config(cls, config: PipelineConfig) -> Self:
        return cls(
            working_dir=config.working_dir,
            save_prefix=config.save_prefix,
            run_date=config.run_date,
        )

    # Directory containing AnnData checkpoint files.
    #
    # The name is not stored separately because it is always derived from
    # working_dir. That keeps the root directory authoritative.
    @property
    def h5ad_dir(self) -> Path:
        return self.working_dir / "h5ad_objects"

    # Directory for QC figures and related plots.
    @property
    def qc_dir(self) -> Path:
        return self.working_dir / "qc_images"

    # Directory for workspace artifacts such as saved configs and pickled
    # helper objects.
    @property
    def workspace_dir(self) -> Path:
        return self.working_dir / "workspace_files"

    # Directory for clustering results and cluster-based figures.
    @property
    def cluster_dir(self) -> Path:
        return self.working_dir / "cluster_analysis"

    # Resolution-specific clustering folder.
    #
    # Why a method instead of a stored path?
    # Because this folder is parameterized by resolution and can be built
    # on demand from the same root directory.
    def resolution_dir(self, resolution: float) -> Path:
        return self.cluster_dir / f"{resolution:.2f}"

    # Create the standard directory tree used by the pipeline.
    #
    # Advantage:
    # - Prevents missing-folder errors later.
    # - Ensures a predictable project layout.
    #
    # Disadvantage:
    # - Assumes the project structure is fixed and standardized.
    def ensure_directories(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.h5ad_dir.mkdir(parents=True, exist_ok=True)
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

    # defined path and file name for the metadata that is calculated on the project level
    # metrics and saved as a pickle file and works like an unstructured dictionary
    @property
    def analysis_metadata_pickle(self) -> Path:
        return (self.workspace_dir / f"analysis_metadata_{self.save_prefix}_{self.run_date}.pkl")

    # Standard filename for the saved config snapshot.
    @property
    def pipeline_config_pickle(self) -> Path:
        return (self.workspace_dir / f"pipeline_config_{self.save_prefix}_{self.run_date}.pkl")

    # Standard filename for the saved doublet-detection object.
    @property
    def scdblfinder_pickle(self) -> Path:
        return self.workspace_dir / f"unfiltered_scdblfinder_obj_{self.save_prefix}_{self.run_date}.pkl"

    # Raw AnnData checkpoint file.
    @property
    def original_h5ad(self) -> Path:
        return self.h5ad_dir / f"unfiltered_original_{self.save_prefix}_{self.run_date}.h5ad"

    # Post-conversion but pre-filtering AnnData checkpoint file.
    @property
    def unfiltered_gene_symbol_h5ad(self) -> Path:
        return self.h5ad_dir / f"unfiltered_geneSymbol_{self.save_prefix}_{self.run_date}.h5ad"

    # Post-filtering AnnData checkpoint file.
    @property
    def filtered_h5ad(self) -> Path:
        return self.h5ad_dir / f"filtered_geneSymbol_{self.save_prefix}_{self.run_date}.h5ad"
