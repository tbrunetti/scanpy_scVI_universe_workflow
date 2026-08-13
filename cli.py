from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipeline_config import PipelineConfig


def convert_to_tuple(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f"Metadata must be key=value, got: {item!r}"
        )
    key, value = item.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Metadata key cannot be empty: {item!r}")
    return key, value


def add_environment_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Environment and project setup")
    g.add_argument("--working_dir", type=Path, default=Path.cwd(), help = "Path to directory where project will be hosted")
    g.add_argument("--save_prefix", type=str, default="", help = "String that represents the project name; no special characters, only alphanumeric and hyphens, no whitespace.")
    g.add_argument("--filtered_feature_bc_matrix", type=Path,  help = "Path to counts matrix, barcodes file, and feature/gene files")
    g.add_argument("--sample_name", type=str, help = "name of sample being processed; alphanumeric, no special characters but hyphens acceptable. No whitespace.")
    g.add_argument("--platform", choices=["parse", "10x", "bdrhapsody"], default="parse", help = "single cell platform used for library preparation")
    g.add_argument("--convertEnsembl", action="store_true", dest="convert_ensembl", help = "If features/gene names are ensembl be sure to convert to gene names; most relevant for Parse platforms. Check features file to determine if this should/needs to be converted.")
    g.add_argument("--metadata", nargs="+", type=convert_to_tuple, default=None, help = "Ex: --metadata sex=female batch=A tissue=spleen age=100 group=\"Control group\"") # each time a key value pair is listed, it converts to a tuple and the tuple will be collected as a list based on argparse nargs
    g.add_argument("--seed", type=int, default=42, help = "When algorithms are non-deterministic, this is the seed that be used for reproducibility purposes.")
    g.add_argument("--threads", type=int, default=5, help = "The number of parallel processes to spawn off when a function can be parallelized.")


def add_filtering_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Cell filtering and QC")
    g.add_argument("--min_cells_expressed", type=int, default=0, help = "Keep genes only if they are expressed in X number of cells.  The default is 0 and genes are not removed (recommended).")
    g.add_argument("--min_unique_genes", type=int, default=300, help = "Filter out any cells that do not have at least x number of unique genes expressed.  Default: 300")
    g.add_argument("--min_umi_counts", type=int, default=500, help = "Filter out any cells that do not have at least x number of UMIs/reads in the cells.  Default: 500")
    g.add_argument("--mito_contam", type=float, default=10.0, help = "Filter out any cells that have a x percent of reads expressed in mitochondrial genes.  Generally a good inidcation of cell death. Default: 10.0")
    g.add_argument("--max_unique_genes", type=int, default=None, help = "Filter out any cells that have more than x number of unique genes expressed.  Default: None, this upper limit is not applied." )
    g.add_argument("--max_umi_counts", type=int, default=None, help = "Filter out any cells that have more than x number of UMIs/reads in the cell.  Default: None, this upper limit is not applied.")
    g.add_argument("--min_complexity", type=float, default=0.8)
    g.add_argument("--mito_regex", type=str, default="MT-", help = "The regular expression string to use to extract mitochondrial genes.  The best way to test this is to try the regex on the feature/gene matrix to ensure only mitochondrial genes are being extracted.")
    g.add_argument("--ribo_regex", nargs="+", type=str, default=["RPL", "RPS"], help = "The regular expression string to use to extract ribosomal genes.  The best way to test this is to try the regex on the feature/gene matrix to ensure only mitochondrial genes are being extracted." )
    g.add_argument("--dbl_rate", type=float, default=0.076, help = "Look up the doublet detection rate expecation for the technology you are using; for Parse in 2026, it was estimated to be <3%% see ParseBioScience_What_is_the_expected_doublet_rate_Support_Suite.html in supplemental_files, so you can set this to 0.03; 10x doublet rate is ~7.6%% for 10,000 cells captured/sequenced and it is depenent on the numberof cells captured, so refer to 10x-How-To-Technical-Seminar_Sample-Prep.pdf under supplemental_files to determine this value for 10x. BD Rhapsody, see BD-Rhapsody-HT-Single-Cell-Analysis-System-Instrument-User-Guide_page33.pdf page 33 in supplemental_files, but estimated about 1.7%% for 10k or 3.7%% for 20k cells.  The default is 10x at 10,000 cells of 7.6%% = 0.076")
    g.add_argument("--removeDoublets", action="store_true", dest="remove_doublets", help = "When this flag is specified, doublets are removed during the cell filtering step of the pipeline.  If this flag is not set, doublets are calculated and marked but not removed/fitlered.")


def add_normalization_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Normalization / transformation")
    g.add_argument("--size_factor", type=int, default=None, help = "If size_factor is set to None, then scanpy's default depth normalization of median total counts is used.  Otherwise, this can be set to an integer to depth normalize.  CPM normalization would mean setting this value to 1000000, or to mimic Seurat's depth normalization, set this value to 10000" )
    g.add_argument("--save_memory", action="store_true", help = "A boolean to save memory footprint of storing the whole array in memory vs chunks of the array at a time. This does not parallelize, still single threaded, just sotred the dense values as needed.")
    g.add_argument("--data_chunk_size", type=int, default=None, help = "If --save_memory flag is set, the paramter is required to be set to an int.  The int roughly represents the number of cells you want processed at a time to reduce memory footprint - will not accelerate via parallelization")
    g.add_argument("--n_hvgs", type=int, default=2000, help = "The number of highly variable genes to select from the dataset for use in PCA.")
    g.add_argument("--regress_vars", nargs="+", type=str, default=["total_counts", "pct_counts_is_mito"], help ="a space-delimited list of strings to regress out; total_counts and pct_counts_is_mito are calculated in the pipeline and regressed by default but any metadata field can be added here as well.  Note, if the default and the metadata field of sex for example is to be regress is should be specifed as: --regress-vars total_counts pct_counts_is_mito sex")
    g.add_argument("--hvg_ignore", type=str, default=r"^IG[HKL]([VJ]|V[IVX]+|D[0-9])", help ="a regex that identifies genes to ignore in HVG selection to prevent biasing clustering; For human, the following is recommended for B cell subclustering to prevent clustering bias by clonotype: ^IG[HKL]([VJ]|V[IVX]+|D[0-9]), for mouse, the following is recommended: ^Ig[hkl][vj]|^Ighd[0-9]. WARNING!! Place regex inbetween quotes to prevent shell/bash from interpreting special characters such a | as a pipe!  However, any regex is supported and will flag genes that match the regex to be ignored during selection of HVG.")


def add_clustering_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Dimensionality reduction and clustering")
    g.add_argument("--pca_var_change", type=float, default=0.05, help = "Threshold used to select the number of PCs. Uses the first PC where the change in percent variance explained between consecutive PCs is less than this value.")
    g.add_argument("--neighbors", type=int, default=30, help = "Number of nearest neighbors used to construct the neighborhood graph for downstream dimensionality reduction and clustering.")
    g.add_argument("--resolutions", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], help = "Clustering resolutions to evaluate. Lower values produce fewer, broader clusters, while higher values produce more granular clusters.")
    g.add_argument("--dist_metric", choices=["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "braycurtis", "canberra", "chebyshev", "correlation", "dice", "hamming", "jaccard", "kulsinski", "mahalanobis", "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath", "sqeuclidean", "yule"], default="euclidean", help = "Method to use for caluclating distances in the neighborhood graph." )
    g.add_argument("--top_genes_per_cluster_to_plot", type=int, default=5, help = "When generating dot plots, how many top genes to plot per cluster.")
    g.add_argument("--core_genes_to_plot", nargs="+", type=str, default=["CD3E", "CD3D", "CD4", "CD8A", "CD8B", "CD19", "MS4A1", "CD79A", "CD79B"], help = "The set of genes that are always plotted regardless if they are top genes or not.")


'''
SAVE FOR A DIFFERENT PIPELINE
def add_de_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Differential expression")
    g.add_argument("--de_test", choices=["pyMAST", "wilcoxon", "logreg", "t-test", "t-test_overestim_var"], default="pyMAST")
    g.add_argument("--min_pct_plot_filter", type=float, default=0.0)
    g.add_argument("--max_padj_plot_filter", type=float, default=0.05)
'''

def add_resume_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Resume")
    g.add_argument("--env_vars_config", type=Path, required=True)
    g.add_argument("--update_working_dir", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-sample scRNA-seq analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    subparsers = parser.add_subparsers(dest="run_mode", required=True)

    new_project = subparsers.add_parser("newProject", help="Start a new sample analysis")
    add_environment_args(new_project)
    add_filtering_args(new_project)
    add_normalization_args(new_project)
    add_clustering_args(new_project)
    #add_de_args(new_project)

    resume = subparsers.add_parser("resume", help="Resume from a saved config")
    add_resume_args(resume)

    return parser


# determines if to generate an object of type PipelineConfig because it is a new project and will be populated
# based on argparse namespaces or if resuming, it should just load in the argparse namespace from the original
# new project creation
def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.run_mode == "newProject":
        return PipelineConfig.from_namespace(ns)

    if ns.run_mode == "resume":
        config = PipelineConfig.load(ns.env_vars_config)

    if ns.update_working_dir is not None:
        config = config.with_updates(update_working_dir=ns.update_working_dir)
        return config
