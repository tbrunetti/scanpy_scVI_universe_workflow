from cli import parse_args
from path_config import PathConfig


def run_pipeline():
    # --------------------------------------------------
    # 1. Parse CLI arguments
    # --------------------------------------------------
    config = parse_args()

    # config is now a PipelineConfig object
    print(f"Sample:     {config.sample_name}")
    print(f"Platform:   {config.platform}")
    print(f"Run date:   {config.run_date}")
    print(f"Working dir: {config.working_dir}")

    # --------------------------------------------------
    # 2. Build all project paths
    # --------------------------------------------------
    paths = PathConfig.from_config(config)

    # Create standard project directories
    paths.ensure_directories()

    # --------------------------------------------------
    # 3. Save configuration for reproducibility/resume
    # --------------------------------------------------
    config.save(paths.pipeline_config_pickle)

    print(f"Saved config: {paths.pipeline_config_pickle}")

    # --------------------------------------------------
    # 4. Run pipeline steps
    # --------------------------------------------------

    # Example:
    #
    # adata = load_data(
    #     config.filtered_feature_bc_matrix,
    #     platform=config.platform,
    # )
    #
    # adata = filter_cells(
    #     adata,
    #     min_genes=config.min_unique_genes,
    #     min_counts=config.min_umi_counts,
    #     mito_cutoff=config.mito_contam,
    # )
    #
    # adata.write(paths.filtered_h5ad)

    print("Pipeline configuration initialized.")
    print(f"H5AD directory: {paths.h5ad_dir}")
    print(f"QC directory:   {paths.qc_dir}")
    print(f"Cluster dir:    {paths.cluster_dir}")


if __name__ == "__main__":
    run_pipeline()