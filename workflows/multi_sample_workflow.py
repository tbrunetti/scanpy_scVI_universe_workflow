import path_config
import anndata
from cli import parse_args
from path_config import PathConfig
from per_sample_functions import normalize_and_transform, identify_and_transform_hvgs, pca, neighbors_umap_clust, final_qc_images
from visualization import qc_figures, plot_reductions
from pipeline_reporting import generate_quarto_report
from multi_sample_functions import merge_h5ads_for_joint_preprocessing

def run_pipeline():
    # --------------------------------------------------
    # 1. Parse CLI arguments
    # --------------------------------------------------
    config = parse_args()

    # config is now a PipelineConfig object
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
    print("Pipeline configuration initialized.")
    print(f"H5AD directory: {paths.h5ad_dir}")
    print(f"QC directory:   {paths.qc_dir}")
    print(f"Cluster dir:    {paths.cluster_dir}")

    # --------------------------------------------------
    # 4. Run pipeline steps
    # --------------------------------------------------

    # first time project level metadata is initiated
    analysis_metadata = {}

    # STEP0: create an unfiltered anndata object
    merged_anndata_obj = merge_h5ads_for_joint_preprocessing(h5ad_paths = config.workflow.anndata_paths,
                layer = "counts",
                barcode_prefix_column = "sample_name",
                h5ad_save_name = paths.merged_h5ad)

    # STEP1: normalize and transform data
    merged_anndata_obj = normalize_and_transform(anndata_obj = merged_anndata_obj,
                size_factor = config.workflow.size_factor,
                save_memory = config.workflow.save_memory,
                data_chunk_size = config.workflow.data_chunk_size,
                h5ad_save_name = paths.merged_h5ad)

    #STEP2: idenfity high variably genes to use for clustering
    merged_anndata_obj, analysis_metadata = identify_and_transform_hvgs(anndata_obj = merged_anndata_obj,
                n_hvgs = config.workflow.n_hvgs,
                vars_to_regress = list(config.workflow.regress_vars),
                hvg_ignore = config.workflow.hvg_ignore,
                threads = config.threads,
                analysis_metadata = analysis_metadata,
                h5ad_save_name = paths.merged_h5ad,
                paths_config = paths)

    #STEP3: pca
    merged_anndata_obj, analysis_metadata = pca(anndata_obj = merged_anndata_obj,
                pca_var_change = config.workflow.pca_var_change,
                seed = config.seed,
                analysis_metadata = analysis_metadata,
                h5ad_save_name = paths.merged_h5ad,
                paths_config = paths)

    #STEP4: find neighbors, generate umap, and define cluster partitions
    merged_anndata_obj = neighbors_umap_clust(anndata_obj = merged_anndata_obj,
                n_neighbors = config.neighbors,
                n_pcs = analysis_metadata["pcs_to_use"],
                dist_metric = config.dist_metric,
                seed = config.seed,
                resolutions = config.resolutions,
                addl_genes = config.core_genes_to_plot,
                h5ad_save_name = paths.merged_h5ad,
                paths_config = paths)

    #STEP5: genearte final set of QC metic images on different embeddings
    plot_embeddigs = ["X_pca", "merged_umap"]
    for embedding in plot_embeddigs:
        final_qc_images(anndata_obj = merged_anndata_obj,
                    reduction = embedding,
                    layer = "normalized",
                    paths_config = paths)


    # --------------------------------------------------
    # FINAL: generate HTML pipeline report
    # --------------------------------------------------
    generate_quarto_report(paths=paths, config=config)



# initiate pipeline
if __name__ == "__main__":
    run_pipeline()
