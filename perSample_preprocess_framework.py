from initial_scRNAseq_analysis_per_sample import neighbors_umap_clust
import path_config
import anndata
from cli import parse_args
from path_config import PathConfig
from per_sample_preprocessing import generate_h5ad, add_qcmetrics_and_metadata, cell_filtering, normalize_and_transform, calculate_cell_cycle, identify_and_transform_hvgs, pca, neighbors_umap_clust
from visualization import qc_figures, plot_reductions
from pipeline_reporting import generate_quarto_report

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
    print("Pipeline configuration initialized.")
    print(f"H5AD directory: {paths.h5ad_dir}")
    print(f"QC directory:   {paths.qc_dir}")
    print(f"Cluster dir:    {paths.cluster_dir}")

    # --------------------------------------------------
    # 4. Run pipeline steps
    # --------------------------------------------------

    # STEP0: create an unfiltered anndata object
    anndata_obj = generate_h5ad(filtered_matrix_files = config.filtered_feature_bc_matrix,
                platform = config.platform, 
                convert_ensembl = config.convert_ensembl,
                paths_config = paths)

    # STEP1: calculate QC metrics and add any addition metadata
    anndata_obj, analysis_metadata = add_qcmetrics_and_metadata(anndata_obj = anndata_obj,
                sample_name = config.sample_name,
                mito_regex = config.mito_regex,
                ribo_regex = config.ribo_regex,
                add_metadata = config.metadata, 
                doublet_rate = config.dbl_rate,
                seed = config.seed,
                paths_config = paths)
    
    qc_figures(anndata_obj = anndata_obj, 
                min_counts = config.min_umi_counts, 
                min_genes = config.min_unique_genes, 
                max_counts = config.max_umi_counts, 
                max_genes = config.max_unique_genes, 
                max_mito = config.mito_contam,
                status = "unfiltered",
                paths_config = paths)

    # STEP2: filter cells
    anndata_obj, analysis_metadata = cell_filtering(anndata_obj = anndata_obj, 
                min_counts = config.min_umi_counts,
                min_genes = config.min_unique_genes,
                max_counts = config.max_umi_counts,
                max_genes = config.max_unique_genes,
                max_mito = config.mito_contam,
                remove_doublets = config.remove_doublets,
                analysis_metadata = analysis_metadata,
                paths_config = paths) 
    
    qc_figures(anndata_obj = anndata_obj, 
                min_counts = config.min_umi_counts, 
                min_genes = config.min_unique_genes, 
                max_counts = config.max_umi_counts, 
                max_genes = config.max_unique_genes, 
                max_mito = config.mito_contam,
                status = "filtered",
                paths_config = paths)
    
    # STEP3: normalize and transform data
    anndata_obj = normalize_and_transform(anndata_obj = anndata_obj,
                size_factor = config.size_factor,
                save_memory = config.save_memory,
                data_chunk_size = config.data_chunk_size,
                paths_config = paths)

    # STEP4: calculate and infer cell cycle
    anndata_obj = calculate_cell_cycle(anndata_obj = anndata_obj,
                s_genes = config.s_genes,
                g2m_genes = config.g2m_genes)
    
    #STEP5: idenfity high variably genes to use for clustering
    anndata_obj, analysis_metadata = identify_and_transform_hvgs(anndata_obj = anndata_obj,
                n_hvgs = config.n_hvgs,
                vars_to_regress = list(config.regress_vars),
                hvg_ignore = config.hvg_ignore,
                threads = config.threads,
                analysis_metadata = analysis_metadata,
                paths_config = paths)

    #STEP6: pca
    anndata_obj, analysis_metadata = pca(anndata_obj = anndata_obj,
                pca_var_change = config.pca_var_change,
                seed = config.seed,
                analysis_metadata = analysis_metadata,
                paths_config = paths)
    
    #STEP7: find neighbors, generate umap, and define cluster partitions
    anndata_obj = neighbors_umap_clust(anndata_obj = anndata_obj,
                n_neighbors = config.neighbors,
                n_pcs = analysis_metadata["pcs_to_use"],
                dist_metric = config.dist_metric,
                seed = config.seed,
                resolutions = config.resolutions,
                addl_genes = config.core_genes_to_plot,
                paths_config = paths)
    
    # --------------------------------------------------
    # FINAL: generate HTML pipeline report
    # --------------------------------------------------
    generate_quarto_report(paths=paths, config=config)



# initiate pipeline
if __name__ == "__main__":
    run_pipeline()