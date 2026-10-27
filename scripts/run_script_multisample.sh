#!/bin/bash

export PYTHONPATH="/home/tonya/Downloads/github/scanpy_scVI_universe_workflow:$PYTHONPATH"
export PYTHONPATH="/home/tonya/Downloads/github/scanpy_scVI_universe_workflow/processing:$PYTHONPATH"
export PYTHONPATH="/home/tonya/Downloads/github/scanpy_scVI_universe_workflow/reporting:$PYTHONPATH"
export PYTHONPATH="/home/tonya/Downloads/github/scanpy_scVI_universe_workflow/scripts:$PYTHONPATH"
export PYTHONPATH="/home/tonya/Downloads/github/scanpy_scVI_universe_workflow/workflows:$PYTHONPATH"

python multi_sample_workflow.py newProject multi_sample \
    --working_dir /home/tonya/test_multi_sample_project \
    --save_prefix multi_sample_analysis \
    --anndata_file /home/tonya/Downloads/github/scanpy_scVI_universe_workflow/anndata_files.txt \
    --seed 42 \
    --threads 5 \
    --size_factor 10000 \
    --n_hvgs 2000 \
    --regress_vars total_counts pct_counts_is_mito \
    --hvg_ignore "^IG[HKL]([VJ]|V[IVX]+|D[0-9])" \
    --pca_var_change 0.01 \
    --neighbors 30 \
    --resolutions 0.2 0.4 \
    --dist_metric euclidean \
    --top_genes_per_cluster_to_plot 5 \
    --core_genes_to_plot CD3D CD3E CD4 CD8A CD8B CD19 MS4A1 CD79A CD79B
