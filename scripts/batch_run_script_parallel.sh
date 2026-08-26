#!/usr/bin/env bash

CSV_FILE="Parse_TCR_Mega_Metadata.csv"

gnuParallelCommand="parallel --tag --memfree 30G --delay 0.2 --jobs 10 -u --progress --joblog jobStatus.parallel.out"

# Read the CSV one row at a time. The first line (header) is skipped.
tail -n +2 "$CSV_FILE" | while IFS=',' read -r ID AGE STUDY_GROUP SURGICAL_PALLIATION_STAGE THYMECTOMY SEX RACE ETHNICITY
do
    # Remove a possible carriage return from the last CSV field.
    ETHNICITY=${ETHNICITY%$'\r'}

    echo "python perSample_preprocess_framework.py newProject \
         --working_dir /home/tonya/test_scanpy_jordan_08032026/${ID} \
         --save_prefix ${ID}_independent_analysis \
         --filtered_feature_bc_matrix /mnt/IM_drive/Jordan_Abbott/parse_1M_cell_cardiac_surgery_Tcell_data_01072025/parse_trailmaker_demux_pipeline_output_02162026/WTA_output/output_combined/${ID}/DGE_filtered/ \
         --sample_name ${ID} \
         --platform parse \
         --convertEnsembl \
         --metadata \"Sex=${SEX}\" \"Age=${AGE}\" \"Study Group=${STUDY_GROUP}\" \"Surgical Palliation Stage=${SURGICAL_PALLIATION_STAGE}\" \"Thymectomy=${THYMECTOMY}\" \"Race=${RACE}\" \"Ethnicity=${ETHNICITY}\" \
         --seed 42 \
         --threads 5 \
         --min_cells_expressed 0 \
         --min_unique_genes 300 \
         --min_umi_counts 500 \
         --mito_contam 10.0 \
         --min_complexity 0.80 \
         --mito_regex \"^MT-\" \
         --ribo_regex \"RPL\" \"RPS\" \
         --dbl_rate 0.03 \
         --removeDoublets \
         --cell_cycle_model human \
         --n_hvgs 2000 \
         --hvg_ignore \"^IG[HKL]([VJ]|V[IVX]+|D[0-9])\" \
         --pca_var_change 0.01 \
         --neighbors 30 \
         --resolutions 0.2 0.4 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
         --dist_metric euclidean \
         --top_genes_per_cluster_to_plot 5 \
         --core_genes_to_plot CD3D CD3E CD4 CD8A CD8B NCAM CD19 MS4A1 CD79A CD79B"

done | $gnuParallelCommand
