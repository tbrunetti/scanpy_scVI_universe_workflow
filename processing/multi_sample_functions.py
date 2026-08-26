from pathlib import Path
import anndata
import scanpy
import scipy.sparse

@profile
def merge_h5ads_for_joint_preprocessing(h5ad_paths:Iterable, layer:str, barcode_prefix_column:str) -> None:
    """
    Create a cohort-level AnnData containing only:

        X   = raw counts from layers["counts"]
        obs = cell metadata
        var = gene metadata

    Existing X, layers, obsm, varm, and uns from the source files
    are not retained.

    Cell barcodes are made globally unique by prepending values of barcode_prefix_column

    Inputs:
        h5ad_paths: a list of paths to all h5ad files to merge
        layer: name of the layer to extract (should be raw counts matrix)
        barcode_prefix_column: the name of a metadata column to user to prepend the value of the cell to the cell barcode
    Outputs:
        None
    """

    anndata_objs = []

    for path in h5ad_paths:
        path = Path(path)
        print(f"Reading: {path}")

        # reduces memory footprint (from scanpy docs):
        # If 'r', load AnnData in backed mode instead of fully loading it into memory (memory mode). 
        # If you want to modify backed attributes of the AnnData object, you need to choose 'r+'.
        # Currently, backed only support updates to X. That means any changes to other slots like obs 
        # will not be written to disk in backed mode. If you would like save changes made to these slots of a 
        # backed AnnData, write them to a new file (see write())
        source = scanpy.read_h5ad(path, backed="r") # backed = 'r' since we are not modifying the data

        if layer not in source.layers:
            raise KeyError(f"{path}: no {layer} layer")

        if barcode_prefix_column not in source.obs.columns:
            raise KeyError(f"{path}: no {barcode_prefix_column} column in obs")

        raw_counts = source.layers[layer]

        # Materialize backed counts into memory.
        # only the extract raw counts layer is now stored in RAM, not the full
        # original AnnData h5ad objects, which are still in backed mode and not in memory
        if hasattr(raw_counts, "to_memory"):
            raw_counts = raw_counts.to_memory()

        # Ensure CSR. - to save memory by forcing matrix to be sparse
        if not sp.isspmatrix_csr(raw_counts):
            raw_counts = sp.csr_matrix(raw_counts)

        # Copy only the metadata from cell and genes
        obs = source.obs.copy()
        var = source.var.copy()

        # Make cell barcodes globally unique by prepending to the front of the 
        # cell barcodes; particularly important if merging many batches of data as 
        # barcodes are resused
        barcode_prefix = obs[barcode_prefix_column].astype(str)
        obs_names = (barcode_prefix + "_" + obs.index.astype(str))

        # replace barcodes, with new barcodes with prepended barcode string
        obs.index = obs_names

        # Construct a minimal AnnData object.
        minimal_anndata_obj = anndata.AnnData(X=raw_counts, obs=obs, var=var)

        anndata_objs.append(minimal_anndata_obj)

    print(f"Concatenating {len(anndata_objs)} datasets...")

    merged_anndata_obj = anndata.concat(anndata_objs, axis="obs", join="inner", merge="same", index_unique=None)

    merged_anndata_obj.X = merged_anndata_obj.X.tocsr()

    if not merged_anndata_obj.obs_names.is_unique:
        raise ValueError(f"Merged AnnData contains duplicated cell barcodes after prepending {barcode_prefix_column} values.")

    merged_anndata_obj.write(filename = paths_config.unfiltered_gene_symbol_h5ad,
                    convert_strings_to_categoricals = True,
                    compression = "gzip")
    
    return merged