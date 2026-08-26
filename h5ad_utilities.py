from pathlib import Path
import scanpy
import scipy.sparse as sp
import numpy as np
import typing


def check_h5ad_sparsity(h5ad_path:str, layer:str|None = None) -> None:
    """
    Inspect an H5AD file and report memory, sparsity, and structure
    relevant to an AnnCollection/scVI workflow.

    Parameters
    ----------
    h5ad_path : str, path to the H5AD file.
    layer: str, name of data layer to check, if None, use active layer (.X)
    Returns
    -------
    dict
        Metrics for layers specified by parameter input.
    """

    anndata_obj = scanpy.read_h5ad(h5ad_path)

    print(f"\nFile: {h5ad_path}")
    print("-" * 60) # make 60 dash marks; make pretty table

    # Basic AnnData information
    print(f"Shape: {anndata_obj.shape}")
    print(f"Cells: {anndata_obj.n_obs:,}")
    print(f"Genes: {anndata_obj.n_vars:,}")
    print(f"Layers: {list(anndata_obj.layers.keys())}")
    print(f"obsm: {list(anndata_obj.obsm.keys())}")

    if layer == None: # when None, means to use active layer (.X)
        # X
        print("\nX")
        print(f"  Type: {type(anndata_obj.X)}")

        if sp.issparse(anndata_obj.X):
            print(f"  Sparse: True")
            print(f"  Format: {anndata_obj.X.getformat()}")
            print(f"  dtype: {anndata_obj.X.dtype}")

            x_bytes = (anndata_obj.X.data.nbytes + anndata_obj.X.indices.nbytes + anndata_obj.X.indptr.nbytes)

            print(f"  Memory: {x_bytes / 1024**3:.3f} GB")
        else:
            print(f"  Sparse: False")
            print(f"  dtype: {anndata_obj.X.dtype}")
            print(f"  Memory: {anndata_obj.X.nbytes / 1024**3:.3f} GB")
    else:
        # Counts
        print(f"\nlayers[{layer}]")

        if layer not in anndata_obj.layers:
            print("  NOT PRESENT")
        else:
            counts = anndata_obj.layers[layer]

            print(f"  Type: {type(counts)}")
            print(f"  dtype: {counts.dtype}")

            if sp.issparse(counts):
                print("  Sparse: True")
                print(f"  Format: {counts.getformat()}")

                counts_bytes = (counts.data.nbytes + counts.indices.nbytes + counts.indptr.nbytes)

                nonzero_fraction = counts.nnz / counts.size

                print(f"  Memory: {counts_bytes / 1024**3:.3f} GB")
                print(f"  Nonzero fraction: {nonzero_fraction:.4%}")
                print(f"  Zero fraction: {1 - nonzero_fraction:.4%}")
                print(f"  CSR: {sp.isspmatrix_csr(counts)}")

            else:
                print("  Sparse: False")

                counts_bytes = counts.nbytes
                nonzero_fraction = np.count_nonzero(counts) / counts.size

                print(f"  Memory: {counts_bytes / 1024**3:.3f} GB")
                print(f"  Nonzero fraction: {nonzero_fraction:.4%}")
                print(f"  Zero fraction: {1 - nonzero_fraction:.4%}")
                print(f"  CSR: False")


# if there is a lyser that is not sparse already and you would like to turn it into a sparse CSR matrix
# then you can run this function
def convert_counts_to_csr(h5ad_path:Path, layer:str|None = None, output_path:Path|None = None) -> None:
    """
    Convert parameter specified layer in an H5AD file to CSR sparse format.

    The original file is not modified unless input_path == output_path.
    """

    h5ad_path = Path(h5ad_path)

    if output_path is None:
        output_path = h5ad_path.with_name(h5ad_path.stem + "_sparse.h5ad")
    else:
        output_path = Path(output_path)

    anndata_obj = scanpy.read_h5ad(h5ad_path)


    if layer == None:
        print("Working on active layer (.X)")
        counts = anndata_obj.X
    elif layer not in anndata_obj.layers:
        raise KeyError(f"No {layer} layer found in {h5ad_path}")
    else:
        counts = anndata_obj.layers[layer]

    if sp.isspmatrix_csr(counts):
        print(f"{h5ad_path.name}: already CSR")
    else:
        print(f"{h5ad_path.name}: converting {type(counts)} -> CSR")
        anndata_obj.layers[layer] = sp.csr_matrix(counts)

    assert sp.isspmatrix_csr(anndata_obj.layers[layer])

    anndata_obj.write_h5ad(output_path)

    print(f"Written: {output_path}")