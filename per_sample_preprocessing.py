from triton.language import cat
from path_config import PathConfig
from zipfile import Path
from typing import Iterable
import sys
import os
import pathlib
import pickle
import numpy
import scipy
import pandas
import pyarrow
import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn
import anndata
import scanpy
import argparse
import pyclustree
import pyscdblfinder # doublet detection library
from memory_profiler import profile
from time import time
import logging
import pymast
import re
from pipeline_config import PipelineConfig
import gc
from visualization import plot_reductions, plot_violin, pretty_pca_loadings, pretty_highly_variable_genes


logger = logging.getLogger(__name__)

# force pandas 3 to be compatible with anndata v 0.12.16
pandas.options.mode.string_storage = "python"
anndata.settings.allow_write_nullable_strings = True

# a function that accepts another function as an input parameter
# so that it can time the execution of the passed in function
def timer(func) -> Callable:
     # the wrap_func is the part of the code that wraps
     # the function to be timed
     def wrap_func(*args, **kwargs):
          start_time = time()
          func_result = func(*args, **kwargs)
          finish_time = time()
          print(f'Function {func.__name__!r} executed in {(finish_time-start_time):.4f}s')
          return func_result

     return wrap_func



@profile
def generate_h5ad(filtered_matrix_files:Path, platform:str, convert_ensembl:bool, paths_config:PathConfig) -> AnnData:


     #anndata_obj = original anndata object that contains a column that has the ensembl and gene symbol mapping dataframe located in .var
     #gene_symbol_col = column name in .var that contains the gene symbols (human readable, non-Ensembl)
     #cell_barcode_col = column name in .obs that contains the barcode/cell names
     @timer
     @profile
     def ensembl_to_symbol_pddense(anndata_obj:anndata.AnnData, gene_symbol_col: str, cell_barcode_col: str) -> AnnData:
          # generate all multi-index columns from tuple of the .var in anndata object
          list_of_tuples = pandas.MultiIndex.from_tuples([tuple(x) for x in anndata_obj.var[anndata_obj.var.columns].to_numpy()],
                         names = anndata_obj.var.columns)


          '''
          Based on Gemini, this will run faster if I avoid the pandas sparse type
          and then trade off in memory due to the following.
          If you use a dense DataFrame, the chain dense_df.T.groupby(...).sum().T becomes highly
          optimized:
               Instant Transposes (.T): In dense pandas, transposing a single-dtype matrix doesn't
               copy or rewrite data in memory. It just shifts the internal metadata instructions
               (creating a memory "view"), making both .T steps virtually instant.
               C-Optimized Math: Once the data is dense, groupby().sum() hands the execution off to
               lightning-fast, C-compiled NumPy routines instead of struggling through unoptimized
               sparse Python code loops
          '''
          dense_pd_df = anndata_obj.to_df() # 13.9s
          dense_pd_df.index = anndata_obj.obs[cell_barcode_col] #0s bc_wells
          dense_pd_df.columns = list_of_tuples #0s
          dense_pd_df = dense_pd_df.T.groupby(level=gene_symbol_col).sum().T #

          # convert to anndata
          anndata_obj_collapsed = anndata.AnnData(dense_pd_df)

          # make sure the index doesn't have a name or it causes problems with object formatting and saving the object
          anndata_obj.obs_names = anndata_obj.obs[cell_barcode_col] # need to do this so it can merge on same index; right now this could be index ranges, not cell names so ensuring this is cell names
          anndata_obj_collapsed.obs_names = anndata_obj.obs[cell_barcode_col]
          anndata_obj_collapsed.obs.index.name = None

          # since on= is not specified as a parameter, it matches the dataframe based on row index; 
          # # note the bc_wells, sample, species, gene_count, tscp_count, mread_count, bc1_well, bc2_well, bc3_well metadata comes from Parse, 
          # so may be incaccurate for gene counts due to being based on Ensembl IDs rather than gene symbols
          anndata_obj_collapsed.obs = anndata_obj_collapsed.obs.join(anndata_obj.obs, how = "left", sort = False) 
          
          # must cast all dtypes that are type str/categorical to type object, because anndata doesn't support arrow types yet and string/str/category come from new pandas arrow types
          #for col in anndata_obj_collapsed.obs.columns:
          #     if anndata_obj_collapsed.obs[col].dtype == 'str' or str(anndata_obj_collapsed.obs[col].dtype) == 'string' or anndata_obj_collapsed.obs[col].dtype == 'category':
          #          anndata_obj_collapsed.obs[col] = anndata_obj_collapsed.obs[col].astype(object)

          return anndata_obj_collapsed


     if platform == "parse":
          anndata_obj = scanpy.read_mtx(filename= os.path.join(filtered_matrix_files, "count_matrix.mtx.gz"))

          # read in gene and cell metadata
          gene_metadata = pandas.read_csv(filepath_or_buffer= os.path.join(filtered_matrix_files, "all_genes.csv.gz"),
                                compression="gzip", 
                                engine="pyarrow") #pyarrow is more memory efficient for large csv files, although parquet may be even better and c engine is very fast but may be less memory efficient than pyarrow

          cell_metadata = pandas.read_csv(filepath_or_buffer= os.path.join(filtered_matrix_files, "cell_metadata.csv.gz"),
                                compression="gzip",
                                engine="pyarrow")

          # add gene and cell level data to anndata object
          anndata_obj.var = gene_metadata # a dataframe that contains gene_id (ensembl), gene_name (symbol), genome (genome version used) - based on Parse input
          anndata_obj.obs = cell_metadata # a dataframe that contain the cell level metadata, very similar to Seurat @meta.data, however need to reassign row names to be cell barcodes

          # change index to be cell ids instead of range 0-n
          anndata_obj.obs_names  = anndata_obj.obs['bc_wells']

          logging.info("Saving unfiltered Ensembl anndata object to disk.")
          # file type is inferred from filename extension
          anndata_obj.write(filename = paths_config.original_h5ad, 
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
          

          if convert_ensembl:
               anndata_obj = ensembl_to_symbol_pddense(anndata_obj = anndata_obj, 
                                                       gene_symbol_col = "gene_name", 
                                                       cell_barcode_col = "bc_wells")
               
               anndata_obj.write(filename = paths_config.unfiltered_gene_symbol_h5ad, 
                                             convert_strings_to_categoricals = True,
                                             compression = "gzip")
          
          # can remove large pandas dfs to save memory since now contained within anndata object
          del gene_metadata
          del cell_metadata
          gc.collect()
          
          return anndata_obj






@timer
@profile
def ensembl_to_symbol_scipy(anndata_obj:anndata.AnnData, gene_symbol_col: str, cell_barcode_col: str) -> AnnData:
     # make a new column in the gene level metadata (.var) that should be for the gene symbols - copying over symbol column to a new variable
     anndata_obj.var["gene_symbol"] = anndata_obj.var[gene_symbol_col]  # or whatever column has symbols
     # if there is no gene_symbol that maps to an ensembl ID, keep ensembl ID
     no_gene_symbol_mapping = anndata_obj.var["gene_symbol"].isna() | (anndata_obj.var["gene_symbol"] == "") # boolean table of whether there are missing gene mappings
     anndata_obj.var.loc[no_gene_symbol_mapping, "gene_symbol"] = anndata_obj.var.loc[no_gene_symbol_mapping, "gene_id"] # locations where Ture, find the location of the ensemble and replace with ensembl mapping
     gene_names_to_groupby = anndata_obj.var["gene_symbol"].values # returns a list of all gene names (symbols and ensemble if no gene symbol is mapped - same order as sparse matrix columns)
     unique_genes, orig_idx = numpy.unique(gene_names_to_groupby, return_inverse=True) # sorted unique gene names (less than gene columns of sparse as it is only unique occurences), the orig_idx in the sparse array where that each gene listed in the original sparse matrix column belongs - not same order as sorted unique gene names (same # columns as sparse)
     # convert the counts data in .X to a compressed sparse row format (CSR) whihc is not manipulating the data, just changing how the data is stored in memory
     # CSR is efficient for matrix muliplication and for row slicing
     # CSC (Compressed sparse column) is effecicnet for column slicing
     # COO (coordinate format) is good for construction but not computation
     # ensure sparse CSR
     csr_x = anndata_obj.X.tocsr()

     # returns a compressed sparse row matrix where rows are unique gene names (less than the total number of ensembl genes in matrix) and column names which is the total number of ensemble genes in the the matrix
     # a matrix of 1s and 0s, where 1 means that gene is a member of the index in the original matrix
     mapping_matrix = scipy.sparse.csr_matrix(
          (numpy.ones(len(orig_idx)), (orig_idx, numpy.arange(len(orig_idx)))),
          shape=(len(unique_genes), len(orig_idx))
          )

     # take a peak at the mapping matrix:
     mapping_matrix.shape
     pandas.DataFrame(
          mapping_matrix[:15, :10].toarray(),
          index=unique_genes[:15],
          columns=[f"orig_{i}" for i in range(10)]
          )

     collapsed_sparse = (mapping_matrix @ csr_x.T).T # transpose to get the matrix multiplication to work mapping_matrix = (unique genes x orig genes) @ (matrix multiply) csr.x.T = (orig genes x cells) , then transpose the matrix mult output should get collapsed_sparse = (cells x unique genes)
     anndata_obj_collapsed = anndata.AnnData(X = collapsed_sparse,
                                             obs = anndata_obj.obs.copy(),
                                             var = pandas.DataFrame(index = unique_genes))

     anndata_obj_collapsed.var["gene_symbol"] = anndata_obj_collapsed.var.index # set the names to be the gene_symbol names

     # set the rownames of cell metadata to be the barcoes/cellID
     anndata_obj_collapsed.obs.set_index(cell_barcode_col, inplace=True) # row names of the cell metadata matrix should be the barcode and well info for each cell
     anndata_obj_collapsed.obs.index.name = None # don't name the index/rowname
     anndata_obj_collapsed.obs_names_make_unique()

     # since on= is not specified as a parameter, it matches the dataframe based on row index; 
     # # note the bc_wells, sample, species, gene_count, tscp_count, mread_count, bc1_well, bc2_well, bc3_well metadata comes from Parse, 
     # so may be incaccurate for gene counts due to being based on Ensembl IDs rather than gene symbols
     anndata_obj_collapsed.obs = anndata_obj_collapsed.obs.join(anndata_obj.obs, how = "left", sort = False) 

     return anndata_obj_collapsed


@timer
@profile
def ensembl_to_symbol_pdsparse(anndata_obj:anndata.AnnData, gene_symbol_col: str, cell_barcode_col: str) -> AnnData:
    # generate all multi-index columns from tuple of the .var in anndata object
    list_of_tuples = pandas.MultiIndex.from_tuples([tuple(x) for x in anndata_obj.var[anndata_obj.var.columns].to_numpy()],
            names = anndata_obj.var.columns)

    # create a pandas sparse object from the anndata scipy sparse object
    # the fill values here is a nan and that is the compressed value
    sparse_pd_df = pandas.DataFrame.sparse.from_spmatrix(anndata_obj.X, index = anndata_obj.obs[cell_barcode_col], columns=list_of_tuples)

    # transpose so gene names are row names, and then groupby
    # the rowname index name that contains the gene symols and then sum
    # this can be slow because once transposed, it does need to be dense again when
    # groupby and sum are used, however, it is much more readable and easier
    # for code review, thereofore keeping it until, until we need to really scale it
    # the data then needs to be transposed back to match the expectation in
    # anndata
    sparse_pd_df = sparse_pd_df.T.groupby(level=gene_symbol_col).sum().T

    # convert to anndata
    anndata_obj_collapsed = anndata.AnnData(sparse_pd_df)

    # since on= is not specified as a parameter, it matches the dataframe based on row index; 
    # # note the bc_wells, sample, species, gene_count, tscp_count, mread_count, bc1_well, bc2_well, bc3_well metadata comes from Parse, 
    # so may be incaccurate for gene counts due to being based on Ensembl IDs rather than gene symbols
    anndata_obj_collapsed.obs = anndata_obj_collapsed.obs.join(anndata_obj.obs, how = "left", sort = False) 

    # make sure the index doesn't have a name or it causes problems with object formatting and saving the object
    anndata_obj.obs.index.name = None

    return anndata_obj_collapsed




@profile
def add_qcmetrics_and_metadata(anndata_obj:AnnData, sample_name:str, mito_regex:str, ribo_regex:list, add_metadata:list, doublet_rate:float, seed:int, paths_config:PathConfig) -> tuple(AnnData, dict):
    # first time project level metadata is initiated
    analysis_metadata = {}

    # add sample_name based on user input to metadata
    anndata_obj.obs["sample_name"] = sample_name
     
    # by running this, you wil get a list of boolean indicating if the condition is met in the same order as myAnnDataObj.var_names.  
    # Recall, this contains an array of genes names
    # args.mito_regex
    anndata_obj.var_names.str.startswith(mito_regex).sum() # count the number of true instances
    # to get the actual values that meet this criteria, a simple python list comprehension can be used; in this case we see our "regex" is correct
    mito_genes = [gene for gene in anndata_obj.var_names if re.search(mito_regex, gene)]
    analysis_metadata['mito_genes'] = mito_genes
    analysis_metadata['total_mito_genes'] = len(mito_genes)

    logging.info("There were a total of {} mitochondrial genes detected.".format(len(mito_genes)))

    # since the mito list looks correct, we want to add this boolean list to our gene/feature metadata.  NOTE!! This is different from our sample/cell metadata
    # remember .var in our gene/feature metadata
    anndata_obj.var["is_mito"] = anndata_obj.var_names.str.match(mito_regex)



    # by running this, you wil get a list of boolean indicating if the condition is met in the same order as myAnnDataObj.var_names.  
    # Recall, this contains an array of genes names
    # args.ribo_regex
    anndata_obj.var_names.str.startswith(tuple(ribo_regex)).sum() # count the number of true instances
    # to get the actual values that meet this criteria, a simple python list comprehension can be used; in this case we see our "regex" is correct
    ribo_genes = [gene for gene in anndata_obj.var_names if gene.startswith(tuple(ribo_regex))]
    analysis_metadata['ribo_genes'] = ribo_genes
    analysis_metadata['total_ribo_genes'] = len(ribo_genes)


    logging.info("There were a total of {} ribosomal genes detected.".format(len(ribo_genes)))


    # since the ribosomal list looks correct, we want to add this boolean list to our gene/feature metadata.  NOTE!! This is different from our sample/cell metadata
    # remember .var in our gene/feature metadata
    anndata_obj.var['is_ribo'] = anndata_obj.var_names.str.startswith(tuple(ribo_regex))

    # calculate mitochondrial and ribosomal percentages per cell and add to metadata
    logging.info("Calculating mitochondrial and ribosomal metrics.")
    scanpy.pp.calculate_qc_metrics(anndata_obj, qc_vars = ['is_mito', 'is_ribo'], log1p=True, expr_type="counts", var_type="genes", inplace=True, layer = None, percent_top = None)

    # estimate doublets - not yet removed
    logging.info("Predicting singlets vs doublets in the dataset.")
    scdblfinder_obj = pyscdblfinder.ScDblFinder(anndata_obj, random_state=seed) # automatically addess class and score columns to the anndata_obj that is passed into the function
    scdblfinder_obj.run(dbr=doublet_rate) 
    # save scDblFinder object as pickled file
    with open(paths_config.scdblfinder_pickle, "wb") as f:
        pickle.dump(scdblfinder_obj, f)
     
    try:
        analysis_metadata['predicted_singlets'] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["singlet"]
    except KeyError:
        logging.critical("Singlet cells not found. There is a critical error in your dataset.")
        analysis_metadata['predicted_singlets'] = 0
        sys.exit()
    try:
        analysis_metadata['predicted_doublets'] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["doublet"]
    except KeyError:
        logging.warning("No doublets detected in the data set.  This is unusual and you may want to take a deeper dive into your input data or double rate parameter.")
        analysis_metadata['predicted_doublets'] = 0
     
    # if users has metadata parameter populated, add all metadata listed to annData cell level metadata
    if add_metadata != None:
        logging.info("Adding user provided cell level metadata to the .obs data slot of anndata object.")
        for colname, cellvalue in add_metadata:
            anndata_obj.obs[colname] = cellvalue


    # save h5ad; file type is inferred from filename extension
    anndata_obj.write(filename = paths_config.unfiltered_gene_symbol_h5ad,
            convert_strings_to_categoricals = True,
            compression = "gzip")
     
     # save dictionary of data generated from functions
    with open(paths_config.analysis_metadata_pickle, "wb") as f:
        pickle.dump(analysis_metadata, f)
    
    # scdblfinder object is already saved as a pickled object so can remove to save memory
    del scdblfinder_obj
    gc.collect()

    return anndata_obj, analysis_metadata



@profile
def cell_filtering(anndata_obj:AnnData, remove_doublets:bool, analysis_metadata:dict, paths_config:PathConfig, min_counts:int | None = None, min_genes:int | None = None, max_counts:int | None = None, max_genes:int | None = None, max_mito:float| None = None) -> tuple(AnnData, dict):
    logging.info("Cell level filtering")

    # store the number of cells before and after filtering in env_vars 
    analysis_metadata["total_cells_prefilter"] = len(anndata_obj.obs.index)
    analysis_metadata["total_genes_prefilter"] = len(anndata_obj.var.index)

    if min_counts != None:
        scanpy.pp.filter_cells(anndata_obj, min_counts=min_counts, inplace = True)
    if min_genes != None:
        scanpy.pp.filter_cells(anndata_obj, min_genes=min_genes, inplace = True)
    if max_counts != None:
        scanpy.pp.filter_cells(anndata_obj, max_counts=max_counts, inplace = True)
    if max_genes != None:
        scanpy.pp.filter_cells(anndata_obj, max_genes=max_genes, inplace = True)
     
    # Keep cells that have < mito_contam percent
    anndata_obj = anndata_obj[anndata_obj.obs['pct_counts_is_mito'] < max_mito, :]

    if remove_doublets == True:
        anndata_obj = anndata_obj[anndata_obj.obs['scDblFinder_class'] == "singlet", :]
     
    # store the number of cells after filtering in env_vars 
    analysis_metadata["total_cells_remaining_postfilter"] = len(anndata_obj.obs.index)
    analysis_metadata["total_genes_remaining_postfilter"] = len(anndata_obj.var.index)

     
    # save h5ad; file type is inferred from filename extension
    anndata_obj.write(filename = paths_config.filtered_h5ad,
            convert_strings_to_categoricals = True,
            compression = "gzip")
     
    # save metrics generated from function as pickled file
    with open(paths_config.analysis_metadata_pickle, "wb") as f:
        pickle.dump(analysis_metadata, f)

    return anndata_obj, analysis_metadata


# when if size_factor is set to None, then scanpy's default depth normalization of median total counts
# if want to normalize to 10k, then set to 10000, if want CPM normalization then set to 1000000
# normalization if then followed by a log transformation +1 pseudocount
# save_memory is a boolean to save memory footprint of storing the whole array in memory vs chunks of the array at a time
# this does not parallelize, still single threaded, just sotred the dense values as needed
# if save_memory is True, data_chunk_size is required and that roughly equates to the bumber of cells per batch to split the array for memory saving purposes
# if save_memory is False then data_chunk_size is set to None as the parameter does not matter; full array is stored in memory
@profile
def normalize_and_transform(anndata_obj:AnnData, paths_config = PathConfig, size_factor: int | None = None, save_memory:bool = False, data_chunk_size:int|None = None) -> AnnData:
    logging.info("Normalization and transformation of counts")

    # save a copy of the raw counts to the counts layer (X is the active layer and when normalization is applied it applies to X)
    anndata_obj.layers['counts'] = anndata_obj.X.copy()
    scanpy.pp.normalize_total(anndata_obj, target_sum = None, layer = None, exclude_highly_expressed = False, inplace = True)
    scanpy.pp.log1p(anndata_obj, base = None, chunked = None, chunk_size = None, layer = None)

    # save h5ad; file type is inferred from filename extension
    anndata_obj.write(filename = paths_config.filtered_h5ad,
            convert_strings_to_categoricals = True,
            compression = "gzip")
     
    return anndata_obj
     
@profile
def calculate_cell_cycle(anndata_obj:AnnData, s_genes:Iterable[str], g2m_genes:Iterable[str], analysis_metadata:dict, paths_config:PathConfig) -> AnnData:
    scanpy.tl.score_genes_cell_cycle(anndata_obj, s_genes=s_genes, g2m_genes=g2m_genes)
    analysis_metadata["total_S_phase_cells"] = int((anndata_obj.obs["phase"] == "S").sum())
    analysis_metadata["total_G1_phase_cells"] = int((anndata_obj.obs["phase"] == "G1").sum())
    analysis_metadata["total_G2M_phase_cells"] = int((anndata_obj.obs["phase"] == "G2M").sum())
    
    # save metrics generated from function as pickled file
    with open(paths_config.analysis_metadata_pickle, "wb") as f:
        pickle.dump(analysis_metadata, f)
    return anndata_obj, analysis_metadata


@profile     
def identify_and_transform_hvgs(anndata_obj:AnnData, analysis_metadata:dict, paths_config:PathConfig, n_hvgs:int, hvg_ignore:str, vars_to_regress:Iterable[str], threads:int) -> tuple(AnnData, dict): 
    logging.info("Scale data, regress covariates, and identify highly variable genes")

    '''
    FROM SCANPY DOCS FOR: scanpy.pp.highly_variable_genes()
    The following may help when comparing to Seurat’s naming: If batch_key=None and flavor='seurat', 
    this mimics Seurat’s FindVariableFeatures(…, method='mean.var.plot'). If batch_key=None and 
    flavor='seurat_v3'/flavor='seurat_v3_paper', this mimics Seurat’s FindVariableFeatures(..., method='vst'). 
    If batch_key is not None and flavor='seurat_v3_paper', this mimics Seurat’s SelectIntegrationFeatures.
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html#scanpy.pp.highly_variable_genes
    '''   
    # hvgs are returned in .var
    scanpy.pp.highly_variable_genes(anndata_obj, n_top_genes=n_hvgs, flavor = "seurat", filter_unexpressed_genes = False, batch_key=None, inplace = True) # note, chose to stick to always Seurat method to reduce complexity; other options are seurat_v3 which we should avoid and cell_ranger; also batch is always None here because it is for independent samples so there will not be a batch
    anndata_obj.layers["normalized"] = anndata_obj.X.copy() # keep a copy of the just the normalized transformed data before regression and scaling (X is the active layer)
    scanpy.pp.regress_out(anndata_obj, keys = vars_to_regress, n_jobs = threads)
    scanpy.pp.scale(anndata_obj, max_value=None, zero_center = True)

  
    # for HVG, set genes to false that are part of B cell clonotypes
    # gets hvg list of top x most variable genes and has
    genes_to_ignore_for_clustering = anndata_obj.var[anndata_obj.var.index.str.contains(hvg_ignore, regex=True)].index.tolist() #list of all genes that should be removed from HVG
    # ensures that if this list is empty it won't throw any errors due to list being empty;
    if len(genes_to_ignore_for_clustering) > 0:
        anndata_obj.var.loc[genes_to_ignore_for_clustering, "highly_variable"] = False # set all of these genes to False under the highly variable column
        analysis_metadata['genes_ignored_for_hvg_selection'] = genes_to_ignore_for_clustering
    else:
        analysis_metadata['genes_ignored_for_hvg_selection'] = "None" # keeping as None string not None object, as this is just human-readable metadata to keep track of methods


    # plot HVGs and save in qc_images subdirectory
    pretty_highly_variable_genes(anndata_obj = anndata_obj, 
            genes_to_ignore_for_clustering = genes_to_ignore_for_clustering,
            paths_config = paths_config,
            n_top_genes = 10)


    # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
    #anndata_obj.layers["regressed_scaled"] = anndata_obj.X.toarray()
    # save h5ad; file type is inferred from filename extension
    anndata_obj.write(filename = paths_config.filtered_h5ad,
            convert_strings_to_categoricals = True,
            compression = "gzip")
     
    # save metrics generated from function as pickled file
    with open(paths_config.analysis_metadata_pickle, "wb") as f:
        pickle.dump(analysis_metadata, f)
     
    return anndata_obj, analysis_metadata


@profile  
def pca(anndata_obj:AnnData, analysis_metadata:dict, pca_var_change:float, seed:int, paths_config = PathConfig) -> tuple(AnnData, dict):
    logging.info("Running PCA")

    '''
    FROM SCANPY DOCS FOR: scanpy.pp.pca()
    when svd_solver = auto, choose automatically depending on the size of the problem: Will use 'full' 
    for small shapes and 'randomized' for large shapes.
    the embedding is stored as obsm['X_pca'] (PCA representation data), 
    the loadings as varm['PCs'] (gene loadings), and the the parameters in 
    uns['pca']['variance_ratio'] (ratio of variance explained), 
    uns['pca']['variance'] (explained variance; equalivalent to eigenvalues of the covariance matrix)
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html#scanpy.pp.pca
    '''
    scanpy.pp.pca(anndata_obj, n_comps = 50, zero_center = True, svd_solver = 'auto',  mask_var="highly_variable", random_state = seed, chunked = False, layer = None) # layer=None means to use the values in the active layer (.X) for the PCA

    #################################################################
    ##### store number of PCs to use for clustering in env_vars #####
    #################################################################
    # convert to numpy array so can take advantage of the efficent diff function in numpy that subtracts and store the diffrence of consecutive values in an array
    pct_var_explained = numpy.asarray((anndata_obj.uns['pca']['variance_ratio'])/(anndata_obj.uns['pca']['variance_ratio'].sum())*100)
    change_in_var = -numpy.diff(pct_var_explained)  # diffs[i] = pct_var[i] - pct_var[i+1], the (-) is multiplying it by -1, since normally we want to calc  pct_var[i+1] - pct_var[i], but default is 1st value - 2nd value, so just multiply by -1 to make the values non-negative

    # returns the first PC where the change in varinces is less than thte change specified by user
    '''
    np.argmax() on a boolean array is a common trick to find the first index where a condition is True, without writing an explicit loop.
    Why it works: change_in_var < 0.1 produces a boolean array ([False, False, True, True, ...]). Python treats True as 1 and False as 0. 
    argmax() returns the index of the first occurrence of the maximum value in an array — and since True (1) is the max possible value in 
    a boolean array, argmax() finds the first True, i.e. the first index satisfying your condition. It's a fast, vectorized way to do 
    "find first index where X" without a Python-level loop.
    '''
    idx = numpy.argmax(change_in_var < pca_var_change)  # first True index (note our numpy array is naturally sorted), or 0 if none are True
    # the if/else logic is only necessary to confirm argmax wored and then to add 2 to the index number in order to return the correct PC cutoff; this is because python is 0 indexed, and we want to go 1 past the index to get the index to get the proper value, hence +2
    if change_in_var[idx] < pca_var_change:
        analysis_metadata["pcs_to_use"] = idx+2
        logging.info("Total PCs to use: {}".format(analysis_metadata["pcs_to_use"]))
    else:
        analysis_metadata["pcs_to_use"] = len(change_in_var)+1 #if there are no PCs that meet the criteria, use all PCs calculated
        logging.info("Total PCs to use: {}".format(analysis_metadata["pcs_to_use"]))

    # plot elbow plot/
    ax = scanpy.pl.pca_variance_ratio(anndata_obj, n_pcs=50, log=True, show = False)
    fig = plt.gcf()
    ax = fig.axes[0]
    # Vertical dashed line
    ax.axvline(x=analysis_metadata["pcs_to_use"], linestyle="--", color="red")

    # Vertical text along the line, near the top
    ax.text(analysis_metadata["pcs_to_use"], 0.97, f"PC={analysis_metadata["pcs_to_use"]}", rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())
    plt.gcf().savefig(paths_config.qc_dir / "pca_elbow_plot_of_hvgs.png", bbox_inches = "tight")
    plt.close()
    
    # plot pc loadings - top hvgs driving each PC
    scanpy.pl.pca_loadings(anndata_obj, components = '1,2,3,4,5,6,7,8,9,10', include_lowest = True, show = False) # include_lowest means to show the features that have the highest and lowest loadings
    plt.gcf().savefig(paths_config.qc_dir / "pca_gene_loadings_of_hvgs.png", bbox_inches = "tight")
    plt.close()

    pretty_pca_loadings(anndata_obj = anndata_obj, 
            total_pcs_to_summarize = 10, 
            n_genes_to_plot_per_direction = 5,
            paths_config = paths_config)


     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     # but there will be added PCA calculations and data to obsm, varm, and uns
    anndata_obj.write(filename = paths_config.filtered_h5ad,
            convert_strings_to_categoricals = True,
            compression = "gzip")
     
    # save metrics generated from function as pickled file
    with open(paths_config.analysis_metadata_pickle, "wb") as f:
        pickle.dump(analysis_metadata, f)
     
    return anndata_obj, analysis_metadata


@profile
def neighbors_umap_clust(anndata_obj:AnnData, n_neighbors:int, n_pcs:int, dist_metric:str, seed:int, resolutions:Iterable, addl_genes:Iterable, paths_config:PathConfig) -> AnnData:
    logging.info("Identify neighbors, generate UMAP embeddings, calulate clustering resolutions")

    '''
    FROM SCANPY DOCS FOR: scanpy.pp.neighbors()
    If not specified, the neighbors data is stored in .uns['neighbors'], distances and connectivities are stored in .obsp['distances'] and 
    .obsp['connectivities'] respectively. If specified, the neighbors data is added to .uns[key_added], distances are stored in 
    .obsp[f'{key_added}_distances'] and connectivities in .obsp[f'{key_added}_connectivities'].
    https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html#scanpy.pp.neighbors
    '''
    scanpy.pp.neighbors(anndata_obj, n_neighbors=n_neighbors, n_pcs = n_pcs, knn=True, method="umap", metric = dist_metric, random_state = seed)
     
    # calculate umap embeddings
    # n_components is the number of dimisions to plot umap embedding into; 2 is typical, and 3 is if you want a 3D
    # umap; this is not in reference to the number of PC compoenents to use, which is used to determine the neighborhood connectivity
    # method: umap or rapids (rapids for GPU accelerated and umap for CPU/non-accelerated)
    # min_dist defaults to 0.5
    # If not specified, the embedding is stored as obsm['X_umap'] and the the parameters in uns['umap']. If specified, the embedding is stored 
    # as obsm[key_added] and the the parameters in uns[key_added].
    scanpy.tl.umap(anndata_obj, n_components = 2, random_state = seed, method = 'umap', min_dist = 0.5, key_added = "initial_umap")

    '''
     scanpy recommends using the leiden algorithm for clustering using scanpy.tl.leiden(). Clustering is based upon the previously calculated neighborhood graphs 
     on the higher dimensional space.  If you prefer to use the louvain algorithm similar to what Seurat implements, you can use scanpy.tl.louvain().

     The options I haev set for the Leiden algorithm for identifying cluster boundries are the following:

     * key_added: if you don't set this, it defaults to leiden but since want to calulcate many resolutions are one time using the for loop, I make a key that stores all resolutions as leiden_res_ followed by the reolution to two decimal places.
     *resolution: this is an float of the resolution you want to calculate; smaller resolutions results in broader and fewer clusters (more global), while larger resolutions result in more granular smaller clusters but more clusters (more localized/specific clusters)
     *flavor: the default is igraph but this allows you to specify which implementation of the leiden package you want to use
     *n_iterations: this is the number of iterations you want the algorithm to go through; I set it to -1 which is the default, which means go through as many iterations as it needs before it completes/converges. No limit.
     *random_state: is essentially a seed since all of the clustering is unsupervised, so can be set to an integer for keeping results reproducible between runs
     *use_weights: if True, edge weights from the graph are used in the computation (placing more emphasis on stronger edges).
     *neighbors:_key by default this is to None, which means the neighbors calcaulated before are stored in the default location of .obsp["connectivities"]. This shouldn't be changed unless you have multiple neighbor graphs calculated or you changed the name of the key when running the neighbors command above.
     '''     
    # for loop through a few different resolutions to test a few options and then refine from here if needed
    for res in resolutions:
        print(res)
        scanpy.tl.leiden(anndata_obj, key_added=f"leiden_res_{res:.2f}", resolution=res, flavor="igraph", n_iterations= -1, random_state = seed, use_weights = True, neighbors_key = None)
     

     #clustree resolution check
    fig = pyclustree.clustree(anndata_obj, 
        [f"leiden_res_{res:.2f}" for res in resolutions],
        title="Clustree",
        x_spacing = 7, # space between nodes along the x-axis; default is 2.5
        y_spacing = 2,
        edge_weight_threshold=0.05,  # the minimum fraction of the parent cluster assigned to the child cluster to plot
        show_fraction=True  # show the fraction of cells in each cluster
        )
    fig.set_size_inches(15, 12)
    fig.set_dpi(100)

    # Increase text size
    for ax in fig.axes:
        for text in ax.texts:
            text.set_fontsize(10)

    fig.savefig(os.path.join(paths_config.cluster_dir, f"clustree_overall_stability.png"),
        dpi=300,
        bbox_inches="tight"
        )

    # clusteree by gene expression

    # Keep only genes that exist
    addl_genes = [genes for genes in addl_genes if genes in anndata_obj.var_names]
    idx = [anndata_obj.var_names.get_loc(genes) for genes in addl_genes]

    # Small AnnData containing only the genes of interest
    tmp = anndata_obj[:, addl_genes].copy()

    # Use the normalized layer as the expression matrix
    tmp.X = anndata_obj.layers["normalized"][:, idx].copy()

    # Set raw from this normalized matrix
    tmp.raw = tmp.copy()

    for gene in addl_genes:
        fig = pyclustree.clustree(tmp,
            [f"leiden_res_{res:.2f}" for res in resolutions],
            title=f"Clustree colored by {gene}",
            edge_weight_threshold=0.05,
            x_spacing=7,
            y_spacing=2,
            node_color_gene=gene,
            node_colormap="Reds",
            show_colorbar=True,
            show_fraction=True  # show the fraction of cells in each cluster
            )
        fig.set_size_inches(15, 12)
        fig.set_dpi(100)
        
        # Increase text size
        for ax in fig.axes:
            for text in ax.texts:
                text.set_fontsize(10)

        fig.savefig(os.path.join(paths_config.cluster_dir, f"clustree_{gene}.png"),
            dpi=300,
            bbox_inches="tight"
            )
    
     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     # but there will be added PCA calculations and data to obsm, varm, and uns
    anndata_obj.write(filename = paths_config.filtered_h5ad,
        convert_strings_to_categoricals = True,
        compression = "gzip")

    # save memory and remove tmp, only required for clustree
    del tmp
    gc.collect()

    pattern = re.compile(r"^leiden_res_[0-9]*")
    all_resolutions = [x for x in anndata_obj.obs_keys() if pattern.search(x)]
    for cluster_res in all_resolutions:
        resolution_dir = paths_config.resolution_dir(cluster_res)
        resolution_dir.mkdir(parents=True, exist_ok=True)
        plot_reductions(anndata_obj = anndata_obj , 
                            reduction_name = "initial_umap", 
                            layer = "normalized", 
                            ncol_layout = 1,   # should always be 1 if clsuter label = True
                            continuous_col = "magma",  # for continuous values
                            categorical_col = "Set3", # for categorical values
                            marker = "o" ,
                            groupby_col = cluster_res,
                            cluster_label = True, # labeling of clusters should be reserved for categorical variables only
                            file_savename = f"{cluster_res}_initial_umap",
                            save_path = resolution_dir)
        
        # plot the normalized expression of the core genes on a violin plot
        for genes in addl_genes:
            plot_violin(anndata_obj = anndata_obj,
                        layer = "normalized",
                        groupby_col = cluster_res,
                        data_to_plot=genes,
                        file_savename=f"{genes}_normalized_expression_violin_plot",
                        save_path = resolution_dir
                        )

    # these plots are not at the per resolution level
    plot_reductions(anndata_obj = anndata_obj , 
                        reduction_name = "initial_umap", 
                        layer = "normalized", 
                        ncol_layout = 3, 
                        continuous_col = "magma",  # for continuous values
                        categorical_col = "Set3", # for categorical values
                        marker = "o",
                        groupby_col = addl_genes, # core_genes_to_plot
                        cluster_label = False, 
                        file_savename = "core_genes_norm_expression_umap",
                        save_path = paths_config.cluster_dir)

    return anndata_obj

@profile
def final_qc_images(anndata_obj:AnnData, reduction:str, layer:str, paths_config:PathConfig, continuous_color_pal:str="magma", categorical_color_pal:str="Set3") -> None:
    logging.info("Generating final images for plotting QC metadata and cell cycle phase on different embeddings")

    # final set of QC checks on umap embeddings and pca  
    plot_reductions(anndata_obj = anndata_obj, 
                reduction_name = reduction, 
                layer = layer, 
                ncol_layout = 2,  
                continuous_col = continuous_color_pal,  # for continuous values: magma
                categorical_col = categorical_color_pal,
                marker = "o",
                groupby_col = ['n_counts', 'n_genes', 'pct_counts_is_mito', 'pct_counts_is_ribo'],
                cluster_label = False,
                file_savename = f"final_qc_{reduction}",
                save_path = paths_config.qc_dir)
    
    plot_reductions(anndata_obj = anndata_obj, 
                reduction_name = reduction, 
                layer = layer, 
                ncol_layout = 1,  
                continuous_col = continuous_color_pal,
                categorical_col = categorical_color_pal,
                marker = "o",
                groupby_col = "phase",
                cluster_label = True,
                file_savename = f"phase_{reduction}",
                save_path = paths_config.qc_dir)
    


'''
TESTING TIME AND MEMORY
with open("/mnt/IM_drive/Jordan_Abbott/parse_1M_cell_cardiac_surgery_Tcell_data_01072025/scRNA_analysis_04032026/B2107_initial_analysis/workspace_files/vars_and_params_B2107_independent_analysis_04032026.pkl", 'rb') as f:
    env_vars = pickle.load(f)

# read in previously analyzed data
orig_anndata_obj = scanpy.read_h5ad(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "unfiltered_original_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])))

# set working directory
os.chdir(env_vars["workingDir"])

print("pandas dense", end="", flush=True)
test1 = ensembl_to_symbol_pddense(anndata_obj = orig_anndata_obj, gene_symbol_col = "gene_name", cell_barcode_col = "bc_wells")
print("scipy", end = "", flush = True)
test1 = ensembl_to_symbol_scipy(anndata_obj = orig_anndata_obj, gene_symbol_col = "gene_name", cell_barcode_col = "bc_wells")
print("pandas sparse", end = "", flush= True)
test1 = ensembl_to_symbol_pdsparse(anndata_obj = orig_anndata_obj, gene_symbol_col = "gene_name", cell_barcode_col = "bc_wells")
'''