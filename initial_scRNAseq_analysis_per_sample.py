import scanpy
from typing import Iterable
from jsonschema.benchmarks.subcomponents import v
from matplotlib.pylab import sca
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
def env_setup() -> dict:
     import datetime

     # initialize environmental variable config file for parameters used in the analysis
     env_vars = {
          "working_dir": args.working_dir,
          "save_prefix": args.save_prefix,
          "date": datetime.date.today().strftime("%m%d%Y"),
          "filtered_feature_bc_matrix": args.filtered_feature_bc_matrix,
          "sample_name": args.sample_name,
          "min_cells_expressed": args.min_cells_expressed,
          "min_unique_genes": args.min_unique_genes,
          "min_umi_counts": args.min_umi_counts,
          "mito_contam": args.mito_contam,
          "min_complexity": args.min_complexity,
          "hvg_features": args.hvg_features,
          "pca_var_change": args.pca_var_change,
          "neighbors": args.neighbors,
          "resolutions": args.resolutions,
          "top_genes_per_cluster_to_plot": args.top_genes_per_cluster_to_plot,
          "additional_genes_to_plot": args.additional_genes_to_plot
          }

     os.chdir(env_vars["working_dir"])

     # create directory architecture
     os.mkdir("h5ad_objects")
     os.mkdir("qc_images")
     os.mkdir("workspace_files")
     #os.mkdir("adt_analysis") # if CITE or LIBRA is used
     #os.mdkir("vdj") # if VDJ is used

     # create subdirectories based on all resolutions listed at runtime
     for res in env_vars["resolutions"]:
          pathlib.Path("/".join(["cluster_analysis", res])).mkdir(parents=True, exist_ok=True) # exist_ok = True means don't throw error if exits; but if does not exist make directory; does not overwrite anything if dir exists

     # save env variables under workspace_files under a vars_and_params pickle file
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)

     return env_vars


@profile
def resume() -> tuple[dict, AnnData, AnnData]:
     # load a pre-saved pickle file
     with open(args.env_vars_config, 'rb') as f:
          env_vars = pickle.load(f)

     # read in previously analyzed data
     orig_anndata_obj = scanpy.read_h5ad(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "unfiltered_original_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])))
     anndata_obj = scanpy.read_h5ad(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_geneSymbol_converted_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])))

     # set working directory
     if args.update_working_dir != None:
          env_vars["working_dir"] == args.update_working_dir
     os.chdir(env_vars["working_dir"])

     return env_vars, orig_anndata_obj, anndata_obj




'''
anndata_obj = original anndata object that contains a column that has the ensembl and gene symbol mapping dataframe located in .var
gene_symbol_col = column name in .var that contains the gene symbols (human readable, non-Ensembl)
cell_barcode_col = column name in .obs that contains the barcode/cell names
'''
@timer
@profile
def ensembl_to_symbol_pddense(anndata_obj:anndata.AnnData, gene_symbol_col: str, cell_barcode_col: str) -> AnnData:
     # force pandas 3 to be compatible with anndata v 0.12.16
     pandas.options.mode.string_storage = "python"
     anndata.settings.allow_write_nullable_strings = True

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
def generate_object(env_vars):
     if args.platform == "10x":
          pass
     elif args.platform == "parse":
          # create annData object
          orig_anndata_obj = scanpy.read_mtx(filename= os.path.join(env_vars["filtered_feature_bc_matrix"], "count_matrix.mtx.gz"))

          # read in gene and cell metadata
          gene_metadata = pandas.read_csv(filepath_or_buffer= os.path.join(env_vars["filtered_feature_bc_matrix"], "all_genes.csv.gz"),
                                          compression="gzip",
                                          engine="pyarrow") #pyarrow is more memory efficient for large csv files, although parquet may be even better and c engine is very fast but may be less memory efficient than pyarrow
          cell_metadata = pandas.read_csv(filepath_or_buffer= os.path.join(env_vars["filtered_feature_bc_matrix"], "cell_metadata.csv.gz"),
                                          compression="gzip",
                                          engine="pyarrow")

          # add gene and cell level data to anndata object
          orig_anndata_obj.var = gene_metadata # a dataframe that contains gene_id (ensembl), gene_name (symbol), genome (genome version used) - based on Parse input
          orig_anndata_obj.obs = cell_metadata # a dataframe that contain the cell level metadata, very similar to Seurat @meta.data, however need to reassign row names to be cell barcodes

          # save h5ad object; file type is inferred from filename extension
          orig_anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "unfiltered_original_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                                 convert_strings_to_categoricals = True,
                                 compression = "gzip")

     elif args.platform == "bdrhapsody":
          pass


@profile
def add_qcmetrics_and_metadata(anndata_obj:AnnData, env_vars:dict, mito_regex:str, ribo_regex:list, add_metadata:list, doublet_rate:float, seed:int) -> tuple(AnnData, dict):
     # add sample_name based on user input to metadata
     anndata_obj.obs["sample_name"] = env_vars["sample_name"]
     
     # by running this, you wil get a list of boolean indicating if the condition is met in the same order as myAnnDataObj.var_names.  
     # Recall, this contains an array of genes names
     # args.mito_regex
     anndata_obj.var_names.str.startswith(mito_regex).sum() # count the number of true instances
     # to get the actual values that meet this criteria, a simple python list comprehension can be used; in this case we see our "regex" is correct
     mito_genes = [gene for gene in anndata_obj.var_names if gene.startswith(mito_regex)]
     env_vars['mito_genes'] = mito_genes

     logging.info("There were a total of {} mitochondrial genes detected.".format(len(mito_genes)))

     # since the mito list looks correct, we want to add this boolean list to our gene/feature metadata.  NOTE!! This is different from our sample/cell metadata
     # remember .var in our gene/feature metadata
     anndata_obj.var['is_mito'] = anndata_obj.var_names.str.startswith(mito_regex)



     # by running this, you wil get a list of boolean indicating if the condition is met in the same order as myAnnDataObj.var_names.  
     # Recall, this contains an array of genes names
     # args.ribo_regex
     anndata_obj.var_names.str.startswith(tuple(ribo_regex)).sum() # count the number of true instances
     # to get the actual values that meet this criteria, a simple python list comprehension can be used; in this case we see our "regex" is correct
     ribo_genes = [gene for gene in anndata_obj.var_names if gene.startswith(tuple(ribo_regex))]
     env_vars['ribo_genes'] = ribo_genes

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

     env_vars["double_rate_param"] = doublet_rate
     env_vars["scdblfinder_object"] = pathlib.Path("".join(["workspace_files/unfiltered_scdblfinder_obj_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"]))

     # save scDblFinder object as pickled file
     with open(pathlib.Path("".join(["workspace_files/unfiltered_scdblfinder_obj_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), 'wb') as object_file:
          pickle.dump(scdblfinder_obj, object_file)
     
     try:
          env_vars['predicted_singlets'] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["singlet"]
     except KeyError:
          logging.critical("Singlet cells not found. There is a critical error in your dataset.")
          env_vars['predicted_singlets'] = 0
          sys.exit()
     try:
          env_vars['predicted_doublets'] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["doublet"]
     except KeyError:
          logging.warning("No doublets detected in the data set.  This is unusual and you may want to take a deeper dive into your input data or double rate parameter.")
          env_vars['predicted_doublets'] = 0
     
     # if users has metadata parameter populated, add all metadata listed to annData cell level metadata
     if add_metadata != None:
          logging.info("Adding user provided cell level metadata to the .obs data slot of anndata object.")
          for colname, cellvalue in add_metadata:
               anndata_obj.obs[colname] = cellvalue


     # save h5ad; file type is inferred from filename extension
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "unfiltered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)

     return anndata_obj, env_vars




@profile
def qc_figures(anndata_obj:AnnData, env_vars:dict, status:str) -> None:
     logging.info("Generating QC images on {} data".format(status))

     plt.figure(figsize=(10, 6))
     seaborn.histplot(data=anndata_obj.obs,
                      x="total_counts",
                      hue="sample_name",
                      kde=True,
                      alpha=0.2,
                      log_scale=True).set_title('Distribution of UMI/reads/counts per cell ({})'.format(status))
     plt.axvline(env_vars["min_umi_counts"],linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
     plt.tight_layout()
     plt.savefig("qc_images/{}_umis.png".format(status), dpi=300)
     #plt.show()


     plt.figure(figsize=(10, 6))
     seaborn.histplot(data=anndata_obj.obs,
                      x="n_genes_by_counts",
                      hue="sample_name",
                      kde=True,
                      alpha=0.2,
                      log_scale=True).set_title('Distribution of uniquely expressed genes per cell ({})'.format(status))
     plt.axvline(env_vars["min_unique_genes"],linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
     plt.tight_layout()
     plt.savefig("qc_images/{}_features.png".format(status), dpi=300)
     #plt.show()


     plt.figure(figsize=(10, 6))
     seaborn.histplot(data=anndata_obj.obs,
                      x="pct_counts_is_mito",
                      hue="sample_name",
                      kde=True,
                      alpha=0.2,
                      log_scale=True).set_title('Distribution of mitochondrial counts per cell'.format(status))
     plt.axvline(env_vars["mito_contam"],linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
     plt.tight_layout()
     plt.savefig("qc_images/{}_mito.png".format(status), dpi=300)
     #plt.show()


     fig, ax = plt.subplots(figsize=(10, 6))
     grey_cmap = LinearSegmentedColormap.from_list("custom_grey", ["#d3d3d3", "#1a1a1a"])

     scatter = ax.scatter(x = anndata_obj.obs["total_counts"],
                          y = anndata_obj.obs["n_genes_by_counts"],
                          c=anndata_obj.obs["pct_counts_is_mito"],
                          cmap=grey_cmap,
                          s=3,       # dot size
                          alpha=1.0,
                          vmin=anndata_obj.obs["pct_counts_is_mito"].min(),
                          vmax=anndata_obj.obs["pct_counts_is_mito"].max()
                          )

     # Linear fit in log space
     x = anndata_obj.obs["total_counts"]
     y = anndata_obj.obs["n_genes_by_counts"]
     log_x = numpy.log10(x)
     log_y = numpy.log10(y)
     m, b = numpy.polyfit(log_x, log_y, 1)
     x_line = numpy.logspace(numpy.log10(x.min()), numpy.log10(x.max()), 300)
     y_line = 10 ** (m * numpy.log10(x_line) + b)
     ax.plot(x_line, y_line, color="red", linewidth=1.5, linestyle="--", label=f"fit: slope={m:.2f}")


     ax.axvline(x=env_vars["min_umi_counts"], linestyle="dotted", color="orange", linewidth=2)
     ax.axhline(y=env_vars["min_unique_genes"], linestyle="dotted", color="orange", linewidth=2)
     plt.colorbar(scatter, ax=ax, label="pct_counts_is_mito")
     ax.set_xscale("log")
     ax.set_yscale("log")
     ax.set_xlabel("total_counts")
     ax.set_ylabel("n_genes_by_counts")
     ax.set_title("total_counts vs n_genes_by_counts")

     plt.tight_layout()
     plt.savefig("qc_images/{}_summary_joint.png".format(status), dpi=300)
     #plt.show()
 

     # highlighting singlets vs doublets
     fig, ax = plt.subplots()

     # define color + marker per category
     style_map = {
     "singlet": {"color": "steelblue", "marker": "o"},
     "doublet": {"color": "crimson", "marker": "^"},
     }

     # must loop through each category
     for cls, style in style_map.items():
          subset = anndata_obj.obs[anndata_obj.obs["scDblFinder_class"] == cls]

          ax.scatter(
               x=subset["total_counts"],
               y=subset["n_genes_by_counts"],
               c=style["color"],
               marker=style["marker"],
               s=3,
               alpha=1.0,
               label=cls
          )

     ax.set_xlabel("total_counts")
     ax.set_ylabel("n_genes_by_counts")
     ax.legend()
     plt.tight_layout()
     plt.savefig("qc_images/{}_doublet_summary_joint.png".format(status), dpi=300)
     # plt.show()

@profile
def cell_filtering(anndata_obj:AnnData, env_vars:dict, remove_doublets:bool, min_counts:int | None = None, min_genes:int | None = None, max_counts:int | None = None, max_genes:int | None = None, max_mito:float| None = None) -> tuple(AnnData, dict):
     logging.info("Cell level filtering")

     # store the number of cells before and after filtering in env_vars 
     env_vars["total_cells_prefilter"] = len(anndata_obj.obs.index)
     env_vars["total_genes_prefilter"] = len(anndata_obj.var.index)

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
     env_vars["total_cells_remaining_postfilter"] = len(anndata_obj.obs.index)
     env_vars["total_genes_remaining_postfilter"] = len(anndata_obj.var.index)

     
     # save h5ad; file type is inferred from filename extension
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     # save additions to env_vars workspace file metadata
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)


     return anndata_obj, env_vars


# when if size_factor is set to None, then scanpy's default depth normalization of median total counts
# if want to normalize to 10k, then set to 10000, if want CPM normalization then set to 1000000
# normalization if then followed by a log transformation +1 pseudocount
# save_memory is a boolean to save memory footprint of storing the whole array in memory vs chunks of the array at a time
# this does not parallelize, still single threaded, just sotred the dense values as needed
# if save_memory is True, data_chunk_size is required and that roughly equates to the bumber of cells per batch to split the array for memory saving purposes
# if save_memory is False then data_chunk_size is set to None as the parameter does not matter; full array is stored in memory
@profile
def normalize_and_transform(anndata_obj:AnnData, env_vars:dict, size_factor: int | None = None, save_memory:bool = False, data_chunk_size:int|None = None) -> tuple(AnnData, dict):
     logging.info("Normalization and transformation of counts")

     # save a copy of the raw counts to the counts layer (X is the active layer and when normalization is applied it applies to X)
     anndata_obj.layers['counts'] = anndata_obj.X.copy()
     scanpy.pp.normalize_total(anndata_obj, target_sum = None, layer = None, exclude_highly_expressed = False, inplace = True)
     scanpy.pp.log1p(anndata_obj, base = None, chunked = None, chunk_size = None, layer = None)

     # save h5ad; file type is inferred from filename extension
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     # save additions to env_vars workspace file metadata
     env_vars['normalization_scale_factor'] = size_factor
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)
     
     return anndata_obj, env_vars
     
@profile
def calculate_cell_cycle(anndata_obj:AnnData, env_vars:dict, s_genes:Iterable[str], g2m_genes:Interable[str]) -> tuple(AnnData, dict):
     scanpy.tl.score_genes_cell_cycle(anndata_obj, s_genes=s_genes, g2m_genes=g2m_genes)
     return anndata_obj, env_vars

@profile     
def identify_and_transform_hvgs(anndata_obj:AnnData, env_vars:dict, n_hvgs:int, hvg_ignore:str, vars_to_regress:Iterable[str], threads:int) -> tuple(AnnData, dict): 
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
          env_vars['genes_ignored_for_hvg_selection'] = genes_to_ignore_for_clustering
     else:
          env_vars['genes_ignored_for_hvg_selection'] = "None" # keeping as None string not None object, as this is just human-readable metadata to keep track of methods


     # plot HVGs and save in qc_images subdirectory
     scanpy.pl.highly_variable_genes(anndata_obj, log = False, highly_variable_genes=True, save =  pathlib.Path("qc_images") / f'{env_vars["save_prefix"]}_hvg.png')


     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     #anndata_obj.layers["regressed_scaled"] = anndata_obj.X.toarray()
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     env_vars["n_top_hvg"] = n_hvgs
     env_vars["vars_to_regress"] = vars_to_regress
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)
     
     return anndata_obj, env_vars


@profile  
def pca(anndata_obj:AnnData, env_vars:dict, pca_var_change:float, seed:int) -> tuple(AnnData, dict):
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
     # plot elbow plot/
     scanpy.pl.pca_variance_ratio(anndata_obj, n_pcs=50, log=True, save = pathlib.Path("qc_images") / f'{env_vars["save_prefix"]}_pca_elbow_plot_of_hvgs.png')
     # plot pc loadings - top hvgs driving each PC
     scanpy.pl.pca_loadings(anndata_obj, components = '1,2,3,4,5,6,7,8,9,10', include_lowest = True, save = pathlib.Path("qc_images") / f'{env_vars["save_prefix"]}_pca_gene_loadings_of_hvgs.png') # include_lowest means to show the features that have the highest and lowest loadings

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
          env_vars["pcs_to_use"] = idx+2
          logging.info("Total PCs to use: {}".format(env_vars["pcs_to_use"]))
     else:
          env_vars["pcs_to_use"] = len(change_in_var)+1 #if there are no PCs that meet the criteria, use all PCs calculated
          logging.info("Total PCs to use: {}".format(env_vars["pcs_to_use"]))

    

     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     # but there will be added PCA calculations and data to obsm, varm, and uns
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     env_vars["pca_var_change"] = pca_var_change
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)
     
     return anndata_obj, env_vars


@profile
def neighbors_umap_clust(anndata_obj:AnnData, env_vars:dict, n_neighbors:int, n_pcs:int, dist_metric:str, seed:int, resolutions:Iterable) -> tuple(AnnData, dict):
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
          edge_weight_threshold=0.05,  # the minimum fraction of the parent cluster assigned to the child cluster to plot
          show_fraction=True  # show the fraction of cells in each cluster
          )
     fig.set_size_inches(15, 15)
     fig.set_dpi(100)
     fig.savefig(pathlib.Path("cluster_analysis", f"clustree_overall_stability.png"),
          dpi=300,
          bbox_inches="tight"
          )

     # clusteree by gene expression
     genes = ["CD3G", "CD3D", "CD8A", "CD4", "MS4A1", "CD79A", "NKG7", "LYZ"]

     # Keep only genes that exist
     genes = [g for g in genes if g in anndata_obj.var_names]
     idx = [anndata_obj.var_names.get_loc(g) for g in genes]

     # Small AnnData containing only the genes of interest
     tmp = anndata_obj[:, genes].copy()

     # Use the normalized layer as the expression matrix
     tmp.X = anndata_obj.layers["normalized"][:, idx].copy()

     # Set raw from this normalized matrix
     tmp.raw = tmp.copy()

     for gene in genes:
          fig = pyclustree.clustree(tmp,
               [f"leiden_res_{res:.2f}" for res in resolutions],
               title=f"Clustree colored by {gene}",
               edge_weight_threshold=0.05,
               node_color_gene=gene,
               node_colormap="Reds",
               show_colorbar=True,
               show_fraction=True  # show the fraction of cells in each cluster
               )
          fig.set_size_inches(15, 15)
          fig.set_dpi(100)
          fig.savefig(pathlib.Path("cluster_analysis", f"clustree_{gene}.png"),
               dpi=300,
               bbox_inches="tight"
               )
     

     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     # but there will be added PCA calculations and data to obsm, varm, and uns
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)

     return anndata_obj, env_vars


@profile
def plot_reductions(anndata_obj:AnnData, env_vars:dict, reduction_name:str, layer:str, ncol_layout:int,  categorical_col:str, continuous_col:str, marker:str, groupby_col:str, file_savename:str) -> None:
     '''
     For plotting umap:
     gene_symbols: str | None (default: None)
     Column name in .var DataFrame that stores gene symbols. By default var_names refer to the index column of the .var DataFrame. Setting this option allows alternative 
     names to be used.
     '''

     # color_map = continuous colors: magma, viridis
     # palette = categorical colors: Set1, Set2, Set3, Accent, tab20, okabe_ito, Dark2
     '''
     Marker	Symbol
     '.'	point (default, small)
     'o'	circle
     ','	pixel (tiny square, fastest to render)
     's'	square
     '^'	triangle up
     'v'	triangle down
     'D'	diamond
     'd'	thin diamond
     '+'	plus
     'x'	x
     '*'	star
     '''

     scanpy.pl.embedding(anndata_obj,  
                         basis = reduction_name, 
                         layer = layer, 
                         color = groupby_col, 
                         projection = '2d',
                         ncols = ncol_layout,
                         add_outline = False, 
                         wspace=0.5, 
                         colorbar_loc = "bottom", 
                         color_map = continuous_col, 
                         palette= categorical_col,
                         hspace=0.5, 
                         frameon=False, 
                         marker = marker,
                         show = False)

     plt.savefig("cluster_analysis/{}.png".format(file_savename), dpi=300, bbox_inches="tight")
     plt.savefig("cluster_analysis/{}.pdf".format(file_savename), bbox_inches="tight")
     plt.close()    

     '''
     #clustree on top of a umap or any other dim reference
     fig = clustree(
     adata,
     [f"leiden_{str(resolution).replace('.', '_')}" for resolution in [0.1, 1.0]],
     title="UMAP Clustree of PBMC68k",
     scatter_reference="X_umap",
     node_size_range=(200, 300),
     edge_width_range=(0.1, 2.0),
     graph_plot_kwargs={"font_color": "black", "font_size": 10, "alpha": 0.75},
     )
     fig.set_size_inches(7.5, 5)
     fig.set_dpi(100)
     '''




     '''
     # code written by Claude and function for split.by since scanpy doesn't have a split.by option
     import scanpy as sc
     import matplotlib.pyplot as plt
     import math

     def umap_split_by(
     adata,
     basis,
     color_col,          # single column to color by (categorical or continuous)
     split_by,            # metadata column to split/facet by
     layer=None,
     palette="tab20",
     marker=".",
     size=None,
     frameon=False,
     ncols=5,
     figsize_per_panel=(3.5, 3.5),
     legend_loc="right margin",   # set to None if you don't want any legend
     save=None,                    # e.g. "umap_leiden_split.png"
     ):
     # get split categories
     categories = adata.obs[split_by].cat.categories if hasattr(adata.obs[split_by], "cat") \
                    else sorted(adata.obs[split_by].unique())

     n = len(categories)
     nrows = math.ceil(n / ncols)

     fig, axes = plt.subplots(
          nrows, ncols,
          figsize=(figsize_per_panel[0]*ncols, figsize_per_panel[1]*nrows),
          squeeze=False,
     )
     axes = axes.flatten()

     # shared axis limits across all panels, taken from the full embedding
     coords = adata.obsm[f"X_{basis}"] if f"X_{basis}" in adata.obsm else adata.obsm[basis]
     xlim = (coords[:, 0].min(), coords[:, 0].max())
     ylim = (coords[:, 1].min(), coords[:, 1].max())

     # lock in consistent category colors before subsetting, so cluster 0
     # is the same color in every panel
     if f"{color_col}_colors" not in adata.uns:
          sc.pl.embedding(adata, basis=basis, color=color_col, show=False)
          plt.close()

     for i, cat in enumerate(categories):
          ax = axes[i]
          subset = adata[adata.obs[split_by] == cat]
          sc.pl.embedding(
               subset,
               basis=basis,
               layer=layer,
               color=color_col,
               palette=palette,
               marker=marker,
               size=size,
               frameon=frameon,
               ax=ax,
               show=False,
               title=str(cat),
               legend_loc=legend_loc if i == n - 1 else None,  # legend only on last panel
          )
          ax.set_xlim(xlim)
          ax.set_ylim(ylim)

     # turn off any unused axes (when categories don't fill the grid evenly)
     for j in range(n, len(axes)):
          axes[j].axis("off")

     fig.suptitle(color_col, fontsize=14, y=1.02)
     plt.tight_layout()

     if save:
          fig.savefig(save, dpi=150, bbox_inches="tight")

     return fig


     # ---- usage ----
     fig = umap_split_by(
     anndata_obj,
     basis="initial_umap",
     color_col="leiden_res_0.60",
     split_by="sample",
     layer="normalized",
     palette="tab20",
     marker="*",
     ncols=5,
     )
     plt.show()
     '''


@profile
def one_vs_all_de(anndata_obj:AnnData, env_vars:dict, num_genes_to_plot:int, de_test:str, core_genes_to_plot:Iterable, min_pct:float, max_padj:float) -> tuple(AnnData, dict):
     # obtain cluster-specific differentially expressed genes using Wilcoxon
     pattern = re.compile(r"^leiden_res_[0-9]*")
     all_resolutions = [x for x in anndata_obj.obs_keys() if pattern.search(x)]
     
     if de_test == "pyMAST":
          for res in all_resolutions:
               # obtrain cluster-specific differentially expressed genes using MAST
               pymast.tl.rank_genes_groups(anndata_obj,
                    groupby=res,          # obs column with group labels
                    groups="all",              # test all grnum_genes_to_plotoups vs. rest
                    reference="rest",
                    layer="normalized",                # use adata.X (pass layer name for a specific layer)
                    ebayes=True,               # empirical Bayes variance shrinkage
                    method="bayesglm",         # "bayesglm" | "glm"
                    n_jobs=-1,                 # parallelism (-1 = all cores)
                    key_added='{}_rank_genes_groups'.format(res)
                    #cdr_key = '{}_cdr'.format(res),
                    #pts = True
               )

               # get df of the results of the MAST DE of 1 vs all for all clusters
               # setting group=None means you want all clusters DE.  If you only want to see a specific
               # cluster group, such as just cluster 0 results, you can set group = "0" or group = "1" to get 
               # cluster ID 1 results for MAST    
               de_df = scanpy.get.rank_genes_groups_df(anndata_obj, group=None, key='{}_rank_genes_groups'.format(res))
               # keep only significant hits
               de_df = de_df[de_df["pvals_adj"] < max_padj]
               # to prevent rare clonotypes from being selected, also apply a minimum percent expressed per group
               de_df = de_df[de_df["pct_nz_group"] > min_pct]

               # top n genes by logfoldchanges within each cluster
               topn_per_group = (de_df.groupby("group", sort=False, group_keys=False).apply(lambda x: x.nlargest(num_genes_to_plot, "logfoldchanges")))
               genes_to_plot = topn_per_group["names"].tolist() + core_genes_to_plot

               # plot mean expression
               dp_mean_norm_expr = scanpy.pl.DotPlot(anndata_obj, 
                                   groupby = res, 
                                   var_names=genes_to_plot, 
                                   layer="normalized", 
                                   standard_scale=None, 
                                   dendrogram=False)
               dp_mean_norm_expr.legend(colorbar_title="Mean Gene Expression", size_title="Fraction Expressing (%)")
               dp_mean_norm_expr.make_figure()
               ax = dp_mean_norm_expr.ax_dict["mainplot_ax"]
               ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
               dp_mean_norm_expr.savefig("cluster_analysis/{}_mean_expr_dotplot.png".format(res), bbox_inches="tight")
               
               # plot z-scaled expression
               dp_scaled_expr = scanpy.pl.DotPlot(anndata_obj, 
                                   groupby = res, 
                                   var_names=genes_to_plot, 
                                   layer="normalized", 
                                   standard_scale="var",
                                   cmap = "RdBu_r",
                                   vcenter = 0, 
                                   var_group_rotation = 45,
                                   dendrogram=False)
               dp_scaled_expr.legend(colorbar_title="Z-scaled Expression", size_title="Fraction Expressing (%)")
               dp_scaled_expr.make_figure()
               ax = dp_scaled_expr.ax_dict["mainplot_ax"]
               ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
               dp_scaled_expr.savefig("cluster_analysis/{}_scaled_expr_dotplot.png".format(res), bbox_inches="tight")
     
     else:
          for res in all_resolutions:
               scanpy.tl.rank_genes_groups(anndata_obj, 
                                        groupby=res, 
                                        groups="all",              # test all grnum_genes_to_plotoups vs. rest
                                        reference="rest",
                                        layer="normalized", 
                                        method = de_test,
                                        pts = True,
                                        key_added='{}_rank_genes_groups'.format(res),
                                        cdr_key = '{}_cdr'.format(res)
                                        )

     env_vars["de_test"] = de_test
     env_vars["min_pct_plot_filter"] = min_pct
     env_vars["max_padj_plot_filter"] = max_padj

     # at the end of this function, the active layer (.X) will be normalized, regressed and scaled counts for all genes
     # but there will be added PCA calculations and data to obsm, varm, and uns
     anndata_obj.write(filename = os.path.join(env_vars["workingDir"], "h5ad_objects", "testing_save_filtered_gene_symbol_{}_{}.h5ad".format(env_vars["save_prefix"], env_vars["date"])),
                       convert_strings_to_categoricals = True,
                       compression = "gzip")
     
     with open(pathlib.Path("".join(["workspace_files/vars_and_params_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), "wb") as f:
          pickle.dump(env_vars, f)

     return anndata_obj, env_vars



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


'''
if __name__ == "__main__":

     def convert_to_tuple(kv_pair) -> tuple:
          key, value = kv_pair.split("=")
          return key.strip(), value.strip()


     parser = argparse.ArgumentParser(description= "Initial QC, filtering, clustering per sample")
     subparser = parser.add_subparsers(dest = "run_mode", required = True) # dependig on the mode selected it will revert to the parser with different options and requirements

     newProj_parser = subparser.add_parser("newProject") # if user selects newProject, then newProj parser arguments become available
     resume_parser = subparser.add_parser("resume") # if user selects resume, then resume parser arguments become available

     # --- Subcommand: newProject ---
     env_setup_group = newProj_parser.add_argument_group("Environment and project setup options")
     env_setup_group.add_argument("--working_dir", default=os.getcwd(), type = pathlib.Path)
     env_setup_group.add_argument("--save_prefix", default="", type = str)
     env_setup_group.add_argument("--filtered_feature_bc_matrix", type = pathlib.Path, help = "path to counts matrix, barcodes file, and feature/gene files")
     env_setup_group.add_argument("--sample_name", type = str)
     env_setup_group.add_argument("--convertEnsembl", action = "store_true", help = "if this flag is set, will collapse Ensembl IDs to human readable gene IDs")
     env_setup_group.add_argument("--platform", choices = ["parse", "10x", "bdrhapsody"])
     env_setup_group.add_argument("--metadata", default = None, nargs = "+", type = convert_to_tuple, help = "Ex: --metadata sex=female batch=A tissue=spleen age=100 group=\"Control group\"") # each time a key value pair is listed, it converts to a tuple and the tuple will be collected as a list based on argparse nargs
     env_setup_group.add_argument("--seed", default = 42, type = int, help = "When algorithms are non-deterministic, this is the seed that be used for reproducibility purposes."
     env_setup_group.add_argument("--threads", default = 5, type = int, help = "The number of parallel processes to spawn off when a function can be parallelized.")

     ## cell filtering and QC options
     filtering_group = newProj_parser.add_argument_group("Cell filtering and QC options")
     filtering_group.add_argument("--min_cells_expressed", default = 0, type = int, help = "Keep genes only if they are expressed in X number of cells.  The default is 0 and genes are not removed (recommended).")
     filtering_group.add_argument("--min_unique_genes", default = 300, type = int, help = "Filter out any cells that do not have at least x number of unique genes expressed.  Default: 300 genes")
     filtering_group.add_argument("--min_umi_counts", default = 500, type = int, help = "Filter out any cells that do not have at least x number of UMIs/reads in the cells.  Default: 500.")
     filtering_group.add_argument("--mito_contam", default= 10, type = float, help = "Filter out any cells that have a x percent of reads expressed in mitochondrial genes.  Generally a good inidcation of cell death. Default: 10")
     filtering_group.add_argument("--max_unique_genes", default = None, type = int, help = "Filter out any cells that have more than x number of unique genes expressed.  Default: None, this upper limit is not applied.")
     filtering_group.add_argument("--max_umi_counts", default = None, type = int, help = "Filter out any cells that have more than x number of UMIs/reads in the cell.  Default: None, this upper limit is not applied.")
     filtering_group.add_argument("--min_complexity", default = 0.8, type = float)
     filtering_parser.add_argument("--mito_regex", default = "MT-", type = str)
     filtering_parser.add_argument("--ribo_regex", default = ["RPL", "RPS"], nargs= "+", type = str)
     filtering_parser.add_argument("--dbl_rate", default = "0.076", type = float32, help = "Look up the doublet detection rate expecation for the technology you are using; for Parse in 2026, it was estimated to be <3% see ParseBioScience_What_is_the_expected_doublet_rate_Support_Suite.html in supplemental_files, so you can set this to 0.03; 10x doublet rate is ~7.6% for 10,000 cells captured/sequenced and it is depenent on the numberof cells captured, so refer to 10x-How-To-Technical-Seminar_Sample-Prep.pdf under supplemental_files to determine this value for 10x. BD Rhapsody, see BD-Rhapsody-HT-Single-Cell-Analysis-System-Instrument-User-Guide_page33.pdf page 33 in supplemental_files, but estimated about 1.7% for 10k or 3.7% for 20k cells.  The default is 10x at 10,000 cells of 7.6% = 0.076."
     filtering_parser.add_argument("--removeDoublets", action = "store_true, help = "When this flag is specified, doublets are removed during the cell filtering step of the pipeline.  If this flag is not set, doublets are calculated and marked but not removed/fitlered."


     ## normalization and transformation options
     normalization_group = newProj_parser.add_argument_group("Normalization/Transformation options")
     normalization_group.add_argument("--size_factor, default = None, type = int, help = "If size_factor is set to None, then scanpy's default depth normalization of median total counts is used.  Otherwise, this can be set to an integer to depth normalize.  CPM normalization would mean setting this value to 1000000, or to mimic Seurat's depth normalization, set this value to 10000")
     normalization_group.add_argument("--save_memory", action = "store_true", help = "A boolean to save memory footprint of storing the whole array in memory vs chunks of the array at a time. This does not parallelize, still single threaded, just sotred the dense values as needed."
     normalization_group.add_argument("--data_chunk_size", type = int, default = None, help ="If --save_memory" flag is set, the paramter is required to be set to an int.  The int roughly represents the number of cells you want processed at a time to reduce memory footprint - will not accelerate via parallelization")
     normalization_group.add_argument("--n_hvgs", default = 2000, type = int, help = "The number of highly variable genes to select from the dataset for use in PCA.")
     normalization_group.add_argument("--regress_vars", default = ["total_counts", "pct_counts_mt"], nargs = "+",  help ="a space-delimited list of strings to regress out; total_counts and pct_counts_mt are calculated in the pipeline and regressed by default but any metadata field can be added here as well.  Note, if the default and the metadata field of sex for example is to be regress is should be specifed as: --regress-vars total_counts pct_counts_mt sex")
     normalization_group.add_argument("--hvg_ignore", default = "^IG[HKL]([VJ]|V[IVX]+|D[0-9])",  help ="a regex that identifies genes to ignore in HVG selection to prevent biasing clustering; For human, the following is recommended for B cell subclustering to prevent clustering bias by clonotype: ^IG[HKL]([VJ]|V[IVX]+|D[0-9]), for mouse, the following is recommended: ^Ig[hkl][vj]|^Ighd[0-9]. WARNING!! Place regex inbetween quotes to prevent shell/bash from interpreting special characters such a | as a pipe!  However, any regex is supported and will flag genes that match the regex to be ignored during selection of HVG.")



     ## clustering and dimensionality reduction options 
     neighbors_umap_clust_group = newProj_parser.add_argument_group("Clustering and dimensionality reduction options")
     neighbors_umap_clust_group.add_argument("--hvg_features", default = 2000, type = int)
     neighbors_umap_clust_group.add_argument("--pca_var_change", default = 0.05, type = float)
     neighbors_umap_clust_group.add_argument("--neighbors", default = 30, type = int)
     neighbors_umap_clust_group.add_argument("--resolutions", default=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0], nargs = "+", type = float)
     neighbors_umap_clust_group.add_argument("--dist_metric", default = "euclidean")
     neighbors_umap_clust_group.add_argument("--core_genes_to_plot", default = "euclidean")
     neighbors_umap_clust_group.add_argument("--top_genes_per_cluster_to_plot", default = 5, type = int)

     ## differential clustering one vs all options 
     one_vs_all_de_group = newProj_parser.add_argument_group("Differential expression options")
     one_vs_all_de_group.add_argument("--de_test", default = "pyMAST", choices = ["pyMAST", "wilcoxon", "logreg", "t-test", "t-test_overestim_var"])
     one_vs_all_de_group.add_argument("--min_pct_plot_filter", default = "pyMAST", choices = ["pyMAST", "wilcoxon", "logreg", "t-test", "t-test_overestim_var"])
     one_vs_all_de_group.add_argument("--max_padj_plot_filter", default = "pyMAST", choices = ["pyMAST", "wilcoxon", "logreg", "t-test", "t-test_overestim_var"])




     newProj_parser.add_argument("--additional_genes_to_plot", default=["CD3E", "CD3D", "CD4", "CD8A", "CD8B", "CD19", "MS4A1", "CD79A", "CD79B"], nargs = "+", type = str)
  
     # --- Subcommand: resume ---
     resume_parser.add_argument("--env_vars_config", required = True, type  = pathlib.Path)
     resume_parser.add_argument("--update_working_dir", default = None, type = pathlib.Path, help = "This needs to be the full path to there the file structure of h5ad_objects/, workspace_files/, qc_images/, cluster_analysis/ live")
     args = parser.parse_args()




     #TODO:
     # parser validation
     #need logic to make sure this holds true
     #min_umi_counts <= x <= max_umi_counts
     
     # Post-parse validation
     if args.save_memory and args.data_chunk_size is None:
          parser.error("--data_chunk_size is required when --save_memory is set.")
     if not args.save_memory and args.data_chunk_size is not None:
          parser.error("--chunk_size can only be specified when --save_memory is set.")
'''
