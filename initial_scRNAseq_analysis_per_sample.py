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
import pyscdblfinder # doublet detection library
from memory_profiler import profile
from time import time
import logging

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
     scdblfinder_obj = pyscdblfinder.ScDblFinder(anndata_obj, random_state=seed)
     scdblfinder_obj.run(dbr=doublet_rate) 
     env_vars["double_rate_param"] = doublet_rate
     env_vars["scdblfinder_object"] = pathlib.Path("".join(["workspace_files/unfiltered_scdblfinder_obj_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"]))

     # save scDblFinder object as pickled file
     with open(pathlib.Path("".join(["workspace_files/unfiltered_scdblfinder_obj_", env_vars["save_prefix"], "_", env_vars["date"], ".pkl"])), 'wb') as object_file:
          pickle.dump(scdblfinder_obj, object_file)
     
     try:
          env_vars[['predicted_singlets']] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["singlet"]
     except KeyError:
          logging.critical("Singlet cells not found. There is a critical error in your dataset.")
          env_vars[['predicted_singlets']] = 0
          sys.exit()
     try:
          env_vars[['predicted_doublets']] = scdblfinder_obj.adata.obs["scDblFinder_class"].value_counts()["doublet"]
     except KeyError:
          logging.warning("No doublets detected in the data set.  This is unusual and you may want to take a deeper dive into your input data or double rate parameter.")
          env_vars[['predicted_doublets']] = 0
     
     # merge doublet metadata information into anndata object
     logging.info("Merging single/doublet prediction columns into anndata.obs metadata.")
     anndata_obj.obs = anndata_obj.obs.join(scdblfinder_obj.adata.obs[["scDblFinder_score", "scDblFinder_class"]], how="left") # join defaults to joining my index match



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
     plt.show()

@profile
def cell_filtering(anndata_obj:AnnData, env_vars:dict) -> tuple(AnnData, dict):
     scanpy.pp.filter_cells(anndata_obj, min_counts=env_vars["min_umi_counts"], 
                            min_genes=env_vars["min_unique_genes"], 
                            max_counts=env_vars["max_umi_counts"], 
                            max_genes=env_vars["max_unique_genes"], 
                            inplace = True)
     
     # Keep cells that have < mito_contam percent
     anndata_obj = anndata_obj[anndata_obj.obs['pct_counts_is_mito'] < env_vars["mito_contam"], :]
     
     return anndata_obj, env_vars

@profile
def normalization():
     pass

@profile
def clustering_dim_red():
     pass




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

     newProj_parser.add_argument("--working_dir", default=os.getcwd(), type = pathlib.Path)
     newProj_parser.add_argument("--save_prefix", default="", type = str)
     newProj_parser.add_argument("--filtered_feature_bc_matrix", type = pathlib.Path, help = "path to counts matrix, barcodes file, and feature/gene files")
     newProj_parser.add_argument("--sample_name", type = str)
     newProj_parser.add_argument("--min_cells_expressed", default = 0, type = int, help = "Keep genes only if they are expressed in X number of cells.  The default is 0 and genes are not removed (recommended).")
     newProj_parser.add_argument("--min_unique_genes", default = 300, type = int, help = "Filter out any cells that do not have at least x number of unique genes expressed.  Default: 300 genes")
     newProj_parser.add_argument("--min_umi_counts", default = 500, type = int, help = "Filter out any cells that do not have at least x number of UMIs/reads in the cells.  Default: 500.")
     newProj_parser.add_argument("--mito_contam", default= 10, type = float, help = "Filter out any cells that have a x percent of reads expressed in mitochondrial genes.  Generally a good inidcation of cell death. Default: 10")
     newProj_parser.add_argument("--max_unique_genes", default = None, type = int, help = "Filter out any cells that have more than x number of unique genes expressed.  Default: None, this upper limit is not applied.")
     newProj_parser.add_argument("--max_umi_counts", default = None, type = int, help = "Filter out any cells that have more than x number of UMIs/reads in the cell.  Default: None, this upper limit is not applied.")
     newProj_parser.add_argument("--min_complexity", default = 0.8, type = float)
     newProj_parser.add_argument("--hvg_features", default = 2000, type = int)
     newProj_parser.add_argument("--pca_var_change", default = 0.10, type = float)
     newProj_parser.add_argument("--neighbors", default = 30, type = int)
     newProj_parser.add_argument("--resolutions", default=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0], nargs = "+", type = float)
     newProj_parser.add_argument("--top_genes_per_cluster_to_plot", default = 5, type = int)
     newProj_parser.add_argument("--additional_genes_to_plot", default=["CD3E", "CD3D", "CD4", "CD8A", "CD8B", "CD19", "MS4A1", "CD79A", "CD79B"], nargs = "+", type = str)
     newProj_parser.add_argument("--convertEnsembl", action = "store_true")
     newProj_parser.add_argument("--platform", choices = ["parse", "10x", "bdrhapsody"])
     newProj_parser.add_argument("--metadata", default = None, nargs = "+", type = convert_to_tuple, help = "Ex: --metadata sex=female batch=A tissue=spleen age=100 group=\"Control group\"") # each time a key value pair is listed, it converts to a tuple and the tuple will be collected as a list based on argparse nargs
     newProj_parser.add_argument("--mito_regex", default = "MT-", type = str)
     newProj_parser.add_argument("--ribo_regex", default = ["RPL", "RPS"], nargs= "+", type = str)
     newProj_parser.add_argument("--dbl_rate", default = "0.076", type = float32, help = "Look up the doublet detection rate expecation for the technology you are using; for Parse in 2026, it was estimated to be <3% see ParseBioScience_What_is_the_expected_doublet_rate_Support_Suite.html in supplemental_files, so you can set this to 0.03; 10x doublet rate is ~7.6% for 10,000 cells captured/sequenced and it is depenent on the numberof cells captured, so refer to 10x-How-To-Technical-Seminar_Sample-Prep.pdf under supplemental_files to determine this value for 10x. BD Rhapsody, see BD-Rhapsody-HT-Single-Cell-Analysis-System-Instrument-User-Guide_page33.pdf page 33 in supplemental_files, but estimated about 1.7% for 10k or 3.7% for 20k cells.  The default is 10x at 10,000 cells of 7.6% = 0.076."
     newProj_parser.add_argument("--seed", default = 42, type = int, help = "When algorithms are non-deterministic, this is the seed that be used for reproducibility purposes."


     resume_parser.add_argument("--env_vars_config", required = True, type  = pathlib.Path)
     resume_parser.add_argument("--update_working_dir", default = None, type = pathlib.Path, help = "This needs to be the full path to there the file structure of h5ad_objects/, workspace_files/, qc_images/, cluster_analysis/ live")
     args = parser.parse_args()

     #TODO:
     #need logic to make sure this holds true
     #min_umi_counts <= x <= max_umi_counts
'''
