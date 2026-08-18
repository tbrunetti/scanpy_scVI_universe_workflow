from typing import Iterable
import sys
import os
import pathlib
import pickle
import numpy
import scipy
import pandas
from memory_profiler import profile
from time import time
import logging
from matplotlib.colors import LinearSegmentedColormap
import seaborn
import anndata
import scanpy
from path_config import PathConfig
from pipeline_config import PipelineConfig
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from adjustText import adjust_text
from pathlib import Path

@profile
def qc_figures(anndata_obj:AnnData, status:str, paths_config:PathConfig, min_counts:int | None = None, min_genes:int | None = None, max_counts:int | None = None, max_genes:int | None = None, max_mito:float| None = None) -> None:
    logging.info("Generating QC images on {} data".format(status))

    # visual of read/umi/count filtering thresholds
    plt.figure(figsize=(10, 6))
    seaborn.histplot(data=anndata_obj.obs,
                    x="total_counts",
                    hue="sample_name",
                    kde=True,
                    alpha=0.2,
                    log_scale=True).set_title(f"Distribution of UMI/reads/counts per cell ({status})")
     
    if min_counts !=None:
        plt.axvline(min_counts,linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
    if max_counts !=None:
        plt.axvline(max_counts, linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
    plt.tight_layout()
    plt.savefig(os.path.join(paths_config.qc_dir, f"{status}_umis.png"), dpi=300)
    #plt.show()

    # visual of unique gene filtering thresholds
    plt.figure(figsize=(10, 6))
    seaborn.histplot(data=anndata_obj.obs,
                    x="n_genes_by_counts",
                    hue="sample_name",
                    kde=True,
                    alpha=0.2,
                    log_scale=True).set_title('Distribution of uniquely expressed genes per cell ({})'.format(status))
     
    if min_genes != None:
        plt.axvline(min_genes, linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
    if max_genes != None:
        plt.axvline(max_genes, linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line

    plt.tight_layout()
    plt.savefig(os.path.join(paths_config.qc_dir, f"{status}_features.png"), dpi=300)
    #plt.show()

    # visual of mitochondrial contamination thresholds
    plt.figure(figsize=(10, 6))
    seaborn.histplot(data=anndata_obj.obs,
                    x="pct_counts_is_mito",
                    hue="sample_name",
                    kde=True,
                    alpha=0.2,
                    log_scale=True).set_title('Distribution of mitochondrial counts per cell'.format(status))
     
    if max_mito != None:
        plt.axvline(max_mito, linestyle = '--', color = 'orange', linewidth = 2) # draws a vertical orange dashed line
    plt.tight_layout()
    plt.savefig(os.path.join(paths_config.qc_dir, f"{status}_mito.png"), dpi=300)
    #plt.show()


    # joint plot visual
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

    if min_counts != None:
        ax.axvline(x=min_counts, linestyle="dotted", color="orange", linewidth=2)
    if min_genes != None:
        ax.axhline(y=min_genes, linestyle="dotted", color="orange", linewidth=2)
    if max_counts != None:
        ax.axvline(x=max_counts, linestyle="dotted", color="orange", linewidth=2)
    if max_genes != None:
        ax.axhline(y=max_genes, linestyle="dotted", color="orange", linewidth=2)
        

    plt.colorbar(scatter, ax=ax, label="pct_counts_is_mito")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total_counts")
    ax.set_ylabel("n_genes_by_counts")
    ax.set_title("total_counts vs n_genes_by_counts")

    plt.tight_layout()
    plt.savefig(os.path.join(paths_config.qc_dir, f"{status}_summary_joint.png"), dpi=300)
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
    plt.savefig(os.path.join(paths_config.qc_dir, f"{status}_doublet_summary_joint.png"), dpi=300)
    # plt.show()



@profile
def plot_reductions(anndata_obj:AnnData, reduction_name:str, layer:str, ncol_layout:int,  categorical_col:str, continuous_col:str, marker:str, groupby_col:str, cluster_label:bool, file_savename:str, save_path:Path) -> None:
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

    ax = scanpy.pl.embedding(anndata_obj,
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
    
    

    if cluster_label:
        # Store text labels for adjustText
        texts = []

        # Get the embedding coordinates dynamically
        embedding = anndata_obj.obsm[f"{reduction_name}"]

        # Add each cluster label at the median position of the cluster
        for cluster in anndata_obj.obs[groupby_col].dropna().unique():
            mask = anndata_obj.obs[groupby_col] == cluster

            x = numpy.median(embedding[mask, 0])
            y = numpy.median(embedding[mask, 1])

            text = ax.text(
                x,
                y,
                str(cluster),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")]
                )

            texts.append(text)

        # Automatically move labels to reduce overlap
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", lw=0.5)
        )

    fig = plt.gcf()

    fig.savefig(
        save_path / f"{file_savename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        save_path / f"{file_savename}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

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



