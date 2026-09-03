import pandas as pd
import scanpy as sc
import string
import anndata as ad
import os
import numpy as np
import matplotlib.pyplot as plt
sc.settings.set_figure_params(dpi=100, facecolor="white")
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
# sc.logging.print_header()

my_color25 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                '#911eb4', '#42d4f4', '#f032e6', "#adff2f", '#469990', 
                "#000080",'#9A6324', '#800000', '#808000', '#2f4f4f', 
                '#a9a9a9', '#dcbeff',"#7b68ee", "#483d8b", '#aaffc3',
                "#cd853f", "#f5deb3", "#ee82ee", "#005000",'#000000']
cluster_list = [str(ii) for ii in range(25)]
color25 = dict(zip(cluster_list, my_color25))

def plot_umaps(data1_name, count1, geneid1, cluster1, data2_name, count2, geneid2, cluster2):
    
    adata = sc.read_mtx(count1)
    adata= adata.T

    adata_features=pd.read_csv(geneid1, header=None, sep='\t')
    adata_barcode=[f'data1_{t}' for t in range(adata.shape[0])]

    adata.obs['cell_id']= adata_barcode
    adata.var['gene_ids']= adata_features[0].tolist()
    # adata.var.index= adata.var['gene_ids']

    cluster_adata=pd.read_csv(cluster1)
    
    print(f'data 1: {adata}')
    
    bdata = sc.read_mtx(count2)
    bdata= bdata.T

    bdata_features=pd.read_csv(geneid2, header=None, sep='\t')
    bdata_barcode=[f'data2_{t}' for t in range(bdata.shape[0])]

    bdata.obs['cell_id']= bdata_barcode
    bdata.var['gene_ids']= bdata_features[0].tolist()
    # adata.var.index= adata.var['gene_ids']

    cluster_bdata=pd.read_csv(cluster2)
    print(f'data 2: {bdata}')
    
    adatas = {data1_name: adata, data2_name: bdata} #"reconst": adata_rec,
    adata_all = ad.concat(adatas, label="dataset_name",  join="outer")
    adata_all.var['gene_ids']= adata.var['gene_ids']

    print(f'combined data: {adata_all}')
    
    adata_all.layers["counts"] = adata_all.X.copy()

    # Normalizing to median total counts
    sc.pp.normalize_total(adata_all)
    # Logarithmize the data
    sc.pp.log1p(adata_all)

    sc.tl.pca(adata_all)

    sc.tl.pca(adata_all)

    sc.pp.neighbors(adata_all)
    sc.tl.umap(adata_all, min_dist=0.375)
    
    dataset_list = [data1_name, data2_name]
    colorpick= ['blue', 'red']
    color_dataset = dict(zip(dataset_list, colorpick))

    sc.pl.umap(adata_all, color="dataset_name", alpha=0.3,
               size=12, palette=color_dataset, 
               save=f"_{data1_name}_v_{data2_name}.png"
              )
    
    sc.pl.umap(adata_all[adata_all.obs['dataset_name']==data1_name], 
               color="dataset_name", alpha=0.4,
               size=12, palette=color_dataset,
               save=f"_{data1_name}.png"
              )

    sc.pl.umap(adata_all[adata_all.obs['dataset_name']==data2_name],
               color="dataset_name",alpha=0.4,
               size=12, palette=color_dataset,
               save=f"_{data2_name}.png"
              )
    
    cluster_all = pd.concat([cluster_adata['partition'], cluster_bdata['partition']])
    cluster_str = [f'{ii}' for ii in cluster_all.values.tolist()]
    adata_all.obs['cluster']= cluster_str

    sc.pl.umap(adata_all, color=["cluster"], alpha=0.5, size=12, 
           palette=color25, save=f"_{data1_name}_v_{data2_name}_cluster.png")

    sc.pl.umap(adata_all[adata_all.obs['dataset_name']==data1_name],
               color=["cluster"], alpha=0.5, size=12, title=f'{data1_name}',
               palette=color25, save=f"_{data1_name}_cluster.png")

    sc.pl.umap(adata_all[adata_all.obs['dataset_name']==data2_name],
               color=["cluster"], alpha=0.5, size=12, title=f'{data2_name}',
               palette=color25, save=f"_{data2_name}_cluster.png")
    return 1


def plot_proportions(data1_name, cluster1, data2_name, cluster2):
    
    if not os.path.isdir("figures"):
        os.makedirs("figures")
    
    cluster_adata=pd.read_csv(cluster1)
    proportion1 = cluster_adata['partition'].value_counts()/cluster_adata.shape[0]
    # count1 = cluster_adata['partition'].value_counts()
    print(f'{data1_name} proportion: {proportion1.sort_index().tolist()}')
    # proportion1.sort_index().tolist()
    
    cluster_bdata=pd.read_csv(cluster2)
    proportion2 = cluster_bdata['partition'].value_counts()/cluster_bdata.shape[0]
    # count2 = cluster_bdata['partition'].value_counts()
    print(f'{data2_name} proportion: {proportion2.sort_index().tolist()}')
    # proportion2.sort_index().tolist()
    
    mae=(abs(np.array(proportion1.sort_index().tolist())-np.array(proportion2.sort_index().tolist()))).sum()/len(proportion1.sort_index().tolist())


    x_limit=round(max(max(proportion1.sort_index().tolist()), max(proportion2.sort_index().tolist()))+0.015,2)

    fig, ax = plt.subplots()

    ax.scatter(proportion1.sort_index().tolist(), proportion2.sort_index().tolist(), s = 20, alpha = 0.5)
    ax.set_aspect('equal')
    ax.set_xlim([0, x_limit])
    ax.set_ylim([0, x_limit])
    ax.plot([0, x_limit],[0, x_limit])

    plt.xlabel(f'{data1_name}')
    plt.ylabel(f'{data2_name}')
    plt.title(f'MAE: {mae}')
    plt.savefig(f"figures/{data1_name}_v_{data2_name}_proportion.png", dpi=300)
    
    return 1