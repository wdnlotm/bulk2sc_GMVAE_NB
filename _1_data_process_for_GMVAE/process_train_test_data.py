import scipy.io as sio
from scipy.sparse import csr_matrix
import pandas as pd
import scanpy as sc
import os

sc.settings.set_figure_params(dpi=100, facecolor="white")
sc.settings.verbosity = 0

## parameters for lung data

num_of_color=20 #up to 25
color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                '#911eb4', '#42d4f4', '#f032e6', "#adff2f", '#469990', 
                "#000080",'#9A6324', '#800000', '#808000', '#2f4f4f', 
                '#a9a9a9', '#dcbeff',"#7b68ee", "#483d8b", '#aaffc3',
                "#cd853f", "#f5deb3", "#ee82ee", "#005000",'#000000']

geneid_file="../_0_data_download/genes_from_raw.tsv"
barcodes_file="../_0_data_download/barcodes_from_raw.tsv"
count_file="../_0_data_download/matrix_raw_counts.mtx"

max_n_genes_by_counts = 2500
max_total_counts = 6000
max_pct_counts_mt = 15
min_gene_num = 200
min_cell_num = 3

leiden_res=0.7
min_cell_count_in_cluster=30
test_frac = 0.25
rseed=123
#################################################################################
# set the color to choose in plots 
color_dict = dict(zip( [str(ii) for ii in range(num_of_color)] , color_list[0:num_of_color]  ))

# read count, geneid, barcodes
adata_bc=pd.read_csv(barcodes_file,header=None)
adata_features=pd.read_csv(geneid_file,header=None)
adata = sc.read_mtx(count_file)
adata= adata.T
adata.obs['cell_id']= adata_bc[0].tolist()
adata.var['gene_ids']= adata_features[0].tolist()
adata.var.index= adata.var['gene_ids']

# Quality Control
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

adata = adata[adata.obs["n_genes_by_counts"] < max_n_genes_by_counts]
adata = adata[adata.obs["total_counts"] < max_total_counts]
adata = adata[adata.obs["pct_counts_mt"] < max_pct_counts_mt]

sc.pp.filter_cells(adata, min_genes = min_gene_num)
sc.pp.filter_genes(adata, min_cells = min_cell_num)

sc.pp.scrublet(adata)
adata = adata[adata.obs["predicted_doublet"] == False]

adata.layers["counts"] = adata.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
sc.tl.umap(adata, min_dist=0.25)

sc.tl.leiden(adata, flavor="igraph", n_iterations=100, resolution=leiden_res, random_state=rseed)

cluster_cell_count = pd.DataFrame(adata.obs["leiden"].value_counts())

cluster_2_remove = list(cluster_cell_count.index[cluster_cell_count['count']<min_cell_count_in_cluster])

adata = adata[ ~ adata.obs["leiden"].isin( cluster_2_remove )] 

sc.pl.umap(adata, color=["leiden"], alpha=0.5, palette=color_dict, save='_with_leiden.png', show=False)

leiden_clusters=pd.DataFrame(adata.obs["leiden"])
leiden_clusters['training']=1
leiden_proportion = leiden_clusters['leiden'].value_counts()

for ii in leiden_proportion.index.tolist():
    leiden_clusters_onecluster=leiden_clusters[leiden_clusters['leiden']==ii]
    leiden_clusters_onecluster_test = leiden_clusters_onecluster.sample(frac = test_frac, random_state=rseed)
    leiden_clusters.loc[leiden_clusters_onecluster_test.index.tolist(),'training']=0
    
adata.obs["training"]=leiden_clusters['training']


## train data
if not os.path.isdir("train"):
    os.makedirs("train")
    
adata_training = adata[adata.obs["training"]==1]

count_table_training_sparse = csr_matrix( adata_training.layers["counts"].todense() )
sio.mmwrite(f"train/matrix_lung_train.mtx", count_table_training_sparse.T)

adata_training.var['gene_ids'].to_csv("train/genes_lung_train.tsv", sep="\t", header=False, index=False)

leiden_cluster_np = adata_training.obs["leiden"].to_numpy(dtype=int)
cluster_df = pd.DataFrame({'partition': leiden_cluster_np})
cluster_df.to_csv('train/cluster_lung_train.csv', index=False)

barcodes_df=pd.DataFrame(adata_training.obs['cell_id'])
barcodes_df.to_csv("train/barcodes_lung_train.tsv", sep="\t", index=False, header=False)

### test data
if not os.path.isdir("test"):
    os.makedirs("test")

adata_test = adata[adata.obs["training"]==0]

count_table_test_sparse = csr_matrix(adata_test.layers["counts"].todense())
sio.mmwrite(f"test/matrix_lung_test.mtx", count_table_test_sparse.T)

adata_test.var['gene_ids'].to_csv("test/genes_lung_test.tsv", sep="\t", header=False, index=False)
barcodes_df=pd.DataFrame(adata_test.obs['cell_id'])
barcodes_df.to_csv("test/barcodes_lung_test.tsv", sep="\t", index=False, header=False)

leiden_cluster_np = adata_test.obs["leiden"].to_numpy(dtype=int)
cluster_df = pd.DataFrame({'partition': leiden_cluster_np})
cluster_df.to_csv('test/cluster_lung_test.csv', index=False)
