import pandas as pd
import scanpy as sc
import os
import string
import numpy as np
# sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)

num_of_pbsample=5

def counts2FPKM(counts, genelen):
    genelen = pd.read_csv(genelen, sep=',')
    genelen['TranscriptLength'] = genelen['Transcript end (bp)'] - genelen['Transcript start (bp)']
    genelen = genelen[['Gene name', 'TranscriptLength']]
    genelen = genelen.groupby('Gene name').max()
    # intersection
    inter = counts.columns.intersection(genelen.index)
    samplename = counts.index
    counts = counts[inter].values
    genelen = genelen.loc[inter].T.values
    # transformation
    totalreads = counts.sum(axis=1)
    counts = counts * 1e9 / (genelen * totalreads.reshape(-1, 1))
    counts = pd.DataFrame(counts, columns=inter, index=samplename)
    return counts

def FPKM2TPM(fpkm):
    genename = fpkm.columns
    samplename = fpkm.index
    fpkm = fpkm.values
    total = fpkm.sum(axis=1).reshape(-1, 1)
    fpkm = fpkm * 1e6 / total
    fpkm = pd.DataFrame(fpkm, columns=genename, index=samplename)
    return fpkm

def counts2TPM(counts, genelen):
    fpkm = counts2FPKM(counts, genelen)
    tpm = FPKM2TPM(fpkm)
    return tpm

data_dir='../_1_data_process_for_GMVAE/train/'

for f in os.listdir(data_dir):
	if ('.mtx' in f):
		raw_count_file=f
	if ('cluster' in f):
		cluster_file=f
	if ('gene' in f):
		geneid_file=f    
	if ('bar' in f):
		barcode_file=f
        
adata_bc=pd.read_csv(data_dir+barcode_file, header=None)
adata_features=pd.read_csv(data_dir+geneid_file,header=None,sep='\t')

cluster_df=pd.read_csv(data_dir+cluster_file)
adata = sc.read_mtx(data_dir+raw_count_file)

adata= adata.T
adata.obs['cell_id']= adata_bc
adata.var['gene_ids']= adata_features[0].tolist()
adata.var.index= adata.var['gene_ids']

counts_df=pd.DataFrame(adata.X.todense(),dtype='int')

column_mapper = dict(zip(list(range(counts_df.shape[1])), adata.var['gene_ids'].values))
counts_df=counts_df.rename(columns=column_mapper)

celltype_mapper = dict(zip(list(range(len(cluster_df['partition'].unique()))), [f'cell_{t}' for t in string.ascii_uppercase[0:len(cluster_df['partition'].unique())]]))
cluster_df['partition']=cluster_df['partition'].map(celltype_mapper)

celltype_counts_df = pd.concat([cluster_df, counts_df], axis=1)
celltype_counts_df=celltype_counts_df.set_index('partition')
celltype_counts_df.index.name = ''
celltype_counts_df.to_csv("lung_data_to_train_scTAPE_from_lung_train.txt", sep="\t", index=True)

size_median=int(celltype_counts_df.shape[0]/3)
np.random.seed(123)
random_sample_size = np.random.randint(size_median, int(size_median*1.2), num_of_pbsample)


for ii in range(num_of_pbsample):
    counts_df_sample=celltype_counts_df.sample(random_sample_size[ii], random_state=123)
    counts_df_sample.to_csv(f'sample_4_pseudo_bulk{ii}.csv', index=True)
    
bulk_sample=pd.DataFrame()
for ii in range(num_of_pbsample):
    sample_temp=pd.read_csv(f'sample_4_pseudo_bulk{ii}.csv')
    sample_temp=sample_temp.iloc[:,1:sample_temp.shape[1]]
    pseudo_bulk=pd.DataFrame((sample_temp.sum(axis=0)))
    pseudo_bulk=pseudo_bulk.T
    pseudo_bulk=pseudo_bulk.rename(index={0:f'sample{ii}'})
    bulk_sample=pd.concat([bulk_sample,pseudo_bulk],axis=0)
    
bulk_TPM_data=counts2TPM(bulk_sample, './GeneLength.txt')

bulk_TPM_data.to_csv(f'lung_pseudo_bulk_tpm_{num_of_pbsample}samples.txt', index=True, sep='\t')

pseudo_sample_prop = pd.DataFrame()
for ii in range(num_of_pbsample):
    sample_temp=pd.read_csv(f'sample_4_pseudo_bulk{ii}.csv', index_col=0)
    sample_temp_prop=pd.DataFrame(sample_temp.index.value_counts()).sort_index()/sample_temp.shape[0]
    sample_temp_prop=sample_temp_prop.T.rename(index={'count': f'sample{ii}'})
    pseudo_sample_prop=pd.concat([pseudo_sample_prop, sample_temp_prop], axis=0)
    
pseudo_sample_prop.to_csv(f"lung_pseudo_sample_prop_{num_of_pbsample}samples.csv", index=True)

del adata, cluster_df, celltype_counts_df

data_dir='../_1_data_process_for_GMVAE/test/'

for f in os.listdir(data_dir):
	if ('.mtx' in f):
		raw_count_file=f
	if ('cluster' in f):
		cluster_file=f
	if ('gene' in f):
		geneid_file=f    
	if ('bar' in f):
		barcode_file=f
        
adata_bc=pd.read_csv(data_dir+barcode_file, header=None)
adata_features=pd.read_csv(data_dir+geneid_file,header=None,sep='\t')

cluster_df=pd.read_csv(data_dir+cluster_file)
adata = sc.read_mtx(data_dir+raw_count_file)

adata= adata.T
adata.obs['cell_id']= adata_bc
adata.var['gene_ids']= adata_features[0].tolist()
adata.var.index= adata.var['gene_ids']

counts_df=pd.DataFrame(adata.X.todense(),dtype='int')

column_mapper = dict(zip(list(range(counts_df.shape[1])), adata.var['gene_ids'].values))
counts_df=counts_df.rename(columns=column_mapper)

celltype_mapper = dict(zip(list(range(len(cluster_df['partition'].unique()))), [f'cell_{t}' for t in string.ascii_uppercase[0:len(cluster_df['partition'].unique())]]))
cluster_df['partition']=cluster_df['partition'].map(celltype_mapper)

celltype_counts_df = pd.concat([cluster_df, counts_df], axis=1)
celltype_counts_df=celltype_counts_df.set_index('partition')
celltype_counts_df.index.name = ''
celltype_counts_df.to_csv("lung_data_to_train_scTAPE_from_lung_test.txt", sep="\t", index=True)
