"""
read scRNA test that is saved 
"""

import os
import anndata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import random
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import TAPE
from TAPE import Deconvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#### NEEDED FILES
# 1. GeneLength.txt
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

def predicted_proportion_by_TAPE_model(modelfile:str, genelenfile:str, intersect_gene:str, rna_data=None, cellcounts=None):
    scTAPE_model=modelfile
    # scTAPE_model="../_4_train_scTAPE/results/model_1024_500_15000_1200_0.85.pt"
    model_new2 = torch.load(scTAPE_model, weights_only=False)
    model_new2=model_new2.to(device)
    model_new2.eval()
    model_new2.state = 'test'
    
    if not rna_data == None:
        celltype_counts_df=pd.read_csv(rna_data, sep="\t", index_col=0)
        pseudo_bulk_whole_test=pd.DataFrame((celltype_counts_df.sum(axis=0)))
        bulk_whole=pd.DataFrame()
        for ii in range(2):
            sample_temp=celltype_counts_df
            pseudo_bulk=pd.DataFrame((sample_temp.sum(axis=0)))
            pseudo_bulk=pseudo_bulk.T
            pseudo_bulk=pseudo_bulk.rename(index={0:f'sample{ii}'})
            bulk_whole=pd.concat([bulk_whole,pseudo_bulk],axis=0)
            bulk_whole_tpm = counts2TPM(bulk_whole, genelenfile)
            # print(bulk_whole_tpm.head())
            
        inter=pd.read_csv(intersect_gene)
        inter=inter['gene'].values.tolist()

        bulk_whole_tpm = bulk_whole_tpm[inter]
        bulk_whole_tpm = np.log(bulk_whole_tpm + 1)
        
        mms = MinMaxScaler()
        bulk_whole_tpm = mms.fit_transform(bulk_whole_tpm.T).T
        data = torch.from_numpy(bulk_whole_tpm).float().to(device)
        
        _, pred, _ = model_new2(data)
        prediction=pred.detach().cpu().numpy()

    return prediction[0,:], celltype_counts_df.shape[0]
