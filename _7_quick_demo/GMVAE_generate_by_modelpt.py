"""

"""
import torch
from torch import optim
from torch.nn import functional as F

import datetime
import math
import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, '../_model_source_codes/')

from GMVAE_scrna_zinb import GMVAE_ZINB
from GMVAE_losses_zinb import *
from GMVAE_utils import train as gmvae_train

import pickle
import scipy.io as sio
from scipy.sparse import csr_matrix


def zinb_gen(rand, dropout, mean, disp):
    take_NB=(rand>=dropout)*1
    success_prob=(disp)/(mean+disp)
    gen_value=take_NB*np.random.negative_binomial(disp, success_prob)
    return gen_value

def reparameterize_manual(mu, logvar, factor):
    std = torch.exp(0.5 * logvar)
    # sample epsilon from a normal distribution with mean 0 and
    # variance 1
    eps = torch.randn_like(std)*factor
    return eps.mul(std).add_(mu)

# device =  "cpu"  #Myles: come back here and check
device = torch.device('cpu')



"""
    generate_from_pt_cellcounts function generates
    cell_counts number of cell from each cell type
    using saved_model.
    factor is the random noise in reparaterization.  
    suffix is for the matrix file name and cluster file name. If '', a default name will be used.
    Generation uses cpu, not gpu.
"""
def generate_from_pt_cellcounts(saved_model:str, cell_counts:list, factor=1, suffix=''):
    # gmvae = torch.load(saved_model, weights_only=False)
    gmvae = torch.load(saved_model, weights_only=False, map_location=torch.device('cpu'))
    gmvae.device = torch.device('cpu')
    # gmvae = gmvae.to(device)  #Myles: come back here and check
    gmvae.eval()
    
    print(f'This model is trained with {gmvae.args.input_dim} genes')
    print(f'This model is trained with {gmvae.args.K} cell types.')
    print(f'{len(cell_counts)} cell counts are provided. If this does not match {gmvae.args.K}, either it will be filled zeroes or be chopped.')
    
    if len(cell_counts) > gmvae.args.K:
        cell_counts = [cell_counts[ii] for ii in range(gmvae.args.K)]
    if len(cell_counts) < gmvae.args.K:
        for ii in range(gmvae.args.K-len(cell_counts)):
            cell_counts.append(0)
            
    ct = datetime.datetime.now()
    time_stamp=f'{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}'
    if suffix=='':
        suffix=f'generated_from_test_data_on_{time_stamp}'
    ## generation begins ##
            
    generated_zerop=torch.empty([0,gmvae.args.input_dim])
    generated_mean=torch.empty([0,gmvae.args.input_dim])
    generated_disp=torch.empty([0,gmvae.args.input_dim])
    
    for ii in range(gmvae.args.K):

        generate_seed = gmvae.onehot[ii,:].unsqueeze(0).expand(cell_counts[ii], gmvae.args.K)
        mu_genz = gmvae.mu_w2(F.relu(gmvae.mu_w1(generate_seed)))
        logvar_genz = gmvae.logvar_w2(F.relu(gmvae.logvar_w1(generate_seed)))
        gen_z = reparameterize_manual(mu_genz,logvar_genz, factor=factor)
        gen_zerop, gen_mean, gen_disp = gmvae.decoder(gen_z)

        generated_zerop = torch.cat([generated_zerop, gen_zerop.cpu()],0)
        generated_mean = torch.cat([generated_mean, gen_mean.cpu()],0)
        generated_disp = torch.cat([generated_disp, gen_disp.cpu()],0)

    cluster_gen = torch.cat([ii*torch.ones(cell_counts[ii], dtype=torch.int) for ii in range(gmvae.args.K)])

    generated_zerop_np = generated_zerop.cpu().detach().numpy()
    generated_mean_np = generated_mean.cpu().detach().numpy()
    generated_disp_np = generated_disp.cpu().detach().numpy()
    
    x_gen_rand_num=np.random.rand(generated_zerop_np.shape[0], generated_zerop.shape[1])
    generated_disp_np = generated_disp_np + 1e-16
    generated_matrix = zinb_gen(x_gen_rand_num, generated_zerop_np, generated_mean_np,  generated_disp_np)
    
    generated_matrix = csr_matrix(generated_matrix)
    sio.mmwrite(f"matrix_{suffix}.mtx", generated_matrix.T)
    
    cluster_gen_np = cluster_gen.cpu().detach().numpy()
    cluster_df=pd.DataFrame({'partition': cluster_gen_np})
    cluster_df.head()
    cluster_df.to_csv(f"cluster_{suffix}.csv")
    
    return cell_counts
