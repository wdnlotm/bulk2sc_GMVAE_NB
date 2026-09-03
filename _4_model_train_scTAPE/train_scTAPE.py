import scTAPE_for_bulk2scGMVAE
from scTAPE_for_bulk2scGMVAE import Deconvolution
import pandas as pd
import os

train_file='../_2_data_process_for_scTAPE/lung_data_to_train_scTAPE_from_lung_train.txt'
bulk_sample_file='../_2_data_process_for_scTAPE/lung_pseudo_bulk_tpm_5samples.txt'
gene_len_file='../_2_data_process_for_scTAPE/GeneLength.txt'
true_proportion_file='../_2_data_process_for_scTAPE/lung_pseudo_sample_prop_5samples.csv'
true_proportion = pd.read_csv(true_proportion_file, index_col='Unnamed: 0')

Sigm, predicted_proportion = Deconvolution(train_file, bulk_sample_file, sep='\t',scaler='mms',
                           datatype='TPM', genelenfile=gene_len_file, mode='overall', 
                           save_model_name = 'scTAPE_lung', adaptive=False,  
                           variance_threshold=0.85, batch_size=600, epochs=640, 
                           samplenumber=15000, sampling_num=1200, seed=15640)

# predicted_proportion=Pred
prediction_save_file=f'predicted_proportion_lung_test_data.csv'
predicted_proportion.to_csv(prediction_save_file)

mae=abs(true_proportion-predicted_proportion).sum().sum()/true_proportion.shape[1]/true_proportion.shape[0]

print(f'MAE: {mae}')

os.rename("genename.csv", f"genename_scTAPE_lung.csv")


    
    
