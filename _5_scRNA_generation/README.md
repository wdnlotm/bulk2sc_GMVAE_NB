# Generate using saved models and test data.
For comparison purposes, synthetic scRNA-seq data is generated from the scRNA-seq data in the test data set.
```python
from scTAPE_predictor_by_modelpt import *
modelpt="../_4_model_train_scTAPE/scTAPE_lung.pt"
geneleng='../_2_data_process_for_scTAPE/GeneLength.txt'

scrna_file='../_2_data_process_for_scTAPE/lung_data_to_train_scTAPE_from_lung_test.txt'
intersect_list='../_4_model_train_scTAPE/genename_scTAPE_lung.csv'

pred_prop, cell_count = predicted_proportion_by_TAPE_model(modelfile=modelpt, 
                                   genelenfile=geneleng, 
                                   intersect_gene=intersect_list, 
                                   rna_data=scrna_file, cellcounts=None)
```
The last command in the above script will
1. create pseudo-bulk RNA-seq data from the scRNA-seq data, `rna_file`, in the test data.
2. perform TPM-normalization using the `genelenfile`.
3. trim genes using the gene list used in `scTAPE` training.
4. produce outputs: `pred_prop`, predicted proportion and `total_count`, the number of cells in the input scRNA-seq data.

Subsequently, executing the scripts below will generate scRNA-seq data based on the cell counts specified in `cell_counts` using the `modelfile`. The results will be saved with the following filenames: `matrix_generated_using_test_data.mtx` and `cluster_generated_using_test_data.csv`. These are scRNA-seq raw count table and cell type data, respectively.
```python
from GMVAE_generate_by_modelpt import *
cell_counts = [round(x*total_count) for x in pred_prop]  #cell counts in each cell types
modelfile="../_3_model_train_GMVAE/models/lung_zdim_32_bs_37_4000_GMVAE_w_nb_wholePTmodel.pt" #model saved after the GMVAE training
generate_from_pt_cellcounts(modelfile, cell_counts, suffix="generated_using_test_data")
```
## Generate from one bulk RNA-seq data.
This is demonstrated in `_7_quick_demo`
