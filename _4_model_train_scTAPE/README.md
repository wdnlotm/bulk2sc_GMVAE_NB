# Train scTAPE 
`scTAPE` is used to predict cell type proportions in bulk RNA-seq. However, many hyperparameters are inaccessible through the default interface of the `scTAPE` package. To address this, the source code of `scTAPE` was modified to expose additional hyperparameters, resulting in a new package: `scTAPE_for_bulk2scGMVAE-0.1-py3-none-any.whl`. This modified package can be installed by running the following command in `_4_model_train_scTAPE` directory. 
```
pip install dist/scTAPE_for_bulk2scGMVAE-0.1-py3-none-any.whl
```
The packages listed in the requirements for `bulk2sc` cover all dependencies required by the modified `scTAPE`.

The input files for scTAPE training are: 
```
train_file='../_2_data_process_for_scTAPE/lung_data_to_train_scTAPE_from_lung_train.txt'
bulk_sample_file='../_2_data_process_for_scTAPE/lung_pseudo_bulk_tpm_5samples.txt'
gene_len_file='./GeneLength.txt'
```
With these input files and specified hyperparameters, the following script will train an `scTAPE` model, save the model in a `.pt` file, and store the gene list used in the saved `scTAPE` model.

```
from scTAPE_for_bulk2scGMVAE import Deconvolution

Sigm, Pred = Deconvolution(scRNA_train_data, bulk_data, sep='\t',scaler='mms',
                           datatype='TPM', genelenfile='Genelength.txt', mode='overall',
                           adaptive=False, variance_threshold=0.85, save_model_name = "_lung_scRNA",
                           batch_size=600, epochs=640, samplenumber=15000, sampling_num=1200, seed=15640)
```
For additional details on running the training, refer to `train_scTAPE.py`.
