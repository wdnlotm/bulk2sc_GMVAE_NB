## Proportion comparison
The `scTAPE` predicted cell type proportions and the ground truth proportions (if available) are compared via scatter plot. 
```
from visualize_results import *
cluster_file1="../_1_data_process_for_GMVAE/test/cluster_lung_test.csv" #ground truth cell typeproportion
cluster_file2="../_5_scRNA_generation/cluster_generated_using_test_data.csv" #predicted cell type proportion

plot_proportions('original', cluster_file1, 'generated', cluster_file2)
```
Running the above script will produce a scatter plot like this.

<img src="fig/original_v_generated_proportion.png" width="30%" alt="Image description">

## UMAPs for dataset & cluster
Since the trained models are evaluated on the test dataset, the original data and the generated data are compared through UMAP projections.
```
from visualize_results import *
# original data
count_file1="../_1_data_process_for_GMVAE/test/matrix_lung_test.mtx"
genename_file1="../_1_data_process_for_GMVAE/test/genes_lung_test.tsv"
cluster_file1="../_1_data_process_for_GMVAE/test/cluster_lung_test.csv"
# generated data
count_file2="../_5_scRNA_generation/matrix_generated_using_test_data.mtx"
genename_file2="../_1_data_process_for_GMVAE/test/genes_lung_test.tsv" #same as data 1
cluster_file2="../_5_scRNA_generation/cluster_generated_using_test_data.csv"

plot_umaps('original', count_file1, genename_file1, cluster_file1, 
           'generated', count_file2, genename_file2, cluster_file2)
```
This script will illustrate UMAP plots as shown below.

<img src="fig/original_v_generated_umap_demo.png" width="75%" alt="Image description">

<img src="fig/original_v_generated_umap_cluster_demo.png" width="75%" alt="Image description">
