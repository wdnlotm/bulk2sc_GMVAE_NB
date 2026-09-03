# Download data
Executing the R script file, `download_lung_GSE130148.R`, downloads a `.RData` file and extract/save a raw count table file, a gene name file, and a barcode file. It also downloads a human gene length file `Genelength.txt` from the [`scTAPE`](https://github.com/poseidonchan/TAPE/tree/main/data) GitHub repository and save it in `_2_data_process_for_scTAPE`.

```
$ Rscript download_lung_GSE130148.R
```
