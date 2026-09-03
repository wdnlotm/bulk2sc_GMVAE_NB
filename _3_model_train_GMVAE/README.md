# Train GMVAE
The main source code for `bulk2sc` is located in the `_model_source_codes` directory. Executing the provided Python script trains the GMVAE model using the data from the `_1_data_process_for_GMVAE` directory for 4000 epochs. The model is saved every 1000 epochs. During training, saved models are stored in the `models` directory, while the TensorBoard event file is saved in the `runs` directory.
```python
epoch=4001; savefreq=1000; h_dim1=128; h_dim2=64; z_dim=32

traindir=../_1_data_process_for_GMVAE/train/
testdir=../_1_data_process_for_GMVAE/test/

python main_nb.py --epochs ${epoch} --h_dim1 ${h_dim1} --h_dim2 ${h_dim2} --z_dim ${z_dim} \
--savefreq ${savefreq} -dl ${traindir} -tdl ${testdir} --dataset lung
```

`main_nb.py` models gene counts with a negative binomial (NB) likelihood. The decoder emits a mean and a dispersion per gene, and because the NB distribution is well defined at a count of zero, the reconstruction term needs no separate zero/non-zero case.

The modules used in `_model_source_codes`:

| | |
| --- | --- |
| entry point | `main_nb.py` |
| model | `GMVAE_scrna_nb.py` |
| loss | `GMVAE_losses_nb.py` |
| train/test loop | `GMVAE_utils_nb.py` |
| saved arguments | `args_save_nb.pickle` |
| default `--model-name` | `GMVAE_w_nb` |

Checkpoint names embed `--model-name`, and the TensorBoard scalar tags are `total_loss`, `KLD_gaussian`, `KLD_pi`, `reconst_loss` and `accuracy`, each logged for both train and test.

### Notes
- Pass a relative path to `--modelsaveloc` (`-msl`). The checkpoint path is built as `./{modelsaveloc}/...`, so an absolute path fails when the first model is written.
- Training requires `tensorboard` to be installed in the active environment.
