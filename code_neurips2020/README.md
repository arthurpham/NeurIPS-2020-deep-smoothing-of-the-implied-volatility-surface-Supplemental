## End-to-end example

Load the data, create and train the model, and display the results with `R-ex/run_example.R`. 
The figure generated is similar as in the paper.
Note that the file assumes that the working directory is the main folder (i.e., where `code.Rproj` is located, see below).

## Software

To run this code, you will need the Python packages `tensorflow` (>= 2.0.0), `tensorflow_probability`,
and the R packages `tidyverse`, `lubridate`, `patchwork`, `ggthemes`, `tensorflow`, and `scam`.

## Data

- Synthetic prices and IVs for the Bates model are in `data/train_data.csv`.
- Because the real prices provided by OptionMetrics through WRDS are proprietary, they can't be included.
  However, the file `R-ex/data_preprocess.R` was used to process the data from the SPX data that can be 
  downloaded from WRDS.

## Other files

- `code.Rproj`: an RStudio project file to ensure that the working directory is set properly.
- `R/utils_bsm.R`: functions to price options and compute IV values for the Black-Scholes model.
- `R/utils_data_preprocess.R`: functions to preprocess the real data from WRDS.
- `R/utils_plot.R`: functions to display results (IVS, total variance, etc)
- `R/utils_ivsmoother_models.R`: functions to build the model and the losses.
- `R/utils_ivsmoother_fit.R`: functions to fit the model.
- `R/utils_tensorflow.R`: utils to reset tensorflow sessions, memory, etc (warning: compatibility issues v1 versus v2).

## Warning!!!

This version of the code runs slow because it builds 4 times the same NN model (for the different loss functions described in the paper).