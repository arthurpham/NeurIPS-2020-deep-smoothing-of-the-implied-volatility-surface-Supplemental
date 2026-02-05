# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Supplemental code for the NeurIPS 2020 paper "Deep Smoothing" - a neural network approach for modeling implied volatility surfaces (IVS) that enforces absence of arbitrage and respects parametric priors from financial models.

- Paper: https://proceedings.neurips.cc/paper/2020/hash/858e47701162578e5e627cd93ab0938a-Abstract.html
- Presentation: https://videos.neurips.cc/search/volatility/video/slideslive-38937756

## Running the Code

```bash
cd code_neurips2020
# Open code.Rproj in RStudio, then run R-ex/run_example.R
# Or from command line:
Rscript R-ex/run_example.R
```

The working directory must be `code_neurips2020/` (where `code.Rproj` is located).

## Dependencies

**Python:**
- tensorflow >= 2.0.0
- tensorflow_probability

**R:**
- tidyverse, lubridate, patchwork, ggthemes, tensorflow, scam
- furrr (for data preprocessing)

## Architecture

### Neural Network Design

The model combines a neural network with financial priors:

```
Input (ttm, logm) → Hidden Layers → Output (exp activation)
                                         ↓
                              [Prior Integration]
                              ├─ No Prior: Direct output
                              └─ With Prior: NN × Prior model (SVI/BS)
                                         ↓
                              Loss = Fit + C4 (calendar) + C5 (butterfly) + C6 (moneyness) + ATM reg
```

### Core Modules (in `R/`)

- **utils_ivsmoother_models.R** - NN architecture, loss functions, prior models (SVI, Black-Scholes)
- **utils_ivsmoother_fit.R** - Training loop with Adam optimizer, early stopping, LR scheduling, multi-restart
- **utils_bsm.R** - Black-Scholes-Merton pricing, IV computation, Greeks
- **utils_plot.R** - Visualization with ggplot2
- **utils_data_preprocess.R** - Forward price extraction, rate/yield inference
- **utils_tensorflow.R** - TF session management, GPU memory control

### Key Design Patterns

- Functional R style with tidyverse pipelines (not OOP)
- TensorFlow v1 compatibility layer (`tf$compat$v1$`) for TF 2.x
- Multiplicative prior integration: `NN_output × Prior_model`
- Multi-restart optimization for robustness

## Data

- `data/train_data.csv` - Synthetic Bates model data (only data included)
- `data/df_fit.csv` - Pre-computed model fitting results
- Real OptionMetrics SPX data from WRDS is proprietary and not included; see `R-ex/data_preprocess.R` for processing pipeline

## Known Issues

- Current version builds 4 identical NN models for different loss functions (slow, but intentional for research comparison)
- TensorFlow v1/v2 compatibility handled via compat layer
