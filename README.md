# PC-KAN: Per-Channel Kolmogorov-Arnold Calibration with LLM-Driven Joint Tuning for Photovoltaic Power Forecasting

This repository provides the code and processed data to reproduce the experiments
in the paper *PC-KAN: A Per-Channel Kolmogorov–Arnold Calibration with LLM-Driven
Joint Tuning for Photovoltaic Power Forecasting*.

## Repository layout

```
code/
  mode_code.py                  Core model + training pipeline (CNN_LSTM_Head, Trainer)
  config.py                     Global configuration (seeds, splits, HPO ranges, API)
  utils.py                      Utilities (set_seed, metrics, paired test, plotting)
  kanconv/kan_layers.py         KANConv layers (SplineAct, KANConv1d, gate, etc.)
  kanconv/kanconv_*.py          Main-table and figure scripts
  llm_project/llm_*.py          LLM-driven joint-tuning scripts and figures
  preprocess/                   Data-cleaning scripts
data/
  site11_2016_15min.csv ...     Cleaned DKASC site data (15-min)
  hkust/                        Cleaned HKUST site data (five main-table sites)
  kanconv_4sig_hkust.csv        Main-table HKUST results
  kanconv_8site_design.csv      Main-table SQ1 + DKASC results
  kanconv_ltsf_ettm2.csv        ETTm2 generalization
  kanconv_ltsf_electricity*.csv Electricity generalization
  llm_trial_registry_*.json     LLM trial registries (SHAP / temperature / figures)
```

## Dependencies

- Python 3.9+
- PyTorch
- pandas, numpy, scipy
- matplotlib (for figure scripts)
- scikit-learn, shap (for the SHAP analysis)
- requests / openai or DeepSeek API client (for `llm_agent.py`)

## Quick start

### 1. Reproduce the main comparison (PC-KAN vs FC / SV-KAN / ConvKAN)

The ten-site main table is produced by two scripts with an identical protocol.
`kanconv_4sig_hkust.py` covers HKUST-Zone_J1 / UG3 / SQ2 / Zone_D, and
`kanconv_8site_design.py` covers HKUST-SQ1 plus the five DKASC sites
(11 / 56 / 67 / 73 / 79). Their union is exactly the ten sites reported in the paper.

```bash
python code/kanconv/kanconv_4sig_hkust.py    # HKUST-4 (Zone_J1, UG3, SQ2, Zone_D)
python code/kanconv/kanconv_8site_design.py  # HKUST-SQ1 + DKASC-5 (11,56,67,73,79)
```

### 2. Reproduce the generalization benchmarks

```bash
python code/kanconv/kanconv_ltsf_ettm2.py
python code/kanconv/kanconv_ltsf_electricity_supp.py
```

### 3. Reproduce the ablation figures

```bash
python code/kanconv/kanconv_ablation_fig.py
python code/kanconv_ablation_c10.py           # NoLSTM removal ablation
```

### 4. Reproduce the LLM-driven joint tuning (requires a DeepSeek API key)

Set your API key in `code/config.py` (or a local secrets file), then run:

```bash
python code/llm_project/llm_e2_ablation.py
```

The figures can be regenerated with:

```bash
python code/llm_project/llm_fig_joint_search_212_213.py
python code/llm_project/llm_fig_r3_spread.py
python code/llm_project/llm_fig_temperature.py
python code/llm_project/llm_fig_lr_factor_interaction.py
python code/llm_project/llm_shap_analysis.py
python code/llm_project/llm_shap_interaction.py
```

### 5. Reproduce Figure 1 (per-channel scatter)

```bash
python code/kanconv/kanconv_fig_intro_channels.py
```

## Data

- **HKUST** photovoltaic data: retrieved from the Dryad repository
  (DOI: 10.5061/dryad.m37pvmd99, https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99).
- **DKASC** (Alice Springs) photovoltaic and meteorological data:
  https://dkasolarcentre.com.au/.

The cleaned 15-min processed data used by the experiments are included under `data/`.

## Notes

- The LLM agent calls an external model API and is not deterministic across API
  versions; the paper mitigates this with a fixed temperature and a frozen prompt.
- Figure 6 (`kanconv_vis_DKASC11_seed456_test_3days.png`) requires prediction
  caches produced by the model before plotting; run the model scripts first.
- Only the ten main-table sites used in the paper are included; intermediate
  experimental sites are omitted.

## License

To be determined.

## Contact

For questions or issues, please open an issue or contact the corresponding author.
