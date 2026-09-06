# Processed Data Index (paper figures and tables)

Every file under `data/` is used by at least one figure or table in the paper.
The table below maps each paper figure/table to its data source and to the
plotting/experiment script under `code/`. Run the scripts from the repository
root after cloning.

Legend:
- columns with a figure number refer to the figure number in the **final PDF**
  (`fig:intro_channels` = Fig. 1, ..., `fig:llm_temp` = Fig. 13).
- `fig3.eps` (architecture) and `fig4.eps` (workflow) are hand-made schematic
  figures without data or a generating script.

| Figure/Table in paper | Data file(s) in `data/` | Script under `code/` |
|---|---|---|
| Fig. 1 `fig_intro_channels` | `hkust/hkust_Zone_J1.csv` | `kanconv/kanconv_fig_intro_channels.py` |
| Fig. 2 `fig:hp_vs_aug` | `llm_hp_scale_212.csv`, `llm_scale_effect_213.csv` | `llm_project/llm_fig_hp_vs_aug_212.py` |
| Fig. 5 `fig_enh_family` | (no data; schematic waveforms) | `llm_project/draw_fig4_enh_family.py` |
| Fig. 6 `fig:vis_dkasc11` | `kanconv_vis_pred/DKASC-11/*.npy` (prediction caches) | `kanconv/kanconv_best_seed.py` |
| Fig. 7 `fig:ablation` | `kanconv_4sig_hkust.csv`, `kanconv_8site_design.csv`, `ablation/ablation_C8.csv` | `kanconv/kanconv_ablation_fig.py` |
| Fig. 8 `fig:llm_joint` | `llm_e2_small_{a_llm,c_tpe,f_random,i_dehb}_aug_{212,213}[,_run1,_run2].csv` | `llm_project/llm_fig_joint_search_212_213.py` |
| Fig. 9 `fig:llm_r3` | `llm_e2_small_{a_llm,c_tpe,f_random,i_dehb,h_hebo}_aug_{212,213}[,_run1,_run2].csv` | `llm_project/llm_fig_r3_spread.py` |
| Fig. 10 `fig:shap_importance` | `llm_trial_registry_212_fixed85_sub0.25.json` | `llm_project/llm_shap_analysis.py` |
| Fig. 11 `fig:shap_interaction` | `llm_trial_registry_212_fixed85_sub0.25.json` | `llm_project/llm_shap_interaction.py` |
| Fig. 12 `fig:lr_factor` | `llm_trial_registry_212_fixed85_sub0.25.json` | `llm_project/llm_fig_lr_factor_interaction.py` |
| Fig. 13 `fig:llm_temp` | (values hard-coded in script; see `llm_interactions/`) | `llm_project/llm_fig_temperature.py` |
| Table 1 `tab:main` (ten-site) | `kanconv_4sig_hkust.csv` (HKUST-4) + `kanconv_8site_design.csv` (SQ1 + DKASC-5) | `kanconv/kanconv_4sig_hkust.py`, `kanconv/kanconv_8site_design.py` |
| Table 2 `tab:gen` (benchmarks) | `kanconv_ltsf_ettm2.csv`, `kanconv_ltsf_electricity.csv`, `kanconv_ltsf_electricity_std.csv` | `kanconv/kanconv_ltsf_ettm2.py`, `kanconv/kanconv_ltsf_electricity_supp.py` |
| Table 5 `tab:llm_ablation` | `llm_e2_small_*.csv` (+ `llm_trial_registry_*.json`) | `llm_project/llm_e2_ablation.py` |
| Table A1 `tab:params` (params) | `kanconv_4sig_hkust.csv`, `kanconv_8site_design.csv` | — (computed in main-table scripts) |
| Table A2 `tab:searchspace` (search space) | — (schema fixed in `code/config.py` / `llm_project/llm_search_space.py`) | — |

## Raw site data

- `site11/56/67/73/79_2016_15min.csv` — cleaned 15-min DKASC sites used in the
  main ten-site comparison.
- `site212_2016_15min.csv`, `site213_2016_15min.csv` — DKASC cold-start LLM sites.
- `hkust/*.csv` — cleaned HKUST sites (`Zone_J1`, `UG3`, `SQ2`, `Zone_D`, `SQ1`).

## LLM interaction records

The full request/response traces of the LLM-driven joint search are under
`../llm_interactions/` (committed separately from the regenerated `llm_logs/`
so that re-running scripts does not dirty the release):

- `agent_ctx_t0.1.jsonl`  — main joint search with full context, temperature 0.1 (sites 212/213/56)
- `agent_ctx_t0.3/0.5/0.9.jsonl` — temperature-sensitivity scan on DKASC-212
- `agent_noctx_t0.1.jsonl` — context-ablation (no-context) runs

Files are JSON Lines: one `{kind: request|response}` per line. API keys are
never stored in these logs.
