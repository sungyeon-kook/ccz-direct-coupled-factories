# Direct-coupled architectures for compact CCZ magic-state factories

Code and data for the manuscript **"Direct-Coupled Architectures for Compact CCZ Magic-State Factories"**
(S. Kook, Y. Kang, I. Sohn, and J. Heo).

The code implements a patch-level layout model and a logical-error model of two-level (15-to-1) x (8-to-CCZ)
factories with three interfaces between the levels: Litinski's interface (L) and the direct-coupled interfaces
D1 (in-place consumption) and D2 (one buffer per supplier). It searches code distances and supplier counts for each
interface, validates the selected designs with a stochastic schedule, and contains the Stim circuits used for the
noise calibration and the merge-circuit checks. All results are model estimates; see the Limitations section of the
manuscript.

## Requirements
Python 3.11+ with `numpy numba scipy mpmath matplotlib` for the resource model, and `stim sinter pymatching`
(PyMatching 2.4) for the Stim scripts:

    pip install -r requirements.txt

## Layout
All scripts are run from the folder they live in, unless noted.

| folder | contents |
|---|---|
| `code/audit/` | resource model (`model.py`), dispatch scheduler (`timing.py`), distance search (`search.py`), stochastic validation (`validate_fronts.py`), regression and precision checks, scheduling-policy and merge-circuit drivers |
| `code/corrected_results/` | baseline search outputs used in the manuscript (`front_*`, `validated_*`, `validated_front_*` for the scenarios `lit`, `q1_abs_d7`, `q5_abs_d7`, `p4`), calibration, merge-circuit, and scheduling-policy results |
| `code/hook2_results/` | search outputs with merge-ancilla rows of height 2 d_X2 |
| `code/sensitivity_v12/` | search outputs with the in-place Y-readout fault, R = 5 and 20, and a 20% error margin |
| `code/scripts/stim/` | Stim memory/stability experiments and merge-circuit construction |
| `code/analysis/` | tables derived from the fronts, design re-evaluation, reproducibility checks |
| `code/verification/` | state-vector check of the signed adaptive X/Y consumption |
| `code/figures/` | scripts of the layout figures |

## Manuscript items

| manuscript item | how to reproduce | stored output |
|---|---|---|
| Sec. II-B, adaptive X/Y rule over all 4^8 branches | `python3 verification/verify_signed.py` | printed |
| Table 2, Table 3, Fig. 5 (baseline fronts) | in `audit/`: `python3 search.py lit` then `python3 validate_fronts.py lit`; `python3 analysis/caps_from_fronts.py corrected_results` (from `code/`) | `corrected_results/*_lit_*` |
| Table 4 (fixed qubit budget) | `python3 analysis/budget_packing.py corrected_results` | printed |
| Table 5 (noise anchors, p = 1e-4) | `search.py` / `validate_fronts.py` with `q1_abs_d7`, `q5_abs_d7`, `p4` | `corrected_results/*_q1_abs_d7_*` etc. |
| Table 6 (Y-readout fault, R, error margin) | `analysis/run_sensitivity.sh` (usage in the file) | `sensitivity_v12/` |
| Table 7 (two-level T factories) | `python3 analysis/t_factory_interface.py` | printed |
| Table 8 (merge-ancilla height) | `OUTDIR=../hook2_results HOOKH=2 python3 search.py lit`, then `validate_fronts.py lit`; `analysis/hook_height_reeval.py audit` | `hook2_results/` |
| Table 9 (scheduling policies) | in `audit/`: `python3 schedule_sensitivity.py lit q1_abs_d7 q5_abs_d7 p4`; `check_schedule_policies.py` | `corrected_results/schedule_sensitivity/` |
| Fig. 6, App. B (S, T anchors) | `scripts/stim/timelike_rates.py`, `fit_timelike.py`, then `audit/calibrate.py` | `scripts/stim/timelike_stats.csv`, `tables/stim_timelike.csv`, `corrected_results/calibration_absolute.csv` |
| App. C (merge circuits) | in `audit/`: `python3 geometry_probe.py 5 3 3 5`, `7 5 5 7`, `5 5 3 5` | `corrected_results/geometry_*.json`, `probe_*.stim` |
| App. D (reproducibility) | `analysis/scheduler_crosscheck.py audit`, `analysis/search_completeness.py audit corrected_results 20000`, `audit/checks.py`, `audit/precision_check.py` | `corrected_results/checks.json`, `precision_check.json` |
| Figs. 2-4 (layouts, boundaries, timeline) | `python3 figures/fig_architectures.py out.pdf` (likewise `fig_boundary.py`, `fig_timeline.py`) | - |
| Figs. 5 and 6 (fronts, calibration) | `python3 audit/make_report.py` (writes `figs/` and `generated/` next to `code/`) | - |

Environment parameters of `search.py` (defaults reproduce the baseline): `HOOKH` (merge-ancilla height in units of
d_X2), `KY` (in-place Y-readout fault factor, Eq. (10)), `RREACT` (reaction time R), `TARGET` (output-error target),
`OUTDIR` (output folder; default `code/corrected_results`, so set it for any non-baseline run).

## Notes
* Fig. 1 (circuit diagram) is a drawing without data and is not included.
* `audit/checks.py` compares the model with Litinski's published implementation
  (https://github.com/litinski/magicstates) and a legacy NumPy port, which are not redistributed here; its stored
  result is `corrected_results/checks.json`.
* The search keeps the fastest candidate for each qubit count; see the Limitations section for the consequence
  (a 28,436-qubit, 280-cycle D1 design, shown in `sensitivity_v12/d1_screen_KY0/`).

## License
MIT (see `LICENSE`).
