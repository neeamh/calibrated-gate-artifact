# SynTrustBench v0.4.1 gate-focused experiment notebook

**v0.4.1 in one paragraph.** The claim under test: averaging dimension scores lets one serious failure hide, and a gate that requires every dimension to pass catches it, but only once each dimension is calibrated against real-vs-real comparisons. The gate now decides on fidelity, utility and privacy; equity and robustness are computed and reported as non-gating diagnostics. Calibration uses 100 fresh null runs (50 fit / 50 audit) and a full-conformal rule. A prespecified admission rule decides on the audit whether each dimension may gate. Every judge (gate, mean, geometric, harmonic, native SDMetrics) is calibrated to the same honest-data rejection rate, so known-bad acceptance is compared at matched cost. The one-page spec is `PROTOCOL_v0.4.1.md`; the reasons for every change are in `JUDGEMENT_CALLS.md`, under "v0.4.1 amendment". Results go to `runs/main_v5`, and the v0.4 run in `runs/main_v3` is kept unchanged.

**Run the full campaign (Mac, JupyterLab):** open the notebook with the `stb04` kernel, check that the settings cell says `'output_root': 'runs/main_v5'`, `smoke_mode: False`, `dry_run: False`, then Run All. Keep `caffeinate -dimsu` running and the lid open. Expect about 8–9 hours with 4 workers (the null stage is about 3 hours of that).

Open **`syntrustbench_v04_experiment.ipynb`** in Jupyter or Colab with Python 3.11+ (3.12 recommended). It is self-contained and delivered unexecuted. It embeds the complete additive library and original-reference regression fixture. The accompanying Python package is provided for inspection, not required to run the notebook.

## Run smoke mode, then the full campaign

1. In the notebook's single configuration cell, set `data.raw_csv` to `diabetic_data.csv` and choose `execution.output_root`. Alternatively, supply `data.frozen_dir` and `data.frozen_manifest` as described below.
2. For the first run, leave `install_dependencies=True`. Use a fresh kernel. The setup installs exact numerical pins and SynTrustBench v0.3.0 at commit `3dd0f7ce088994d84f4925ca7e0949d7e0ecd3d8`, then verifies the installed sources against the checkout. Apple Silicon uses CPU; optional CUDA requires `generators.allow_cuda=True`.
3. Keep `dry_run=True` and run all cells to inspect the full job plan without fitting models. Setup still installs/verifies dependencies.
4. For an end-to-end smoke campaign, set `dry_run=False`, `smoke_mode=True`, and retain all ten stages. Smoke uses at most the configured 200 rows per prepared split (maximum permitted: 500), one epoch, one seed per enabled generator (DPAIM included), five null repetitions split 3/2 (too few for the calibrated rule to reject at these α levels; recorded as `finite_sample_valid: false`), one control replicate, and one nonzero dose per family. Every family, generator, judge and stage runs. Smoke thresholds are not paper evidence.
5. For the full campaign, set `smoke_mode=False`, keep all stages, and run top to bottom. The full `smoke` stage first runs a separate complete small campaign using only T+C. Main E is excluded from that preflight and from null calibration. Full mode uses 100 null runs (50 fit/50 audit, fresh v0.4.1 seeds), all twelve default generator fits, three replicates of each control, all doses on positive controls and Gaussian Copula baselines, and both provisional/calibrated variants.

The full plan contains **542 jobs**, including **193 actual profile evaluations** (100 null) and **24 dose-zero profile aliases**. Each actual profile writes the eight canonical official-v0.3 files, v0.4 evidence, and SDMetrics evidence. No generator/control/injection scoring starts before the calibration artifact is saved and hashed. The small plan has 69 jobs. These are plan counts, not experimental results.

Start with `workers=1`; increase conservatively for available RAM. Independent profile jobs use a spawn-based `ProcessPoolExecutor`, with per-worker thread limits. The notebook runs without CUDA. Cross-hardware bitwise identity is not claimed.

## Added generators

The existing notebook structure and all original cell IDs are preserved. Four new library/explanation cells extend the generator layer. The frozen v0.3 code and the prior `reference.py`/`results.py` are unchanged. Injections remain exclusively on positive controls and Gaussian Copula baselines.

| Generator | Default | Environment and settings | Added jobs (full / smoke) |
|---|---|---|---|
| DPAIM | On, required | SmartNoise Synth 1.0.8; ε=1.0; δ=`1/(10*n_train)` unless configured | 6 / 2 |
| DPMST | Off: `generators.enable_dpmst` | Same SmartNoise/public-domain preprocessing | 6 / 2 when enabled |
| TabDDPM | Off: `generators.enable_tabddpm` | Synthcity 0.2.12 `ddpm`, isolated Python 3.10, CPU | 6 / 2 when enabled |

Default counts are 542 full / 69 smoke. One optional generator gives 548 / 71; both give 554 / 73. The 192 full injection jobs (32 smoke) are unchanged. Each optional generator adds three full official profiles (approximately 1.75–2.25 hours of official evaluation sequentially) plus its fit and v0.4 evaluation time.

SmartNoise preprocessing uses **only configured public domains**. `generators.dp_domains` declares each integer lower bound, cap, bin width and finite overflow upper bound. `dp_categories` declares public category supersets. No training extrema, quantiles, observed categories or frequencies define bins or labels. The fitted public `TableTransformer` is passed explicitly with zero preprocessing epsilon. Samples are decoded uniformly to integers in their bins using a sampling seed, with hospital stay restricted to 1–14 and all counts nonnegative. The persisted manifest records every interval and category. This follows SmartNoise's [explicit-transformer interface](https://docs.smartnoise.org/synth/synthesizers/index) and the [UCI schema](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), with analyst choices listed in JUDGEMENT_CALLS.md.

Seeds 17/29/43 identify separate DP fits. **DP fit noise is fresh and private**, as agreed; public seeds determine sampling and within-bin decoding from a fixed fitted model. Same-seed independent refits are not claimed to be identical. Reuse the completed hashed fit/sample for reproducibility. Do not publish a private noise tape or replace DP noise with publicly seeded noise.

Per-fit `generator_provenance.json` and the separate analysis table `generator_formal_privacy.csv` record nominal ε/δ, resolved δ, public n, preprocessing budget, scope and assumptions. They do not change the empirical Privacy evaluator or gate. This is upstream mechanism accounting, not independent certification. It excludes raw inputs, exact cohort summaries, input hashes, evaluator reports, and other generator outputs. Multiple fitted releases compose: three ε=1 fits are not a single ε=1 release. No guarantee is made for publishing the campaign directory as a whole.

TabDDPM is never imported into the main numerical environment. Set `generators.tabddpm_python` to a Python 3.10 executable if needed. The optional environment installs the complete `syntrustbench_campaign/tabddpm-requirements.txt` lock using `--require-hashes --no-deps`, followed by `pip check`; every runtime dependency is already pinned in the resolved lock. Its package versions, Synthcity source hashes, worker hash and lock hash are verified/recorded. CSV requests, models, subprocess logs and outputs are hashed with the job artifacts. Missing Python, an unbuildable environment or a failed import probe becomes **Unavailable** for its generator/profile jobs; judges and reports retain those unavailable rows and the other generators continue. Full-mode preflight also tolerates only these declared optional TabDDPM gaps.

Local TabDDPM validation installed the pinned Python 3.10 environment and passed its dependency check, but the plugin import was **Unavailable** because XGBoost could not load `libomp.dylib` on this Mac. A working OpenMP runtime is needed before retrying here. No TabDDPM fit or sample is claimed as locally validated. AIM and MST were fitted and sampled successfully.

After fixing an optional environment, enable `retry_failed=True`. Unchanged Unavailable jobs otherwise resume without rewriting their states. A successful retry can refresh saved judges/analysis/report in new attempts; changed scientific inputs still refuse reuse. Unexpected fit/sample errors after a successful environment probe retain the original traceback-and-raise policy.

Additional CPU planning estimates: AIM/MST can take minutes to hours per full fit depending on public domain sizes and graphical-model complexity; TabDDPM can take tens of minutes to hours per fit, plus a potentially lengthy environment build. These are broad estimates, not full-UCI measurements. Full AIM uses the upstream adaptive-round default; small mode uses 32 initial-budget rounds. TabDDPM uses 1,000 epochs/timesteps by default, reduced to 1 epoch/10 timesteps in smoke mode. All settings and fit/sample times are recorded. CUDA and cross-platform execution were not validated locally.

## Frozen inputs

The split directory must contain `real_train.csv`, `calibration_pool.csv`, and `real_test.csv`, in the exact analytic column order in the configuration. A JSON manifest must contain `split_hashes`, mapping those three names to SHA-256 values. Include `patient_audit` (absolute path, or relative to the manifest) and preferably `patient_audit_sha256`.

The patient audit CSV must contain `role`, zero-based `analytic_row`, and `patient_id_sha256`, with one aligned, unique hash per patient. The previous notebook's `patient_role_audit.csv` is supported. When `patient_audit` is omitted, the file is sought in the split directory. Without patient provenance the job is explicitly unavailable; the notebook does not silently resplit or infer patient independence from anonymous rows.

## Resume and inspect evidence

Rerun with the same scientific settings and input files. Use a new output root for this extension; existing v0.4 runs retain their original implementation lock. Completed jobs are reused only after input/artifact hash verification. Numerical-version or implementation changes refuse resume. Use a new `output_root` for changed implementation, environment or inputs. Scientific configuration changes produce a different campaign directory automatically. After inspecting a failed job, enable `retry_failed=True` to create a new attempt; prior attempts remain intact.

The campaign folder contains the expected-job plan, frozen configuration, numerical lock, session manifests, append-only event/failure ledgers, job states and attempt folders. Unexpected errors retain full tracebacks and are re-raised. Only declared `UnavailableAnalysis` cases become unavailable records.

The `analysis` and `report` stages read saved evidence only. They create:

- Cohort/split, missingness and subgroup tables; official-v0.3/provisional-v0.4/calibrated-v0.4 comparisons and metric evidence.
- **Gate admission table** (prespecified rule, held-out audit runs), **known-bad acceptance** (headline set: real data with one injected flaw, plus the copy control) and **honest-data acceptance** for every judge, under the primary matched-cost scheme and the secondary τ* scheme; acceptance by injection family and dose; judge provenance.
- Controls versus truth by judge; injection minimum doses, response and selectivity; held-out null false alarms with Wilson intervals; threshold sensitivity; ranking inversions.
- Status heatmap (gate dimensions, then diagnostics), injection response heatmap with intended dimensions outlined, a judge-comparison figure with an error trade-off panel, and `report.md`, whose headline section is computed from the tables. Figures are saved as PNG and PDF.
- Dated completion audits, including missing metric-level analyses inside otherwise completed profiles. After an interrupted/erroring stage, run the final audit cell manually.

`raw_reference` preserves the original Fidelity/Utility outputs. Evidence-sufficiency and calibrated decisions are separate wrappers. Calibrated metric thresholds/flags are under `calibrated_signals`; the original metric descriptions are retained for provenance.

## CPU runtime estimates

These are planning estimates, not guarantees. The supplied brief reports **35–45 minutes per full v0.3 profile** on Colab CPU and **about 25 seconds for reference Fidelity+Utility per half-size table**. The new v0.4 robustness refits add time beyond that reference-only estimate.

| Stage | CPU planning estimate |
|---|---|
| `prepare` | Seconds to a few minutes, excluding download/setup |
| `smoke` | Several minutes on small data; includes all three one-epoch SDV generators and DPAIM and all stage paths |
| `null` | 100 profiles: measured about 6.4 min each on an M5 Pro (about 3 h wall with 4 workers) |
| `calibrate` | Seconds to minutes once saved null evidence exists |
| `clean` | Twelve default profiles: roughly 7–9 hours official evaluation, plus generator fitting and v0.4 work |
| `controls` | Nine profiles: roughly 5–7 hours official evaluation, plus v0.4 work |
| `injections` | 72 nonzero profiles: measured about 6.7 min each on an M5 Pro (about 2.7 h wall with 4 workers); dose-zero profiles are reused |
| `judges` / `analysis` | Seconds to minutes, depending on artifact size |
| `report` / completion audit | Seconds to minutes for figures and hash verification |

Measured v0.4 campaign on an M5 Pro (64 GB, 4 workers × 2 threads): 5 h 47 min for 362 jobs. v0.4.1 adds 180 null jobs, about 8–9 h in total. Parallel wall time depends on memory, thread contention and disk speed.

## Validation boundary

The delivered notebook has empty outputs and execution counts. v0.4.1 was checked before delivery: the full-mode dry run (frozen v0.3 install and source verification, 542-job plan) and a complete smoke campaign on the frozen UCI splits (69 of 69 jobs, no failures) both ran in a clean Python 3.12 environment with the pinned packages. The new calibration, judge and analysis code was also run over the saved full-size v0.4 profiles as an integration test; that run is not evidence, because it reuses the runs that motivated the amendment. The self-tests check that diagnostics never change the gate, that any gating failure fails it, and that the honest rejection rate of the gate and of every matched compensatory judge is nominal on simulated heavy-tailed data. **No v0.4.1 paper result is claimed** until `runs/main_v5` completes.

Read **`JUDGEMENT_CALLS.md`** before using the output for a paper. It documents the unsettled score normalization, role sizes, reused positive pool, missing-evidence policy, calibration limits, robustness interpretation, and official-comparison data choices. Execution completion does not establish the claim; read the admission table first.
