# PASS-ML: ML-assisted pinching-antenna placement (Major Project, Phase 1)

Python continuation of the MATLAB minor project (MINOR-PROJECT-PASS).
It keeps Stages 1-4 and adds **Stage 5**, where a machine-learning model
chooses the coarse antenna placement and a short grid search does the
fine phase alignment.

## Run

    pip install -r requirements.txt
    python test_sanity.py        # quick checks (seconds)
    python run_all.py            # dataset -> training -> evaluation -> figures

`run_all.py` takes roughly 20 minutes on a 2-core machine; almost all of it
is `s01_generate_dataset.py`. The generated dataset, trained model, results
and figures are already included, so steps 2-4 can be run on their own.
For a fast trial: `python s01_generate_dataset.py 200 50`.

## Files

| File | Purpose |
|---|---|
| `pass_sim.py` | Simulator: parameters, room and LoS test, channel, ZF, ZF-AO search |
| `pass_ml.py` | Stage 5: candidate placements, features, nearest-waveguide rule, ML placement |
| `s01_generate_dataset.py` | Builds training and test sets with the simulator |
| `s02_train_model.py` | Trains and compares Ridge, k-NN, random forest, gradient boosting |
| `s03_evaluate.py` | Runs all schemes on unseen user drops |
| `s04_make_figures.py` | Writes the 8 report figures to `figures/` |
| `matlab_crosscheck/` | Same fixed scenario in MATLAB and Python, to confirm the port |

## Method

1. **Coarse decision.** Each waveguide packs its 6 antennas above one
   user. With 5 waveguides and 4 users (every user served) there are 240
   such assignments.
2. **ML model.** A regressor predicts the transmit power [dBm] an
   assignment will need. Inputs (49 features): waveguide-to-user offsets,
   LoS flags, waveguides per user, distance to the nearest other user, and
   the large-scale link gain of each waveguide-user pair.
3. **Fine search.** The 3 assignments with the lowest predicted power are
   refined with one grid-search sweep within +/-5 cm; the best is kept.
4. **Stage 5+ (optional).** Two ordinary ZF-AO sweeps starting from the
   Stage 5 result.

Training data: 1,500 user drops x 60 random assignments = 90,000 rows,
labelled by the simulator. Test data: 200 different drops with all 240
assignments evaluated, so the best possible choice is known.

## Results (200 unseen drops, 10 dB SINR target, fading on)

| Scheme | Mean Tx power | Cost evaluations | Time per drop |
|---|---|---|---|
| Stage 2: random + ZF | 15.9 dBm | 0 | - |
| Stage 3: ZF-AO, random start | -2.4 dBm | 14,051 | 151 ms |
| Stage 4: ZF-AO, smart start | -2.4 dBm | 13,600 | 147 ms |
| Nearest-waveguide rule + fine search (no ML) | 0.3 dBm | 521 | 13 ms |
| **Stage 5: ML + fine search** | **-1.3 dBm** | **1,521** | **59 ms** |
| **Stage 5+: ML + 2 ZF-AO sweeps** | **-2.2 dBm** | **4,325** | **88 ms** |

Stage 1 (random positions, matched filter at the Stage 2 power) reaches a
mean per-user SINR of 2.4 dB. Power is averaged in dBm. Times were measured
with two worker processes running in parallel. Full numbers are in
`results/summary.txt` and `results/train_report.json`.

Model comparison (validation drops): gradient boosting R2 = 0.64, random
forest 0.58, k-NN 0.58, ridge 0.51. All four pick placements of similar
quality (within about 0.15 dB), and 50 training drops are nearly as good
as 1,500.

## Differences from the MATLAB code

- **Fading is frozen per trial.** One Rician and one Rayleigh coefficient
  per user-waveguide link, drawn once. The MATLAB code redrew fading on
  every channel call.
- **NLoS loss** is the 10 dB wall loss in all cases (MATLAB used 15 dB with
  fading on and 10 dB with fading off).
- **Stage 1** uses the matched filter at the same total power as ZF, so its
  SINR is comparable.
- **MIMO benchmarks are not part of Phase 1.** The comparison with
  conventional MIMO is cited from Wang et al. instead of re-simulated.
- The grid search evaluates tr((Psi^H Psi)^-1) for all candidates in one
  batched 4x4 inverse instead of the Sherman-Morrison update; the objective
  is identical.

## Limitations

Static users; one user per waveguide in the coarse stage; 2-D LoS test;
equal-power radiation model only; the model is trained for this room and
this N, M, K.
