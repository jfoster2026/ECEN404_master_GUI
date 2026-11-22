# SMAJ400A Master GUI

The master GUI is a desktop app that predicts the voltage waveform of an SMAJ400A diode during a
radiation pulse and compares the prediction with the measured waveform. It
includes five models (Siamese, KAN, PINN, DPM and Tabular) and scores them all
the same way: each test is held out, predicted from the other tests, then
compared with its measurement.

**The master GUI runs from `smaj_gui.py`.** Start it with `python smaj_gui.py`
(full steps below).

## Files

| File | What it is |
| --- | --- |
| `smaj_gui.py` | The GUI. Run this file. |
| `smaj_models.py` | The models and the scoring code. Must be in the same folder as `smaj_gui.py`. |
| `siamese_ridge_metadata.csv` | The 21 tests: dose rate, bias, load, PCD width, scope scale factor and the path to each waveform file. |

## 1. Install Python and the packages

You need **Python 3.12** (Tkinter is included with the python.org installer).
In PowerShell, from the folder that holds the three files:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "numpy>=2.0,<3" "pandas>=2.2,<4" "scipy>=1.12,<2" "matplotlib>=3.8,<4" "torch>=2.5,<3" "pykan==0.2.8" "scikit-learn>=1.3,<2" "tqdm>=4.66,<5" "pyyaml>=6,<7"
```

On macOS or Linux, use `python3.12 -m venv .venv` and `.venv/bin/python`
instead. The GUI was developed and tested on Windows.

## 2. Add the waveform data

The measured waveforms are **not in this repository**: they total about 400 MB,
and one file is larger than GitHub's 100 MB limit. Get them from the team share
and put them in a `data` folder next to `smaj_gui.py`:

```
smaj_gui.py
smaj_models.py
siamese_ridge_metadata.csv
data/
    27271_data.csv
    27272_data.csv
    ...
```

The GUI expects these 21 files: `27271_data.csv` through `27289_data.csv`,
plus `27306_data.csv` and `27307_data.csv`. Each is a scope export containing
the `Diode` and `PCD3_B` channels. If a file is somewhere else, change its path
in the `waveform_csv` column of `siamese_ridge_metadata.csv`.

## 3. Run the GUI

The GUI starts from `smaj_gui.py`. From the same folder:

```powershell
.\.venv\Scripts\python.exe smaj_gui.py
```

## Using the GUI

1. Pick a model with the **Model: Siamese | KAN | PINN | DPM | Tabular** buttons
   at the top.
2. Choose a test from the list on the left, then click **Model selected test**.
   The plot shows the measured waveform (solid) and the prediction (dashed). The
   cards above the plot give the error (NRMSE, %) and R².
3. Use **First 200 ns | Prompt window | Full waveform** to change the time range.
   **View waveform stats** shows the full statistics for that test.
4. **Run all tests** predicts every test with the current model and shows a
   table of results, with a **Download CSV** button. From that window,
   **Input sensitivity** plots the error against dose rate and PCD width.
5. **Compare models** runs every model and opens a side-by-side comparison:
   summary statistics, per-test results, sensitivity plots, and every model's
   prediction for one test on the same plot.
6. **Choose tests…** lets you leave tests out of any model, or add new tests
   (drop the scope CSV into a `new_tests` folder and enter its conditions).

## The five models

| Model | How it predicts | Works with just these files? |
| --- | --- | --- |
| **Siamese** | Kernel ridge regression that maps a test's conditions (dose rate, bias, load, PCD width) to its waveform. | Yes, a few seconds per test. |
| **KAN** | A Kolmogorov-Arnold network that predicts the voltage at each time from the measured PCD3_B dose-rate signal, bias and load. | Yes. The first run trains one network per held-out test (a few minutes each) and saves them in `.smaj_gui/kan_networks/`; later runs reuse them. |
| **PINN** | A physics-informed neural network driving a diode-circuit model. | No. Needs the team's `pinn_dpm/` package and its own Python environment. A full 21-test run takes about 9 hours. |
| **DPM** | A deterministic physics model of the diode circuit; not fitted to any test. | No. Needs the `pinn_dpm/` package. |
| **Tabular** | Google's pretrained TabFM model, given sample points from the other tests as examples. | No. Needs the team's `tabfm_handoff/` package, its environment and the 6 GB model weights (non-commercial license). |

Without its package, the PINN, DPM or Tabular model shows a message saying
what is missing. The Siamese and KAN models still work. To set up those
packages, place `pinn_dpm/` or `tabfm_handoff/` next to `smaj_gui.py` and follow
the setup instructions that come with each package.

## Troubleshooting

- **"CSV not found"**: a waveform file is missing from `data/`, or its path in
  `siamese_ridge_metadata.csv` is wrong.
- **"No module named …"**: run the GUI with the environment's Python
  (`.\.venv\Scripts\python.exe`), not the system `python`.
- **KAN is slow the first time**: it is training its networks. Copy a
  `.smaj_gui/kan_networks/` folder from a computer that already has them to
  skip this.
- Running `python smaj_models.py` directly only prints a reminder to start the
  GUI with `smaj_gui.py`; it is used that way when the Tabular model runs.
