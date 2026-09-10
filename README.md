# Time Series Forecasting with Global Models

Code for a series of research projects on **how often global forecasting models
should be retrained**, and on the accuracy, stability and cost trade-offs involved.

## Papers

1. [On the retraining frequency of global models in retail demand forecasting](https://www.sciencedirect.com/science/article/pii/S2666827025001525?via%3Dihub)
2. [Analyzing the retraining frequency of global forecasting models: exploring the accuracy-stability trade-off](https://arxiv.org/abs/2506.05776)
3. [The cost of ensembling: is it always worth combining global retail forecasting models?](https://arxiv.org/abs/2506.04677)
4. [Analyzing the retraining frequency of global forecasting models in the presence of structural breaks]()

## What the experiments do

Each experiment simulates a forecasting system over a rolling test window and
repeats it under several **retraining scenarios**, from retraining at every period to
training the models only once at the beginning of the window. For every scenario the
code records forecast accuracy, forecast stability and computational time, so that the
scenarios can be compared.

- **Datasets**: M4 (daily), M5 (daily), VN1 (weekly), Hapag region (weekly)
- **Models**: statistical (ETS, ARIMA), machine learning (Linear Regression, Random Forest,
  XGBoost, LightGBM, CatBoost), deep learning (MLP, LSTM, TCN, NBEATSx, NHITS), plus ensembles
- **Built on**: the [Nixtla](https://github.com/Nixtla) ecosystem (`statsforecast`, `mlforecast`, `neuralforecast`)

## Repository structure

| Folder | Content |
| --- | --- |
| `src/env-setup/` | conda environment file to replicate the results |
| `src/Python/` | scripts to run the analysis, with the `utils/` module containing the implementation |
| `src/R/` | scripts to produce the plots and tables of results |
| `config/` | YAML configuration files with the parameters of each experiment |
| `docs/` | final plots and aggregated results, one folder per project |

Running the code creates three more folders that are **not part of the repository**:
`data/` (the datasets), `results/` (forecasts, evaluations and timings of every run) and
`logs/` (run logs).

## Setup

```bash
conda env create -f src/env-setup/conda_env_setup.yml
conda activate global_retrain
```

The datasets and the aggregated results of the projects are not stored in this repository and can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1Ff2hSvYhSMO2PZL_TX_OyKzkYjkpHID3?usp=sharing).
Place them in `data/` and `results/` at the root of the repository, or recreate them by
running the pipeline below.

## Usage

Every script reads a YAML file from `config/` whose path is set at the top of the
script itself: edit that line to switch dataset or experiment. All scripts are meant
to be run from the root of the repository.

```bash
python src/Python/download.py       # download and prepare the datasets into data/
python src/Python/retrain_sf.py     # retrain statistical models over all scenarios
python src/Python/retrain_ml.py     # retrain machine learning models
python src/Python/retrain_dl.py     # retrain deep learning models
python src/Python/preds_model.py    # collect the forecasts of the runs
python src/Python/eval_model.py     # evaluate forecast accuracy
python src/Python/stab_model.py     # evaluate forecast stability
```

`src/Python/experiment.py` chains fitting, prediction and evaluation into a single run.
The `*_dataset.py` variants apply the same steps to all the models of a dataset at once.
Every step writes into `results/` and logs its progress into `logs/`.

Results are then analysed in R through `src/R/anal/main_anal.R`, which takes a config
from `config/anal/` and writes the plots and tables of a project into `docs/`, the only
outputs shared through this repository.

## License

Released under the [MIT License](LICENSE).
