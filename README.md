# Anomaly detection on time-series data

## Environment setup
Package management and virtual environment management utilizes `pipenv`.

### Initial setup
1. Install `pipenv`.
2. Navigate to the project root directory.
3. Setup the virtual environment with the project dependecies using `pipenv install`.

## Basic usage

This section describes how to utilize the implemented functionality.

### Data pulling
1. Setup a `.env` file within the project root directory, with its content structure according to the provided `.env.example` and your given inputs.
2. Setup a PromQL query config conforming to the schema defined in `query_config_file_schema.yaml` and according to the given Prometheus configuration of your target Prometheus API server. Ensure to correcly refer to the query config file in `.env`.
3. Run the data pulling script using `pipenv run data_retriever.py`.
4. If successful, files containing the time series data will be located in the directory structure with its root being `TARGET_DATA_DIR` specified in `.env`, with subdirectories and file names specified as per the query config file.

### Model workflows

Model workflows are showcased via the example Jupyter notebooks provided in the project root directory.
Utilize the created virtual environment as the Jupyter kernel for your notebooks if you wish to re-run them.

Example datasets and model configurations are provided in `examples/` directory.

Provided example notebooks serving:
- ARIMA model family:
    - `arima_example.ipynb` - univariate ARIMA workflow example, model parameters set manually via config
    - `auto_arima_example.ipynb` - univariate ARIMA workflow example
    - `arimax_example.ipynb`- multivariate input example using ARIMAX
    - `varmax_example.pynb` - multivariate input and output example using VARMAX
- LSTM
    - `lstm_example.ipynb` - univariate LSTM example
    - `multivar_lstm_example.ipynb` - multivariate input and output LSTM example
