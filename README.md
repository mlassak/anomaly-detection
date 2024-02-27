# Anomaly detection on time-series data

## Environment setup
Package management and virtual environment management using `pipenv`.

### Initial setup
1. Install `pipenv`.
2. Navigate to the project root directory.
3. Setup the virtual environment with the project dependecies using `pipenv install`.

## Basic usage

This section describes how to utilize the implemented functionality.

### Data pulling
1. Setup a `.env` file within the project root directory, with its content structure according to the provided `.env.example` and your given inputs.
2. Setup a PromQL query config according to the given Prometheus configuration of your Prometheus API server, and ensure to correcly refer to it in `.env`.
3. Run the data pulling script using `pipenv run data_retriever.py`.

### Model usage
For now, refer to the provided Jupyter notebooks (`arima_example.ipynb` and `lstm_example.ipynb`) and data/config examples provided in `examples/` as reference for usage.
Utilize the created virtual environment as Jupyter kernel for your notebooks.
