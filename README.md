# NeuralForecast-Based Electricity Load Forecasting

This repository contains a NeuralForecast-based experimental pipeline for multi-horizon electricity load forecasting using N-HiTS, N-BEATS, and N-BEATSx models.

The project focuses on electricity-demand forecasting across different datasets, forecast horizons, and input configurations, with particular attention to the role of calendar-based exogenous variables.

## Project Overview

Electricity load forecasting is an important task in modern power and energy systems. Reliable demand forecasts can support grid planning, demand management, renewable integration, capacity assessment, and data-driven decision-making.

This project compares MLP-based neural forecasting architectures for electricity-demand forecasting and evaluates how model architecture, forecast horizon, dataset characteristics, and exogenous variables affect forecasting performance.

## Models

This repository focuses on NeuralForecast models:

* N-HiTS
* N-BEATS
* N-BEATSx

A separate repository is planned for Temporal Fusion Transformer (TFT)-based electricity load forecasting.

## Main Focus

* Multi-horizon electricity load forecasting
* Time-series forecasting for energy systems
* Deep learning for electricity-demand modeling
* Calendar-based exogenous variables
* Forecast horizon analysis
* Model comparison across datasets
* Fixed and rolling-origin evaluation
* Multi-seed experimental runs
* Metrics reporting and computational cost analysis

## Datasets

The study uses large-scale electricity-demand datasets, including:

* ECL electricity consumption dataset
* PJM hourly electricity load dataset

## Pipeline Features

* Configurable forecast horizons
* Configurable input window lengths
* Experiments with and without exogenous variables
* Fixed and rolling-origin evaluation modes
* Multi-seed execution
* Export of raw and normalized forecasting metrics
* Timing and computational cost summaries
* Diagnostic visualization and reporting

## Repository Structure

```text
large-scale-electricity-load-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── train_nf_models.py
│   ├── run_nf_pipeline.py
│   ├── metrics_export.py
│   ├── cost_export.py
│   ├── reporting_nf.py
│   └── data_visualisation.py
├── figures/
└── docs/
```

## Code Availability

This repository includes selected implementation code for research and portfolio demonstration. Some experimental outputs, large datasets, checkpoints, and unpublished result files may be excluded to keep the repository clean and reproducible.

The code will continue to be updated as the related research work is refined and prepared for publication.

## Status

Research project based on my MSc thesis in Machine Learning, Systems and Control at Lund University.
