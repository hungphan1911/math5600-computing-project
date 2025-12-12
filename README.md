# Regression–Prediction Project

## Overview
This repository contains our **Regression–Prediction computing project** for **MATH 5600 (Numerical Computing)**.  
The objective of this project is to explore and compare different **interpolation and regression methods** on real-world data and evaluate their predictive performance using standard error metrics.

## Dataset
- **File:** `4182195.csv`
- The dataset contains daily observations, including temperature measurements, measured at Salt Lake International Airport Station.
- For this project, we focus on:
  - `DATE`: date of measurements, act as a time index
  - `TMAX`: max temperature of the day, response variable

Missing values are filtered out during preprocessing.

## Methods Implemented
The following methods are implemented and evaluated:
- Polynomial interpolation
- Newton interpolation
- Cubic spline interpolation
- Linear regression

Each method is trained on a subset of the data and evaluated on a same test set.

## Evaluation Metrics
To compare model performance, we compute:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

These metrics are evaluated on the test set to assess predictive accuracy.

## Repository Structure
```
.
├── computing-project.jl   # Main Julia script (data processing, modeling, evaluation)
├── 4182195.csv             # Dataset
├── figures/                # Generated plots and result figures
├── README.md               # Project documentation
└── .gitignore
````

## Running the Project
### Requirements
- Julia
- Required packages:
  - CSV
  - DataFrames
  - Dates
  - Random
  - StatsBase
  - Polynomials
  - Statistics
  - LinearAlgebra
  - FundamentalsNumericalComputation
  - Plots

### Execution
From the project root directory, run:
```bash
julia computing-project.jl
````

This script will:

* Load and preprocess the data
* Train each interpolation/regression model
* Compute MAE and MSE
* Generate plots and save them to the `figures/` directory

## Output

The script automatically generates:

* Model fit visualizations
* Prediction comparison plots
* Error metric plots (MAE and MSE)

All output figures are saved under the `figures/` folder.

## Authors

* Khoa Minh Ngo
* Hung Phan Quoc Viet
* Markeya Gu

## Course Information

* **Course:** MATH 5600 – Numerical Computing
* **Project Type:** Computing Project (Regression–Prediction)