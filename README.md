SARlib: Statistical Agnostic Regression Library
================================================================================

This library provides tools for statistical analysis, regression modeling, 
sample size analysis, and visualization. It includes OLS and SAR models, as 
well as utilities for data preprocessing and plotting.

A formal description and analysis are included in the following reference:

J. M. Gorriz, J. Ramirez, F. Segovia, C. Jimenez-Mesa, F. J. Martinez-Murcia, 
y J. Suckling, _«Statistical agnostic regression: A machine learning method 
to validate regression models»_, Journal of Advanced Research, may 2025, 
doi: [10.1016/j.jare.2025.04.026](https://doi.org/10.1016/j.jare.2025.04.026).


Features
--------------------------------------------------------------------------------

- **Data Preprocessing**:
  Standardize and clean input data, handle missing values.
- **Visualization**:
  Scatter plots for exploratory data analysis.
- **OLS Regression**:
  Permutation-based p-value, power, R², and coefficient estimation.
- **SAR Regression**:
  Support Vector Regression with advanced generalization bounds.
- **Sample Size Analysis**:
  Empirical study of how sample size affects model statistics.
- **Plotting Utilities**:
  Visualize loss, threshold, power, p-value, and coefficients as functions of 
  sample size.

Installation
--------------------------------------------------------------------------------

SARlib can be installed via PyPI:

    pip install sarlib

Alternatively, you can install it manually by downloading the source code. In
that case, make sure you have the following dependencies installed:

- numpy
- matplotlib
- statsmodels
- scikit-learn
- scipy


Main Components
--------------------------------------------------------------------------------

Classes:

- `SAR`:                     Statistical Analysis Regression with PAC-Bayes, 
                             Vapnik, and IGP bounds.

- `OLS`:                     Ordinary Least Squares regression with
                             permutation-based significance and power analysis.

- `SampleSizeAnalysis`:      Analyzes the effect of sample size on model 
                             performance and statistics.

Functions:

- `fix_data(x, y)`:          Standardizes and cleans input data.

- `show_scatter(x, y, ...)`: Visualizes predictors vs. response.


Usage
--------------------------------------------------------------------------------

1. Import packages and prepare your data as numpy arrays:

    ```python
    from sarlib import SAR, OLS, SampleSizeAnalysis, show_scatter
    import numpy as np
    x = np.random.randn(100, 3)  # predictors
    y = np.random.randn(100)     # response
    ```

2. Visualize data:

    ```python
    show_scatter(x, y)
    ```

3. Fit SAR model:

    ```python
    model_sar = SAR(n_realiz=100, norm='epsins', alpha=0.05)
    stats_sar = model_sar.fit(x, y, verbose=True)
    ```

4. Compare with an OLS model:

    ```python
    model_ols = OLS(n_realiz=100, alpha=0.05)
    stats_ols = model_ols.fit(x, y, verbose=True)
    ```

5. Analyze sample size effect:

    ```python
    analysis = SampleSizeAnalysis(model_sar, x, y, steps=7)
    analysis.plot_loss()
    analysis.plot_pvalue()
    analysis.plot_coef()
    ```


Function/Class Documentation
--------------------------------------------------------------------------------

All functions and classes are documented with docstrings. Please refer to the 
code for parameter details and usage.


Author & License
--------------------------------------------------------------------------------

Author: Sipba Group, UGR, https://sipba.ugr.es/

Please cite:
J. M. Gorriz, J. Ramirez, F. Segovia, C. Jimenez-Mesa, F. J. Martinez-Murcia, y 
J. Suckling, _«Statistical agnostic regression: A machine learning method to 
validate regression models»_, Journal of Advanced Research, may 2025, 
doi: [10.1016/j.jare.2025.04.026](https://doi.org/10.1016/j.jare.2025.04.026).


License: GPL Version 3
