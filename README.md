SARlib: Statistical Analysis and Regression Library
================================================================================

This library provides tools for statistical analysis, regression modeling, 
sample size analysis, and visualization. It includes OLS and SAR models, as 
well as utilities for data preprocessing and plotting.


Installation
--------------------------------------------------------------------------------

SARlib can be installed via PyPI:

    pip install sarlib

Alternatively, you can install it manually by downloading the source code. In that
case, make sure you have the following dependencies installed:

- numpy
- matplotlib
- statsmodels
- scikit-learn
- scipy


Usage
--------------------------------------------------------------------------------

Import the module in your Python script:

   ```python
   import sarlib
   ```

Or copy the code into your project and import the classes/functions as needed.


Main Components
--------------------------------------------------------------------------------

- fix_data(x, y):          Standardizes and cleans input data.

- show_scatter(x, y, ...): Visualizes predictors vs. response.

- OLS:                     Ordinary Least Squares regression with
                           permutation-based significance and power analysis.

- SAR:                     Statistical Analysis Regression with PAC-Bayes, 
                           Vapnik, and IGP bounds.

- SampleSizeAnalysis:      Analyzes the effect of sample size on model 
                           performance and statistics.


Example Workflow
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


License & Author
--------------------------------------------------------------------------------

Author: Sipba Group, UGR, https://sipba.ugr.es/
License: GPL Version 3
