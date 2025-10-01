"""
================================================================================
SARLIB: Statistical Analysis and Regression Library
================================================================================

sarlib - Statistical Analysis and Regression Library

This library provides tools for statistical analysis, regression modeling, 
sample size analysis, and visualization. It includes OLS and SAR models, as 
well as utilities for data preprocessing and plotting.

--------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------

sarlib is a standalone Python module. To use it, ensure you have the following 
dependencies installed:

- numpy
- matplotlib
- statsmodels
- scikit-learn
- scipy

You can install these with pip:

    pip install numpy matplotlib statsmodels scikit-learn scipy

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------

Import the module in your Python script:

    import sarlib

Or copy the code into your project and import the classes/functions as needed.

--------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------
Example Workflow
--------------------------------------------------------------------------------

1. Prepare your data as numpy arrays:

    import numpy as np
    x = np.random.randn(100, 3)  # predictors
    y = np.random.randn(100)     # response

2. Visualize data:

    show_scatter(x, y)

3. Fit OLS model:

    model_ols = OLS(n_realiz=100, alpha=0.05)
    stats_ols = model_ols.fit(x, y, verbose=True)

4. Fit SAR model:

    model_sar = SAR(n_realiz=100, norm='epsins', alpha=0.05)
    stats_sar = model_sar.fit(x, y, verbose=True)

5. Analyze sample size effect:

    analysis = SampleSizeAnalysis(model_sar, x, y, steps=7)
    analysis.plot_loss()
    analysis.plot_pvalue()
    analysis.plot_coef()

--------------------------------------------------------------------------------
Function/Class Documentation
--------------------------------------------------------------------------------

All functions and classes are documented with docstrings. Please refer to the 
code for parameter details and usage.

--------------------------------------------------------------------------------
License & Author
--------------------------------------------------------------------------------

Author: Sipba Group, UGR, https://sipba.ugr.es/
License: []

================================================================================

"""

# Import required libraries
import sys                                 # System parameters and functions
import numpy as np                         # Numerical operations
import matplotlib.pyplot as plt            # Plotting library
import statsmodels.api as sm               # Statistical models
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.svm import SVR                # Support Vector Regression
from sklearn.model_selection import KFold  # Cross-validation
from scipy.stats import iqr                # Interquartile range
from scipy.special import comb             # Combinatorial functions

# Set default parameters for matplotlib for consistent plotting style
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.linewidth'] = 0.25
plt.rcParams['axes.xmargin'] = 0.01
plt.rcParams['axes.ymargin'] = 0.01
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['figure.constrained_layout.use'] = True

def fix_data(x, y):
    """
    Standardize predictors and response, and remove rows with NaN values.

    This function concatenates the response and predictors, standardizes them 
    to zero mean and unit variance, and removes any rows containing NaN values.
    It ensures that the input data is clean and ready for modeling.

    Parameters:
        x (np.ndarray): Predictor variables, shape (n_samples, n_features).
        y (np.ndarray): Response variable, shape (n_samples,).

    Returns:
        tuple: (x, y) standardized and cleaned. Returns None if input shapes do
        not match.
    """

    # If the number of samples does not match, return
    if x.shape[0] != len(y): return
    
    data = np.c_[y,x]                 # Concatenate response and predictors
    scaler = StandardScaler()         # Initialize scaler
    data = scaler.fit_transform(data) # Standardize data

    # Remove rows with NaN values in any column
    for j in range(data.shape[1]):
        data = data[~np.isnan(data[:, j]).ravel()]

    y = data[:,0]    # Extract standardized response
    x = data[:,1:]   # Extract standardized predictors
    return x, y


def show_scatter(x, y, x_name=None, y_name=None, block=False):
    """
    Display scatter plots of each feature in x against the target variable y.

    This function creates a grid of scatter plots, one for each predictor
    variable against the response. Useful for visualizing relationships and
    detecting outliers or patterns.

    Parameters:
        x (np.ndarray): Predictor variables, shape (n_samples, n_features).
        y (np.ndarray): Response variable, shape (n_samples,).
        x_name (list, optional): Names of predictor variables.
        y_name (str, optional): Name of response variable.
        block (bool, optional): Whether to block execution until plot is closed.
    """

    fig = plt.figure('Scatter plots')           # Create figure
    ylabel = 'Y' if y_name is None else y_name  # Set ylabel
    x = x.reshape(-1,1) if x.ndim == 1 else x   # Ensure x is 2D

    # Plot each predictor against y
    for j in range(x.shape[1]):
        n_cols = int(np.ceil(np.sqrt(x.shape[1])))  # Number of columns
        n_rows = int(np.ceil(x.shape[1] / n_cols))  # Number of rows
        subplot = fig.add_subplot(n_rows, n_cols, j+1)
        subplot.scatter(x[:, j], y, s=0.3, c='g')
        subplot.grid(True, linewidth=0.1)
        xlabel = f'X_{j+1}' if x_name is None else x_name[j]
        subplot.set(xlabel=xlabel, ylabel=ylabel)
    
    plt.show(block=block)  # Show plot


class OLS:
    """
    Ordinary Least Squares regression with permutation-based significance and
    power analysis.

    This class implements OLS regression and uses Monte Carlo permutations 
    to estimate p-value, power, R^2, and regression coefficients. Useful for
    statistical inference and model evaluation.
    """

    def __init__(self, n_realiz, alpha=0.05):
        """
        Initialize OLS model.

        Parameters:
            n_realiz (int): Number of Monte Carlo realizations for permutation
            testing.
            alpha (float): Significance level for hypothesis testing.
        """

        self.n_realiz = n_realiz  # Number of realizations
        self.alpha = alpha        # Significance level

        # Validate n_realiz
        if ((not isinstance(self.n_realiz, int)) 
        or (isinstance(self.n_realiz, int) and not (1 <= self.n_realiz))): 
            print('Wrong parameter: ', self.n_realiz, file=sys.stderr)
            print('Default value (100) was used', file=sys.stderr)
            self.n_realiz = 100

        # Validate alpha
        if ((not isinstance(self.alpha, float)) 
        or (isinstance(self.alpha, float) and not (0 < self.alpha < 1))):
            print('Wrong parameter: ', self.alpha, file=sys.stderr)
            print('Default value (0.05) was used', file=sys.stderr)
            self.alpha = 0.05  

    def fit(self, x, y, n=None, seed=None, verbose=True):
        """
        Fit OLS model and compute statistics over multiple realizations.

        This method performs OLS regression on random subsamples of the data,
        computes p-value, R^2, power, and regression coefficients, and averages
        the results over multiple realizations.

        Parameters:
            x (np.ndarray): Predictor variables, shape (n_samples, n_features).
            y (np.ndarray): Response variable, shape (n_samples,).
            n (int, optional): Sample size per realization. If None, use all samples.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool, optional): Print summary table.

        Returns:
            dict: Statistics including p-value, power, R^2, and coefficients.
        """

        # Validate n
        if ((not isinstance(n, (int, np.integer)) and n is not None) 
        or ((isinstance(n, (int, np.integer)) and n is not None) and n < 3)):
            print('Wrong parameter: ', n, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)
            n = None

        # Validate seed
        if ((not isinstance(seed, (int, np.integer)) and seed is not None) 
        or ((isinstance(seed, (int, np.integer)) and seed is not None) 
        and not (0 <= seed <= 2**32 - 1))):
            print('Wrong parameter: ', seed, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)
            seed = None            

        # Validate verbose
        if (not isinstance(verbose, bool)):
            print('Wrong parameter: ', verbose, file=sys.stderr)
            print('Default value (True) was used', file=sys.stderr)
            verbose = True
            
        x, y = fix_data(x, y)          # Clean and standardize data
        if n is None: n = x.shape[0]   # Use all samples if n not specified
        if seed is not None: np.random.seed(seed)   # Set random seed

        pvalue, r2, regress, beta = [], [], [], []  # Initialize lists
        for r in range(self.n_realiz):
            # Shuffle data for permutation
            ids = np.random.choice(x.shape[0], n, replace=False)
            x_const = sm.add_constant(x[ids])     # Add intercept
            model = sm.OLS(y[ids], x_const).fit() # Fit OLS model
            pvalue.append(model.f_pvalue)         # F-test p-value
            r2.append(model.rsquared)             # R^2 statistic
            regress.append(model.f_pvalue < self.alpha)  # Significant regression
            beta.append(model.params)             # Regression coefficients
        
        # Aggregate statistics
        pvalue = np.mean(pvalue)
        power = np.mean(regress)
        r2 = np.mean(r2)
        beta = np.mean(beta, axis=0)

        # Print summary table if verbose
        if verbose:
            table = [['Sample size:',       x.shape[0]],
                     ['Significance level', self.alpha],
                     ['P-value',            pvalue],
                     ['R^2',                r2],
                     ['Power',              power]
                     ]
            
            print("-" * 41)

            for h, v in table:
                v = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"| {h:<20}: {v:<15} |")
            
            print("-" * 41)
        
        return {
            'pvalue': pvalue,
            'power':  power,
            'r2':     r2,
            'beta':   beta
        }


class SAR:
    """
    Statistical Analysis Regression (SAR) model with support for PAC-Bayes, 
    Vapnik, and IGP bounds.

    This class implements a regression model using Support Vector Regression 
    (SVR) and provides statistical bounds for generalization error using 
    PAC-Bayes, Vapnik, and IGP methods. It supports different validation modes
    and loss functions.
    """

    def __init__(self, n_realiz, norm='epsins', eps_0=None, alpha=0.05, 
                 mode='resusb', bound='pacbayes', eta=0.5, dropout_rate=0.5):
        """
        Initialize SAR model.

        Parameters:
            n_realiz (int): Number of Monte Carlo realizations.
            norm (str): Norm of loss function ('epsins' for epsilon-insensitive,
                       'rmse' for root mean squared error).
            eps_0 (float, optional): Threshold for loss. If None, uses SVR epsilon.
            alpha (float): Significance level for hypothesis testing.
            mode (str): Validation mode ('resusb' for resubstitution, 'kfold'
                        for k-fold CV, 'leaveoo' for leave-one-out CV).
            bound (str): Bound type ('pacbayes', 'vapnik', 'igp', 'igp_approx').
            eta (float): Confidence parameter for bounds.
            dropout_rate (float): Dropout rate for PAC-Bayes bound.
        """

        self.n_realiz = n_realiz           # Number of realizations
        self.norm = norm                   # Loss function norm
        self.eps_0 = eps_0                 # Loss threshold
        self.alpha = alpha                 # Significance level
        self.mode = mode                   # Validation mode
        self.bound = bound                 # Bound type
        self.eta = eta                     # Confidence parameter
        self.dropout_rate = dropout_rate   # Dropout rate for PAC-Bayes
        self.sample_stats = []             # Store sample statistics

        # Validate parameters
        if ((not isinstance(self.n_realiz, int)) 
        or (isinstance(self.n_realiz, int) and not (1 <= self.n_realiz))):
            print('Wrong parameter: ', self.n_realiz, file=sys.stderr)
            print('Default value (100) was used', file=sys.stderr)
            self.n_realiz = 100

        if self.norm not in ['epsins', 'rmse']:
            print('Wrong parameter: ', self.norm, file=sys.stderr)
            print('Default value (epsins) was used', file=sys.stderr)
            self.norm = 'epsins'

        if ((not isinstance(self.eps_0, float) and self.eps_0 is not None)
        or ((isinstance(self.eps_0, float) and self.eps_0 is not None)
        and self.eps_0 < 0)):
            print('Wrong parameter: ', self.eps_0, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)
            self.eps_0 = None

        if ((not isinstance(self.alpha, float)) 
        or (isinstance(self.alpha, float) and not (0 < self.alpha < 1))):
            print('Wrong parameter: ', self.alpha, file=sys.stderr)
            print('Default value (0.05) was used', file=sys.stderr)
            self.alpha = 0.05     

        if self.mode not in ['resusb', 'kfold', 'leaveoo']:
            print('Wrong parameter: ', self.mode, file=sys.stderr)
            print('Default value (resusb) was used', file=sys.stderr)
            self.mode = 'resusb'

        if self.bound not in ['pacbayes', 'vapnik', 'igp', 'igp_approx']:
            print('Wrong parameter: ', self.bound, file=sys.stderr)
            print('Default value (pacbayes) was used', file=sys.stderr)
            self.mode = 'pacbayes'

        if ((not isinstance(self.eta, float)) 
        or (isinstance(self.eta, float) and not (0 < self.eta < 1))): 
            print('Wrong parameter: ', self.eta, file=sys.stderr)
            print('Default value (0.5) was used', file=sys.stderr)
            self.eta = 0.5    

        if ((not isinstance(self.dropout_rate, float)) 
        or (isinstance(self.dropout_rate, float) 
        and not (0 <= self.dropout_rate <= 1))):           
            print('Wrong parameter: ', self.dropout_rate, file=sys.stderr)
            print('Default value (0.5) was used', file=sys.stderr)
            self.dropout_rate = 0.5 


    def fit(self, x, y, n=None, seed=None, verbose=True):
        """
        Fit SAR model and compute statistics over multiple realizations.

        This method fits the SAR model using SVR, computes empirical loss, 
        threshold, power, and generalization bounds over multiple random 
        subsamples, and averages the results.

        Parameters:
            x (np.ndarray): Predictor variables, shape (n_samples, n_features).
            y (np.ndarray): Response variable, shape (n_samples,).
            n (int, optional): Sample size per realization. If None, use all 
                               samples.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool, optional): Print summary table.

        Returns:
            dict: Statistics including loss, threshold, power, and bounds.
        """

        # Validate data types
        if x.dtype != int and x.dtype != float:
            print("Predictors data type must be numeric (int or float).", 
                  file=sys.stderr)
            return
        
        if y.dtype != int and y.dtype != float:
            print("Response data type must be numeric (int or float).", 
                  file=sys.stderr)
            return
        
        # Validate n
        if ((not isinstance(n, (int, np.integer)) and n is not None) 
        or ((isinstance(n, (int, np.integer)) and n is not None) and n < 3)):
            print('Wrong parameter: ', n, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)        
            n = None

        # Validate seed
        if ((not isinstance(seed, (int, np.integer)) and seed is not None) 
        or ((isinstance(seed, (int, np.integer)) and seed is not None) 
        and not (0 <= seed <= 2**32 - 1))):          
            print('Wrong parameter: ', seed, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)
            seed = None             

        # Validate verbose
        if (not isinstance(verbose, bool)):
            print('Wrong parameter: ', verbose, file=sys.stderr)
            print('Default value (True) was used', file=sys.stderr)
            verbose = True

        x, y = fix_data(x, y)                     # Clean and standardize data
        if n is None: n = x.shape[0]              # Use all samples if n not specified
        if seed is not None: np.random.seed(seed) # Set random seed

        # Initialize lists to store results for each permutation
        stats_p = []
        for r in range(self.n_realiz):
            # Shuffle data for permutation
            ids = np.random.choice(x.shape[0], n, replace=False)

            # Run SAR validation for this permutation
            stats_p.append(self._cubv_(x[ids], y[ids]))

        # Compute mean statistics across permutations
        stats = {}
        for key in stats_p[0].keys():
            stats[key] = np.mean([d[key] for d in stats_p], axis=0)
        stats['stdloss'] = np.std([d['loss'] for d in stats_p])
        stats['stdthres'] = np.std([d['thres'] for d in stats_p])
        stats['stdvarloss'] = np.std([d['varloss'] for d in stats_p])

        # Print summary table if verbose
        if verbose:
            table = [['Sample size:',       x.shape[0]],
                     ['Loss',               stats['loss']],
                     ['Threshold',          stats['thres']],
                     ['Power',              stats['power']],
                     ]

            print("-" * 41)

            for h, v in table:
                v = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"| {h:<20}: {v:<15} |")

            print("-" * 41)
        
        return stats


    def _cubv_(self, x, y):
        """
        Internal method to compute statistics using cross-validation or
        resubstitution.

        Depending on the selected mode, this method performs k-fold or 
        leave-one-out cross-validation, or resubstitution (train and test on 
        the same data), and computes statistics for each fold.

        Parameters:
            x (np.ndarray): Predictor variables, shape (n_samples, n_features).
            y (np.ndarray): Response variable, shape (n_samples,).

        Returns:
            dict: Computed statistics for the given mode.
        """

        if x is None or y is None: return
        stats = None

        # Cross-validation modes: k-fold or leave-one-out
        if (self.mode == 'kfold') or (self.mode == 'leaveoo'):
            k = 10 if  self.mode == 'kfold' else len(y)

            stats_f = []
            for trn, tst in KFold(n_splits=k).split(x):
                stats_f += [self._compute_stats_(x[trn], y[trn], x[tst], y[tst])]

            # Compute mean statistics across folds
            stats = {}
            for key in stats_f[0].keys():
                stats[key] = np.mean([d[key] for d in stats_f], axis=0)
            
            stats['varloss'] = np.var([d['emp_loss'] for d in stats_f])

        # Resubstitution mode (train and test on the same data)
        elif self.mode == 'resusb':
            stats = self._compute_stats_(x, y, x, y) 
        
        return stats


    def _compute_stats_(self, x_train, y_train, x_test, y_test):
        """
        Internal method to fit SVR and compute loss, threshold, and bounds.

        This method fits a linear SVR model, computes the empirical loss and 
        threshold according to the selected norm, and calculates the 
        generalization bound.

        Parameters:
            x_train (np.ndarray): Training predictors, shape (n_train, n_features).
            y_train (np.ndarray): Training response, shape (n_train,).
            x_test (np.ndarray): Test predictors, shape (n_test, n_features).
            y_test (np.ndarray): Test response, shape (n_test,).

        Returns:
            dict: Computed statistics for the fold or resubstitution.
        """

        scaler = StandardScaler()                # Initialize scaler
        x_train = scaler.fit_transform(x_train)  # Standardize training data
        x_test = scaler.transform(x_test)        # Standardize test data

        eps = iqr(y_train) / 13.49  # Set SVR epsilon using IQR
        svr_model =  SVR(kernel='linear', C=1, epsilon=eps).fit(x_train, y_train)

        loss, thres = None, None
        if self.norm == 'epsins':
            eps_0 = svr_model.epsilon if self.eps_0 is None else self.eps_0

            loss = np.maximum(0, (np.abs(y_test - svr_model.predict(x_test)) - eps))
            loss = loss[loss != 0]
            loss = 0 if len(loss) == 0 else np.mean(loss)

            thres = np.maximum(0, (np.abs(y_test) - eps_0))
            thres = thres[thres != 0]
            thres = 0 if len(thres) == 0 else np.mean(thres)

        elif self.norm == 'rmse':
            loss = np.sqrt(np.mean((y_test - svr_model.predict(x_test)) ** 2))
            thres = np.sqrt(np.mean(y_test**2))

        fcn = eval('self.' + self.bound + '_bound')  # Select bound function
        bound = fcn(x_train, y_train, svr_model, loss)  # Compute bound

        # Compute regression coefficients in original scale
        b = np.concatenate(
            (svr_model.intercept_ - (scaler.mean_ / scaler.scale_) @ 
            svr_model.coef_.transpose(),
            svr_model.coef_ / scaler.scale_.transpose()),
            axis=None
        )
        return {'beta': b.reshape(-1, 1),
                'power': 1 if (loss+bound) < thres else 0,
                'pvalue': 1 - ((loss+bound) < thres),
                'thres': thres,
                'bound': bound,
                'emp_loss': loss,
                'loss': loss + bound,
                'varloss': 0
               }

    def pacbayes_bound(self, x_train=None, y_train=None, svr_model=None, loss=None):
        """
        Compute PAC-Bayes generalization bound for a linear SVR model.

        This method calculates the PAC-Bayes bound for the generalization error
        using the SVR coefficients and dropout rate.

        Parameters:
            x_train (np.ndarray): Training predictors.
            y_train (np.ndarray): Training response.
            svr_model (SVR): Trained SVR model.
            loss (float): Empirical loss.

        Returns:
            float: Minimum PAC-Bayes bound.
        """

        # Compute PAC-Bayes bound over a range of lambda values and return the minimum
        lmax = np.max(np.maximum(0, np.abs(
            y_train - svr_model.predict(x_train)) - svr_model.epsilon))
        d, _ = x_train.shape
        lambda_val = np.arange(0.6, 10.1, 0.1)
        a = 1 / (1 - 1 / (2 * lambda_val))
        theta = np.concatenate((svr_model.coef_, svr_model.intercept_), axis=None)
        k = np.arange(1, len(a) + 1)
        bound = np.min(
            (a - 1) * loss +
            a * (lambda_val * lmax / d) *
            ((1 - self.dropout_rate) / 2 * (np.linalg.norm(theta) ** 2) + 
            np.log(k / self.eta))
        )
        return bound

    def vapnik_bound(self, x_train=None, y_train=None, svr_model=None, loss=None):
        """
        Compute Vapnik's upper bound for generalization error.

        This method calculates the Vapnik bound for the generalization error
        based on the VC dimension and sample size.

        Parameters:
            x_train (np.ndarray): Training predictors.
            y_train (np.ndarray): Training response.
            svr_model (SVR): Trained SVR model.
            loss (float): Empirical loss.

        Returns:
            float: Vapnik bound.
        """

        # Vapnik´s upper bound
        d, n = x_train.shape
        bound = np.sqrt(np.abs(((d+1)*(np.log(2*n/(d+1))+1)-np.log(self.eta/4))/n))
        return bound


    def igp_bound (self, x_train=None, y_train=None, svr_model=None, loss=None):
        """
        Compute IGP upper bound for generalization error.

        This method calculates the IGP bound for the generalization error
        using combinatorial calculations.

        Parameters:
            x_train (np.ndarray): Training predictors.
            y_train (np.ndarray): Training response.
            svr_model (SVR): Trained SVR model.
            loss (float): Empirical loss.

        Returns:
            float: IGP bound.
        """

        # Method: igp upper bound
        d, n = x_train.shape
        cld = 0
        for k in range(1, d+1):
            cld = cld+2*comb(n-1, k-1)
        
        bound = np.sqrt(np.log(cld/self.eta)/(2*n))
        return bound


    def igp_approx_bound(self, x_train=None, y_train=None, svr_model=None, loss=None):
        """
        Compute approximate IGP upper bound for generalization error.

        This method calculates an approximate IGP bound for the generalization
        error using nested combinatorial calculations.

        Parameters:
            x_train (np.ndarray): Training predictors.
            y_train (np.ndarray): Training response.
            svr_model (SVR): Trained SVR model.
            loss (float): Empirical loss.

        Returns:
            float: Approximate IGP bound.
        """

        # Method 3: igp upper bound
        d, n = x_train.shape
        cld = 0
        for z in range(d):
            for k in range(1, d-z+1):
                cld = cld+2*comb(n, z)*comb(n-1-z, k-1)
        
        bound = np.sqrt(np.log(cld/self.eta)/(2*n))
        return bound


class SampleSizeAnalysis:
    """
    Analyze the effect of sample size on model performance and statistics.

    This class runs the provided model on increasing sample sizes and collects
    statistics to study how sample size affects loss, threshold, power, p-value,
    and coefficients. It provides plotting utilities for visualization.
    """

    def __init__(self, model, x, y, steps=7, seed=None, verbose=True):
        """
        Initialize sample size analysis.

        Parameters:
            model (OLS or SAR): Model instance to analyze.
            x (np.ndarray): Predictor variables, shape (n_samples, n_features).
            y (np.ndarray): Response variable, shape (n_samples,).
            steps (int): Number of sample sizes to analyze.
            seed (int, optional): Random seed for reproducibility.
            verbose (bool, optional): Print progress.
        """

        self.model = model                          # Model instance
        self.n_realiz = model.n_realiz              # Number of realizations
        self.alpha = model.alpha                    # Significance level
        self.model_name = model.__class__.__name__  # Model name
        self.cv_model = hasattr(model, 'mode') and model.mode in ['kfold', 'leaveoo']  # Is cross-validation model

        # Validate model type
        if not isinstance(model, (OLS,SAR)):
            print('Wrong parameter: ', model, file=sys.stderr)
            print('Default value (SAR) was used', file=sys.stderr)
            model = SAR
        
        # Validate n_realiz
        if ((not isinstance(self.n_realiz, int)) 
        or (isinstance(self.n_realiz, int) and not (1 <= self.n_realiz))):
            print('Wrong parameter: ', self.n_realiz, file=sys.stderr)
            print('Default value (100) was used', file=sys.stderr)
            self.n_realiz = 100

        # Validate alpha
        if ((not isinstance(self.alpha, float)) 
        or (isinstance(self.alpha, float) and not (0 < self.alpha < 1))):
            print('Wrong parameter: ', self.alpha, file=sys.stderr)
            print('Default value (0.05) was used', file=sys.stderr)
            self.alpha = 0.05   
            
        # Validate steps
        if ((not isinstance(steps, int)) 
        or (isinstance(steps, int) and not (2 <= steps <= 20))): 
            print('Wrong parameter: ', steps, file=sys.stderr)
            print('Default value (7) was used', file=sys.stderr)
            steps = 7         

        # Validate seed
        if ((not isinstance(seed, (int, np.integer)) and seed is not None) 
        or ((isinstance(seed, (int, np.integer)) and seed is not None) 
        and not (0 <= seed <= 2**32 - 1))):          
            print('Wrong parameter: ', seed, file=sys.stderr)
            print('Default value (None) was used', file=sys.stderr)
            seed = None             

        # Validate verbose
        if (not isinstance(verbose, bool)):
            print('Wrong parameter: ', verbose, file=sys.stderr)
            print('Default value (True) was used', file=sys.stderr)
            verbose = True

        x, y = fix_data(x, y)                      # Clean and standardize data
        if seed is not None: np.random.seed(seed)  # Set random seed

        max_val = int(np.floor(x.shape[0] / 10) * 10)  # Maximum sample size
        log_values = np.logspace(np.log10(10), np.log10(max_val), num=steps)  # Log-spaced sample sizes
        self.n_sample = np.round(log_values / 10).astype(int) * 10  # Round to nearest 10

        self.sample_stats = []  # Store statistics for each sample size
        for n in self.n_sample:
            if verbose: print(f"Analyzing sample size: {n} ...")
            self.sample_stats += [model.fit(x, y, n, verbose=verbose)]


    def plot_loss(self, block=False):
        """
        Plot model loss and threshold as a function of sample size.

        This method visualizes how the empirical loss and threshold change as
        the sample size increases. It also plots the variance of the loss if
        the model uses cross-validation.

        Parameters:
            block (bool, optional): Whether to block execution until plot is
            closed.
        """

        required_fields = {'loss', 'thres', 'stdloss', 'stdthres'}

        if not required_fields.issubset(self.sample_stats[0]):
            print("Model not supported", file=sys.stderr)
            return

        colors = {
            'blue': [0, 0, 1],
            'lightblue': [0, 0, 1, 0.5],
            'green': [0, 1, 0],
            'lightgreen': [0, 1, 0, 0.5]
        }

        loss     = np.array([stats['loss']     for stats in self.sample_stats])
        stdloss  = np.array([stats['stdloss']  for stats in self.sample_stats])
        thres    = np.array([stats['thres']    for stats in self.sample_stats])
        stdthres = np.array([stats['stdthres'] for stats in self.sample_stats])

        fig = plt.figure("Model loss and threshold")
        subplot = fig.add_subplot(1+self.cv_model, 1, 1)
        subplot.plot(self.n_sample, loss, color=colors['blue'])
        subplot.fill_between(self.n_sample, loss+stdloss, loss-stdloss, 
                             color=colors['lightblue'], 
                             label='$\\mathbf{y}$ vs. $\\mathbf{X}$')

        subplot.plot(self.n_sample, thres, color=colors['green'])
        subplot.fill_between(self.n_sample, thres+stdthres, thres-stdthres, 
                             color=colors['lightgreen'], label='Threshold')
        subplot.legend()
        subplot.set_xticks(self.n_sample)
        subplot.set(xlabel='$N$', ylabel='$\\mathcal{R}$')
        subplot.set_title(f'Monte Carlo simulation ({self.n_realiz} trials) for '
                           '$\\mathcal{{R}}$')
        subplot.grid(True, linewidth=0.1)

        if self.cv_model:   # Plot variance of loss
            varloss    = np.array([stats['varloss']    for stats in self.sample_stats])
            stdvarloss = np.array([stats['stdvarloss'] for stats in self.sample_stats])

            subplot = fig.add_subplot(2, 1, 2)
            subplot.plot(self.n_sample, varloss, color=colors['blue'])
            subplot.fill_between(self.n_sample, varloss+stdvarloss, varloss-stdvarloss, 
                                color=colors['lightblue'], 
                                label='$\\mathbf{y}$ vs. $\\mathbf{X}$')
            subplot.legend()
            subplot.set_xticks(self.n_sample)
            subplot.set(xlabel='$N$', ylabel='$var(\\mathcal{R})$')
            subplot.grid(True, linewidth=0.1)  

        plt.show(block=block)

    def plot_pvalue(self, block=False):
        """
        Plot p-value, R^2, and power as a function of sample size.

        This method visualizes how the p-value, R^2, and statistical power
        change as the sample size increases.

        Parameters:
            block (bool, optional): Whether to block execution until plot is
            closed.
        """

        required_fields = {'pvalue', 'power'}
        if not required_fields.issubset(self.sample_stats[0]):
            print("Model not supported", file=sys.stderr)
            return

        pvalue = np.array([stats['pvalue'] for stats in self.sample_stats])
        power  = np.array([stats['power'] for stats in self.sample_stats])

        fig = plt.figure("Model significance")
        n_axes = 3 if 'r2' in self.sample_stats[0] else 2

        # Plot p-value
        subplot = fig.add_subplot(n_axes, 1, 1)
        subplot.plot(self.n_sample, pvalue, label='$\\mathbf{y}$ vs. $\\mathbf{X}$')
        subplot.plot(self.n_sample, np.full(len(self.n_sample), self.alpha), 
                     'k-.', label='$\\alpha$')
        subplot.set_xticks(self.n_sample)
        subplot.set_title('p-value')
        subplot.set_ylabel('p-value')
        subplot.grid(True, linewidth=0.1)
        subplot.legend()

        # Plot r^2
        if 'r2' in self.sample_stats[0]:
            r2 = np.array([stats['r2'] for stats in self.sample_stats])

            subplot = fig.add_subplot(n_axes, 1, 2)
            subplot.plot(self.n_sample, r2, c='g', label='$\\mathbf{y}$ vs. $\\mathbf{X}$')
            subplot.set_xticks(self.n_sample)
            subplot.set_ylabel('$R^{2}$')
            subplot.set_ylim(-0.01, 1.01)
            subplot.set_title('$R^{2}$')
            subplot.grid(True, linewidth=0.1)
            subplot.legend()

        # Plot power
        subplot = fig.add_subplot(n_axes, 1, n_axes)
        subplot.plot(self.n_sample, power, label='$\\mathbf{y}$ vs. $\\mathbf{X}$')
        subplot.set(xlabel='$N$', ylabel='$1-\\beta$')
        subplot.set_xticks(self.n_sample)
        subplot.set_ylim(-0.01, 1.01)
        subplot.set_title('Power')
        subplot.grid(True, linewidth=0.1)
        subplot.legend()

        plt.show(block=block)

    def plot_coef(self, block=False):
        """
        Plot model coefficients as a function of sample size.
        This method visualizes how the regression coefficients change as the 
        sample size increases.

        Parameters:
            block (bool, optional): Whether to block execution until plot is
            closed.
        """

        required_fields = {'beta'}

        if not required_fields.issubset(self.sample_stats[0]):
            print("Model not supported", file=sys.stderr)
            return

        beta = np.array([stats['beta'] for stats in self.sample_stats])

        fig = plt.figure("Model coefficients")
        fig.suptitle('$\\mathbf{\\beta}$ for $\\mathbf{y}$ vs. $\\mathbf{X}$ model')
        subplot = fig.add_subplot(1, 1, 1)
        for j in range(beta.shape[1]):
            subplot.plot(self.n_sample, beta[:, j], label=f'$\\beta_{j}$')
        subplot.grid(True, linewidth=0.1)
        subplot.set(xlabel='$N$', ylabel='$\\beta_{i}$')
        subplot.set_xticks(self.n_sample)
        subplot.legend()

        plt.show(block=block)

