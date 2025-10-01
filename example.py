import numpy as np
import pandas as pd
from sarlib import SAR, OLS, SampleSizeAnalysis, show_scatter

file = 'example_data.csv'
data = pd.read_csv(file)

x = data['Variable6'].to_numpy()
y = data['Variable9'].to_numpy()

# Dataset evaluation with SAR
print('Dataset evaluation with SAR')
model_sar = SAR(n_realiz=100, eta=0.05, eps_0=None, mode='resusb')
results = model_sar.fit(x, y)

# Dataset evaluation with OLS
print('Dataset evaluation with OLS')
model_ols = OLS(n_realiz=100)
results = model_ols.fit(x, y)

# Sample size analysis with SAR
print('Sample size analysis with SAR')
model_sar = SAR(n_realiz=100, eta=0.05, eps_0=None, mode='resusb')
ssa = SampleSizeAnalysis(model_sar, x, y)
show_scatter(x,y,block=False)
ssa.plot_pvalue()
ssa.plot_loss()
ssa.plot_coef()

# Sample size analysis with OLS
print('Sample size analysis with OLS')
model_ols = OLS(n_realiz=100)
ssa = SampleSizeAnalysis(model_ols, x, y)
ssa.plot_pvalue()
ssa.plot_coef(block=True)

