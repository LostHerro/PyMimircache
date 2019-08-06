import numpy as np
import pandas as pd
import sys
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from shap import summary_plot

file_name = sys.argv[1]

df = pd.read_csv('eval/shap/' + file_name + '_shap_results.csv', index_col=0)
df = df[:50]

col_names = df.columns
shap_values = df.to_numpy().astype('float64')

summary_plot(shap_values, feature_names=col_names)
plt.show()