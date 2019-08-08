import numpy as np
import pandas as pd
import sys
import matplotlib
if len(sys.argv) > 2 and sys.argv[2] == 'save':
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from shap import summary_plot

file_name = sys.argv[1]

df = pd.read_csv('eval/shap/' + file_name + '_shap_results.csv', index_col=0)
df = df[:250]

col_names = df.columns
shap_values = df.to_numpy().astype('float64')

summary_plot(shap_values, feature_names=col_names)
plt.xscale('symlog')

if len(sys.argv) > 2 and sys.argv[2] == 'save':
    plt.savefig('eval/shap/' + file_name + '_img.png')
else:
    plt.show()