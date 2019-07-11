import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv('fut_rd_stats/akamai1TrafficID458_5000.csv')
time_arr = df['0'].to_numpy()
med_arr = df['1'].to_numpy()
q3_arr = df['2'].to_numpy()

length = len(time_arr)

red_factor = 25
time_arr = [time_arr[i] for i in range(length) if i % red_factor == 0]
med_arr = [med_arr[i] for i in range(length) if i % red_factor == 0]
q3_arr = [q3_arr[i] for i in range(length) if i % red_factor == 0]

length = len(time_arr)

""" acf_med = plot_acf(med_arr, lags=int((length)**(0.8)))
pacf_med = plot_pacf(med_arr, lags=int((length)**(0.4)))
plt.show() """

factor = 0.9

med_train = med_arr[:int(factor*length)]
med_eval = med_arr[int(factor*length):]

q3_train = q3_arr[:int(factor*length)]
q3_eval = q3_arr[int(factor*length):]

""" res_med = seasonal_decompose(med_train, freq=2500)
res_med.plot()
res_q3 = seasonal_decompose(q3_train, freq=2500)
res_q3.plot()
plt.show() """

sarimax_med = SARIMAX(med_train, order=(1, 0, 13), seasonal_order=(1, 0, 1, 93))
#sarimax_q3 = SARIMAX(q3_train, order=(5, 1, 5), seasonal_order=(1, 1, 0, 8))

fit_med = sarimax_med.fit(disp=0)
#fit_q3 = sarimax_q3.fit(disp=0)

pred_med = fit_med.predict(start=1, end=length-1)
#pred_q3 = fit_q3.predict(start=1, end=length-1)

time_arr = time_arr[1:]
med_arr = med_arr[1:]
q3_arr = q3_arr[1:]

plt.figure(0)
plt.scatter(time_arr, med_arr, c='g', s=2.5, edgecolors='none')
plt.scatter(time_arr, pred_med, c='b', s=2.5, edgecolors='none')

#plt.figure(1)
#plt.scatter(time_arr, q3_arr, c='g', s=0.5)
#plt.scatter(time_arr, pred_q3, c='b', s=0.5)

#plt.show()
plt.savefig('fut_rd_stats/akamai1TrafficID458_5000_' + str(red_factor) + '_' + str(factor) + '.png')