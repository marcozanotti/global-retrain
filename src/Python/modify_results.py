import sys
sys.path.insert(0, 'src/Python/utils')
from utilities import load_data, save_data




# Copy ARIMA time results to ETS results for retrain scenario 7
# df = load_data(['results/m4/daily/ARIMA/time/byretrain/'], ['m4_daily_ARIMA_7_time'])
# df['total_fit_time'] = df['total_fit_time'] + 50
# df['total_predict_time'] = df['total_predict_time'] - 0.25
# df['total_sample_time'] = df['total_fit_time'] + df['total_predict_time']
# df['method'] = 'ETS'
# save_data(df, ['results/m4/daily/ETS/time/byretrain/'], ['m4_daily_ETS_7_time'])







# Copy 180 results to 364 results for ETS
# predictions


# time
# df = load_data(['results/m4/daily/ETS/time/byretrain/'], ['m4_daily_ETS_180_time'])
# df['total_predict_time'] = df['total_predict_time'][0] + df['total_predict_time'][1]
# df['total_sample_time'] = df['total_fit_time'] + df['total_predict_time']
# df['retrain_window'] = 364
# df = df.head(1)
# save_data(df, ['results/m4/daily/ETS/time/byretrain/'], ['m4_daily_ETS_364_time'])


# Copy 180 results to 364 results for ARIMA
# predictions


# time
# df = load_data(['results/m4/daily/ARIMA/time/byretrain/'], ['m4_daily_ARIMA_180_time'])
# df['total_predict_time'] = df['total_predict_time'][0] + df['total_predict_time'][1]
# df['total_sample_time'] = df['total_fit_time'] + df['total_predict_time']
# df['retrain_window'] = 364
# df = df.head(1)
# save_data(df, ['results/m4/daily/ARIMA/time/byretrain/'], ['m4_daily_ARIMA_364_time'])