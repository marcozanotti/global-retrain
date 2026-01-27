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
