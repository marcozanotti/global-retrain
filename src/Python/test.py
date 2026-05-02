
# Assuming 'fit_tmp' is your trained XGBoost model object
# and 'X_train' is the DataFrame used for training
import pandas as pd

fit_tmp
train_df_tmp.columns


# importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
fit_tmp.models_['XGBRegressor'].feature_importances_
fit_tmp.models_['XGBRegressor'].get_booster().get_score(importance_type='gain')

f_importance = fit_tmp.models_['XGBRegressor'].get_booster().get_score(importance_type='gain')



f_importance_original = pd.DataFrame(data = fit_tmp.models_['XGBRegressor'].feature_importances_, index = f_importance.keys())

importance_df = pd.DataFrame.from_dict(data=f_importance, orient='index')

f_importance_original.plot.bar()
importance_df.plot.bar()



# extract feature importance from xgbregressor

