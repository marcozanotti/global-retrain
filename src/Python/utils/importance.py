import sys
sys.path.insert(0, 'src/Python/utils')
import pandas as pd
from utilities import create_file_path
from mlforecast import MLForecast
import plotly.express as px
from utilities import save_data

import logging
module_logger = logging.getLogger('importance')

def compute_feature_importance(config = None):
    
    """ Compute feature importance for the specified models and retrain scenarios.
    Args:
        config (dict): Configuration dictionary containing all necessary parameters.
    Returns:
        pd.DataFrame: DataFrame containing feature importance for each model and retrain scenario.
    """

    module_logger.info('===============================================================')

    # dataset parameters
    dataset_name = config['dataset']['dataset_name']
    frequency = config['dataset']['frequency']
    has_xregs_mapping = config['dataset']['has_xregs_mapping']
    ext = config['dataset']['ext']
    # fitting parameters
    retrain_window = config['fitting']['retrain_scenarios']
    # model parameters
    model_names = config['model_names']
    save_results = True

    for rs in retrain_window:

        module_logger.info('---------------------------- START ----------------------------')
        rows = []

        for m in model_names:

            module_logger.info(f"Processing model: {m}, retrain scenario: {rs}")

            path_tmp = create_file_path(["results", dataset_name, frequency, m, str(rs), "models"])
            fit_tmp = MLForecast.load(path_tmp).models_[m]

            module_logger.info("Computing feature importance...")

            if m in ['LinearRegression', 'Lasso']:
                f_imp = dict(zip(fit_tmp.feature_names_in_, fit_tmp.coef_)) 
                f_imp = {k: abs(v) for k, v in f_imp.items()}
            elif m in ['RandomForestRegressor']:
                f_imp = dict(zip(fit_tmp.feature_names_in_, fit_tmp.feature_importances_))
            elif m in ['XGBRegressor']:            
                f_imp = fit_tmp.get_booster().get_score(importance_type='weight')
            elif m in ['LGBMRegressor']:
                f_imp = dict(zip(fit_tmp.feature_name_, fit_tmp.feature_importances_))
            else:
                module_logger.warning(f"Feature importance not implemented for model: {m}")
                continue
            
            f_imp_df = pd.DataFrame.from_dict(f_imp, orient="index", columns=["importance"]).reset_index()
            f_imp_df = f_imp_df.rename(columns={"index": "xregs"})
            f_imp_df = f_imp_df.assign(method = m, retrain_window = rs)
            # add column relative importance
            f_imp_df["relative_importance"] = f_imp_df["importance"] / f_imp_df["importance"].max()
            rows.append(f_imp_df)

        res_df = pd.concat(rows, ignore_index=True)

        module_logger.info('---------------------------- END ----------------------------')

    if has_xregs_mapping:
        xregs_map = pd.read_csv(f"data/{dataset_name}/{dataset_name}_xregs_mapping.csv")
        res_df = res_df.merge(xregs_map, left_on="xregs", right_on="xregs", how="left")
        res_df["feature"] = res_df.apply(
            lambda row: row["Name"] if pd.notnull(row["Name"]) else row["xregs"], axis=1
        )
        res_df = res_df[["feature", "xregs", "method", "retrain_window", "importance", "relative_importance"]]
    else:
        res_df["feature"] = res_df["xregs"]
        res_df = res_df[["feature", "xregs", "method", "retrain_window", "importance", "relative_importance"]]

    if save_results:
        save_data(res_df, ["results", dataset_name, frequency, "importance"], ["imp", f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"])

    module_logger.info('===============================================================')

    return res_df

def plot_feature_importance(f_imp_df, retrain_scenario: int, top_n: int = 50):

    """ Plot feature importance for the specified retrain scenario.
    Args:
        f_imp_df: DataFrame containing feature importance for each model and retrain scenario.
        retrain_scenario (int): Retrain scenario to plot.
        top_n (int, optional): Number of top features to display. Defaults to 10.
    Returns:
        plotly.graph_objects.Figure: Plotly figure object containing the feature importance plot.
    """ 

    f_imp_df_rs = f_imp_df[f_imp_df["retrain_scenario"] == retrain_scenario]
    plot_df = f_imp_df_rs.groupby('method').tail(top_n)

    fig = px.bar(
        plot_df,
        x="importance",
        y="feature",
        orientation="h",
        facet_col="model",
        facet_col_wrap=3,
        title=f"Retrain Scenario: {retrain_scenario}",
        labels={"importance": "", "feature": ""},
        height=900
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)

    return fig
