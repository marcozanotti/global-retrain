import sys
sys.path.insert(0, 'src/Python/utils')
import pandas as pd
from utilities import create_file_path
from mlforecast import MLForecast
import plotly.express as px

import logging
module_logger = logging.getLogger('importance')

def compute_feature_importance(
    dataset_name: str,
    frequency: str,
    model_names: list[str],
    retrain_scenarios: list[int]
):
    
    """ Compute feature importance for the specified models and retrain scenarios.
    Args:
        dataset_name (str): Name of the dataset.
        frequency (str): Frequency of the data (e.g., 'weekly').
        model_names (list[str]): List of model names to compute feature importance for.
        retrain_scenarios (list[int]): List of retrain scenarios to consider.
        importance_type (str, optional): Type of importance to compute. Defaults to "gain".
    Returns:
        pd.DataFrame: DataFrame containing feature importance for each model and retrain scenario.
    """

    xregs_map = pd.read_csv(f"data/{dataset_name}/{dataset_name}_xregs_mapping.csv")
    xregs_map["Name"] = xregs_map["Name"].apply(lambda x: x[1:] if x.startswith("_") else x)

    for rs in retrain_scenarios:
        rows = []

        for m in model_names:

            module_logger.info(f"Processing model: {m}, retrain scenario: {rs}")
            path_tmp = create_file_path(["results", dataset_name, frequency, m, str(rs), "models"])
            fit_tmp = MLForecast.load(path_tmp).models_[m]

            module_logger.info("Computing feature importance...")

            if m in ['LinearRegression']:
                f_imp = dict(zip(fit_tmp.feature_names_in_, fit_tmp.coef_)) 
                f_imp = {k: abs(v) for k, v in f_imp.items()}
            elif m in ['XGBRegressor']:            
                f_imp = fit_tmp.get_booster().get_score(importance_type='weight')
            elif m in ['LGBMRegressor']:
                f_imp = dict(zip(fit_tmp.feature_name_, fit_tmp.feature_importances_))
            else:
                module_logger.warning(f"Feature importance not implemented for model: {m}")
                continue
            
            f_imp_df = (
                pd.DataFrame.from_dict(f_imp, orient="index", columns=["importance"])
                .reset_index()
                .merge(xregs_map, left_on="index", right_on="xregs", how="left")
            )
            f_imp_df["feature"] = f_imp_df.apply(
                lambda row: row["Name"] if pd.notnull(row["Name"]) else row["index"], axis=1
            )
            f_imp_df = (
                f_imp_df[["feature", "importance"]]
                .sort_values("importance", ascending=True)
                .assign(model=m, retrain_scenario=rs)
            )

            rows.append(f_imp_df)

        res_df = pd.concat(rows, ignore_index=True)

    return res_df

def plot_feature_importance(f_imp_df, retrain_scenario: int, top_n: int = 10):

    """ Plot feature importance for the specified retrain scenario.
    Args:
        f_imp_df: DataFrame containing feature importance for each model and retrain scenario.
        retrain_scenario (int): Retrain scenario to plot.
        top_n (int, optional): Number of top features to display. Defaults to 10.
    Returns:
        plotly.graph_objects.Figure: Plotly figure object containing the feature importance plot.
    """ 

    f_imp_df_rs = f_imp_df[f_imp_df["retrain_scenario"] == retrain_scenario]
    plot_df = f_imp_df_rs.groupby('model').tail(top_n)

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
