#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix


# Check if Response is Categorical or Continuous
def Check_response_var(inp_df):
    if inp_df.target.nunique() > 2:
        return "Continuous"
    else:
        return "Boolean"


# Check if Predictor is Categorical
# Categorical have few unique values
def is_pred_categorical(pred):
    if pred.dtypes == "object" or pred.nunique() / pred.count() < 0.05:
        return True
    else:
        return False


# Generate Heatmap Categorical Predictor by Categorical Response
def generate_heatmap(df, col, filename):
    conf_matrix = confusion_matrix(df[col], df["target"])
    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response ",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()
    fig_no_relationship.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


# Generate Violin plot when Continuous Response by Categorical Predictor
def generate_violin_plot(df, col, filename):
    group_labels = pd.unique(df.target)
    fig_1 = go.Figure()
    M = df[df["target"] == group_labels[0]][col]
    B = df[df["target"] == group_labels[1]][col]
    for curr_hist, curr_group in zip([M, B], group_labels):
        fig_1.add_trace(
            go.Violin(
                x=np.repeat(curr_group, len(df)),
                y=curr_hist,
                name=int(curr_group),
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    # fig_1.show()
    fig_1.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


def generate_dist_plot(df, col, filename):
    group_labels = ["0", "1"]
    # Create distribution plot with custom bin_size
    M = df[df["target"] == 0][col]
    B = df[df["target"] == 1][col]
    fig_1 = ff.create_distplot([M, B], group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    # fig_1.show()
    fig_1.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


def generate_scatter_plot(df, col, filename):
    fig = px.scatter(x=df[col], y=df["target"], trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor ",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    # fig.show()
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


max_bin = 10


def calculate_mean_of_response(df, col, pop_prop_1, n=max_bin):
    print(col)
    d1 = pd.DataFrame({"X": df[col], "Y": df["target"], "Bucket": pd.qcut(df[col], 10)})
    d2 = d1.groupby("Bucket", as_index=True)
    d3 = pd.DataFrame({}, index=[])
    d3["LowerBin"] = d2.min().X
    d3["UpperBin"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    pop_prop = d3["COUNT"] / len(df)
    d3["BinMean"] = d2.sum().Y / d3.COUNT
    d3["Mean_sq_diff"] = (d3.BinMean - pop_prop_1) ** 2
    d3["Mean_sq_diffW"] = d3.Mean_sq_diff * pop_prop
    return d3["Mean_sq_diff"].sum(), d3["Mean_sq_diffW"].sum()


def calculate_mean_of_response_cat(df, col, pop_prop_1):
    # bins = len(np.unique(df[col].values))
    d1 = pd.DataFrame({"X": df[col], "Y": df["target"]}).groupby(df[col])
    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = d1.count().Y
    pop_prop = d3["COUNT"] / len(df)
    d3["BinMean"] = d1.sum().Y / d3.COUNT
    d3["Mean_Sq_Diff"] = (d3.BinMean - pop_prop_1) ** 2
    d3["Mean_Sq_DiffW"] = d3.Mean_Sq_Diff * pop_prop
    return d3["Mean_Sq_Diff"].sum(), d3["Mean_Sq_DiffW"].sum()

    print("calculate_mean_of_response_cat")


def read_breast_cancer_data():
    inp_data = pd.read_csv("./Data_Files/wisconsin-breast-cancer-dataset.csv")
    inp_data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    return inp_data


def main(fname, r_name):
    if fname == "":
        inp_data = read_breast_cancer_data()
    else:
        try:
            inp_data = pd.read_csv(fname)
        except FileNotFoundError:
            inp_data = read_breast_cancer_data()
    print(inp_data.head())
    if r_name == "":
        r_name = "diagnosis"
    print(r_name)
    inp_data = inp_data.rename(columns={r_name: "target"})
    inp_data = inp_data.dropna(axis=1, how="any")
    y = inp_data.target.values
    X = inp_data.drop("target", axis=1)

    if not os.path.exists("~/plots"):
        print("Creating plots")
        os.makedirs("~/plots")
    file_path = "~/plots/"

    col_names = [
        "Predictor",
        "Cat/Con",
        "Plot_Link",
        "t-value",
        "p-value",
        "m-plot",
        "RF-FI",
        "MeanSqDiff",
        "MeanSqDiffWeighted",
    ]
    output_df = pd.DataFrame(columns=col_names)

    res_type = Check_response_var(inp_data)
    print("Response variable is " + res_type)
    if res_type == "Boolean":
        inp_data["target"] = inp_data["target"].astype("category")
        inp_data["target"] = inp_data["target"].cat.codes
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, n_jobs=1)
        rf.fit(X, y)
        importance = rf.feature_importances_
    else:
        rf = RandomForestRegressor(n_estimators=50, oob_score=True, n_jobs=1)
        rf.fit(X, y)
        importance = rf.feature_importances_
    output_df["Predictor"] = X.columns

    out = []
    f_path = []
    p_val = []
    t_val = []
    m_plot = []
    msd_uw = []
    msd_w = []
    for col in X.columns:
        col_name = col.replace(" ", "-").replace("/", "-")
        if is_pred_categorical(X[col]) and res_type == "Boolean":
            out.append("Categorical")
            filename = (
                file_path + "cat_response_cat_predictor_heat_map_" + col_name + ".html"
            )
            generate_heatmap(inp_data, col, filename)
            f_path.append("")
        elif is_pred_categorical(X[col]) and res_type != "Boolean":
            out.append("Categorical")
            filename = (
                file_path
                + "cont_response_cat_predictor_violin_plot_"
                + col_name
                + ".html"
            )
            generate_violin_plot(inp_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")
        elif not (is_pred_categorical(X[col])) and res_type == "Boolean":
            out.append("Continuous")
            filename = (
                file_path
                + "cat_response_cont_predictor_dist_plot_"
                + col_name
                + ".html"
            )
            generate_dist_plot(inp_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")
        else:
            out.append("Continuous")
            filename = (
                file_path
                + "cont_response_cont_predictor_scatter_plot_"
                + col_name
                + ".html"
            )
            generate_scatter_plot(inp_data, col, filename)
            f_path.append("<a href=" + filename + ">" + filename + "</a>")

        # Calculating p-value and t-score
        if res_type == "Boolean":
            predictor = sm.add_constant(inp_data[col])
            logit = sm.Logit(inp_data["target"], predictor)
            logit_fitted = logit.fit()
            # Get the stats
            t_value = round(logit_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logit_fitted.pvalues[1])
            t_val.append(t_value)
            p_val.append(p_value)
            filename = "~/plots/ranking_" + col_name + ".html"
            m_plot.append("<a href=" + filename + ">" + filename + "</a>")
            # Plot the figure
            fig = px.scatter(x=inp_data[col], y=inp_data["target"], trendline="ols")
            fig.update_layout(
                title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {col}",
                yaxis_title="y",
            )
            # fig.show()
            fig.write_html(
                file=f"~/plots/ranking_{col_name}.html", include_plotlyjs="cdn"
            )
        else:
            predictor = sm.add_constant(inp_data[col])
            linear_regression_model = sm.OLS(inp_data["target"], predictor)
            linear_regression_model_fitted = linear_regression_model.fit()

            # Get the stats
            t_value = round(linear_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
            t_val.append(t_value)
            p_val.append(p_value)
            # Plot the figure
            fig = px.scatter(x=inp_data[col], y=inp_data["target"], trendline="ols")
            fig.update_layout(
                title=f"Variable: {col}: (t-value={t_value}) (p-value={p_value})",
                xaxis_title=f"Variable: {col}",
                yaxis_title="y",
            )
            # fig.show()
            fig.write_html(
                file=f"{file_path}/ranking_{col_name}.html", include_plotlyjs="cdn"
            )
            filename = "~/plots/ranking_" + col_name + ".html"
            m_plot.append("<a href=" + filename + ">" + filename + "</a>")
        # Mean Squared Difference
        if is_pred_categorical(X[col]):
            pop_prop_1 = inp_data.target.sum() / len(inp_data)
            uw, w = calculate_mean_of_response_cat(inp_data, col, pop_prop_1)
        else:
            pop_prop_1 = inp_data.target.sum() / len(inp_data)
            uw, w = calculate_mean_of_response(inp_data, col, pop_prop_1)
        msd_uw.append(uw)
        msd_w.append(w)

    output_df["Cat/Con"] = out
    output_df["Plot_Link"] = f_path
    output_df["t-value"] = t_val
    output_df["p-value"] = p_val
    output_df["m-plot"] = m_plot
    output_df["RF-FI"] = importance
    output_df["MeanSqDiff"] = msd_uw
    output_df["MeanSqDiffWeighted"] = msd_w

    print(output_df)
    output_df.to_html("Assignment_4.html", render_links=True, escape=False)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else ""
    print(fname)
    r_name = sys.argv[2] if len(sys.argv) > 1 else ""
    print(r_name)
    sys.exit(main(fname, r_name))
