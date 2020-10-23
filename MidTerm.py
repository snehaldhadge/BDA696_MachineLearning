#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats.stats as stats
import seaborn as sns
from plotly import graph_objects as go
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes

import Assignment_4 as a4
import cat_correlation as cc


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


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


# Read the default data which is the auto_mpg
def read_auto_data():
    inp_data = pd.read_csv("./Data_Files/auto_mpg_dataset.csv")
    return inp_data


# Load Data based on user choice either from sklearn or
# user define excel or default auto_mpg
def load_data(fname, r_name):
    # Checking for input csv passed if none passed read the Breast Cancer CSV
    if fname == "1":
        if r_name == "1":
            data = load_boston()
            # res_name = "Species"
        elif r_name == "2":
            data = load_diabetes()
            # res_name = "Outcome"
        else:
            data = load_breast_cancer()
            # res_name = "Outcome"
        inp_data = pd.DataFrame(data.data, columns=data.feature_names)
        inp_data["target"] = pd.Series(data.target)
    else:
        if fname == "":
            inp_data = read_auto_data()
        else:
            try:
                inp_data = pd.read_csv(fname)
            except FileNotFoundError:
                inp_data = read_auto_data()

        # if No response variable passed use the Breast Cancer response variable
        if r_name == "":
            r_name = "mpg"
        # res_name = r_name

        # Rename the response variable to target column to make it generic
        inp_data = inp_data.rename(columns={r_name: "target"})

    # remove any Null values
    inp_data = inp_data.dropna(axis=0, how="any")
    inp_data.reset_index(drop=True, inplace=True)
    return inp_data


# Get List of all Cat-Cont columns
def get_cat_cont_columns(X, inp_data):
    cat_cols = []
    cont_cols = []
    for col in X.columns:
        col_name = col.replace(" ", "-").replace("/", "-")
        if is_pred_categorical(X[col]):
            inp_data[col] = inp_data[col].astype("category")
            inp_data[col] = inp_data[col].cat.codes
            cat_cols.append(col_name)
        else:
            cont_cols.append(col_name)
    return cat_cols, cont_cols


# Call function from Assignment 4 to generate Heatmap
def gen_heatmap(inp_data, col):
    filename = (
        "./Output-File/hw4_plots/cat_response_cat_predictor_heat_map_" + col + ".html"
    )
    a4.generate_heatmap(inp_data, col, filename)
    file_n = "./hw4_plots/cat_response_cat_predictor_heat_map_" + col + ".html"
    column1 = "<a href=" + file_n + ">" + col
    return column1


# Call function from Assignment 4 to generate Violin Plot
def gen_violin_plot(inp_data, col):
    filename = (
        "./Output-File/hw4_plots/cont_response_cat_predictor_violin_plot_"
        + col
        + ".html"
    )
    a4.generate_violin_plot(inp_data, col, filename)
    file_n = "./hw4_plots/cont_response_cat_predictor_violin_plot_" + col + ".html"
    column1 = "<a href=" + file_n + ">" + col
    return column1


# Call function from Assignment 4 to generate Ditsribution Plot
def gen_distri_plot(inp_data, col):
    filename = (
        "./Output-File/hw4_plots/cat_response_cont_predictor_dist_plot_" + col + ".html"
    )
    a4.generate_dist_plot(inp_data, col, filename)
    file_n = "./hw4_plots/cat_response_cont_predictor_dist_plot_" + col + ".html"
    column1 = "<a href=" + file_n + ">" + col
    return column1


# Call function from Assignment 4 to generate Scatter Plot
def gen_scatter_plot(inp_data, col):
    filename = (
        "./Output-File/hw4_plots/cont_response_cont_predictor_scatter_plot_"
        + col
        + ".html"
    )
    a4.generate_scatter_plot(inp_data, col, filename)
    file_n = "./hw4_plots/cont_response_cont_predictor_scatter_plot_" + col + ".html"
    column1 = "<a href=" + file_n + ">" + col
    return column1


# Get DataFrame for Cont-Cont MSD Calculation
def get_meansqdiff_cont_cont(col1, col2, inp_data):
    n = 5
    d1 = pd.DataFrame(
        {
            "X1": inp_data[col1],
            "X2": inp_data[col2],
            "Y": inp_data["target"],
            "Bucket1": pd.qcut(inp_data[col1], n),
            "Bucket2": pd.qcut(inp_data[col2], n),
        }
    )
    d2 = d1.groupby(["Bucket1", "Bucket2"]).agg({"Y": ["count", "mean"]}).reset_index()

    return d2


def get_meansqdiff_cat_cont(cat_col, cont_col, inp_data):
    n = 5
    d1 = pd.DataFrame(
        {
            "X1": inp_data[cat_col],
            "X2": inp_data[cont_col],
            "Y": inp_data["target"],
            "Bucket": pd.qcut(inp_data[cont_col], n),
        }
    )
    d2 = d1.groupby(["X1", "Bucket"]).agg({"Y": ["count", "mean"]}).reset_index()
    return d2


def get_w_uw_msd(col1, col2, inp_data, pop_prop_1, type):
    if type == 3:
        d1_c_c = pd.DataFrame(
            {
                "X1": inp_data[col1],
                "X2": inp_data[col2],
                "Y": inp_data["target"],
            }
        )
        d2_c_c = (
            d1_c_c.groupby(["X1", "X2"]).agg({"Y": ["count", "mean"]}).reset_index()
        )

    elif type == 2:
        d2_c_c = get_meansqdiff_cat_cont(col1, col2, inp_data)
    else:
        d2_c_c = get_meansqdiff_cont_cont(col1, col2, inp_data)

    # Calculate the Bincount and Binmean and also the mean of columns
    d2_c_c.columns = [col1, col2, "BinCount", "BinMean"]
    pop_prop = d2_c_c.BinCount / len(inp_data)

    # Calculate MeansqDiff weighted and unweighted
    d2_c_c["Mean_sq_diff"] = (d2_c_c["BinMean"] - pop_prop_1) ** 2
    d2_c_c["Mean_sq_diffW"] = d2_c_c.Mean_sq_diff * pop_prop

    # MSd Plot
    d_mat = d2_c_c.pivot(index=col1, columns=col2, values="Mean_sq_diffW")
    fig = go.Figure(data=[go.Surface(z=d_mat.values)])
    fig.update_layout(
        title=col1 + " " + col2 + " Plot",
        autosize=True,
        scene=dict(xaxis_title=col2, yaxis_title=col1, zaxis_title="target"),
    )

    filename = "./Output-File/hw4_plots/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    file_n = "./hw4_plots/BruteForce_Plot_" + col1 + "_" + col2 + ".html"
    fname = "<a href=" + file_n + ">Plot Link"
    return d2_c_c["Mean_sq_diff"].sum(), d2_c_c["Mean_sq_diffW"].sum(), fname


# Calculate the correlation for cont-cont data using Pearsonr
# It also creates the correlation matrix to display the correlation heatmap
# It also calculates the Difference in mean weighted and unweighted
def get_cont_cont_cor(cont_cols, inp_data, cor_plot):
    df_cols = ["Continuous1", "Continuous2", "Correlation"]
    bf_cols = ["Continuous1", "Continuous2", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cont_cont = pd.DataFrame(columns=bf_cols)
    cont_cont_corr = pd.DataFrame(columns=df_cols)
    cont_cont_matrix = pd.DataFrame(index=cont_cols, columns=cont_cols)
    pop_prop_1 = inp_data.target.sum() / len(inp_data)
    if len(cont_cols) > 1:
        for i in range(len(cont_cols)):
            for j in range(i, len(cont_cols)):
                if cont_cols[i] != cont_cols[j]:
                    val, _ = stats.pearsonr(
                        inp_data[cont_cols[i]], inp_data[cont_cols[j]]
                    )
                    cont_cont_matrix.loc[cont_cols[i]][cont_cols[j]] = val
                    cont_cont_matrix.loc[cont_cols[j]][cont_cols[i]] = val
                    cont_cont_corr = cont_cont_corr.append(
                        dict(
                            zip(
                                df_cols,
                                [cor_plot[cont_cols[i]], cor_plot[cont_cols[j]], val],
                            )
                        ),
                        ignore_index=True,
                    )
                    w, uw, fname = get_w_uw_msd(
                        cont_cols[i], cont_cols[j], inp_data, pop_prop_1, 1
                    )
                    d2_cont_cont = d2_cont_cont.append(
                        dict(zip(bf_cols, [cont_cols[i], cont_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )
                else:
                    cont_cont_matrix[cont_cols[i]][cont_cols[j]] = 1.0
    return cont_cont_corr, cont_cont_matrix, d2_cont_cont


# Calculate the correlation for cat-cont data using code from cat_correlation.py
# cat_correlation Calculates correlation statistic for categorical-cont association
# using Correlation ratio
# It also creates the correlation matrix to display the correlation heatmap
# It also calculates the Difference in mean weighted and unweighted
def get_cat_cont_cor(cat_cols, cont_cols, inp_data, cor_plot):
    df_cols = ["Categorical", "Continuous", "Correlation"]
    bf_cols = ["Categorical", "Continuous", "MeanSqDiff", "MeanSqDiffW", "PLot Link"]
    d2_cat_cont = pd.DataFrame(columns=bf_cols)
    cat_cont_corr = pd.DataFrame(columns=df_cols)
    cat_cont_matrix = pd.DataFrame(index=cat_cols, columns=cont_cols)
    pop_prop_1 = inp_data.target.sum() / len(inp_data)
    # To calculate the correlation there has to be atleast 1 cat and cont variable
    if (len(cont_cols) >= 1) and (len(cat_cols) >= 1):
        for i in range(len(cat_cols)):
            for j in range(len(cont_cols)):
                # Calculate the correlation for cat-cont data using code from
                # cat_correlation.py
                val = cc.cat_cont_correlation_ratio(
                    inp_data[cat_cols[i]], inp_data[cont_cols[j]]
                )

                cat_cont_corr = cat_cont_corr.append(
                    dict(
                        zip(
                            df_cols,
                            [cor_plot[cat_cols[i]], cor_plot[cont_cols[j]], val],
                        )
                    ),
                    ignore_index=True,
                )
                cat_cont_matrix.loc[cat_cols[i]][cont_cols[j]] = val
                w, uw, fname = get_w_uw_msd(
                    cat_cols[i], cont_cols[j], inp_data, pop_prop_1, 2
                )
                d2_cat_cont = d2_cat_cont.append(
                    dict(zip(bf_cols, [cat_cols[i], cont_cols[j], w, uw, fname])),
                    ignore_index=True,
                )
    return cat_cont_corr, cat_cont_matrix, d2_cat_cont


# Calculate the correlation for cat-cat data using code from cat_correlation.py
# cat_correlation Calculates correlation statistic for categorical-categorical
# association using Cramers-V
# It also creates the correlation matrix to display the correlation heatmap
# It also calculates the Difference in mean weighted and unweighted
def get_cat_cat_cor(cat_cols, inp_data, cor_plot):
    df_cols = ["Categorical1", "Categorical2", "Correlation"]
    bf_cols = ["Categorical1", "Categorical2", "MeanSqDiff", "MeanSqDiffW", "Plot Link"]
    d2_cat_cat = pd.DataFrame(columns=bf_cols)
    cat_cat_corr = pd.DataFrame(columns=df_cols)
    cat_cat_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
    pop_prop_1 = inp_data.target.sum() / len(inp_data)
    # Loop through entire combination of cat columns
    # Skip calculation if there is just one categorical column
    # Skip calculation of same column
    if len(cat_cols) > 1:
        for i in range(len(cat_cols)):
            for j in range(i, len(cat_cols)):
                if cat_cols[i] != cat_cols[j]:
                    # Creating dataframe with cat column and target to calculate diff
                    # in mean
                    w, uw, fname = get_w_uw_msd(
                        cat_cols[i], cat_cols[j], inp_data, pop_prop_1, 3
                    )
                    d2_cat_cat = d2_cat_cat.append(
                        dict(zip(bf_cols, [cat_cols[i], cat_cols[j], w, uw, fname])),
                        ignore_index=True,
                    )
                    # It uses Cramers-V to calculate the correlation
                    val = cc.cat_correlation(
                        inp_data[cat_cols[i]], inp_data[cat_cols[j]]
                    )
                    cat_cat_corr = cat_cat_corr.append(
                        dict(
                            zip(
                                df_cols,
                                [cor_plot[cat_cols[i]], cor_plot[cat_cols[j]], val],
                            )
                        ),
                        ignore_index=True,
                    )
                    # get the correlation matrix for plotting
                    cat_cat_matrix.loc[cat_cols[i]][cat_cols[j]] = val
                    cat_cat_matrix.loc[cat_cols[j]][cat_cols[i]] = val
                else:
                    cat_cat_matrix.loc[cat_cols[i]][cat_cols[j]] = 1
    return cat_cat_corr, cat_cat_matrix, d2_cat_cat


file_path = "Output-File/hw4_plots/"


def main(fname, r_name):
    inp_data = load_data(fname, r_name)
    print(inp_data.head())
    # y = inp_data.target.values
    X = inp_data.drop("target", axis=1)

    res_type = Check_response_var(inp_data)
    print("Response variable is " + res_type)

    if res_type == "Boolean":
        inp_data["target"] = inp_data["target"].astype("category")
        inp_data["target"] = inp_data["target"].cat.codes
    # pred_types = {}

    # Part 1: Correlation metrics

    # First get Categorical and Continuous column list
    cat_cols, cont_cols = get_cat_cont_columns(X, inp_data)
    print_heading("Categorical Columns:")
    print(cat_cols)
    print_heading("Continuous Columns:")
    print(cont_cols)
    col_plot = {}

    for col in cat_cols:
        if res_type == "Boolean":
            col_plot[col] = gen_heatmap(inp_data, col)
        else:
            col_plot[col] = gen_violin_plot(inp_data, col)
    for col in cont_cols:
        if res_type == "Boolean":
            col_plot[col] = gen_distri_plot(inp_data, col)
        else:
            col_plot[col] = gen_scatter_plot(inp_data, col)

    # Cont/Cont
    cont_cont_corr, cont_cont_matrix, d2_cont_cont = get_cont_cont_cor(
        cont_cols, inp_data, col_plot
    )
    print_heading("Continuous-Continuous Correlation metrics")
    cont_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cont_cont_corr)

    # Cat/Cont
    cat_cont_corr, cat_cont_matrix, d2_cat_cont = get_cat_cont_cor(
        cat_cols, cont_cols, inp_data, col_plot
    )
    print_heading("Categorical-Continuous Correlation metrics")
    cat_cont_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cont_corr)

    # Cat/Cat
    cat_cat_corr, cat_cat_matrix, d2_cat_cat = get_cat_cat_cor(
        cat_cols, inp_data, col_plot
    )
    print_heading("Categorical-Categorical Correlation metrics")
    cat_cat_corr.sort_values(by=["Correlation"], inplace=True, ascending=False)
    print(cat_cat_corr)

    with open("./Output-File/Midterm-Part1-Corr-Metric.html", "w") as _file:
        _file.write(
            cont_cont_corr.to_html(render_links=True, escape=False)
            + "<br>"
            + cat_cont_corr.to_html(render_links=True, escape=False)
            + "<br>"
            + cat_cat_corr.to_html(render_links=True, escape=False)
        )

    # Part 2: Correlation Matrices
    cont_cont_matrix = cont_cont_matrix.astype(float)
    sns_plot = sns.heatmap(cont_cont_matrix, annot=True)
    fig1 = sns_plot.get_figure()
    fig1.savefig("./Output-File/hw4_plots/cont-cont-corr.png")
    plt.clf()

    # Cat-Cont-corr plot
    cat_cont_matrix = cat_cont_matrix.astype(float)
    sns_plot_1 = sns.heatmap(cat_cont_matrix, annot=True)
    fig2 = sns_plot_1.get_figure()
    fig2.savefig("./Output-File/hw4_plots/cat-cont-corr.png")
    plt.clf()

    # Cat-Cat-corr plot
    cat_cat_matrix = cat_cat_matrix.astype(float)
    sns_plot_2 = sns.heatmap(cat_cat_matrix, annot=True)
    fig3 = sns_plot_2.get_figure()
    fig3.savefig("./Output-File/hw4_plots/cat-cat-corr.png")

    with open("./Output-File/Midterm-Part2-Corr-Matrices.html", "w") as _file:
        _file.write(
            "<h1> Continuous-Continuous Plot </h1> "
            + "<img src='./hw4_plots/cont-cont-corr.png'"
            + "alt='Continuous-Continuous Plot'>"
            + "<h1> Categorical-Continuous Plot </h1> "
            + "<img src = './hw4_plots/cat-cont-corr.png'"
            + "alt='Categorical-Continuous Plot'>"
            + "<h1> Categorical-Categorical Plot </h1>"
            + "<img src ='./hw4_plots/cat-cat-corr.png'"
            + "alt='Categorical-Categorical'>"
        )

    # Brute-Force
    # Cont-Cont Diff of mean
    print_heading("Cont-Cont Brute-Force")
    d2_cont_cont = d2_cont_cont.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cont_cont)

    # Cat-Cont Diff of mean
    print_heading("Cat-Cont Brute-Force")
    d2_cat_cont = d2_cat_cont.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cat_cont)

    # Cat-Cat
    print_heading("Cat-Cat Brute-Force")
    d2_cat_cat = d2_cat_cat.sort_values(by="MeanSqDiffW", ascending=False)
    print(d2_cat_cat)

    with open("./Output-File/Midterm-Part2-BruteForce.html", "w") as _file:
        _file.write(
            d2_cont_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cont.to_html(render_links=True, escape=False)
            + "<br>"
            + d2_cat_cat.to_html(render_links=True, escape=False)
        )

    with open("./Output-File/Midterm-Final-Output.html", "w") as _file:
        _file.write(
            "<p><b> MidTerm Output <table><tr>"
            + "<tr><td><a href= 'Midterm-Part1-Corr-Metric.html'>"
            + "1. Correlation Metrics"
            + "<tr><td> <a href= 'Midterm-Part2-Corr-Matrices.html'>"
            + "2. Correlation Plot"
            + "<tr><td> <a href= 'Midterm-Part2-BruteForce.html'>"
            + "3. Brute-Force"
        )


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else ""
    print(fname)
    r_name = sys.argv[2] if len(sys.argv) > 1 else ""
    print(r_name)
    sys.exit(main(fname, r_name))
