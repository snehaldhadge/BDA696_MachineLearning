#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def main():
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    col_names = [
        "Sepal_Length",
        "Sepal_Width",
        "Petal_Length",
        "Petal_Width",
        "Species",
    ]
    # Load the Iris data set into Pandas DataFrame
    iris_df = pd.read_csv("Iris.data", names=col_names)
    print_heading("Printing first 10 rows")
    print(iris_df.head(10))

    # drop missing values
    iris_df = iris_df.dropna()

    iris_np = np.array(iris_df)

    # summary statistics
    print_heading("Summary Statistics using Numpy:")
    print("Mean:", np.mean(iris_np[:, :-1], axis=0))

    print("Max:", np.max(iris_np[:, :-1], axis=0))

    print("Min:", np.min(iris_np[:, :-1], axis=0))

    print("25%:", np.quantile(iris_np[:, :-1], 0.25, axis=0))

    print("50%:", np.quantile(iris_np[:, :-1], 0.5, axis=0))

    print_heading("Plots")
    fig = px.scatter(
        iris_df,
        x="Sepal_Width",
        y="Sepal_Length",
        color="Species",
        title="Scatter plot between Sepal_Width and Sepal_Length",
    )
    fig.show()

    # Violin plot provides with density estimate on the y-axis
    fig = px.violin(
        iris_df,
        y="Petal_Length",
        x="Species",
        color="Species",
        box=True,
        points="all",
        hover_data=iris_df.columns,
        title=" Violin plot using Petal_Length as feature",
    )
    fig.show()

    # Boxplot using Petal_Length as feature to show what percentile
    # ranges in what region
    fig = px.box(
        iris_df,
        y="Petal_Length",
        x="Species",
        color="Species",
        title=" Boxplot using Petal_Length as feature",
    )
    fig.show()

    # Scatter Plot of different features
    fig = px.scatter_matrix(
        iris_df,
        dimensions=["Sepal_Width", "Sepal_Length", "Petal_Width", "Petal_Length"],
        color="Species",
    )
    fig.show()

    # Pie chart to show Population of each Species in the dataset
    fig = px.pie(
        iris_df,
        values=iris_df["Species"].value_counts(),
        title="Population of different Species",
    )
    fig.show()

    # Splitting Data into Train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        iris_df.iloc[:, :-1].values, iris_df["Species"], test_size=0.3, random_state=100
    )
    print("\n X-train Shape:", X_train.shape)
    print("\n X-test Shape:", X_test.shape)

    # Setting the Pipeline to Normalize the data and using RandomForest
    pipeline = Pipeline(
        [("Normalize", Normalizer()), ("rf", RandomForestClassifier(random_state=1234))]
    )
    # Using Pipeline to Normalize data and fitting RandomForesClassifier
    pipeline.fit(X_train, y_train)
    # Predicting for test data
    prediction = pipeline.predict(X_test)
    print_heading("Accuracy using RandomForestClassifier")
    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy: {accuracy}")

    # Using Logistic Regression
    # Setting the Pipeline to Normalize the data and using LogisticRegression
    pipeline = Pipeline([("Normalize", Normalizer()), ("lf_fit", LogisticRegression())])
    # Using Pipeline to Normalize data and fitting Logistic Regression
    pipeline.fit(X_train, y_train)
    # Predicting for test data
    prediction = pipeline.predict(X_test)
    print_heading("\nAccuracy using LogisticRegression")
    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    sys.exit(main())
