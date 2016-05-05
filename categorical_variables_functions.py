# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn import feature_extraction

# Create a data frame.
d = {"col_1": np.random.choice(["a", "b", "c", "d"], size=1000000,
                             p=[0.4, 0.2, 0.1, 0.3]),
     "col_2": np.random.choice(["z", "e", "d", "a"], size=1000000,
                             p=[0.2, 0.3, 0.4, 0.1]),
     "col_3": np.random.normal(0, scale=2, size=1000000),
     "col_4": np.random.normal(100, scale=25, size=1000000)}

df = pd.DataFrame(d)

# Replace 5% of the values by np.nan.
for col in df.columns:
    df.loc[random.sample(range(0, df.shape[0]), int(df.shape[0] * 0.05)),
           col] = np.nan

# Create TRAIN and TEST datasets.
# Pandas >= 0.16.
TRAIN = df.sample(frac=0.7, replace=False)
TEST = df.drop(TRAIN.index)
# Pandas < 0.16.
# rows = random.sample(df2.index, int(df.shape[0] * 0.7))
# TRAIN = df.ix[rows]
# TEST = df.drop(rows)

# Reset the index.
TRAIN = TRAIN.reset_index(drop=True)
TEST = TEST.reset_index(drop=True)

# Lists of categorical and numerical variables.
categorical = df.select_dtypes(include=["object"]).columns
numerical = df.select_dtypes(exclude=["object"]).columns

def replacing_na_and_combining_values(TRAIN, TEST=None, fill_na_with="median",
                                      threshold=0.01):
    """
    Replace missing values by "missing" in the categorical variables and
    using a given strategy in the numerical variables.
    Replace the values that occur less than the threshold in each categorical
    column by "other".
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        fill_na_with: string or float, optional (default="median").
            The imputation strategy: "median", "mean", "most_frequent"
            or a float.
        threshold: float, optional (default=0.01).
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    
    # Replace missing values in the categorical variables by "missing".
    TRAIN[categorical] = TRAIN[categorical].fillna("missing")

    # Replace missing values in the numerical variables by the strategy chosen.
    if isinstance(fill_na_with, str):
        imputer = preprocessing.Imputer(missing_values="NaN",
                                        strategy=fill_na_with, axis=0)
        TRAIN[numerical] = imputer.fit_transform(TRAIN[numerical])
    else:
        TRAIN[numerical] = TRAIN[numerical].fillna(value=fill_na_with)

    # Combine together the values which appear less than the threshold
    # in the categorical variables.
    for col in categorical:
        counts = TRAIN[col].value_counts(normalize=True)
        TRAIN.loc[TRAIN[col].isin(counts[counts <= threshold].index),
                  col] = "other"

    # If TEST is not None do the same process.
    if TEST is not None:
        TEST[categorical] = TEST[categorical].fillna("missing")

        if isinstance(fill_na_with, str):
            TEST[numerical] = imputer.transform(TEST[numerical])
        else:
            TEST[numerical] = TEST[numerical].fillna(value=fill_na_with)

        for col in categorical:
            TEST.loc[TEST[col].isin(counts[counts <= threshold].index),
                     col] = "other"
        return (TRAIN, TEST)
    else:
        return TRAIN

TRAIN, TEST = replacing_na_and_combining_values(TRAIN, TEST,
                                                fill_na_with="median",
                                                threshold=0.01)

# for col in categorical:
#     print("")
#     print("-----")
#     print("")
#     print(TRAIN[col].value_counts(dropna=False))

"""
----------
"""

def to_numerical_sorted_alphabetically(TRAIN, TEST=None, classes=False):
    """
    Transform categorical features to numerical. The categories are encoded
    alphabetically (0 for the first one, 1 for the second, etc.).
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        classes: boolean, optional (default=False).
            Print the categories and the corresponding value for each
            categorical features.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    le = preprocessing.LabelEncoder()
    if TEST is not None:
        for col in categorical:
            TRAIN[col] = le.fit_transform(TRAIN[col])
            TEST[col] = le.transform(TEST[col])
            if classes:
                print("")
                print("-----")
                print("")
                print("Variable: {0}".format(col))
                for i in range(len(le.classes_)):
                    print("{0}: {1}".format(le.classes_[i],
                          np.sort(TRAIN[col].unique())[i]))
        return (TRAIN, TEST)
    else:
        for col in categorical:
            TRAIN[col] = le.fit_transform(TRAIN[col])
            if classes:
                print("")
                print("-----")
                print("")
                print("Variable: {0}".format(col))
                for i in range(len(le.classes_)):
                    print("{0}: {1}".format(le.classes_[i],
                          np.sort(TRAIN[col].unique())[i]))
        return TRAIN

# TRAIN, TEST = to_numerical_sorted_alphabetically(TRAIN, TEST, classes=True)

"""
----------
"""

def to_numerical_sorted_by_count(TRAIN, TEST=None, classes=False):
    """
    Transform categorical features to numerical. The categories are encoded
    in descending order (0 for the most frequent category, 1 for the second
    most frequent, etc.).
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        classes: boolean, optional (default=False).
            Print the categories and the corresponding value for each
            categorical features.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    if TEST is not None:
        for col in categorical:
            cat_ordered = TRAIN[col].value_counts()
            dict_cat_values = dict(zip(cat_ordered.index,
                                       range(len(cat_ordered))))
            TRAIN[col] = TRAIN[col].replace(dict_cat_values)
            TEST[col] = TEST[col].replace(dict_cat_values)
            if classes:
                print("")
                print("-----")
                print("")
                print("Variable: {0}".format(col))
                for i in range(len(TRAIN[col].unique())):
                    print("{0}: {1}".format(cat_ordered.index[i], i))
        return (TRAIN, TEST)
    else:
        for col in categorical:
            cat_ordered = TRAIN[col].value_counts()
            dict_cat_values = dict(zip(cat_ordered.index,
                                       range(len(cat_ordered))))
            TRAIN[col] = TRAIN[col].replace(dict_cat_values)
            if classes:
                print("")
                print("-----")
                print("")
                print("Variable: {0}".format(col))
                for i in range(len(TRAIN[col].unique())):
                    print("{0}: {1}".format(cat_ordered.index[i], i))
    return TRAIN

# TRAIN, TEST = to_numerical_sorted_by_count(TRAIN, TEST, classes=True)

"""
----------
"""

def to_dummy(TRAIN, TEST=None):
    """
    Transform categorical features to dummy variables.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    dv = feature_extraction.DictVectorizer(sparse=False)
    TRAIN = pd.concat([pd.DataFrame(dv.fit_transform(TRAIN[categorical].\
    to_dict("records"))), TRAIN[numerical]], axis=1)
    features_names = dv.get_feature_names()
    TRAIN.columns = [features_names + list(numerical)]
    TRAIN.columns = TRAIN.columns.str.replace("=", "_")
    if TEST is not None:
        TEST = pd.concat([pd.DataFrame(dv.fit_transform(TEST[categorical].\
        to_dict("records"))), TEST[numerical]], axis=1)
        TEST.columns = [features_names + list(numerical)]
        TEST.columns = TEST.columns.str.replace("=", "_")
        return (TRAIN, TEST)
    else:
        return TRAIN

# TRAIN, TEST = to_dummy(TRAIN, TEST)

"""
----------
"""

def to_numerical_replaced_by_count(TRAIN, TEST=None, classes=False):
    """
    Transform categorical features to numerical. The categories are encoded
    by their respective count (on both data frames if TEST is provided).
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        classes: boolean, optional (default=False).
            Print the categories and the corresponding value for each
            categorical features.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    if TEST is not None:
        data = pd.concat([TRAIN, TEST])
        for col in categorical:
            cat_counts = data[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            TEST[col] = TEST[col].replace(dict_cat_counts)
            if classes:
                print("")
                print("-----")
                print("")
                print(cat_counts)
        return (TRAIN, TEST)
    else:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            if classes:
                print("")
                print("-----")
                print("")
                print(cat_counts)
    return TRAIN

# TRAIN, TEST = to_numerical_replaced_by_count(TRAIN, TEST, classes=True)

"""
----------
"""

def to_numerical_replaced_by_percentage(TRAIN, TEST=None, classes=False):
    """
    Transform categorical features to numerical. The categories are encoded
    by their respective relative frequency (on both data frames if TEST
    is provided).
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        classes: boolean, optional (default=False).
            Print the categories and the corresponding value for each
            categorical features.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    if TEST is not None:
        data = pd.concat([TRAIN, TEST])
        for col in categorical:
            cat_counts = data[col].value_counts(normalize=True, dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            TEST[col] = TEST[col].replace(dict_cat_counts)
            if classes:
                print("")
                print("-----")
                print("")
                print(cat_counts)
        return (TRAIN, TEST)
    else:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(normalize=True, dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            if classes:
                print("")
                print("-----")
                print("")
                print(cat_counts)
    return TRAIN

# TRAIN, TEST = to_numerical_replaced_by_percentage(TRAIN, TEST, classes=True)

"""
----------
"""

def numerical_to_quantiles(TRAIN, TEST=None, n_quantiles=10):
    """
    Transform numerical features to numerical. The categories are encoded
    by their respective relative frequency (on both data frames if TEST
    is provided).
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        n_quantiles: integer , optional (default=10).
            Number of quantiles.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame if a second DataFrame was provided.
    """
    numerical = TRAIN.select_dtypes(exclude=["object"]).columns
    if TEST is not None:
        for col in numerical:
            TRAIN[col], bins = pd.qcut(TRAIN[col], n_quantiles,
                                       labels=False, retbins=True)
            TEST["_bins_" + col] = np.nan
            for idx, item in enumerate(bins):
                if idx <= 1:
                    TEST["_bins_" + col] = np.where(TEST[col] <= item, 0, 
                                                    TEST["_bins_" + col])
                if idx >= (n_quantiles - 1):
                    TEST["_bins_" + col] = np.where(TEST[col] > item,
                                                    (n_quantiles - 1),
                                                    TEST["_bins_" + col])
                if (1 < idx)  & (idx <= (n_quantiles - 1)):
                    TEST["_bins_" + col] = np.where((float(bins[idx - 1]) \
                    < TEST[col]) & (TEST[col] <= item), idx - 1, 
                                                    TEST["_bins_" + col])
        TEST = TEST.drop(numerical, axis=1)
        TEST.columns = TEST.columns.str.replace("_bins_", "")
        return (TRAIN, TEST)
    else:
        for col in numerical:
            TRAIN[col], bins = pd.qcut(TRAIN[col], 10, labels=False,
                                       retbins=True)
        return TRAIN

# TRAIN, TEST = numerical_to_quantiles(TRAIN, TEST, n_quantiles=10)
