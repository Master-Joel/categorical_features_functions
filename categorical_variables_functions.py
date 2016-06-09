# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn import feature_extraction

"""
----------
"""

def replace_na(TRAIN, TEST=None, fill_na_with="median"):
    """
    Replace missing values by "missing" for the categorical variables and
    by using one of the four "strategies" available for the numerical variables.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        fill_na_with: string or float, optional (default="median").
            The imputation strategy: "median", "mean", "most_frequent"
            or a float.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    numerical = TRAIN.select_dtypes(exclude=["object"]).columns

    TRAIN[categorical] = TRAIN[categorical].fillna("missing")
    if isinstance(fill_na_with, str):
        imputer = preprocessing.Imputer(missing_values="NaN",
                                        strategy=fill_na_with, axis=0)
        TRAIN[numerical] = imputer.fit_transform(TRAIN[numerical])
    else:
        TRAIN[numerical] = TRAIN[numerical].fillna(value=fill_na_with)
    if TEST is not None:
        TEST[categorical] = TEST[categorical].fillna("missing")
        if isinstance(fill_na_with, str):
            TEST[numerical] = imputer.transform(TEST[numerical])
        else:
            TEST[numerical] = TEST[numerical].fillna(value=fill_na_with)
        return (TRAIN, TEST)
    else:
        return TRAIN

"""
----------
"""

def combine_values(TRAIN, TEST=None, threshold=0.01):
    """
    Replace the values that occur less than the threshold in each categorical
    feature by "other".
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        threshold: float, optional (default=0.01).
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns

    if TEST is not None:
        for col in categorical:
            counts = TRAIN[col].value_counts(dropna=False, normalize=True)
            TRAIN.loc[TRAIN[col].isin(counts[counts <= threshold].index),
                      col] = "other"
            TEST.loc[TEST[col].isin(counts[counts <= threshold].index),
                     col] = "other"
        return (TRAIN, TEST)
    else:
        for col in categorical:
            counts = TRAIN[col].value_counts(dropna=False, normalize=True)
            TRAIN.loc[TRAIN[col].isin(counts[counts <= threshold].index),
                      col] = "other"
        return TRAIN

"""
----------
"""

def transform_categorical_alphabetically(TRAIN, TEST=None, verbose=0):
    """
    Transform categorical features to numerical. The categories are encoded
    alphabetically (0 for the first one, 1 for the second, etc.).
    To be consistent with scikit-learn transformers having categories 
    in transform that are not present during training will raise an error
    by default.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        verbose: integer, optional (default=0).
            Controls the verbosity of the process.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    le = preprocessing.LabelEncoder()

    if TEST is not None:
        for col in categorical:
            TRAIN[col] = le.fit_transform(TRAIN[col])
            TEST[col] = le.transform(TEST[col])
            if verbose > 0:              
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    for i in range(len(le.classes_)):
                        print("{0}: {1}".format(le.classes_[i],
                              np.sort(TRAIN[col].unique())[i]))
        return (TRAIN, TEST)
    else:
        for col in categorical:
            TRAIN[col] = le.fit_transform(TRAIN[col])
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    for i in range(len(le.classes_)):
                        print("{0}: {1}".format(le.classes_[i],
                              np.sort(TRAIN[col].unique())[i]))
        return TRAIN

"""
----------
"""

def transform_categorical_sorted_by_count(TRAIN, TEST=None, 
                                          handle_unknown="error", 
                                          verbose=0):
    """
    Transform categorical features to numerical. The categories are encoded
    in descending order ("0" for the most frequent category, "1" for the second
    most frequent one, etc.).
    To be consistent with scikit-learn transformers having categories 
    in transform that are not present during training will raise an error
    by default.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        handle_unknown: str, "error", "ignore" or "NaN", 
        optional (default="error").
            Whether to raise an error, ignore or replace by NA if a unknown 
            category is present during transform.
        verbose: integer, optional (default=0).
            Controls the verbosity of the process.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    
    if TEST is not None:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, 
                                       range(len(cat_counts))))
            not_in_train = list(set(TEST[col].unique()) - set(cat_counts.index))
            if len(not_in_train) > 0:
                if handle_unknown == "error":
                    raise ValueError("TEST contains new labels: {0} "
                    "in variable {1}.".format(not_in_train, col))
                if handle_unknown == "ignore":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been ignored.".format(not_in_train))
                if handle_unknown == "NaN":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been replaced by NA.".format(not_in_train))
                    for item in not_in_train:
                        dict_cat_counts[item] = np.nan
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            TEST[col] = TEST[col].replace(dict_cat_counts)
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    for i in range(len(TRAIN[col].unique())):
                        print("{0}: {1}".format(cat_counts.index[i], i))
        return (TRAIN, TEST)
    else:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index,
                                       range(len(cat_counts))))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    for i in range(len(TRAIN[col].unique())):
                        print("{0}: {1}".format(cat_counts.index[i], i))
    return TRAIN

"""
----------
"""

def transform_categorical_to_dummy(TRAIN, TEST=None):
    """
    Transform categorical features to dummy variables.
    To be consistent with scikit-learn transformers having categories 
    in transform that are not present during training will raise an error
    by default.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns
    numerical = TRAIN.select_dtypes(exclude=["object"]).columns
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

"""
----------
"""

def transform_categorical_by_count(TRAIN, TEST=None, handle_unknown="error", 
                                   verbose=0):
    """
    Transform categorical features to numerical. The categories are encoded
    by their respective count (in the TRAIN dataset).
    To be consistent with scikit-learn transformers having categories 
    in transform that are not present during training will raise an error
    by default.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        handle_unknown: str, "error", "ignore" or "NaN", 
        optional (default="error").
            Whether to raise an error, ignore or replace by NA if a unknown 
            category is present during transform.
        verbose: integer, optional (default=0).
            Controls the verbosity of the process.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns

    if TEST is not None:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            not_in_train = list(set(TEST[col].unique()) - set(cat_counts.index))
            if len(not_in_train) > 0:
                if handle_unknown == "error":
                    raise ValueError("TEST contains new labels: {0} "
                    "in variable {1}.".format(not_in_train, col))
                if handle_unknown == "ignore":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been ignored.".format(not_in_train))
                if handle_unknown == "NaN":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been replaced by NA.".format(not_in_train))
                    for item in not_in_train:
                        dict_cat_counts[item] = np.nan
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            TEST[col] = TEST[col].replace(dict_cat_counts)
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    print(cat_counts)
        return (TRAIN, TEST)
    else:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
                    print(cat_counts)
    return TRAIN

"""
----------
"""

def transform_categorical_by_percentage(TRAIN, TEST=None, 
                                        handle_unknown="error", verbose=0):
    """
    Transform categorical features to numerical. The categories are encoded
    by their relative frequency (in the TRAIN dataset).
    To be consistent with scikit-learn transformers having categories 
    in transform that are not present during training will raise an error
    by default.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        handle_unknown: str, "error", "ignore" or "NaN", 
        optional (default="error").
            Whether to raise an error, ignore or replace by NA if a unknown 
            category is present during transform.
        verbose: integer, optional (default=0).
            Controls the verbosity of the process.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
    """
    categorical = TRAIN.select_dtypes(include=["object"]).columns

    if TEST is not None:
        for col in categorical:
            cat_counts = TRAIN[col].value_counts(normalize=True, dropna=False)
            dict_cat_counts = dict(zip(cat_counts.index, cat_counts))
            not_in_train = list(set(TEST[col].unique()) - set(cat_counts.index))
            if len(not_in_train) > 0:
                if handle_unknown == "error":
                    raise ValueError("TEST contains new labels: {0} "
                    "in variable {1}.".format(not_in_train, col))
                if handle_unknown == "ignore":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been ignored.".format(not_in_train))
                if handle_unknown == "NaN":
                    print("")
                    print("-----")
                    print("")
                    print("Variable: {0}".format(col))
                    print("Unknown category(ies) {0} present during transform "
                    "has(ve) been replaced by NA.".format(not_in_train))
                    for item in not_in_train:
                        dict_cat_counts[item] = np.nan
            TRAIN[col] = TRAIN[col].replace(dict_cat_counts)
            TEST[col] = TEST[col].replace(dict_cat_counts)
            if verbose > 0:
                print("")
                print("-----")
                print("")
                print("Feature: {0}".format(col))
                if verbose > 1:
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
                print("Feature: {0}".format(col))
                if verbose > 1:
                    print(cat_counts)
    return TRAIN

"""
----------
"""

def transform_numerical_to_quantiles(TRAIN, TEST=None, n_quantiles=10):
    """
    Transform numerical features to quantiles.
    -----
    Arguments:
        TRAIN: DataFrame.
        TEST: DataFrame, optional (default=None).
        n_quantiles: integer , optional (default=10).
            Number of quantiles.
    -----
    Returns:
        TRAIN: DataFrame.
        TEST: DataFrame.
            This second DataFrame is returned if two DataFrames were provided.
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
            TRAIN[col] = pd.qcut(TRAIN[col], n_quantiles, labels=False)
        return TRAIN
