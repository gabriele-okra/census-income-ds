import pandas as pd


def treat_weighted_dupes(df_train: pd.DataFrame, df_test:pd.DataFrame, weight_col: str) -> pd.DataFrame:
    """This function checks whether there are duplicates in both training and test
    sets and sums the weights for those instances. Furthermore, it assesses data leakage
    and drop records in the test set which are also present in the traning set.

    Args:
        df_train: input training set.
        df_test: input test set.
        weight_col: column constaining the instance weight.

    Returns:
        pd.DataFrame: Training + test set with no duplications.

    """

    # Define set of columns on which duplicates are assessed
    grouping_cols = [col for col in df_train.columns if col != weight_col]

    # Sum weights of duplicates rows separately for training and test sets
    print(" - Using weights to join duplicate rows in each set.")
    df_train = df_train.groupby(grouping_cols, as_index=False)[weight_col].sum()
    df_test = df_test.groupby(grouping_cols, as_index=False)[weight_col].sum()

    # Define boolean value to be used later on for splitting
    df_train["is_train"] = True
    df_test["is_train"] = False

    # Join sets
    df_all = pd.concat([df_train, df_test])
    meta_cols = ["is_train", weight_col]

    # If data leakage exists, drop duplicates in test set
    if _assert_data_leakage(train=df_train, test=df_test, exclude_cols=meta_cols):
        print(" - Fixing data leakage by dropping duplicates in the test set.")
        dupes_cols = [col for col in df_all.columns if col not in meta_cols]
        df_all = df_all.drop_duplicates(subset=dupes_cols, keep="first")

    return df_all

def _assert_data_leakage(train: pd.DataFrame, test:pd.DataFrame, exclude_cols: list[str]) -> bool:
    """This function checks whether there is data leakage between training
    and test sets.

    Args:
        train: Training set with no duplicates.
        test: Test set with no duplicates.
        exclude_cols: Columns not to be considered for duplication assessment.

    Returns:
        bool: True if data leakage exists.

    """

    train_size = train.drop(columns=exclude_cols).drop_duplicates().shape[0]
    test_size = test.drop(columns=exclude_cols).drop_duplicates().shape[0]

    return pd.concat([train, test]).drop(columns=exclude_cols).drop_duplicates().shape[0] != (train_size + test_size)


def target_mapping(df_all: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """This function maps the target income into standard 0's and 1's.

    Args:
        df_all: Input dataframe with traning and test samples.

    Returns:
        pd.DataFrame: Same dataframe with mapped target.

    """

    # Map income classes to standard 0's and 1's.
    print(" - Mapping the target columnto boolean values.")
    df_all[target_col] = df_all[target_col].map({' - 50000.': 0, ' 50000+.': 1})

    # Use instance_weight to compute class imbalance in the real population
    weighted_counts = df_all.groupby(target_col)['instance_weight'].sum()
    weighted_proportions = weighted_counts / weighted_counts.sum()
    print(" - Class imbalance in real population:")
    print(weighted_proportions)

    print(" - Class imbalance in training set:")
    print(df_all[df_all["is_train"]][target_col].value_counts(normalize=True))

    return df_all


def flag_missing_values(df: pd.DataFrame) -> None:
    """This function raises an error in case of existing NaN values in the
    dataframe. The reason why this is treated as an error is that in the
    current pipeline there is no mechanism to handle NaN values.

    Args:
        df: Input dataframe with training and test samples.

    """

    print(" - Checking for missing values.")
    if df.isna().sum().sum():
        raise ValueError("Existing missing values need treatment.")


def normalise_field_names(df: pd.DataFrame) -> pd.DataFrame:
    """This function normalises the various values for the nominal features
    by stripping heading or trailing spaces, and substitutes intra-string
    white spaces and special characters to make sure feature names are
    compliant with modelling libraries.

    Args:
        df: Input dataframe with training and test samples.

    Returns:
        pd.DataFrame: Same dataframe with normalised nominal feature values.

    """

    # Normalisation is only applied to nominal features
    obj_cols = df.select_dtypes(include='object').columns.to_list()

    print(" - Normalising string fields of categorical features.")
    for col in obj_cols:
        df[col] = df[col].apply(
            lambda x: x.strip().replace(" ", "_").replace("<", "less_than")
        )
    
    return df