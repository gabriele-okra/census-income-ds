import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from globals import CORRELATED_COLS_AFTER_ENCODING, ORDERED_EDU_LEVELS

AGE_BINS = 5
RARE_CATEGORY_THRESHOLD = 0.01


def create_age_group(df: pd.DataFrame, age_col: str) -> pd.Series:
    """This function creates an additional features from the age column
    by grouping age values.

    Args:
        df: Input dataframe with training and test samples.
        age_col: Column name with age information.

    Returns:
        pd.Series: Series with new age group feature.

    """

    # Define bins
    age_max = df[age_col].max()
    age_min = df[age_col].min()
    age_interval = (age_max - age_min)/AGE_BINS
    
    # Generate new grouped feature
    print(" - Generating an explicit age group feature to boost model's predictions.")
    return df[age_col].apply(lambda x: int((x - age_min)/(age_interval + 1e-4)))

def treat_investment_features(df: pd.DataFrame) -> pd.DataFrame:
    """This function treat all investment-related features. First of all,
    it transforms them into boolean values. Then it creates an additional column
    containing information on whether the record has any investment activity.

    Args:
        df: Input dataframe with training and test samples.

    Returns:
        pd.DataFrame: Same dataframe with boolean investment-related features.

    """

    print(" - Creating powerful boolean predictors out of investment-related features.")
    # New feature is 1 if there is any investment activity, and 0 otherwise
    df['has_investment_income'] = (
        (df['capital_gains'] > 0) |
        (df['capital_losses'] > 0) |
        (df['divdends_from_stocks'] > 0)
    ).astype(int)

    # Investment features as booleans
    df["capital_gains"] = df["capital_gains"].astype(bool).astype(int)
    df["capital_losses"] = df["capital_losses"].astype(bool).astype(int)
    df["divdends_from_stocks"] = df["divdends_from_stocks"].astype(bool).astype(int)

    return df


def treat_migration_features(df: pd.DataFrame, migration_cols: list[str]) -> pd.DataFrame:
    """This function treat all migration-related columns by creating a boolean
    value indicating whether the individuals in the record migrated or not.

    Args:
        df: Input dataframe with training and test samples.
        migration_cols: List of migration-related columns.

    Returns:
        pd.DataFrame: Same dataframe with boolean migration features.

    """

    print(" - Simplifying migration-related features.")
    # Group all non-migrant status
    non_migrant_labels = ["Not_in_universe", "Nonmover", "Not_identifiable", "?"]

    for col in migration_cols:
        df[col] = df[col].apply(
            lambda x: 0 if x in non_migrant_labels else 1
        )
    
    return df


def treat_birth_country_features(df: pd.DataFrame) -> pd.DataFrame:
    """This function treats all features related to the birth countries for the
    individual and her parents. First it creates boolean values indicating whether
    the person is born in the US or not, and then it computes the number of US-born
    parents.

    Args:
        df: Input dataframe with training and test samples.

    Returns:
        pd.DataFrame: Same dataframe with new boolean features related to
        birth countries.

    """

    print(" - Creating powerful boolean predictors out of birth-countries features.")
    # Country of birth to integer for both parents and individual
    df["mother_is_from_us"] = (df["country_of_birth_mother"]=="United-States").astype(int)
    df["father_is_from_us"] = (df["country_of_birth_father"]=="United-States").astype(int)
    df = df.drop(columns=["country_of_birth_mother", "country_of_birth_father"])

    df["born_in_us"] = (df["country_of_birth_self"]=="United-States").astype(int)
    df = df.drop(columns=["country_of_birth_self"])

    # Generate new feature by summing up the two on top
    df["number_parents_from_us"] = df["mother_is_from_us"] + df["father_is_from_us"]
    df = df.drop(columns=["mother_is_from_us", "father_is_from_us"])

    return df


def hispanic_to_boolean(df: pd.DataFrame) -> pd.Series:
    """This function transforms the hispanic-origins feature into a boolean,
    grouping together all hispanic origins into one.

    Args:
        df: Input dataframe with training and test samples.

    Returns:
        pd.DataFrame: Same dataframe with boolean hispanic-origin feature.

    """

    print(" - Turning hispanic origin info into a boolean feature.")
    false_value = "All_other"
    return df["hispanic_origin"].apply(lambda x: 0 if x==false_value else 1)


def sex_to_boolean(df: pd.DataFrame) -> pd.DataFrame:
    """This function transforms the sex column into a boolean value.

    Args:
        df: Input dataframe with training and test samples.

    Returns:
        pd.DataFrame: Same dataframe with boolean sex column.

    """

    print(" - Turning sex info into a boolean feature.")
    df["is_female"] = (df["sex"]=="Female").astype(int)
    df = df.drop(columns="sex")
    return df


def group_rare_categories(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """This function loops over nominal features and groups all values which are
    less frequent than a specified threshold into a sigle field named "Other", to
    prevent the model from learning spourious correlations.

    Args:
        df: Input dataframe with training and test samples.
        exclude_cols: Objecg columns to be excluded from the grouping operation.

    Returns:
        pd.DataFrame: Same dataframe with nominal features with reduced cardinality.

    """

    # Grouping of rare categories is only applied to nominal features
    # With some exception, see "education" (check notebook)
    cols_to_group = [
        col for col in df.select_dtypes("object").columns.to_list() if col not in exclude_cols
    ]

    print(f" - Grouping categories as rare as {RARE_CATEGORY_THRESHOLD*100}% to prevent overfitting.")
    for col in cols_to_group:
        value_counts = df[col].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < RARE_CATEGORY_THRESHOLD].index
        df[col] = df[col].replace(rare_categories, 'Other')

    return df


def encode_all(df: pd.DataFrame) -> pd.DataFrame:
    """This function applies different encoding strategies to different features.
    Specifically, it applies frequency encoding, ordinal encoding, and one-hot
    encodings to different feature sets based on the characteristics of such
    features.

    Args:
        df: Input dataframe with training and test samples. At this stage
        all pre-processing has been performed.

    Returns:
        pd.DataFrame: Encoded dataframe ready to be ingested by a classifier model.

    """

    # Because in "major_occupation_code" there is a correlation between the frequency 
    # of a category and the average target value for that category, we apply
    # frequency encoding.
    print(" - Applying frequency encoding to 1 feature.")
    major_occupation_encoding = df["major_occupation_code"].value_counts(normalize=True).to_dict()
    df['major_occupation_code'] = df['major_occupation_code'].map(major_occupation_encoding)

    # Create encoding dictionary to perform ordinal encoding on the "education" feature
    print(" - Applying ordinal encoding to 1 feature.")
    education_encoding = {level: i for i, level in enumerate(ORDERED_EDU_LEVELS)}
    df['education'] = df['education'].map(education_encoding)  

    # All the other nominal features undergo one-hot encoding
    cols_to_one_hot_encode = df.select_dtypes("object").columns
    print(f" - Applying one-hot encoding to {len(cols_to_one_hot_encode)} feature(s).")
    df = pd.get_dummies(df, columns=cols_to_one_hot_encode, drop_first=False)
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)

    # For clusters of correlated features after encoding, we only keep the one with
    # the higher mutual information score between the feature and the target
    n_dropped_features = 0
    for correlated_cols in CORRELATED_COLS_AFTER_ENCODING:
        mi_score = {}
        for col in correlated_cols:
            mi_score[col] = float(mutual_info_classif(df[[col]], df["target"], discrete_features=True)[0])
        max_key = max(mi_score, key=mi_score.get)
        cols_to_drop = [col for col in correlated_cols if col != max_key]
        df = df.drop(columns=cols_to_drop)
        n_dropped_features = n_dropped_features + len(cols_to_drop)
    print(f" - Dropped {n_dropped_features} highly-correlated features.")

    return df
