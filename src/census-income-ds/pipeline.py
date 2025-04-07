import pandas as pd
import pickle
import json

from globals import COLUMN_NAMES, COLS_DROP_NUMERICAL, COLS_DROP_NOMINAL

from pre_processing import (
    flag_missing_values,
    normalise_field_names,
    target_mapping,
    treat_weighted_dupes,
)
from feature_engineering import (
    create_age_group,
    encode_all,
    hispanic_to_boolean,
    group_rare_categories,
    sex_to_boolean,
    treat_birth_country_features,
    treat_investment_features,
    treat_migration_features
)
from modelling import (
    explainability_engine,
    prepare_train_test_sets,
    train_model,
)

def main():
    print("\n======== DATAIKU ASSIGNMENT - START ========")

    print("\n1) Loading data inputs...")
    df_train = pd.read_csv("data/inputs/census_income_learn.csv", header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv("data/inputs/census_income_test.csv", header=None, names=COLUMN_NAMES)
    print(f" - Loaded train and test sets with shapes {df_train.shape} and {df_test.shape} respectively.")
    print(" --> Data loading complete.")

    print("\n2) Pre-processing data...")
    df_all = treat_weighted_dupes(df_train=df_train, df_test=df_test, weight_col="instance_weight")
    df_all = target_mapping(df_all, target_col="target")
    flag_missing_values(df_all)
    df_all = normalise_field_names(df_all)
    df_pre_processed = df_all.copy()
    df_pre_processed.to_parquet("data/outputs/df_pre_processed.parquet")
    print(" --> Data pre-processing complete.")

    print("\n3.1) Engineering numerical features...")
    df_encoded = df_pre_processed.copy()
    df_encoded = df_encoded.drop(columns=COLS_DROP_NUMERICAL)
    df_encoded["age_group"] = create_age_group(df_encoded, age_col="age")
    df_encoded = treat_investment_features(df_encoded)
    print(" --> Numerical features engineering complete.")

    print("\n3.2) Engineering nominal features...")
    df_encoded = df_encoded.drop(columns=COLS_DROP_NOMINAL)
    df_encoded = treat_migration_features(
        df=df_encoded,
        migration_cols = ["migration_code_change_in_msa", "migration_code_change_in_reg", "migration_code_move_within_reg"],
    )
    df_encoded = treat_birth_country_features(df_encoded)
    df_encoded["hispanic_origin"] = hispanic_to_boolean(df_encoded)
    df_encoded = sex_to_boolean(df_encoded)
    df_encoded = group_rare_categories(df_encoded, exclude_cols=["education"])
    df_encoded = encode_all(df_encoded)
    df_encoded.to_parquet("data/outputs/df_encoded.parquet")
    print(" --> Nominal features engineering complete.")

    print("\n4) Modelling step started...")
    X_train, y_train, X_test, y_test = prepare_train_test_sets(df_encoded)
    trained_model = train_model(X_train, y_train, X_test, y_test)
    feat_imp, shap_explanations = explainability_engine(trained_model, X_test)
    with open('data/outputs/trained_model.pkl', 'wb') as file:
        pickle.dump(trained_model, file)
    with open("data/outputs/feat_imp.json", "w") as file:
        json.dump(feat_imp, file, indent=4)
    with open("data/outputs/shap_values.pkl", "wb") as file:
        pickle.dump(shap_explanations, file)
    print(" --> Modelling step complete.")



    print("\n======== DATAIKU ASSIGNMENT - END ========")



if __name__ == "__main__":
    main()
