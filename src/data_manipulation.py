import pandas as pd
from sklearn.preprocessing import StandardScaler

def fill_categorical_nan_with_mode(df: pd.DataFrame, categorical_attr: str):
    mode_value = df[categorical_attr].mode()[0]
    df[categorical_attr] = df[categorical_attr].fillna(mode_value)
    return df


def fill_numerical_nan_with_mean(df: pd.DataFrame, numerical_attr: str):
    df[numerical_attr] = df[numerical_attr].fillna(df[numerical_attr].mean())
    return df


def apply_continuous_label_encoding_to_categorical_columns(df: pd.DataFrame, feature: str):
    df[feature] = df[feature].astype('category')
    df[feature] = df[feature].cat.codes
    return df


def add_scaled_column(df: pd.DataFrame, feature: str):
    scaler = StandardScaler()
    df[[f'{feature}_Scaled']] = scaler.fit_transform(df[[feature]])
    return df