"""Data filtering and NaN handling functions for Crunchbase startup features."""
import pandas as pd
import numpy as np
import os

REPLACE_NAN = False


# filtering: dropping rows and manipulating values
def m_founding_year(df):
    if REPLACE_NAN:
        df["founding_date"].fillna(pd.to_datetime("1900-01-01"), inplace=True)
    return df


def m_replace_invalid_category_values(value):
    if isinstance(value, float):
        value = pd.NA

    if REPLACE_NAN:
        return "unknown" if pd.isna(value) else value

    return value


def m_founder_count(df):
    if REPLACE_NAN:
        df["founder_count"].fillna(0, inplace=True)
    return df


def m_sfc_num_ventures(df):
    if REPLACE_NAN:
        df["num_ventures"].fillna(0, inplace=True)
    return df


def m_ivy_league_count(df):
    if REPLACE_NAN:
        df["ivy_league_count"].fillna(0, inplace=True)
    return df


def m_edu_counts(df):
    # Fill NaN values with 0 and convert counts to int
    for col in [
        "bachelor_count",
        "master_count",
        "phd_count",
        "business_degree_count",
        "it_degree_count",
        "stem_degree_count",
        "law_degree_count",
        "other_edu_count",
    ]:
        if REPLACE_NAN:
            df[col].fillna(0, inplace=True)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def m_replace_nan_with_empty_string(object_string):
    return str(object_string) if not pd.isna(object_string) else ""


def m_funding_rounds(df):
    # Fill NaN values with 0 and convert counts to int
    for col in df.columns:
        if "uuid" not in col:
            if REPLACE_NAN:
                df[col].fillna(0, inplace=True)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def m_top_percent_investor_count(df, percent):
    if REPLACE_NAN:
        df[f"top_{int(percent * 100)}_percent_investor_count"].fillna(0, inplace=True)
    df[f"top_{int(percent * 100)}_percent_investor_count"] = pd.to_numeric(
        df[f"top_{int(percent * 100)}_percent_investor_count"], errors="coerce"
    ).astype("Int64")
    return df


def m_accelerator_part_count(df):
    if REPLACE_NAN:
        df["accelerator_participation_count"].fillna(0, inplace=True)
    df["accelerator_participation_count"] = pd.to_numeric(
        df["accelerator_participation_count"], errors="coerce"
    ).astype("Int64")
    return df


def f_days_before_first_funding_old_companies(df):
    df = df[df["founded_on"] > "1900-01-01"]
    df = df[df["founded_on"] < "2100-01-01"]
    return df


def f_days_unitl_first_funding(df):
    # Remove startups with funding date before founding date
    if REPLACE_NAN:
        df = df[
            df["days_until_first_funding"].isna()
            | (df["days_until_first_funding"] >= 0)
        ]
        df = df[(df["days_until_first_funding"] >= 0)]
    return df


def f_founder_count(df):
    df["founder_count"] = pd.to_numeric(df["founder_count"], errors="coerce").astype(
        "Int64"
    )
    initial_row_count = df.shape[0]
    df = df[df["founder_count"] != 0]
    final_row_count = df.shape[0]
    rows_removed = initial_row_count - final_row_count
    print(
        f"Number of rows removed for founder_count: {rows_removed} / {initial_row_count} ({rows_removed/initial_row_count:.2%})"
    )
    return df


def filtering(df):
    df = f_founder_count(df)
    return df
