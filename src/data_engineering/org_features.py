"""Organization feature engineering: date parsing, media presence, category encoding, and employee metrics."""
import logging
import pandas as pd
import numpy as np

from . import aux_pipeline as ap
from . import filtering as f


def dates(df, is_startup=False):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["founded_on"] = pd.to_datetime(df["founded_on"], errors="coerce")
    if is_startup:
        df["last_funding_on"] = pd.to_datetime(df["last_funding_on"], errors="coerce")
    df["closed_on"] = pd.to_datetime(df["closed_on"], errors="coerce")
    df = f.m_founding_year(df)
    df["founded_on_year"] = df["founded_on"].apply(
        lambda x: x.year if pd.notnull(x) else np.nan
    )
    return df


def organization_types(df):
    """
    Create one-hot encoded columns for each unique organization type.
    
    Args:
        df: DataFrame with 'type' column
        
    Returns:
        DataFrame with type_* columns and original type column dropped
    """
    for org_type in df["type"].dropna().unique():
        df[f"type_{org_type.lower().replace(' ', '_')}"] = (df["type"] == org_type).astype(int)

    df.drop(columns="type", inplace=True)
    return df


def media_presence(df):
    """
    Create boolean columns indicating presence of media/contact information.
    
    Args:
        df: DataFrame with media presence columns
        
    Returns:
        DataFrame with additional has_* columns
    """
    columns_to_check = [
        "domain", "email", "phone", "facebook_url", 
        "linkedin_url", "twitter_url", "logo_url"
    ]
    
    for col in columns_to_check:
        if col in df.columns:
            df[f"has_{col}"] = df[col].notna() & (df[col] != "")
        else:
            df[f"has_{col}"] = False
    
    return df


def industry_sectors(df):
    df["category_list"] = df["category_list"].apply(
        f.m_replace_invalid_category_values
    )
    df["industries"] = df["category_list"].apply(
        lambda x: pd.NA if pd.isna(x) else x.split(",")
    )
    df["category_groups_list"] = df["category_groups_list"].apply(
        f.m_replace_invalid_category_values
    )
    df["industry_groups"] = df["category_groups_list"].apply(
        lambda x: pd.NA if pd.isna(x) else x.split(",")
    )
    return df


def clean_category_list(df):
    df["category_list"] = df["category_list"].apply(f.m_replace_invalid_category_values)
    # Convert category_list to lists, or pd.NA if the value is NaN
    df["category_list"] = df["category_list"].apply(
        lambda x: pd.NA if pd.isna(x) else x.split(",")
    )
    return df

def is_sustainable2(df):
    df["sustainable_group"] = df["category_list"].apply(
        lambda x: (
            1
            if isinstance(x, list)
            and any(item in ap.SUSTAINABLE_CATEGORIES for item in x)
            else 0 if isinstance(x, list) else pd.NA
        )
    )
    df.drop(columns="category_list", inplace=True)
    return df


def filter_top_industries(df, X):
    industries = df["industries"].explode()
    top_industries = industries.value_counts().head(X).index.tolist()
    logging.info(f"Top {X} industries selected: {top_industries}")

    for industry in top_industries:
        df[f"industry_{industry.lower().replace(' ', '_')}"] = df["industries"].apply(
            lambda x: (
                1
                if isinstance(x, list) and industry in x
                else (0 if isinstance(x, list) else pd.NA)
            )
        )

    df.drop(columns="industries", inplace=True)
    return df


def is_sustainable(df):
    df["is_sustainable"] = df["industry_groups"].apply(
        lambda x: (
            1
            if isinstance(x, list) and "Sustainability" in x
            else (0 if isinstance(x, list) else pd.NA)
        )
    )
    return df
