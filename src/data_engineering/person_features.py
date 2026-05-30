"""Founder and team feature engineering: founder counts, education levels, serial entrepreneurship metrics."""
import pandas as pd
from . import aux_pipeline as ap
from . import filtering as f


# preprocessing of personal features
def founder_count(df, people_df):
    founder_count_df = (
        people_df.groupby("featured_job_organization_uuid")
        .size()
        .reset_index(name="founder_count")
    )
    founder_count_df["founder_count"] = founder_count_df["founder_count"].astype(int)

    # Merge founder_count with temp_df
    df = df.merge(
        founder_count_df,
        left_on="startup_uuid",
        right_on="featured_job_organization_uuid",
        how="left",
    )
    df.drop(columns="featured_job_organization_uuid", inplace=True, errors='ignore')
    df = f.m_founder_count(df)
    return df


def female_count(df, people_df):
    female_count_df = (
        people_df.groupby("featured_job_organization_uuid")["gender"]
        .apply(lambda x: (x == "female").sum())
        .reset_index(name="female_count")
    )
    female_count_df["female_count"].astype(int)

    # Merge female_count with temp_df
    df = df.merge(
        female_count_df,
        left_on="startup_uuid",
        right_on="featured_job_organization_uuid",
        how="left",
    )
    df.drop(columns="featured_job_organization_uuid", inplace=True, errors='ignore')
    return df


def serial_founder_count(df, jobs_df):
    filtered_df = jobs_df[jobs_df["title"].str.contains("found", case=False, na=False)]
    num_ventures = (
        filtered_df.groupby("founder_uuid").size().reset_index(name="num_ventures")
    )
    num_ventures = f.m_sfc_num_ventures(num_ventures)
    num_ventures["num_ventures"] = num_ventures["num_ventures"].astype(int)
    # Filter num_ventures to include only rows where num_ventures > 1
    filtered_num_ventures = num_ventures[num_ventures["num_ventures"] > 1]

    # Merge with jobs_df to get startup_uuid for each person
    merged_df = filtered_num_ventures.merge(
        jobs_df[["founder_uuid", "startup_uuid"]], on="founder_uuid", how="left"
    )

    # Count per startup_uuid
    serial_founder_count_df = (
        merged_df.groupby("startup_uuid")
        .size()
        .reset_index(name="serial_founder_count")
    )
    serial_founder_count_df["serial_founder_count"].astype(int)
    sfc_df_renamed = serial_founder_count_df.rename(
        columns={"startup_uuid": "startup_uuid_jobs"}
    )

    # Merge serial_founder_count with temp_df
    df = df.merge(
        sfc_df_renamed[["serial_founder_count", "startup_uuid_jobs"]],
        left_on="startup_uuid",
        right_on="startup_uuid_jobs",
        how="left",
    )
    df.drop(columns="startup_uuid_jobs", inplace=True)
    return df


def determine_ivy_league(university_name):
    if university_name in ap.IVY_LEAGUE_PLUS:
        return 1  # ivy_league
    return 0  # not ivy_league


def _compute_ivy_counts(base_df, people_df, degrees_df, level="startup"):
    """
    Generic helper to compute Ivy League counts either per startup or per founder.
    level: "startup" or "founder"
    """

    # Determine if the institution is Ivy League+
    degrees_df["is_ivy_league_plus"] = degrees_df["university_name"].apply(
        determine_ivy_league
    )

    # Merge degrees with people to link founders to orgs
    degrees_people_df = degrees_df.merge(
        people_df, left_on="founder_uuid", right_on="uuid", how="left"
    )

    if level == "startup":
        group_key = "featured_job_organization_uuid"
        count_col = "ivy_league_plus_count"
    elif level == "founder":
        group_key = "founder_uuid"
        count_col = "ivy_league_plus_count_person"
    else:
        raise ValueError("level must be 'startup' or 'founder'")

    # Count Ivy League+ degrees
    ivy_counts_df = (
        degrees_people_df[degrees_people_df["is_ivy_league_plus"] == 1]
        .groupby(group_key)
        .size()
        .reset_index(name=count_col)
    )

    # Merge ivy counts into base_df
    merged_df = base_df.merge(
        ivy_counts_df,
        left_on='startup_uuid',
        right_on=group_key,
        how="left",
    )

    # Fill NaN with 0 where non-Ivy founders/orgs exist
    merged_df[count_col] = merged_df[count_col].fillna(0).astype(int)

    # Optional: pass through filtering functions depending on level
    if level == "startup":
        merged_df = f.m_ivy_league_count(merged_df)
    else:
        merged_df = f.m_ivy_league_count_person(merged_df) if hasattr(f, "m_ivy_league_count_person") else merged_df

    return merged_df


def ivy_count(temp_df, people_df, degrees_df):
    """Existing startup-level Ivy count wrapper"""
    temp_df = _compute_ivy_counts(temp_df, people_df, degrees_df, level="startup")
    return temp_df


def ivy_count_founder(founder_nodes, people_df, degrees_df):
    """New function: compute Ivy League degree count per founder"""
    founder_nodes = _compute_ivy_counts(founder_nodes, people_df, degrees_df, level="founder")
    return founder_nodes


# Define a function to categorize education types
def categorize_education_type(degree_type, name, subject):
    # Ensure degree_type, name, and subject are strings
    degree_type = f.m_replace_nan_with_empty_string(degree_type)
    name = f.m_replace_nan_with_empty_string(name)
    subject = f.m_replace_nan_with_empty_string(subject)

    if any(keyword in degree_type for keyword in ap.BACHELOR_KEYS) or any(
        keyword in name for keyword in ap.BACHELOR_KEYS
    ):
        return "bachelor"
    elif any(keyword in degree_type for keyword in ap.MASTER_KEYS) or any(
        keyword in name for keyword in ap.MASTER_KEYS
    ):
        return "master"
    elif any(keyword in degree_type for keyword in ap.PHD_KEYS) or any(
        keyword in name for keyword in ap.PHD_KEYS
    ):
        return "phd"
    elif (
        any(keyword in degree_type for keyword in ap.BUSINESS_KEYS)
        or any(keyword in name for keyword in ap.BUSINESS_KEYS)
        or any(keyword in subject for keyword in ap.BUSINESS_KEYS)
    ):
        return "business"
    elif (
        any(keyword in degree_type for keyword in ap.IT_KEYS)
        or any(keyword in name for keyword in ap.IT_KEYS)
        or any(keyword in subject for keyword in ap.IT_KEYS)
    ):
        return "it"
    elif (
        any(keyword in degree_type for keyword in ap.STEM_KEYS)
        or any(keyword in name for keyword in ap.STEM_KEYS)
        or any(keyword in subject for keyword in ap.STEM_KEYS)
    ):
        return "stem"
    elif (
        any(keyword in degree_type for keyword in ap.LAW_KEYS)
        or any(keyword in name for keyword in ap.LAW_KEYS)
        or any(keyword in subject for keyword in ap.LAW_KEYS)
    ):
        return "law"
    return "other"


def edu_counts(temp_df, people_df, degrees_df):
    # Categorize education types
    degrees_df["education_type"] = degrees_df.apply(
        lambda row: categorize_education_type(
            row["degree_type"], row["name"], row["subject"]
        ),
        axis=1,
    )

    # Merge degrees_df with people_df to link degrees to organizations
    degrees_people_df = degrees_df.merge(
        people_df, left_on="founder_uuid", right_on="uuid", how="left"
    )

    # Group by startup_uuid and count the number of each education type
    education_counts_df = (
        degrees_people_df.groupby(["featured_job_organization_uuid", "education_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Rename columns to match the desired output
    education_counts_df.rename(
        columns={
            "bachelor": "bachelor_count",
            "master": "master_count",
            "phd": "phd_count",
            "business": "business_degree_count",
            "it": "it_degree_count",
            "stem": "stem_degree_count",
            "law": "law_degree_count",
            "other": "other_edu_count",
        },
        inplace=True,
    )

    # Merge the education counts with temp_df
    temp_df = temp_df.merge(
        education_counts_df,
        left_on="startup_uuid",
        right_on="featured_job_organization_uuid",
        how="left",
    )
    temp_df.drop(columns="featured_job_organization_uuid", inplace=True, errors='ignore')
    temp_df = f.m_edu_counts(temp_df)
    return temp_df


def extract_degree_features(temp_df, people_df, degrees_df):
    temp_df = ivy_count(temp_df, people_df, degrees_df)
    temp_df = edu_counts(temp_df, people_df, degrees_df)
    return temp_df



# ---------------------------------------
# FOUNDER-LEVEL FEATURE FUNCTIONS
# ---------------------------------------

def founder_serial_features(founder_nodes, jobs_df):
    found_jobs = jobs_df[jobs_df["title"].str.contains("found", case=False, na=False)]
    venture_counts = (
        found_jobs.groupby("founder_uuid")["startup_uuid"]
        .nunique()
        .reset_index(name="num_ventures")
    )
    founder_nodes = founder_nodes.merge(venture_counts, on="founder_uuid", how="left")
    founder_nodes["num_ventures"] = founder_nodes["num_ventures"].fillna(0).astype(int)
    founder_nodes["is_serial_founder"] = (founder_nodes["num_ventures"] > 1).astype(int)
    return founder_nodes


def founder_degree_features(founder_nodes, degrees_df):
    degrees_df = degrees_df.copy()
    degrees_df["is_ivy_league_plus"] = degrees_df["university_name"].apply(determine_ivy_league)
    degrees_df["education_type"] = degrees_df.apply(
        lambda r: categorize_education_type(r["degree_type"], r["name"], r["subject"]),
        axis=1,
    )

    # 1. Ivy count per founder
    ivy_df = (
        degrees_df[degrees_df["is_ivy_league_plus"] == 1]
        .groupby("founder_uuid")
        .size()
        .reset_index(name="ivy_league_plus_count")
    )

    # 2. Education type counts per founder
    edu_df = (
        degrees_df.groupby(["founder_uuid", "education_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    rename_map = {
        "bachelor": "bachelor_count",
        "master": "master_count",
        "phd": "phd_count",
        "business": "business_degree_count",
        "it": "it_degree_count",
        "stem": "stem_degree_count",
        "law": "law_degree_count",
        "other": "other_edu_count",
    }
    edu_df.rename(columns=rename_map, inplace=True)

    founder_nodes = founder_nodes.merge(ivy_df, on="founder_uuid", how="left")
    founder_nodes = founder_nodes.merge(edu_df, on="founder_uuid", how="left")

    # fill NaNs with 0
    degree_cols = list(rename_map.values()) + ["ivy_league_plus_count"]
    for c in degree_cols:
        if c in founder_nodes.columns:
            founder_nodes[c] = founder_nodes[c].fillna(0).astype(int)

    return founder_nodes


def team_composition_features(df):
    """
    Computes team composition ratios and flags based on existing count columns.
    """
    # Avoid division by zero
    # Use a copy of founder_count for division so a 0 count maps to 1 instead of dividing by zero
    founder_count = df["founder_count"].replace(0, 1)
    
    # Ratios
    if "female_count" in df.columns:
        df["female_ratio"] = df["female_count"] / founder_count
    
    if "serial_founder_count" in df.columns:
        df["serial_founder_ratio"] = df["serial_founder_count"] / founder_count
        
    if "ivy_league_plus_count" in df.columns:
        df["ivy_league_ratio"] = df["ivy_league_plus_count"] / founder_count
    
    # Degree Ratios
    degree_cols = ["bachelor_count", "master_count", "phd_count", "stem_degree_count", "business_degree_count", "it_degree_count"]
    for col in degree_cols:
        if col in df.columns:
            ratio_col = col.replace("_count", "_ratio")
            df[ratio_col] = df[col] / founder_count
            
    # Skill Mix
    if "stem_degree_count" in df.columns and "business_degree_count" in df.columns:
        # Check if team has at least one STEM/IT person AND at least one Business person
        has_tech = (df["stem_degree_count"] > 0) | (df.get("it_degree_count", 0) > 0)
        has_biz = df["business_degree_count"] > 0
        df["has_tech_and_biz"] = (has_tech & has_biz).fillna(False).astype(int)
        
    return df
