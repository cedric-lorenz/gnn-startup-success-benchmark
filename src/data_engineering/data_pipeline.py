"""
Run this file with the following command:
    python -m src.data_engineering.data_pipeline
Output:
    - data/crunchbase_df.csv
"""

import logging
import ast
import os
import pandas as pd
from . import aux_pipeline as ap
from . import org_features as of
from . import person_features as pf
from . import finance_features as ff
from . import city_features as cf
from . import targets as t
from sklearn.preprocessing import LabelEncoder
import uuid
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def build_founder_co_study_edges(config):
    """
    Builds edges between founders who studied at the same university during overlapping periods.
    Uses 'degree_type' to infer duration if 'started_on' is missing.
    """
    data_dir = config['paths']['graph_dir']
    base_dir = "data/crunchbase/2023" # Assuming raw data location
    
    print("Building Founder Co-Study Edges...")
    
    # Load Edges
    edges_path = os.path.join(data_dir, "founder_university_edges.csv")
    if not os.path.exists(edges_path):
        print(f"Skipping Co-Study: {edges_path} not found.")
        return

    df = pd.read_csv(edges_path)
    
    # Parse Dates
    df['completed_on'] = pd.to_datetime(df['completed_on'], errors='coerce')
    df['started_on'] = pd.to_datetime(df['started_on'], errors='coerce')
    
    df = df.dropna(subset=['completed_on']).copy()

    degree_map = {
        'PhD': 5, 'Doctorate': 5,
        'BS': 4, 'BA': 4, 'Bachelor': 4, 'Bachelors': 4,
        'MS': 2, 'MA': 2, 'Master': 2, 'Masters': 2, 'MBA': 2
    }
    
    def get_duration(deg_type):
        if pd.isna(deg_type): return 3
        for key, val in degree_map.items():
            if key.lower() in str(deg_type).lower():
                return val
        return 3 # Default
    
    # Only impute where missing
    missing_start = df['started_on'].isna()
    if missing_start.sum() > 0:
        print(f"Imputing start dates for {missing_start.sum()} records...")
        durations = df.loc[missing_start, 'degree_type'].apply(get_duration)
        # Subtract duration years (approx 365.25 days)
        # Timedelta might be safer
        offsets = pd.to_timedelta(durations * 365.25, unit='D')
        df.loc[missing_start, 'started_on'] = df.loc[missing_start, 'completed_on'] - offsets
        
    # Drop any remaining invalid dates
    df = df.dropna(subset=['started_on', 'completed_on'])
    
    print(f"Valid education records with timeline: {len(df)}")
    
    cols = ['founder_uuid', 'university_uuid', 'started_on', 'completed_on']
    merged = df[cols].merge(df[cols], on='university_uuid', suffixes=('_1', '_2'))
    
    # Filter logic
    # 1. Different founders
    merged = merged[merged['founder_uuid_1'] != merged['founder_uuid_2']]
    
    # 2. Duplicate pairs (A-B vs B-A) -> Keep A < B
    merged = merged[merged['founder_uuid_1'] < merged['founder_uuid_2']]
    
    # 3. Time Overlap
    # Max(start1, start2) < Min(end1, end2)
    start_max = pd.concat([merged['started_on_1'], merged['started_on_2']], axis=1).max(axis=1)
    end_min = pd.concat([merged['completed_on_1'], merged['completed_on_2']], axis=1).min(axis=1)
    
    merged = merged[start_max < end_min]
    
    print(f"Found {len(merged)} co-study pairs.")
    
    if len(merged) > 0:
        # Calculate overlap duration (days)
        merged['overlap_days'] = (end_min - start_max).dt.days
        
        # Save
        out_df = merged[['founder_uuid_1', 'founder_uuid_2', 'overlap_days']]
        out_path = os.path.join(data_dir, "founder_co_study_edges.csv")
        out_df.to_csv(out_path, index=False)

def build_founder_role_edges(config, jobs_df, founder_nodes, startup_nodes, investor_nodes):
    """
    Builds edges for founder roles:
    1. Board Member (type='board_member' at Startup)
    2. Director at Startup (title contains 'Director')
    3. Director at Investor (title contains 'Director')
    """
    data_dir = config['paths']['graph_dir']
    print("Building Founder Role Edges (Board, Director)...")
    
    # 1. Valid Node Sets
    valid_founders = set(founder_nodes['founder_uuid'])
    valid_startups = set(startup_nodes['startup_uuid'])

    inv_col = 'investor_uuid' if 'investor_uuid' in investor_nodes.columns else 'uuid'
    valid_investors = set(investor_nodes[inv_col])
    
    # 2. Filter jobs for valid founders only
    founder_jobs = jobs_df[jobs_df['founder_uuid'].isin(valid_founders)].copy()
    
    print(f"  Found {len(founder_jobs)} jobs held by valid founders.")
    
    # --- A. Board Member Edges (Founder -> Startup) ---
    board_jobs = founder_jobs[
        (founder_jobs['job_type'] == 'board_member') & 
        (founder_jobs['startup_uuid'].isin(valid_startups))
    ]
    
    if len(board_jobs) > 0:
        out_df = board_jobs[['founder_uuid', 'startup_uuid']]
        out_path = os.path.join(data_dir, "founder_board_edges.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_df)} Board Member edges to {out_path}")
    else:
        print("  No Board Member edges found.")

    # --- B. Director at Startup (Founder -> Startup) ---
    # Logic: title contains "Director" AND org is startup
    director_mask = founder_jobs['title'].str.contains('Director', case=False, na=False)
    
    # B1. Startup Directors
    startup_dir_jobs = founder_jobs[
        director_mask & 
        (founder_jobs['startup_uuid'].isin(valid_startups))
    ]
    
    if len(startup_dir_jobs) > 0:
        out_df = startup_dir_jobs[['founder_uuid', 'startup_uuid']]
        out_path = os.path.join(data_dir, "founder_startup_director_edges.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_df)} Startup Director edges to {out_path}")
    else:
        print("  No Startup Director edges found.")
        
    # --- C. Director at Investor (Founder -> Investor) ---
    investor_dir_jobs = founder_jobs[
        director_mask & 
        (founder_jobs['startup_uuid'].isin(valid_investors))
    ]
    
    if len(investor_dir_jobs) > 0:
        # Rename 'startup_uuid' column to 'investor_uuid' for the output CSV
        out_df = investor_dir_jobs[['founder_uuid', 'startup_uuid']].rename(
            columns={'startup_uuid': 'investor_uuid'}
        )
        out_path = os.path.join(data_dir, "founder_investor_director_edges.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved {len(out_df)} Investor Director edges to {out_path}")
    else:
        print("  No Investor Director edges found.")


def extract_organization_features(org_df):
    """
    Extracts features from the organization dataframe.
    """
    print("Extracting organization features...")

    df_dates = of.dates(org_df, is_startup=True)
    df_ind = of.industry_sectors(df_dates)
    df_sus = of.is_sustainable(df_ind)
    df_media = of.media_presence(df_sus)
    df_types = of.organization_types(df_media)
    
    df_types.drop(columns=["permalink", "cb_url", "homepage_url", "rank", "legal_name", "roles", "domain", "address", "postal_code", "short_description", "email", "phone", "facebook_url", "linkedin_url", "twitter_url", "logo_url", "alias1", "alias2", "alias3", "category_groups_list", "category_list"], inplace=True)
    df_types = df_types.rename(
        columns={
            "uuid": "startup_uuid",
            "status": "dc_status",
            "closed_on": "dc_closed_on",
        }
    )
    return df_types


def extract_people_features(temp_df, people_df, degrees_df, jobs_df):
    """
    Extracts features from the people dataframe.
    """
    print("Extracting people features...")

    # Use founders which have a valid featured_job_organization_uuid
    filtered_people_df = people_df[
        people_df["featured_job_organization_uuid"].isin(temp_df["startup_uuid"])
    ]
    filtered_jobs_df = jobs_df[jobs_df["startup_uuid"].isin(temp_df["startup_uuid"])]

    # Add people features
    temp_df = pf.founder_count(temp_df, filtered_people_df)
    temp_df = pf.female_count(temp_df, filtered_people_df)
    temp_df = pf.serial_founder_count(temp_df, filtered_jobs_df)

    # Add degree features
    temp_df = pf.extract_degree_features(temp_df, people_df, degrees_df)
    
    # Add team composition features (ratios, skill mix)
    temp_df = pf.team_composition_features(temp_df)

    return temp_df


def extract_financial_features(
    temp_df, investors_df, investments_df, funding_rounds_df
):
    """
    Extracts features from the investors dataframe.
    Extracted features:
        - investor_uuid
        - top_ten_percent_investor_count
        - days_until_first_funding
        - accelerator_participation_count
        - money_pre_seed
        - money_seed
        - money_angel_round
        - money_venture_round
        - pre_seed
        - seed
        - angel_round
        - venture_round
        - investors_pre_seed
        - investors_seed
        - investors_angel_round
        - investors_venture_round
        - total_money_raised
        - total_investors
        - investor_type
    """
    print("Extracting financial features...")

    temp_df = ff.funding_rounds(temp_df, funding_rounds_df)
    temp_df = ff.top_percent_investor_count(
        temp_df, investors_df, investments_df, funding_rounds_df, percent=0.1
    )
    temp_df = ff.top_percent_investor_count(
        temp_df, investors_df, investments_df, funding_rounds_df, percent=0.5
    )
    temp_df = ff.days_until_first_funding(temp_df, funding_rounds_df)
    temp_df = ff.accelerator_participation_count(
        temp_df, investors_df, investments_df, funding_rounds_df
    )
    temp_df = ff.startup_investor_types(temp_df, investors_df, funding_rounds_df)
    
    # Add funding velocity and growth features
    temp_df = ff.funding_velocity_features(temp_df, funding_rounds_df, investments_df)
    
    return temp_df

def add_description(nodes, descriptions_df, left_on):
    nodes = nodes.merge(
        descriptions_df,
        left_on=left_on,
        right_on='uuid',
        how='left'
    )
    
    # Add a binary flag: 1 if description exists and is not empty
    nodes["has_description"] = nodes["description"].notna() & (nodes["description"].str.strip() != "")
    nodes["has_description"] = nodes["has_description"].astype(int)
    
    # Add description length feature
    nodes["description_length"] = nodes["description"].str.len().fillna(0).astype(int)
    
    return nodes


def define_targets(temp_df, funding_rounds_df, acq_df, future_org_df, new_funding_rounds_df, new_acq_df, ipo_df, new_ipo_df):
    """
    Adds success definitions (targets) to the dataframe.
    """
    print("Defining targets...")

    temp_df = t.prepare_success(temp_df, funding_rounds_df, acq_df)
    temp_df = t.add_future_status(temp_df, future_org_df)
    temp_df = t.add_next_funding_round(temp_df, funding_rounds_df, new_funding_rounds_df)
    temp_df = t.add_new_acquisitions(temp_df, acq_df, new_acq_df)
    temp_df = t.add_new_ipos(temp_df, ipo_df, new_ipo_df)
    temp_df = t.add_mixed_target(temp_df)
    return temp_df


def create_startup_nodes(
    organizations_df,
    people_df,
    degrees_df,
    jobs_df,
    investors_df,
    investments_df,
    funding_rounds_df,
    acq_df,
    future_org_df, 
    new_funding_rounds_df,
    org_descriptions_df,
    new_acq_df,
    ipo_df,
    new_ipo_df
):
    startup_nodes = extract_organization_features(organizations_df)
    # Filter for operating companies only (as of 2023)
    startup_nodes = startup_nodes[startup_nodes['dc_status'] == 'operating']
    startup_nodes = extract_people_features(startup_nodes, people_df, degrees_df, jobs_df) # Uncomment if aggregated founder features in node are wanted
    startup_nodes = extract_financial_features(
        startup_nodes, investors_df, investments_df, funding_rounds_df
    )
    startup_nodes = add_description(startup_nodes, org_descriptions_df, left_on="startup_uuid")
    startup_nodes = define_targets(startup_nodes, funding_rounds_df, acq_df, future_org_df, new_funding_rounds_df, new_acq_df, ipo_df, new_ipo_df)
    startup_nodes.drop(columns=['uuid', 'state_code', 'region', 'total_funding', 'uuid_x', 'uuid_y'], inplace=True, errors='ignore')
    return startup_nodes


def create_investor_nodes(investors_df):
    df_keep = investors_df.copy()
    df_keep["is_org"] = df_keep["type"].apply(lambda x: 1 if x == "organization" else 0)
    # rename PK column
    df_keep.rename(columns={"uuid": "investor_uuid"}, inplace=True)

    # compute top 10 percent investor feature
    sorted_investors = (
        df_keep[["name", "total_funding_usd"]]
        .dropna()
        .sort_values(by="total_funding_usd", ascending=False)
    )
    sorted_investors = sorted_investors[: int(len(sorted_investors) * 0.1)]
    top_investors = sorted_investors["name"].to_list()

    df_keep["is_top_10_investor"] = df_keep["name"].apply(
        lambda x: 1 if x in top_investors else 0
    )
    
    df_keep = of.dates(df_keep)
    df_keep = of.media_presence(df_keep)
    
    df_keep = ff.investor_type_columns(df_keep)
    
    df_keep.drop(columns=["permalink", "cb_url", "rank", "roles", "domain", "facebook_url", "linkedin_url", "twitter_url", "logo_url", "state_code", "region", "investor_types","total_funding"], inplace=True, errors='ignore')

    return df_keep

def continents(df):
    """
    Adds one-hot columns for each continent returned by ap.convert_to_continent.
    Example columns: continent_africa, continent_europe, continent_unknown, ...
    The original 'continent' column is dropped.
    """
    df = df.copy()
    df["continent"] = df["country_code"].apply(ap.convert_to_continent)

    def _col_name(val):
        if pd.isna(val):
            return "continent_unknown"
        name = str(val).strip().lower().replace(" ", "_")
        return f"continent_{name}"

    df["__continent_col"] = df["continent"].apply(_col_name)
    unique_cols = df["__continent_col"].dropna().unique().tolist()
    # ensure unknown is present if any NaN
    if df["__continent_col"].isnull().any() and "continent_unknown" not in unique_cols:
        unique_cols.append("continent_unknown")

    for col in unique_cols:
        df[col] = (df["__continent_col"] == col).astype(int)

    df.drop(columns=["continent", "__continent_col"], inplace=True, errors="ignore")
    return df


def add_external_city_data_with_country(city_nodes, city_data):
    """
    Enrich city_nodes with external GeoNames-style data using (city_name + country_code) match.
    Handles common naming variations (e.g., "New York" vs "New York City").
    """

    # --- Normalize ---
    def normalize(s):
        if pd.isna(s):
            return ""
        return str(s).strip().lower()

    def clean_city_name(name):
        """Normalize and remove common city suffixes like 'city', 'town', etc."""
        name = normalize(name)
        for suffix in [" city", " town", " municipality", " village", " metropolitan area"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name.strip()

    city_nodes = city_nodes.copy()
    city_nodes["country_code"] = city_nodes["country_code"].apply(
        lambda c: ap.COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2.get(c, c)
    )
    city_nodes["__city_name_norm"] = city_nodes["city"].apply(clean_city_name)
    city_nodes["__country_norm"] = city_nodes["country_code"].fillna("").str.strip().str.lower()

    city_data = city_data.copy()
    city_data["__name_norm"] = city_data["Name"].apply(clean_city_name)
    city_data["__country_norm"] = city_data["Country Code"].fillna("").str.strip().str.lower()
    city_data["__alt_names_list"] = city_data["Alternate Names"].fillna("").apply(
        lambda x: [clean_city_name(a) for a in x.split(",") if a.strip()]
    )

    # --- Split coordinates ---
    def split_coords(coord_str):
        try:
            lat, lon = map(float, coord_str.split(","))
            return pd.Series({"latitude": lat, "longitude": lon})
        except Exception:
            return pd.Series({"latitude": pd.NA, "longitude": pd.NA})

    coords = city_data["Coordinates"].apply(split_coords)
    city_data = pd.concat([city_data, coords], axis=1)
    
    # Deduplicate by (normalized name, normalized country), keep highest population
    city_data["Population"] = pd.to_numeric(city_data["Population"], errors="coerce")
    city_data = (
        city_data.sort_values("Population", ascending=False)
        .drop_duplicates(subset=["__name_norm", "__country_norm"], keep="first")
    )
    
    # --- Step 1: Direct match (normalized name + country) ---
    merged = city_nodes.merge(
        city_data[
            ["__name_norm", "__country_norm", "Country Code", "Population", "latitude", "longitude"]
        ],
        left_on=["__city_name_norm", "__country_norm"],
        right_on=["__name_norm", "__country_norm"],
        how="left"
    )

    # --- Step 1.5: Retry with 'city' variations if missing ---
    missing_mask = merged["Country Code"].isna()
    if missing_mask.any():
        alt_merge_candidates = city_nodes.loc[missing_mask].copy()
        alt_merge_candidates["__city_name_city_suffix"] = alt_merge_candidates["__city_name_norm"].apply(
            lambda n: f"{n} city" if not n.endswith(" city") else n
        )

        alt_merge = alt_merge_candidates.merge(
            city_data[
                ["__name_norm", "__country_norm", "Country Code", "Population", "latitude", "longitude"]
            ],
            left_on=["__city_name_city_suffix", "__country_norm"],
            right_on=["__name_norm", "__country_norm"],
            how="left"
        )

        # fill only where missing
        fill_cols = ["Country Code", "Population", "latitude", "longitude"]
        for col in fill_cols:
            # alt_merge has reset index, so we assign by values (lengths must match)
            if len(merged.loc[missing_mask, col]) == len(alt_merge[col]):
                merged.loc[missing_mask, col] = alt_merge[col].values
            else:
                pass

    # --- Step 2: Alternate names fallback ---
    still_missing = merged["Country Code"].isna()
    if still_missing.any():
        alt_lookup = {}
        for _, row in city_data.iterrows():
            for alt in row["__alt_names_list"]:
                key = (alt, row["__country_norm"])
                if key not in alt_lookup:
                    alt_lookup[key] = {
                        "Country Code": row["Country Code"],
                        "Population": row["Population"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    }

        alt_matches = merged.loc[still_missing].apply(
            lambda r: alt_lookup.get((r["__city_name_norm"], r["__country_norm"])), axis=1
        )
        alt_df = pd.DataFrame(list(alt_matches))
        if not alt_df.empty:
            alt_df.index = merged.loc[still_missing].index
            merged.loc[still_missing, alt_df.columns] = alt_df

    # --- Step 3: Flag and rename ---
    merged["has_city_data"] = merged["Country Code"].notna().astype(int)
    merged = merged.rename(
        columns={
            "Country Code": "country_code_external",
            "Population": "population"
        }
    )

    # --- Step 4: Cleanup ---
    merged = merged.drop(columns=["__city_name_norm", "__country_norm", "__name_norm", "country_code_external", 0], errors="ignore")

    return merged

    

def create_city_nodes_and_edges(startup_nodes, university_nodes, investor_nodes, city_data):
    """
    Create city nodes and edges linking to startups, universities, and investors.
    Enrich city nodes with external data (GeoNames-style) and continent information.
    """

    # --- Step 1: Collect unique cities from all sources ---
    all_cities = (
        pd.concat(
            [
                startup_nodes[["city", "country_code"]],
                investor_nodes[["city", "country_code"]],
                university_nodes[["city"]].assign(country_code=pd.NA),
            ],
            ignore_index=True
        )
        .drop_duplicates(subset=["city"])
        .reset_index(drop=True)
    )

    # --- Step 2: Add continent based on country_code ---
    all_cities = continents(all_cities)

    # --- Step 3: Assign city_uuid ---
    all_cities["city_uuid"] = [str(uuid.uuid4()) for _ in range(len(all_cities))]
    city_nodes = all_cities.copy()

    # --- Step 4: Create edges ---
    startup_city_edges = pd.merge(
        startup_nodes[["startup_uuid", "city"]],
        city_nodes[["city", "city_uuid"]],
        on="city",
        how="left"
    )[["city_uuid", "startup_uuid"]]

    university_city_edges = pd.merge(
        university_nodes[["university_uuid", "city"]],
        city_nodes[["city", "city_uuid"]],
        on="city",
        how="left"
    )[["city_uuid", "university_uuid"]]

    investor_city_edges = pd.merge(
        investor_nodes[["investor_uuid", "city"]],
        city_nodes[["city", "city_uuid"]],
        on="city",
        how="left"
    )[["city_uuid", "investor_uuid"]]

    # --- Step 5: Attach city_uuid back to universities/investors ---
    university_nodes = university_nodes.merge(
        city_nodes[["city_uuid", "city"]], on="city", how="left"
    )
    investor_nodes = investor_nodes.merge(
        city_nodes[["city_uuid", "city"]], on="city", how="left"
    )

    # --- Step 6: Enrich with external data (country-aware) ---
    city_nodes = add_external_city_data_with_country(city_nodes, city_data)

    return city_nodes, university_nodes, investor_nodes, startup_city_edges, university_city_edges, investor_city_edges



def create_startup_investor_edges(
    startup_nodes, investor_nodes, funding_rounds_df, investments_df
):
    df_rounds_investments = pd.merge(
        left=funding_rounds_df,
        right=investments_df,
        how="inner",
        left_on="funding_round_uuid",
        right_on="funding_round_uuid",
        suffixes=("_rounds", "_investments"),
    ).sort_values(by=["startup_uuid", "funding_round_uuid"])
    df_rounds_investments[
        [
            "funding_round_uuid",
            "investment_type",
            "startup_uuid",
            "org_name",
            "investor_uuid",
            "investor_name",
        ]
    ]
    # We want to process all investment types now to categorize them
    # df_rounds_investments = df_rounds_investments[
    #     df_rounds_investments["investment_type"].isin(selected_investment_types)
    # ]
    
    # Map investment types to stages
    def map_stage(inv_type):
        if inv_type in ["pre_seed", "seed", "angel"]:
            return "early"
        elif inv_type in ["series_a", "series_b"]:
            return "mid"
        elif inv_type in ["series_c", "series_d", "series_e", "series_f", "series_g", "series_h", "series_i", "series_j", "private_equity"]:
            return "late"
        else:
            return "other"
            
    df_rounds_investments["stage"] = df_rounds_investments["investment_type"].apply(map_stage)
    
    # We want unique edges per stage. If an investor invested in multiple rounds of the same stage, it's still one edge.
    final_df = df_rounds_investments[["startup_uuid", "investor_uuid", "stage"]].drop_duplicates()
    
    return final_df


def create_founder_investor_employment_edges(founder_nodes, investor_nodes, jobs_df):
    """
    Creates edges between founders and investors where the founder worked at the investor organization.
    """
    # Filter jobs for founders
    # jobs_df has columns renamed: person_uuid -> founder_uuid, org_uuid -> startup_uuid
    founder_jobs = jobs_df[jobs_df["founder_uuid"].isin(founder_nodes["founder_uuid"])].copy()
    
    # Filter jobs where the org is an investor
    # Note: investor_nodes has 'investor_uuid' which corresponds to 'startup_uuid' (originally org_uuid) in jobs
    employment_jobs = founder_jobs[founder_jobs["startup_uuid"].isin(investor_nodes["investor_uuid"])]
    
    # Select relevant columns and rename
    edges = employment_jobs[["founder_uuid", "startup_uuid", "started_on", "ended_on"]].copy()
    edges.rename(columns={"startup_uuid": "investor_uuid"}, inplace=True)
    
    # Calculate duration
    edges["days_of_employment"] = (
        pd.to_datetime(edges["ended_on"], errors="coerce")
        - pd.to_datetime(edges["started_on"], errors="coerce")
    ).dt.days
    
    return edges


def create_coworking_edges(founder_nodes, jobs_df):
    """
    Creates edges between founders who worked at the same organization at the same time.
    """
    # Filter jobs for founders
    founder_jobs = jobs_df[jobs_df["founder_uuid"].isin(founder_nodes["founder_uuid"])].copy()
    
    # Convert dates
    founder_jobs["started_on"] = pd.to_datetime(founder_jobs["started_on"], errors="coerce")
    founder_jobs["ended_on"] = pd.to_datetime(founder_jobs["ended_on"], errors="coerce")
    
    # Missing end date means 'current'; use a safe future date, not too far to avoid OverflowError (pandas has ~292 year limit)
    future_date = pd.Timestamp("2030-01-01")
    founder_jobs["ended_on"] = founder_jobs["ended_on"].fillna(future_date)
    
    # Drop rows with missing start date as we can't determine overlap
    founder_jobs = founder_jobs.dropna(subset=["started_on"])
    
    # Filter out very old dates to prevent overflow and irrelevant data
    founder_jobs = founder_jobs[founder_jobs["started_on"] > pd.Timestamp("1970-01-01")]
    
    # Self-join on startup_uuid; keep only necessary columns (full self-join over ~1.1M jobs is heavy)
    cols = ["founder_uuid", "startup_uuid", "started_on", "ended_on"]
    jobs_slim = founder_jobs[cols]
    
    merged = pd.merge(
        jobs_slim, 
        jobs_slim, 
        on="startup_uuid", 
        suffixes=("_1", "_2")
    )
    
    # Filter for unique pairs (undirected)
    merged = merged[merged["founder_uuid_1"] < merged["founder_uuid_2"]]
    
    # Check time overlap
    # Overlap exists if max(start1, start2) < min(end1, end2)
    
    start_max = merged[["started_on_1", "started_on_2"]].max(axis=1)
    end_min = merged[["ended_on_1", "ended_on_2"]].min(axis=1)
    
    # Calculate overlap in days
    # We convert to days immediately to avoid carrying Timedeltas that might overflow
    merged["overlap_days"] = (end_min - start_max).dt.days
    
    # Filter for positive overlap (e.g. at least 30 days)
    coworking_edges = merged[merged["overlap_days"] > 30].copy()
    
    # Rename columns
    coworking_edges.rename(columns={
        "founder_uuid_1": "founder_uuid_1",
        "founder_uuid_2": "founder_uuid_2"
    }, inplace=True)
    
    return coworking_edges[["founder_uuid_1", "founder_uuid_2", "startup_uuid", "overlap_days"]]


def create_founder_investor_identity_edges(founder_nodes, investor_nodes):
    """
    Creates identity edges between founders and investors who are the same person (same UUID).
    """
    # Find overlapping UUIDs
    common_uuids = set(founder_nodes["founder_uuid"]).intersection(set(investor_nodes["investor_uuid"]))
    
    if not common_uuids:
        return pd.DataFrame(columns=["founder_uuid", "investor_uuid"])
        
    # Create DataFrame with matches
    # Since UUIDs match, we just need the list of common UUIDs
    common_list = list(common_uuids)
    identity_edges = pd.DataFrame({
        "founder_uuid": common_list,
        "investor_uuid": common_list
    })
    
    return identity_edges


def create_founder_nodes_and_edges(startup_nodes, people_df, jobs_df, degrees_df, people_descriptions_df):
    # Step 1: Filter jobs for founders (e.g., title contains "found")
    founder_jobs = jobs_df[
        jobs_df["title"].str.contains("found", case=False, na=False)
    ].copy()

    # Step 2: Keep only jobs with valid orgs that are in startup_nodes
    founder_jobs = founder_jobs[
        founder_jobs["startup_uuid"].isin(startup_nodes["startup_uuid"])
    ]

    # Step 3: Build founder nodes by merging with people_df
    founder_nodes = pd.merge(
        founder_jobs[["founder_uuid"]].drop_duplicates(),
        people_df,
        left_on="founder_uuid",
        right_on="uuid",
        how="left",
    ).drop(columns=["uuid"])

    # Step 4: Create edges from founders to startups
    startup_founder_edges = founder_jobs[
        ["startup_uuid", "founder_uuid", "title", "started_on", "ended_on"]
    ].copy()

    # Step 5: Create days_of_employment feature
    startup_founder_edges["days_of_employment"] = (
        pd.to_datetime(startup_founder_edges["ended_on"], errors="coerce")
        - pd.to_datetime(startup_founder_edges["started_on"], errors="coerce")
    ).dt.days
    
    # Step 6: Add descriptions
    founder_nodes = add_description(founder_nodes, people_descriptions_df, left_on="founder_uuid")

    founder_nodes = pf.founder_serial_features(founder_nodes, jobs_df)
    founder_nodes = pf.founder_degree_features(founder_nodes, degrees_df)
    
    # Add is_female flag
    founder_nodes["is_female"] = (founder_nodes["gender"] == "female").astype(int)

    founder_nodes.drop(columns=["featured_job_organization_uuid", "permalink", "uuid", "gender"], inplace=True, errors='ignore')
    
    return founder_nodes.reset_index(drop=True), startup_founder_edges.reset_index(
        drop=True
    )


def create_university_nodes_and_edges(founder_nodes, degrees_df):
    # Step 1: Filter degrees_df to only include degrees from people in founder_nodes
    founder_degrees = degrees_df[
        degrees_df["founder_uuid"].isin(founder_nodes["founder_uuid"])
    ].copy()

    # Step 2: Build university nodes (deduplicated)
    university_nodes = (
        founder_degrees[["university_uuid", "university_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Step 3: Build founder-university edges with degree info
    founder_university_edges = founder_degrees[
        [
            "founder_uuid",
            "university_uuid",
            "degree_type",
            "subject",
            "started_on",
            "completed_on",
            "is_completed",
        ]
    ].reset_index(drop=True)

    # Step 4: Create days_of_study feature
    founder_university_edges["days_of_study"] = (
        pd.to_datetime(founder_university_edges["completed_on"], errors="coerce")
        - pd.to_datetime(founder_university_edges["started_on"], errors="coerce")
    ).dt.days

    # Step 5: Create city column in university_nodes
    university_nodes = cf.create_university_cities(university_nodes)

    return university_nodes, founder_university_edges


def create_sector_nodes_and_edges(startup_nodes):
    # Step 1: Parse industries column from string to list (if it's a stringified list)
    startup_nodes = startup_nodes.copy()
    startup_nodes["industries"] = startup_nodes["industries"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Step 2: Explode to get one row per (startup, sector)
    exploded = (
        startup_nodes[["startup_uuid", "industries"]].explode("industries").dropna()
    )
    exploded = exploded.rename(columns={"industries": "sector_name"})

    # Step 3: Create unique sector nodes
    sector_nodes = exploded[["sector_name"]].drop_duplicates().reset_index(drop=True)
    sector_nodes["sector_uuid"] = sector_nodes["sector_name"].apply(
        lambda name: pd.util.hash_pandas_object(pd.Series(name)).values[0]
    )  # optional UUID-like hash

    # Step 4: Merge to get sector_uuid on the exploded edge list
    startup_sector_edges = pd.merge(
        exploded, sector_nodes, on="sector_name", how="left"
    )[["startup_uuid", "sector_uuid"]]

    return sector_nodes, startup_sector_edges


def rename_initial_columns(columns_mapping, dfs):
    for df in dfs:
        df.rename(columns=columns_mapping, inplace=True)


def import_data(crunchbase_dir, old_year=2023, new_year=2025):
    organizations_df = ap.import_csv(
        crunchbase_dir,
        f"{old_year}/organizations.csv",
        #nrows=10000
    )
    future_org_df = ap.import_csv(
        crunchbase_dir,
        f"{new_year}/organizations.csv",
        #nrows=10000
    )
    funding_rounds_df = ap.import_csv(crunchbase_dir, f"{old_year}/funding_rounds.csv")
    new_funding_rounds_df = ap.import_csv(crunchbase_dir, f"{new_year}/funding_rounds.csv")
    investors_df = ap.import_csv(crunchbase_dir, f"{old_year}/investors.csv")
    investments_df = ap.import_csv(crunchbase_dir, f"{old_year}/investments.csv")
    degrees_df = ap.import_csv(
        crunchbase_dir,
        f"{old_year}/degrees.csv",
        columns=[
            "uuid",
            "name",
            "type",
            "person_uuid",
            "institution_uuid",
            "institution_name",
            "degree_type",
            "subject",
            "is_completed",
            "started_on",
            "completed_on",
        ],
    )
    people_df = ap.import_csv(
        crunchbase_dir,
        f"{old_year}/people.csv",
        columns=[
            "uuid",
            "name",
            "permalink",
            "type",
            "gender",
            "featured_job_organization_uuid",
        ],
    )
    jobs_df = ap.import_csv(
        crunchbase_dir,
        f"{old_year}/jobs.csv",
        columns=["person_uuid", "org_uuid", "title", "started_on", "ended_on", "job_type"],
    )
    acq_df = ap.import_csv(
        crunchbase_dir, f"{old_year}/acquisitions.csv", columns=["uuid", "acquirer_uuid", "acquiree_uuid"]
    )
    new_acq_df = ap.import_csv(
        crunchbase_dir, f"{new_year}/acquisitions.csv", columns=["uuid", "acquirer_uuid", "acquiree_uuid"]
    )
    ipo_df = ap.import_csv(
        crunchbase_dir, f"{old_year}/ipos.csv", columns=["uuid", "org_uuid"]
    )
    new_ipo_df = ap.import_csv(
        crunchbase_dir, f"{new_year}/ipos.csv", columns=["uuid", "org_uuid"]
    )
    org_descriptions_df = ap.import_csv(
        crunchbase_dir, f"{old_year}/organization_descriptions.csv", columns=["uuid", "description"]
    )
    people_descriptions_df = ap.import_csv(
        crunchbase_dir, f"{old_year}/people_descriptions.csv", columns=["uuid", "description"]
    )

    rename_initial_columns(
        {
            "org_uuid": "startup_uuid",
            "person_uuid": "founder_uuid",
            "institution_uuid": "university_uuid",
            "institution_name": "university_name",
        },
        [
            organizations_df,
            future_org_df,
            funding_rounds_df,
            new_funding_rounds_df,
            investors_df,
            investments_df,
            degrees_df,
            people_df,
            jobs_df,
            acq_df,
            new_acq_df,
            ipo_df,
            new_ipo_df,
            org_descriptions_df,
            people_descriptions_df
        ],
    )

    return (
        organizations_df,
        funding_rounds_df,
        investors_df,
        investments_df,
        degrees_df,
        people_df,
        jobs_df,
        acq_df,
        future_org_df,
        new_funding_rounds_df,
        new_acq_df,
        ipo_df,
        new_ipo_df,
        org_descriptions_df,
        people_descriptions_df
    )


def main():
    crunchbase_dir = config["paths"]["crunchbase_dir"]
    graph_dir = config["paths"]["graph_dir"]

    print("Importing data...")
    (
        organizations_df,
        funding_rounds_df,
        investors_df,
        investments_df,
        degrees_df,
        people_df,
        jobs_df,
        acq_df,
        future_org_df,
        new_funding_rounds_df,
        new_acq_df,
        ipo_df,
        new_ipo_df,
        org_descriptions_df,
        people_descriptions_df
    ) = import_data(crunchbase_dir)
    
    city_data = pd.read_csv(os.path.join(config["paths"]["other_dir"], "huwise_city_data.csv"), sep=';', usecols=["Name", "Alternate Names", "Country Code", "Population", "DIgital Elevation Model", "Coordinates"])

    organizations_df.rename(columns={"org_uuid": "startup_uuid"}, inplace=True)

    print("Starting data pipeline...")
    print(f"Extracting STARTUP nodes...")
    startup_nodes = create_startup_nodes(
        organizations_df,
        people_df,
        degrees_df,
        jobs_df,
        investors_df,
        investments_df,
        funding_rounds_df,
        acq_df,
        future_org_df,
        new_funding_rounds_df,
        org_descriptions_df,
        new_acq_df,
        ipo_df,
        new_ipo_df
    )

    print(f"Extracting INVESTOR nodes...")
    investor_nodes = create_investor_nodes(investors_df)
    print(f"Extracting FOUNDER nodes...")
    founder_nodes, startup_founder_edges = create_founder_nodes_and_edges(
        startup_nodes, people_df, jobs_df, degrees_df, people_descriptions_df
    )
    print("Extracting UNIVERSITY nodes...")
    university_nodes, founder_university_edges = create_university_nodes_and_edges(
        founder_nodes, degrees_df
    )
    
    edges_path = os.path.join(graph_dir, "founder_university_edges.csv")
    founder_university_edges.to_csv(edges_path, index=False)

    if config["data_processing"].get("role_edges_enabled", False):
        build_founder_role_edges(config, jobs_df, founder_nodes, startup_nodes, investor_nodes)
        
    print("Extracting CITY nodes...")
    (
        city_nodes,
        university_nodes,
        investor_nodes,
        startup_city_edges,
        university_city_edges,
        investor_city_edges,
    ) = create_city_nodes_and_edges(startup_nodes, university_nodes, investor_nodes, city_data)
    print(f"Extracting SECTOR nodes...")
    sector_nodes, startup_sector_edges = create_sector_nodes_and_edges(startup_nodes)

    print(f"Finished nodes.")
    print(f"Creating edges...")
    startup_investor_edges = create_startup_investor_edges(
        startup_nodes, investor_nodes, funding_rounds_df, investments_df
    )
    print(f"Finished edges.")
    
    print(f"Extracting EMPLOYMENT edges...")
    founder_investor_employment_edges = create_founder_investor_employment_edges(
        founder_nodes, investor_nodes, jobs_df
    )
    
    print(f"Extracting CO-WORKING edges...")
    founder_coworking_edges = create_coworking_edges(founder_nodes, jobs_df)
    
    print(f"Extracting IDENTITY edges...")
    founder_investor_identity_edges = create_founder_investor_identity_edges(founder_nodes, investor_nodes)
    
    print("Data pipeline finished.")

    os.makedirs(graph_dir, exist_ok=True)
    print(f"Exporting data to {graph_dir}...")
    startup_nodes.to_csv(os.path.join(graph_dir, "startup_nodes.csv"), index=False)
    investor_nodes.to_csv(os.path.join(graph_dir, "investor_nodes.csv"), index=False)
    startup_investor_edges.to_csv(
        os.path.join(graph_dir, "startup_investor_edges.csv"), index=False
    )
    founder_nodes.to_csv(os.path.join(graph_dir, "founder_nodes.csv"), index=False)
    startup_founder_edges.to_csv(
        os.path.join(graph_dir, "startup_founder_edges.csv"), index=False
    )
    university_nodes.to_csv(
        os.path.join(graph_dir, "university_nodes.csv"), index=False
    )
    founder_university_edges.to_csv(
        os.path.join(graph_dir, "founder_university_edges.csv"), index=False
    )
    city_nodes.to_csv(os.path.join(graph_dir, "city_nodes.csv"), index=False)
    startup_city_edges.to_csv(
        os.path.join(graph_dir, "startup_city_edges.csv"), index=False
    )
    university_city_edges.to_csv(
        os.path.join(graph_dir, "university_city_edges.csv"), index=False
    )
    investor_city_edges.to_csv(
        os.path.join(graph_dir, "investor_city_edges.csv"), index=False
    )
    sector_nodes.to_csv(os.path.join(graph_dir, "sector_nodes.csv"), index=False)
    startup_sector_edges.to_csv(
        os.path.join(graph_dir, "startup_sector_edges.csv"), index=False
    )
    
    founder_investor_employment_edges.to_csv(
        os.path.join(graph_dir, "founder_investor_employment_edges.csv"), index=False
    )
    founder_coworking_edges.to_csv(
        os.path.join(graph_dir, "founder_coworking_edges.csv"), index=False
    )
    founder_investor_identity_edges.to_csv(
        os.path.join(graph_dir, "founder_investor_identity_edges.csv"), index=False
    )

    print("Finished.")

    return


if __name__ == "__main__":
    main()
