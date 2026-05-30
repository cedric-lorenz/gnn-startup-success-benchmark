"""Financial feature engineering: funding round aggregation, investment type flags, and valuation metrics."""
import pandas as pd
import numpy as np
from . import aux_pipeline as ap
from . import filtering as f


def aggregate_funding_data(df, investment_types):
    return (
        df[df["investment_type"].isin(investment_types)]
        .groupby("startup_uuid")
        .agg({"raised_amount_usd": "sum", "investor_count": "sum"})
        .rename(
            columns={
                "raised_amount_usd": f'money_{"_".join(investment_types)}',
                "investor_count": f'investors_{"_".join(investment_types)}',
            }
        )
        .reset_index()
    )


def match_venture_round(df):
    """Return only venture rounds where name contains 'venture' and type is series_unknown."""
    return df[
        (df["investment_type"] == "series_unknown")
        & (df["name"].str.contains("venture", case=False, na=False))
    ]


def aggregate_venture_data(df):
    """Aggregate venture-specific rounds."""
    df = match_venture_round(df)
    return (
        df.groupby("startup_uuid")
        .agg({"raised_amount_usd": "sum", "investor_count": "sum"})
        .rename(
            columns={
                "raised_amount_usd": "money_venture_round",
                "investor_count": "investors_venture_round",
            }
        )
        .reset_index()
    )


def create_binary_feature(df, investment_type):
    """Binary flag: 1 if startup has at least one round of this type."""
    binary_df = (
        df[df["investment_type"] == investment_type]
        .groupby("startup_uuid")
        .size()
        .reset_index(name=f"{investment_type}_round")
    )
    binary_df[f"{investment_type}_round"] = 1
    return binary_df


def create_venture_feature(df):
    """Binary flag: 1 if startup has a venture round."""
    binary_df = (
        match_venture_round(df)
        .groupby("startup_uuid")
        .size()
        .reset_index(name="venture_round")
    )
    binary_df["venture_round"] = 1
    return binary_df


def funding_rounds(temp_df, funding_rounds_df):
    # --- Known round types ---
    # Extended to include late stage rounds for maturity gating
    known_types = [
        "angel", "pre_seed", "seed", 
        "series_a", "series_b", "series_c", 
        "series_d", "series_e", "series_f", "series_g", "series_h", "series_i", "series_j",
        "private_equity"
    ]

    # Aggregate known rounds
    agg_dfs = {t: aggregate_funding_data(funding_rounds_df, [t]) for t in known_types}
    bin_dfs = {t: create_binary_feature(funding_rounds_df, t) for t in known_types}

    # --- Venture rounds ---
    venture_round_df = aggregate_venture_data(funding_rounds_df)
    venture_round_binary_df = create_venture_feature(funding_rounds_df)

    # --- Other (unrecognized) investment types ---
    # --- Identify venture rounds first ---
    venture_mask = (
        (funding_rounds_df["investment_type"] == "series_unknown")
        & (funding_rounds_df["name"].str.contains("venture", case=False, na=False))
    )

    # --- Known types we always exclude ---
    known_set = set(known_types)

    # --- "Other" should exclude known + venture-specific rounds only ---
    other_df = funding_rounds_df[
        (~funding_rounds_df["investment_type"].isin(known_set))
        & (~venture_mask)  # exclude only those series_unknown that are venture rounds
    ]

    other_agg_df = (
        other_df.groupby("startup_uuid")
        .agg({"raised_amount_usd": "sum", "investor_count": "sum"})
        .rename(
            columns={
                "raised_amount_usd": "money_other_rounds",
                "investor_count": "investors_other_rounds",
            }
        )
        .reset_index()
    )

    other_flag_df = (
        other_df.groupby("startup_uuid")
        .size()
        .reset_index(name="other_round")
    )
    other_flag_df["other_round"] = 1

    # --- Total money & investors ---

    total_investors_df = (
        funding_rounds_df.groupby("startup_uuid")["investor_count"]
        .sum()
        .reset_index()
        .rename(columns={"investor_count": "total_investors"})
    )

    # --- Merge everything together ---
    aggregated_df = temp_df[["startup_uuid"]]

    for t in known_types:
        aggregated_df = (
            aggregated_df
            .merge(agg_dfs[t], on="startup_uuid", how="left")
            .merge(bin_dfs[t], on="startup_uuid", how="left")
        )

    aggregated_df = (
        aggregated_df
        .merge(venture_round_df, on="startup_uuid", how="left")
        .merge(venture_round_binary_df, on="startup_uuid", how="left")
        .merge(other_agg_df, on="startup_uuid", how="left")
        .merge(other_flag_df, on="startup_uuid", how="left")
        .merge(total_investors_df, on="startup_uuid", how="left")
    )

    # --- Apply any postprocessing ---
    aggregated_df = f.m_funding_rounds(aggregated_df)

    # Merge back into temp_df
    temp_df = temp_df.merge(aggregated_df, on="startup_uuid", how="left")
    return temp_df


def top_percent_investor_count(
    temp_df, investors_df, investments_df, funding_rounds_df, percent=0.1
):
    # Rename columns for consistency
    investors_df.rename(columns={"uuid": "investor_uuid"}, inplace=True)
    funding_rounds_df.rename(columns={"uuid": "funding_round_uuid"}, inplace=True)

    # Sort investors by total funding and select the top percentage
    sorted_investors = (
        investors_df[["investor_uuid", "name", "total_funding_usd"]]
        .dropna()
        .sort_values(by="total_funding_usd", ascending=False)
    )
    sorted_investors = sorted_investors[: int(len(sorted_investors) * percent)]
    top_investors = sorted_investors["investor_uuid"].to_list()

    # Merge investments with funding rounds
    investments_funding_rounds = investments_df.merge(
        funding_rounds_df, on="funding_round_uuid", how="left"
    )

    # Filter investments to only include top investors
    top_investments = investments_funding_rounds[
        investments_funding_rounds["investor_uuid"].isin(top_investors)
    ]

    # Count the number of top investors for each funding round
    top_investor_counts = (
        top_investments.groupby("startup_uuid")["investor_uuid"]
        .nunique()
        .reset_index()
        .rename(
            columns={
                "investor_uuid": f"top_{int(percent * 100)}_percent_investor_count"
            }
        )
    )

    # Merge the top investor counts back into temp_df
    temp_df = temp_df.merge(top_investor_counts, on="startup_uuid", how="left")
    temp_df = f.m_top_percent_investor_count(temp_df, percent)
    return temp_df


def days_until_first_funding(temp_df, funding_rounds_df):
    funding_rounds_df["announced_on"] = pd.to_datetime(
        funding_rounds_df["announced_on"], errors="coerce"
    )
    funding_rounds_df = funding_rounds_df.dropna(subset=["announced_on"])
    # Join both tables on ORGANIZATION UUID
    grouped_df = funding_rounds_df.loc[
        funding_rounds_df.groupby("startup_uuid")["announced_on"].idxmin()
    ]
    grouped_df = grouped_df.merge(
        temp_df, left_on="startup_uuid", right_on="startup_uuid", how="inner"
    )

    grouped_df = f.f_days_before_first_funding_old_companies(grouped_df)

    # Convert columns to correct type `datetime`
    grouped_df["founded_on"] = pd.to_datetime(grouped_df["founded_on"], errors="coerce")
    grouped_df["announced_on"] = pd.to_datetime(
        grouped_df["announced_on"], errors="coerce"
    )

    # Calculate the difference (in days) between first funding and the day the startup was founded
    grouped_df["days_until_first_funding"] = grouped_df.apply(
        lambda row: (
            (row["announced_on"] - row["founded_on"]).days
            if pd.notna(row["announced_on"]) and pd.notna(row["founded_on"])
            else np.nan
        ),
        axis=1,
    )
    grouped_df = f.f_days_unitl_first_funding(grouped_df)
    temp_df = temp_df.merge(
        grouped_df[["days_until_first_funding", "startup_uuid"]],
        on="startup_uuid",
        how="left",
    )
    return temp_df


def accelerator_participation_count(
    temp_df, investors_df, investments_df, funding_rounds_df
):
    investors = investors_df[["investor_uuid", "investor_types"]]
    investors_investment_merge = pd.merge(
        investments_df,
        investors,
        how="left",
        left_on="investor_uuid",
        right_on="investor_uuid",
    )
    investors_investment_merge.rename(
        columns={"investor_uuid": "investment_uuid"}, inplace=True
    )
    # investors_investment_merge.drop('uuid_y', axis=1, inplace=True)

    # Merge with funding_rounds to get startup_uuid
    funding_rounds = funding_rounds_df[["funding_round_uuid", "startup_uuid"]]
    ii_merge = pd.merge(
        investors_investment_merge,
        funding_rounds,
        how="left",
        left_on="funding_round_uuid",
        right_on="funding_round_uuid",
    )
    # ii_merge.drop('uuid', axis=1, inplace=True)
    investor_types_and_orgs = ii_merge[["startup_uuid", "investor_types"]]
    contains_accelerator = investor_types_and_orgs["investor_types"].str.contains(
        "accelerator", case=False
    )

    result = pd.concat([investor_types_and_orgs, contains_accelerator], axis=1)
    result.columns = ["startup_uuid", "investor_types", "contains_accelerator"]
    result = result[result["contains_accelerator"] == True]
    result["contains_accelerator"] = result["contains_accelerator"].astype(int)
    result.drop("investor_types", axis=1, inplace=True)

    grouped_result = (
        result.groupby("startup_uuid")["contains_accelerator"]
        .sum()
        .rename("accelerator_participation_count")
    )
    df_grouped_result = grouped_result.reset_index()
    temp_df = pd.merge(
        temp_df,
        df_grouped_result[["startup_uuid", "accelerator_participation_count"]],
        how="left",
        left_on="startup_uuid",
        right_on="startup_uuid",
    )
    temp_df = f.m_accelerator_part_count(temp_df)
    return temp_df


def startup_investor_types(temp_df, investors_df, funding_rounds_df):
    # Ensure lead_investors_uuids is a list, then explode it
    funding_rounds_df_exploded = funding_rounds_df.explode("lead_investor_uuids")

    # Merge funding rounds with investors to get investor types
    funding_with_investors = funding_rounds_df_exploded.merge(
        investors_df,
        left_on="lead_investor_uuids",
        right_on="investor_uuid",
        how="left",
    )

    # Merge with temp_df to get the companies
    funding_data = funding_with_investors.merge(
        temp_df, left_on="startup_uuid", right_on="startup_uuid", how="right"
    )

    # Aggregate investor types per company
    investor_types_per_company = funding_data.groupby("startup_uuid")[
        "investor_types"
    ].agg(lambda x: set(x.dropna()))

    # Map aggregated investor types back to temp_df
    temp_df["investor_type"] = temp_df["startup_uuid"].map(investor_types_per_company)

    # Define classification function
    def classify_funding(investor_types):
        if not investor_types:
            return "unknown"

        investor_types = set(investor_types)  # Ensure it's a set

        has_public = "government_office" in investor_types
        has_private = len(investor_types - {"government_office"}) > 0  # Any other type

        if has_public and has_private:
            return "hybrid"
        elif has_public:
            return "public"
        else:
            return "private"

    # Apply classification
    temp_df["investor_type"] = temp_df["investor_type"].apply(classify_funding)

    return temp_df


def investor_type_columns(investor_df, col="investor_types"):
    """
    Takes a DataFrame with a column of comma-separated investor types,
    and creates one boolean column per unique type.
    
    Parameters
    ----------
    investor_df : pd.DataFrame
        DataFrame containing the investor_types column.
    col : str
        Name of the column containing comma-separated investor types.
    
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional boolean columns for each investor type.
    """
    df = investor_df.copy()
    
    # 1. Split the comma-separated strings into lists
    df[col] = df[col].fillna("")  # replace NaN with empty string
    df[col + "_list"] = df[col].apply(lambda x: [t.strip() for t in x.split(",") if t.strip() != ""])
    
    # 2. Find all unique investor types across all rows
    unique_types = set([t for lst in df[col + "_list"] for t in lst])
    
    # 3. Create boolean column for each investor type
    for t in unique_types:
        df[f"is_{t}"] = df[col + "_list"].apply(lambda lst: t in lst)
    
    # 4. Drop the temporary list column
    df.drop(columns=[col + "_list"], inplace=True)
    
    return df


def funding_velocity_features(temp_df, funding_rounds_df, investments_df=None):
    """
    Computes funding velocity, growth, and advanced VC metrics.
    """
    # Ensure dates are datetime
    funding_rounds_df = funding_rounds_df.copy()
    funding_rounds_df["announced_on"] = pd.to_datetime(funding_rounds_df["announced_on"], errors="coerce")

    # --- Last Funding Stage Feature ---
    # Computed BEFORE dropna so rounds with missing amounts still count for stage.
    # Only needs startup_uuid + investment_type (no amount/date required).
    stage_map = {
        # --- Stage 1: The First Checks (Individuals / Angels) ---
        'angel': 1,               # User Source: Earliest stage
        'grant': 1,               # Non-dilutive, often first money in
        'non_equity_assistance': 1,
        'convertible_note': 1,    # Common instrument for Angels
        'product_crowdfunding': 1, # Kickstarter/Indiegogo (Revenue/Pre-product)

        # --- Stage 2: Early Institutional (Pre-Seed) ---
        'pre_seed': 2,            # User Source: Comes after Angel
        'equity_crowdfunding': 2, # Often acts as a formal Pre-Seed/Seed alternative

        # --- Stage 3: Validation (Seed) ---
        'seed': 3,                # The classic "First Institutional Round"

        # --- Stage 4: Early Venture (Product-Market Fit) ---
        'series_a': 4,
        'venture': 4,             # Mapped to Series A (Unspecified Venture Round)

        # --- Stage 5: Growth (Scale) ---
        'series_b': 5,
        'series_c': 6,
        'series_d': 7,
        'series_e': 8,
        'series_f': 9,
        'series_g': 10, 'series_h': 11, 'series_i': 12, 'series_j': 13,

        # --- Stage 6: Liquidity / Exit ---
        'private_equity': 14,     # Late stage buyout / growth
        'post_ipo_equity': 15,
        'post_ipo_debt': 15,
        'post_ipo_secondary': 15,

        # --- Zeros (No Ordinal Signal) ---
        'series_unknown': 0,
        'undisclosed': 0,
        'debt_financing': 0,
        'corporate_round': 0,
        'secondary_market': 0,
        'initial_coin_offering': 0
    }

    # Calculate Max Funding Stage (highest stage achieved) from ALL rounds
    temp_rounds = funding_rounds_df[["startup_uuid", "investment_type"]].copy()
    temp_rounds["stage_code"] = temp_rounds["investment_type"].map(stage_map).fillna(0).astype(int)
    max_stage_df = temp_rounds.groupby("startup_uuid")["stage_code"].max().reset_index()
    max_stage_df.rename(columns={"stage_code": "last_funding_stage"}, inplace=True)

    # --- Filter for velocity/growth features (need both date and amount) ---
    funding_rounds_df = funding_rounds_df.dropna(subset=["announced_on", "raised_amount_usd"])

    # Sort by date
    funding_rounds_df = funding_rounds_df.sort_values(["startup_uuid", "announced_on"])
    
    # 1. Velocity (Days between rounds)
    funding_rounds_df["prev_announced_on"] = funding_rounds_df.groupby("startup_uuid")["announced_on"].shift(1)
    funding_rounds_df["days_since_last"] = (funding_rounds_df["announced_on"] - funding_rounds_df["prev_announced_on"]).dt.days
    velocity_df = funding_rounds_df.groupby("startup_uuid")["days_since_last"].mean().reset_index(name="avg_days_between_rounds")
    
    # 2. Recency (Months since last funding)
    # Reference date = max announced_on in data (dataset end); avoids data leakage vs. "current time".
    max_date = funding_rounds_df["announced_on"].max()
    last_dates = funding_rounds_df.groupby("startup_uuid")["announced_on"].max().reset_index()
    last_dates["months_since_last_funding"] = (max_date - last_dates["announced_on"]).dt.days / 30.0
    
    # 3. Growth Rate & Burn Rate Proxy
    # Get first and last round amounts
    first_rounds = funding_rounds_df.groupby("startup_uuid").first().reset_index()[["startup_uuid", "raised_amount_usd", "announced_on"]]
    last_rounds = funding_rounds_df.groupby("startup_uuid").last().reset_index()[["startup_uuid", "raised_amount_usd", "announced_on"]]
    
    growth_df = first_rounds.merge(last_rounds, on="startup_uuid", suffixes=("_first", "_last"))
    
    # Growth Rate
    growth_df["raised_amount_usd_first"] = growth_df["raised_amount_usd_first"].replace(0, 1)
    growth_df["funding_growth_rate"] = (growth_df["raised_amount_usd_last"] - growth_df["raised_amount_usd_first"]) / growth_df["raised_amount_usd_first"]
    
    # Implied Monthly Burn (Last Amount / Months since last round)
    growth_df = growth_df.merge(last_dates[["startup_uuid", "months_since_last_funding"]], on="startup_uuid")
    growth_df["months_since_last_funding"] = growth_df["months_since_last_funding"].replace(0, 0.5) # avoid div by zero
    growth_df["implied_monthly_burn"] = growth_df["raised_amount_usd_last"] / growth_df["months_since_last_funding"]

    # 4. Down Round Indicator
    # Check if any round was smaller than the previous one
    funding_rounds_df["prev_raised_amount"] = funding_rounds_df.groupby("startup_uuid")["raised_amount_usd"].shift(1)
    funding_rounds_df["is_down_round"] = (funding_rounds_df["raised_amount_usd"] < funding_rounds_df["prev_raised_amount"])
    down_round_df = funding_rounds_df.groupby("startup_uuid")["is_down_round"].any().astype(int).reset_index(name="has_down_round")

    # 5. Investor Follow-on Ratio
    follow_on_df = pd.DataFrame(columns=["startup_uuid", "investor_follow_on_ratio"])
    if investments_df is not None:
        # Merge investments with funding rounds to link Investor -> Startup
        inv_rounds = investments_df.merge(funding_rounds_df[["funding_round_uuid", "startup_uuid"]], on="funding_round_uuid")
        
        # Count rounds per investor per startup
        inv_counts = inv_rounds.groupby(["startup_uuid", "investor_uuid"]).size().reset_index(name="round_count")
        
        # For each startup, count how many investors have round_count > 1
        inv_counts["is_follow_on"] = inv_counts["round_count"] > 1
        
        startup_follow_on = inv_counts.groupby("startup_uuid").agg(
            total_investors=("investor_uuid", "count"),
            follow_on_investors=("is_follow_on", "sum")
        ).reset_index()
        
        startup_follow_on["total_investors"] = startup_follow_on["total_investors"].replace(0, 1)
        startup_follow_on["investor_follow_on_ratio"] = startup_follow_on["follow_on_investors"] / startup_follow_on["total_investors"]
        follow_on_df = startup_follow_on[["startup_uuid", "investor_follow_on_ratio"]]

    # Merge all features back
    temp_df = temp_df.merge(velocity_df, on="startup_uuid", how="left")
    temp_df = temp_df.merge(growth_df[["startup_uuid", "funding_growth_rate", "months_since_last_funding", "implied_monthly_burn"]], on="startup_uuid", how="left")
    temp_df = temp_df.merge(down_round_df, on="startup_uuid", how="left")
    temp_df = temp_df.merge(follow_on_df, on="startup_uuid", how="left")
    
    # Merge last_funding_stage (computed above before dropna)
    temp_df = temp_df.merge(max_stage_df, on="startup_uuid", how="left")
    temp_df["last_funding_stage"] = temp_df["last_funding_stage"].fillna(0).astype(int)

    return temp_df
    