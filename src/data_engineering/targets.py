"""Target variable definitions for startup success: various combinations of acquisition, IPO, and Series A."""
def t_acq_ipo_seriesA(row):  # Ours
    success = 0
    if row.acquired or row.ipo or row.seriesA:
        success = 1
    return success


def t_acq_ipo(row):  # Ours without Series A
    success = 0
    if row.acquired or row.ipo:
        success = 1
    return success


def t_acq_acq_ipo_seriesA(row):  # Bonaventura et al. (2020) + Series A
    success = 0
    if row.acquirer or row.acquired or row.ipo or row.seriesA:
        success = 1
    return success


def t_acq_acq_ipo(row):  # Bonaventura et al. (2020), Braesemann et al. (2024)
    success = 0
    if row.acquirer or row.acquired or row.ipo:
        success = 1
    return success


def t_seriesA(row):  # Te et al. (2023) (XGBoost Reference Paper)
    success = 0
    if row.seriesA:
        success = 1
    return success


def t_acq_seriesA(row):  # Zhang et al. (2021) (GNN Paper)
    success = 0
    if row.acquired or row.seriesA:
        success = 1
    return success


def prepare_success(df, funding_rounds_df, acq_df):
    # create boolean series A column
    funding_df = funding_rounds_df[funding_rounds_df["investment_type"] == "series_a"]
    df["seriesA"] = df["startup_uuid"].isin(funding_df["startup_uuid"]).astype(int)

    # create resp. columns for acq and ipo
    df["acquired"] = df["dc_status"].apply(
        lambda status: 1 if status in ["acquired"] else 0
    )
    df["ipo"] = df["dc_status"].apply(lambda status: 1 if status in ["ipo"] else 0)
    df["closed"] = df["dc_status"].apply(
        lambda status: 1 if status in ["closed"] else 0
    )
    df["operating"] = df["dc_status"].apply(
        lambda status: 1 if status in ["operating"] else 0
    )

    # create acquirer column
    df["acquirer"] = df["startup_uuid"].isin(acq_df["acquirer_uuid"]).astype(int)

    return df


def add_future_status(temp_df, future_org_df):
    # Select only startup_uuid and future dc_status, rename to future_status before merge
    future_status_df = future_org_df[['uuid', 'status']].rename(columns={'status': 'future_status'})

    # Merge into temp_df without overwriting the original dc_status
    temp_df = temp_df.merge(future_status_df, left_on='startup_uuid', right_on='uuid', how='left')

    return temp_df


def add_next_funding_round(temp_df, funding_rounds_df, new_funding_rounds_df):
    # Step 1: Identify new funding round UUIDs
    past_round_uuids = set(funding_rounds_df['funding_round_uuid'])
    future_round_uuids = set(new_funding_rounds_df['uuid'])
    new_round_uuids = future_round_uuids - past_round_uuids

    # Step 2: Get the startup_uuids associated with new rounds
    new_rounds_only = new_funding_rounds_df[new_funding_rounds_df['uuid'].isin(new_round_uuids)]
    startups_with_new_rounds = set(new_rounds_only['startup_uuid'])

    # Step 3: Add boolean column
    temp_df['new_funding_round'] = temp_df['startup_uuid'].isin(startups_with_new_rounds)
    
    # Fill NaN with False (if any) and convert to int/bool
    temp_df['new_funding_round'] = temp_df['new_funding_round'].fillna(False).astype(int)

    return temp_df


def add_new_acquisitions(temp_df, acq_df, new_acq_df):
    # Step 1: Identify new acquisition UUIDs
    past_acq_uuids = set(acq_df['uuid'])
    future_acq_uuids = set(new_acq_df['uuid'])
    new_acq_uuids = future_acq_uuids - past_acq_uuids

    # Step 2: Get the startup_uuids (acquirees) associated with new acquisitions
    new_acqs_only = new_acq_df[new_acq_df['uuid'].isin(new_acq_uuids)]
    startups_with_new_acqs = set(new_acqs_only['acquiree_uuid'])

    # Step 3: Add boolean column
    temp_df['new_acquired'] = ((temp_df['future_status'] == 'acquired') | temp_df['startup_uuid'].isin(startups_with_new_acqs)).astype(int)

    return temp_df


def add_new_ipos(temp_df, ipo_df, new_ipo_df):
    # Step 1: Identify new IPO UUIDs
    past_ipo_uuids = set(ipo_df['uuid'])
    future_ipo_uuids = set(new_ipo_df['uuid'])
    new_ipo_uuids = future_ipo_uuids - past_ipo_uuids

    # Step 2: Get the startup_uuids associated with new IPOs
    new_ipos_only = new_ipo_df[new_ipo_df['uuid'].isin(new_ipo_uuids)]
    startups_with_new_ipos = set(new_ipos_only['startup_uuid'])

    # Step 3: Add boolean column
    temp_df['new_ipo'] = ((temp_df['future_status'] == 'ipo') | temp_df['startup_uuid'].isin(startups_with_new_ipos)).astype(int)

    return temp_df

def add_mixed_target(temp_df):
    temp_df['acq_ipo_funding'] = temp_df['new_acquired'] | temp_df['new_ipo'] | temp_df['new_funding_round']
    return temp_df
