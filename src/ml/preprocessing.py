"""Data preprocessing: loads CSV data, computes features, generates description embeddings, and builds PyG HeteroData."""
import os
import json as _json
import pandas as pd
import numpy as np
from torch import tensor, long, float32
import inspect
import time
from .utils import load_config, load_data, success_ratio, get_maturity_mask
from .graph_assembler import create_graph
from .feature_visualization import plot_embeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense
import torch
from pathlib import Path

config = load_config()

# Feature group definitions for startup feature ablation.
# Each group maps to the CSV column names it contains.
# Groups "description" and "graph_features" are handled via config flags.
STARTUP_FEATURE_GROUPS = {
    "team": [
        "founder_count", "female_count", "serial_founder_count",
        "featured_job_organization_uuid_x", "ivy_league_plus_count",
        "featured_job_organization_uuid_y", "bachelor_count", "business_degree_count",
        "it_degree_count", "law_degree_count", "master_count", "other_edu_count",
        "phd_count", "stem_degree_count", "female_ratio", "serial_founder_ratio",
        "ivy_league_ratio", "bachelor_ratio", "master_ratio", "phd_ratio",
        "stem_degree_ratio", "business_degree_ratio", "it_degree_ratio",
        "has_tech_and_biz",
    ],
    "funding_rounds": [
        "money_angel", "investors_angel", "angel_round",
        "money_pre_seed", "investors_pre_seed", "pre_seed_round",
        "money_seed", "investors_seed", "seed_round",
        "money_series_a", "investors_series_a", "series_a_round",
        "money_series_b", "investors_series_b", "series_b_round",
        "money_series_c", "investors_series_c", "series_c_round",
        "money_series_d", "investors_series_d", "series_d_round",
        "money_series_e", "investors_series_e", "series_e_round",
        "money_series_f", "investors_series_f", "series_f_round",
        "money_series_g", "investors_series_g", "series_g_round",
        "money_series_h", "investors_series_h", "series_h_round",
        "money_series_i", "investors_series_i", "series_i_round",
        "money_series_j", "investors_series_j", "series_j_round",
        "money_private_equity", "investors_private_equity", "private_equity_round",
        "money_venture_round", "investors_venture_round", "venture_round",
        "money_other_rounds", "investors_other_rounds", "other_round",
    ],
    # Kept in the Raw tier: CB-native summary fields and intrinsic temporal
    # trajectories the graph cannot derive without timestamped edge attributes.
    "temporal_intrinsic": [
        "num_funding_rounds", "total_funding_usd", "total_funding_currency_code",
        "last_funding_on", "last_funding_stage",
        "days_until_first_funding", "avg_days_between_rounds",
        "funding_growth_rate", "months_since_last_funding",
        "implied_monthly_burn", "has_down_round",
        "seriesA",
    ],
    # Dropped in the Raw tier: aggregates the GNN recomputes via message
    # passing over typed Startup-Investor edges.
    "investor_aggregates": [
        "total_investors", "top_10_percent_investor_count",
        "top_50_percent_investor_count", "accelerator_participation_count",
        "investor_type", "investor_follow_on_ratio",
    ],
    "online_presence": [
        "has_domain", "has_email", "has_phone",
        "has_facebook_url", "has_linkedin_url", "has_twitter_url",
        "has_logo_url", "type_organization",
    ],
    "description": [
        "has_description", "description_length",
    ],
    "graph_features": [],  # Handled via config flags, not column dropping
}

# Non-startup feature groups — mirrored from STARTUP_FEATURE_GROUPS to allow the
# intrinsic tier to extend beyond startup. "Aggregates" are features that are
# already summed / ranked at the source (Crunchbase native fields or our
# pre-graph derivations) and which a GNN ought to derive via message passing.
INVESTOR_FEATURE_GROUPS = {
    "aggregates": [
        # Native Crunchbase columns that pre-aggregate the investor's activity
        "investment_count",       # count of investments — should emerge from funding-edge aggregation
        "total_funding_usd",      # cumulative dollars deployed — ditto (sum over edge weights)
        "is_top_10_investor",     # ranking over total_funding_usd (leakage-adjacent if outcomes fed in)
    ],
}

FOUNDER_FEATURE_GROUPS = {
    "aggregates": [
        # Counts of a founder's startup-graph participation — derivable via
        # message-passing over founded_by edges.
        "num_ventures",
        "is_serial_founder",
    ],
}

def node_preprocessing(
    df,
    node_type,
    uuid_col,
    name_col=None,
    drop_cols=None,
    target_mode=None,  # 'multi_prediction', 'binary_prediction', or 'multi_task'
    multi_column=None,
    binary_column=None,
    special_preprocessing_fn=None,
    methods_params=None,
    config=None, # Added
):
    if config is None:
        config = load_config()

    # --- ABLATION: DROP NODE TYPE ---
    # If this node type is in the drop list, return empty DataFrame immediately
    drop_node_types = config["data_processing"].get("ablation", {}).get("drop_node_types", [])
    if node_type in drop_node_types:
        print(f"🚫 ABLATION: Dropping node type '{node_type}' completely.")
        # Return None to signal dropped node (handled by safe_node_process)
        return None

    df = df.copy()
    original_len = len(df)

    # Calculate NaN count early (before dropping columns or adding embeddings)
    # This ensures we capture the "missingness" of the original data
    # We exclude uuid and name columns from this count as they are always present/irrelevant for this metric
    nan_check_cols_raw = [c for c in df.columns if c not in [uuid_col, name_col, f"{node_type}_id"]]
    df["nan_count"] = df[nan_check_cols_raw].isna().sum(axis=1)

    # Special logic per node type
    if special_preprocessing_fn:
        sig = inspect.signature(special_preprocessing_fn)
        if len(sig.parameters) == 1:
            df = special_preprocessing_fn(df)
        else:
            df = special_preprocessing_fn(df, methods_params or {})

    # Determine if we need to protect a class column for ArcFace
    protected_cols = {name_col, uuid_col}
    retrieval_loss_type = config["train"]["loss"].get("retrieval_loss_type")
    contrastive_source = config["train"]["loss"].get("contrastive_positive_source", "text")
    
    if retrieval_loss_type == "arcface" or (retrieval_loss_type == "contrastive" and contrastive_source == "label"):
         arc_config = config["train"]["loss"].get("arcface", {})
         class_col = arc_config.get("class_column", "industry_groups")
         protected_cols.add(class_col)

    # Drop columns (protect uuid/name/class_col)
    drop_cols_clean = [
        col for col in (drop_cols or []) if col not in protected_cols
    ]
    df.drop(columns=drop_cols_clean, errors="ignore", inplace=True)

    # Drop rows where target is NaN
    if target_mode in {"multi_prediction", "binary_prediction", "multi_task"}:
        if multi_column and multi_column in df.columns:
            df = df[df[multi_column].notna()]
        if binary_column and binary_column in df.columns:
            df = df[df[binary_column].notna()]
            
    # Class-specific NaN filtering
    nan_config = config["data_processing"].get("nan_filtering", {})
    if nan_config.get("enabled", False) and target_mode in {"multi_prediction", "binary_prediction"}:
        print("\nApplying class-specific NaN filtering...")
        alphas = nan_config.get("alphas", {})
        
        # Determine class column and mapping
        class_col = None
        class_map = {} # map internal value to config key
        
        if target_mode == "multi_prediction":
            class_col = multi_column
        elif target_mode == "binary_prediction":
            class_col = binary_column
            # Map 1 -> funded, 0 -> not_funded
            class_map = {1: "funded", 0: "not_funded", True: "funded", False: "not_funded"}
            
        if class_col and class_col in df.columns:
            initial_len = len(df)
            rows_to_drop = []
            
            # Group by class to calculate stats
            # We need to handle the mapping for binary
            temp_class_series = df[class_col].copy()
            print(f"  Unique values in {class_col} before mapping: {temp_class_series.unique()}")
            
            if class_map:
                # Update map to handle floats if keys are ints
                extended_map = {}
                for k, v in class_map.items():
                    extended_map[k] = v
                    if isinstance(k, int):
                        extended_map[float(k)] = v
                
                temp_class_series = temp_class_series.map(extended_map)
            
            unique_classes = temp_class_series.unique()
            print(f"  Found classes for filtering: {unique_classes}")
            
            for cls in unique_classes:
                if pd.isna(cls): continue
                
                cls_str = str(cls)
                if cls_str in alphas:
                    alpha = alphas[cls_str]
                    cls_mask = temp_class_series == cls
                    cls_data = df[cls_mask]
                    
                    mu = cls_data["nan_count"].mean()
                    sigma = cls_data["nan_count"].std()
                    
                    if pd.isna(sigma): sigma = 0
                    
                    # Drop anything worse than the alpha-th percentile (nan_count > quantile(alpha))
                    # alpha=0.8 means we keep the best 80% (lowest nan_counts) and drop the worst 20%
                    threshold = cls_data["nan_count"].quantile(alpha)
                    
                    # Identify rows to drop: nan_count > threshold
                    drop_mask = (cls_data["nan_count"] > threshold)
                    drop_indices = cls_data[drop_mask].index
                    rows_to_drop.extend(drop_indices)
                    
                    print(f"  Class '{cls_str}': mu={mu:.2f}, sigma={sigma:.2f}, alpha={alpha}, T={threshold:.2f} (Dropping > {alpha:.0%}ile)")
                    print(f"    - Dropping {len(drop_indices)}/{len(cls_data)} rows ({len(drop_indices)/len(cls_data):.1%})")
                else:
                    print(f"  Class '{cls_str}': No alpha defined, keeping all.")
            
            if rows_to_drop:
                df.drop(index=rows_to_drop, inplace=True)
                print(f"  Total removed by class-specific filtering: {len(rows_to_drop)} ({len(rows_to_drop)/initial_len:.1%})")
            else:
                print("  No rows removed by class-specific filtering.")

    # Assign ID and reset index
    df.reset_index(drop=True, inplace=True)
    df[f"{node_type}_id"] = range(len(df))

    status_changed = None
    if target_mode == "multi_prediction":
        # Check if dc_status and future_status are different
        if multi_column == "future_status":
            status_changed = df["dc_status"] != df["future_status"]
        if (
            (multi_column == "future_status" or binary_column == "new_funding_round")
            and not config["data_processing"]["keep_dc_status"]
        ) or multi_column == "dc_status":
            df.drop(columns=["dc_status"], inplace=True, errors="ignore")

    # --- ARC FACE / RETRIEVAL LABEL PREPROCESSING ---
    # Encode industry/cluster column to integers for ArcFace classification
    # Only applicable to the target node type (startup)
    retrieval_loss_type = config["train"]["loss"].get("retrieval_loss_type")
    contrastive_source = config["train"]["loss"].get("contrastive_positive_source", "text")
    
    should_encode_labels = (
        (retrieval_loss_type == "arcface") or 
        (retrieval_loss_type == "contrastive" and contrastive_source == "label")
    )
    
    if node_type == "startup" and should_encode_labels:
        arc_config = config["train"]["loss"].get("arcface", {})
        # Default to industry_groups, but allow config override (e.g. for clusters)
        class_col = arc_config.get("class_column", "industry_groups")
        
        print(f"🔹 ArcFace: Preprocessing class column '{class_col}'...")
        if class_col in df.columns:

             # Handle list-like strings for ArcFace (need single-label)
             # "Software, SaaS, B2B" -> "Software"
             # Assuming format is comma-separated string or list
             def get_primary_label(val):
                 if pd.isna(val): return "Unknown"
                 s_val = str(val)
                 if "," in s_val:
                     return s_val.split(",")[0].strip()
                 # Handle list represented as string if necessary, but simple split is robust for standard CSVs
                 return s_val.strip()

             # Apply primary label extraction
             df[class_col] = df[class_col].apply(get_primary_label)
             
             # Encode
             df["retrieval_class_idx"] = df[class_col].astype("category").cat.codes
             num_ret_classes = df["retrieval_class_idx"].max() + 1
             print(f"   ✅ Encoded {num_ret_classes} retrieval classes from '{class_col}' (Primary Label Only)")

             # Verify we don't leak strings to GNN if this column was meant to be dropped
             # If class_col was in the original drop_cols (which we protected it from), we should now drop it.
             if drop_cols and class_col in drop_cols:
                 print(f"   🧹 Dropping raw '{class_col}' after encoding (cleanup)...")
                 df.drop(columns=[class_col], inplace=True)
             
             # Identify single-instance classes (cannot train ArcFace effectively on 1 sample)
             pass
        else:
            print(f"   ⚠️ Class column '{class_col}' not found! ArcFace will fail or strictly require available column.")
            print(f"   ⚠️ Available columns: {list(df.columns)}")
            
            # Create dummy to prevent crash, but warn heavily
            df["retrieval_class_idx"] = 0
            
    # Build name_df
    name_df_cols = [uuid_col, f"{node_type}_id"]
    if name_col and name_col in df.columns:
        name_df_cols.insert(1, name_col)
    name_df = df[name_df_cols].copy()

    # Save 'founded_on' column if exists
    founded_on_col = df["founded_on"].copy() if "founded_on" in df.columns else None

    # Drop features not used in model
    cols_config = config["data_processing"].get("multi_label", {}).get("columns", {})
    col_fund = cols_config.get("funding", "new_funding_round")
    col_acq = cols_config.get("acquisition", "new_acquired")
    col_ipo = cols_config.get("ipo", "new_ipo")

    feature_drop_cols = [
        f"{node_type}_id",
        uuid_col,
        name_col,
        multi_column,
        binary_column,
        col_fund,
        col_acq,
        col_ipo,
    ]
    feature_df = df.drop(
        columns=[col for col in feature_drop_cols if col in df.columns]
    )
    
    # Convert timestamp columns to datetime type
    timestamp_columns = ['created_at', 'updated_at', 'founded_on', 'dc_closed_on', 'closed_on', 'last_funding_on']
    for col in timestamp_columns:
        if col in feature_df.columns:
            feature_df[col] = pd.to_datetime(feature_df[col], errors='coerce')

    # Encode timestamps
    reference_date = pd.to_datetime(config["data_processing"].get("start_date", "2000-01-01"))
    for col in feature_df.select_dtypes(include=['datetime64[ns]']).columns:
        # Convert to days since reference date
        feature_df[col] = feature_df[col].apply(lambda x: (x.to_pydatetime() - reference_date).days if pd.notnull(x) else np.nan)
        # Replace NaT with NaN
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)

    # Encode categorical features
    for col in feature_df.select_dtypes(include=["object", "string"]).columns:
        feature_df[col] = (
            feature_df[col].astype("category").cat.codes.replace(-1, np.nan)
        )

    node_features = feature_df.astype("float32").to_numpy()
    feature_names = feature_df.columns.tolist()

    # Process targets
    target_multi, target_binary = None, None
    target_name = None
    if target_mode == "multi_prediction":
        target_name = multi_column
        target_multi = df[multi_column].astype("category").cat.codes
        target = tensor(target_multi.values, dtype=long)
    elif target_mode == "binary_prediction":
        target_name = binary_column
        target_binary = df[binary_column].fillna(0).astype(int)
        target = tensor(target_binary.values, dtype=float32)
    elif target_mode == "multi_task":
        target_name = (multi_column, binary_column)
        target_multi = df[multi_column].astype("category").cat.codes
        target_binary = df[binary_column].fillna(0).astype(int)
        target = (
            tensor(target_multi.values, dtype=long),
            tensor(target_binary.values, dtype=float32),
        )
    elif target_mode == "multi_label":
        # Multi-Label: [Funding, Acquisition, IPO]
        target_name = "multi_label (Fund, Acq, IPO)"
        
        # 1. Funding
        if col_fund not in df.columns: raise ValueError(f"Funding column '{col_fund}' missing")
        y_fund = df[col_fund].fillna(0).astype(int).values
        
        # 2. Acquisition
        if col_acq not in df.columns: raise ValueError(f"Acquisition column '{col_acq}' missing")
        y_acq = df[col_acq].fillna(0).astype(int).values
            
        # 3. IPO
        if col_ipo not in df.columns: raise ValueError(f"IPO column '{col_ipo}' missing")
        y_ipo = df[col_ipo].fillna(0).astype(int).values
            
        # Stack into [N, 3]
        target = tensor(np.stack([y_fund, y_acq, y_ipo], axis=1), dtype=float32)

    elif target_mode == "masked_multi_task":
        target_name = "masked_multi_task (Momentum, Liquidity)"
        
        # --- 1. Define Momentum Target ---
        # Logic: Pure Funding only. (IPO/Acq are treated as "Masked", not negative)
        if col_fund not in df.columns: raise ValueError(f"Funding column '{col_fund}' missing")
        y_mom = df[col_fund].fillna(0).astype(int).values
        
        # --- 2. Define Liquidity Target ---
        # Logic: Acquisition OR IPO
        if col_acq not in df.columns: raise ValueError(f"Acq column '{col_acq}' missing")
        if col_ipo not in df.columns: raise ValueError(f"IPO column '{col_ipo}' missing")
        y_liq = ((df[col_acq] == 1) | (df[col_ipo] == 1)).astype(int).values

        # --- 3. Define Masks ---
        # Maturity Logic:
        # - Age > 3 years (approx 1095 days)
        # - Total Funding > $5M
        # - Num Rounds > 2 (Series A/B proxy)
        
        # Check availability
        has_funding = "total_funding_usd" in df.columns
        has_rounds = "num_funding_rounds" in df.columns
        has_founded = "founded_on" in df.columns
        
        age_in_days = np.full(len(df), 0)
        if has_founded:
            snapshot_date = pd.Timestamp("2023-06-01")
            
            founded_dates = pd.to_datetime(df["founded_on"], errors='coerce')
            age_in_days = (snapshot_date - founded_dates).dt.days.fillna(0).values
        
        total_funding = df["total_funding_usd"].fillna(0).values if has_funding else np.zeros(len(df))
        num_rounds = df["num_funding_rounds"].fillna(0).values if has_rounds else np.zeros(len(df))
        
        # Thresholds
        AGE_THRESHOLD = 365 * 5 # 5 years (Optimized from Sweep)
        FUNDING_THRESHOLD = 1_000_000 # 1M USD (Optimized from Sweep)
        ROUNDS_THRESHOLD = 3 # 3 rounds (Optimized from Sweep)
        
        # --- PARAMETER SWEEP ANALYSIS ---
        print("\n" + "-"*80)
        print(f"MATURITY THRESHOLD SWEEP (Computed on filtered dataset of {len(df)} nodes)")
        print("-" * 80)
        print(f"{'Age (Yrs)':<10} | {'Funding ($)':<15} | {'Rounds':<8} | {'Mature Count':<12} | {'Mature Exits':<12} | {'Exit Ratio':<10}")
        print("-" * 80)
        
        sweep_ages = [2, 3, 5]
        sweep_funds = [1_000_000, 5_000_000, 10_000_000]
        sweep_rounds = [2, 3, 4]
        
        current_cfg = (5, 1_000_000, 3) # Updated "Current"
        scenarios = []
        for a in sweep_ages:
            for f in sweep_funds:
                for r in sweep_rounds:
                    scenarios.append((a, f, r))
        
        if current_cfg not in scenarios: scenarios.insert(0, current_cfg)
        
        for age_yr, fund_amt, rnd_cnt in scenarios:
             # Calculate mask for this scenario
             t_age = 365 * age_yr
             
             scen_mature = (
                (age_in_days > t_age) | 
                (total_funding > fund_amt) | 
                (num_rounds >= rnd_cnt)
            )
             
             # Count
             n_mat = scen_mature.sum()
             
             # Count exits among mature
             # y_liq is 1 for exit, 0 otherwise
             # We want sum(y_liq == 1 AND scen_mature == True)
             # y_liq is numpy array
             
             n_mat_exits = (y_liq[scen_mature].sum())
             ratio = n_mat_exits / n_mat if n_mat > 0 else 0.0
             
             marker = " (Current/Opt)" if (age_yr, fund_amt, rnd_cnt) == current_cfg else ""
             print(f"{age_yr:<10} | {fund_amt:<15,.0f} | {rnd_cnt:<8} | {n_mat:<12} | {n_mat_exits:<12} | {ratio:<10.2%}{marker}")

        print("-" * 80 + "\n")

        # Use helper for consistency with config and utils
        is_mature_series = get_maturity_mask(df, config)
        if is_mature_series is not None:
             is_mature = is_mature_series.values # Convert to numpy array
        else:
             print("⚠️ Strict Gating Mask could not be generated (check config). Defaulting to unmasked liquidity.")
             is_mature = np.ones(len(df))

        is_mature = is_mature.astype(int)
        
        # --- Mask 1: Momentum Mask ---
        # Rule: Mask (0) if Exited (Liquidity Event). Train (1) otherwise.
        # "Young/Mature Winner (Exit)" -> Masked.
        # "Young/Mature Winner (Funding)" -> Train.
        # "Young/Mature Failure" -> Train.
        mask_mom = 1 - y_liq 
        
        # --- Mask 2: Liquidity Mask (Strict Gating) ---
        # Rule: 
        # - STRICTLY MASK ALL Young Startups. 
        # - Only train if is_mature == 1.
        # - This avoids "Young = No Exit" bias.
        
        mask_liq = is_mature.astype(int)
        
        # Log Effective Ratios (Strict Gating)
        n_mature = mask_liq.sum()
        n_mature_exits = y_liq[mask_liq == 1].sum()
        effective_liq_ratio = n_mature_exits / n_mature if n_mature > 0 else 0.0
        
        print(f"\n📊 EFFECTIVE LIQUIDITY RATIO (Strict Gating):")
        print(f"   Mature Startups: {n_mature}")
        print(f"   Mature Exits:    {n_mature_exits}")
        print(f"   Ratio:           {effective_liq_ratio:.4%} (Target: ~1.86%)\n")
        
        target = tensor(np.stack([y_mom, y_liq, mask_mom, mask_liq], axis=1), dtype=float32)

        # Append Retrieval/ArcFace Labels if they exist
        if "retrieval_class_idx" in df.columns:
             # We need to pass this alongside the main target.
             # Since 'target' is used for DataLoader, sticking an int column into a float32 tensor is okay 
             # (we will cast back to long in loss).
             # Shape: [N, 5] -> Mom, Liq, MaskMom, MaskLiq, RetClass
             ret_labels = df["retrieval_class_idx"].values
             target = tensor(np.stack([y_mom, y_liq, mask_mom, mask_liq, ret_labels], axis=1), dtype=float32)

        
    elif target_mode is None:
        pass
    else:
        raise ValueError(f"Invalid target_mode: {target_mode}")

    has_target = target_mode in {"multi_prediction", "binary_prediction", "multi_task", "multi_label", "masked_multi_task"}
    if has_target:
        if isinstance(target, tuple):
             print(f"Prediction mode: {target_mode}, Target: {target_name}, Shape: (Binary: {target[1].shape}, Multi: {target[0].shape})")
        else:
             print(f"Prediction mode: {target_mode}, Target: {target_name}, Shape: {target.shape}")
        return (
            df,
            node_features,
            target,
            name_df,
            feature_names,
            founded_on_col,
            status_changed,
        )
    else:
        return df, node_features, name_df, feature_names


def merge_edge_df(first_df, second_df, edge_df, first_id_col, second_id_col, first_uuid_col, second_uuid_col, edge_first_col, edge_second_col):
    # Prepare right-side dataframes with renamed ID columns to avoid collisions
    right_first = first_df[[first_uuid_col, first_id_col]].rename(columns={first_id_col: "__src_node_id"})
    right_second = second_df[[second_uuid_col, second_id_col]].rename(columns={second_id_col: "__dst_node_id"})

    edge_df = pd.merge(
        edge_df,
        right_first,
        left_on=edge_first_col,
        right_on=first_uuid_col,
    )
    edge_df = pd.merge(
        edge_df,
        right_second,
        left_on=edge_second_col,
        right_on=second_uuid_col,
    )

    # Ensure no NaNs snuck in
    if edge_df[["__src_node_id", "__dst_node_id"]].isnull().any().any():
        raise ValueError(
            "NaNs found in edge index after merging. Some UUIDs are not matched to node IDs."
        )

    edge_index = edge_df[["__src_node_id", "__dst_node_id"]].values.transpose()
    edge_index = tensor(np.array(edge_index)).squeeze(0)

    return edge_index


def edge_preprocessing(
    edge_df, first_df, second_df, first_id_col, second_id_col, first_uuid_col, second_uuid_col, edge_first_col=None, edge_second_col=None, edge_attribute_names=None
):
    if edge_first_col is None:
        edge_first_col = first_uuid_col
    if edge_second_col is None:
        edge_second_col = second_uuid_col

    # --- ABLATION SUPPORT ---
    # If either node dataframe is None (dropped via ablation), we must drop the edges too.
    if first_df is None or second_df is None:
        # Return empty edge index (2, 0) and None attributes
        return torch.empty((2, 0), dtype=torch.long), None

    edge_index = merge_edge_df(
        first_df,
        second_df,
        edge_df,
        first_id_col, 
        second_id_col, 
        first_uuid_col, 
        second_uuid_col,
        edge_first_col,
        edge_second_col
    )

    edge_attributes = edge_df[edge_attribute_names] if edge_attribute_names else None

    return edge_index, edge_attributes


def drop_target_columns(startup_df):
    target_mode = config["data_processing"]["target_mode"]
    multi_col = config["data_processing"].get("multi_column")
    binary_col = config["data_processing"].get("binary_column")

    all_target_columns = [
        "acquired",
        "ipo",
        "closed",
        "operating",
        "future_status", 
        "new_funding_round",
        "t_acq_ipo_seriesA",
        "new_acquired",
        "new_ipo",
        "acq_ipo_funding"
    ]

    if target_mode == "multi_prediction":
        keep_targets = {multi_col}
    elif target_mode == "binary_prediction":
        keep_targets = {binary_col}
    elif target_mode == "multi_task":
        keep_targets = {multi_col, binary_col}
    elif target_mode == "multi_label":
        cols_config = config["data_processing"].get("multi_label", {}).get("columns", {})
        col_fund = cols_config.get("funding", "new_funding_round")
        col_acq = cols_config.get("acquisition", "acquired")
        col_ipo = cols_config.get("ipo", "ipo")
        keep_targets = {col_fund, col_acq, col_ipo}
    elif target_mode == "masked_multi_task":
        # Keep all ingredients needed for mask/target calculation
        cols_config = config["data_processing"].get("multi_label", {}).get("columns", {})
        col_fund = cols_config.get("funding", "new_funding_round")
        col_acq = cols_config.get("acquisition", "acquired")
        col_ipo = cols_config.get("ipo", "ipo")
        # Note: total_funding_usd and num_funding_rounds are not in 'all_target_columns' list so they won't be dropped here regardless.
        keep_targets = {col_fund, col_acq, col_ipo, "total_funding_usd", "num_funding_rounds", "founded_on", "retrieval_class_idx"}
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    drop_targets = [col for col in all_target_columns if col not in keep_targets]
    startup_df = startup_df.drop(columns=drop_targets, errors="ignore")

    return startup_df
    
def encode_description(df, params):
    """
    Encode descriptions using Sentence-BERT with dimension reduction (PCA) and CSV caching.
    Uses CSV files for efficient storage of embeddings.
    
    Args:
        df: DataFrame with 'description' column and UUID column
        params: Dictionary containing 'entity_type' ('org' or 'people')
        
    Returns:
        DataFrame with description encoded as vectors
    """
    from sklearn.decomposition import PCA
    
    entity_type = params.get("entity_type", "org")
    
    # Get target dimension from config
    # Can be int (fixed dim) or float (explained variance ratio)
    target_dim = config["data_processing"]["description_embedding_dim"]
    original_dim = 384  # Original dimension from 'all-MiniLM-L6-v2'
    
    # Setup cache directory
    cache_dir = Path("data/embeddings_csv")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Always cache the full 384d embeddings
    cache_file = cache_dir / f"{entity_type}_embeddings_original_{original_dim}d.csv"
    
    # Determine which UUID column to use
    uuid_col = 'startup_uuid' if entity_type == 'org' else 'founder_uuid'
    if uuid_col not in df.columns:
        raise ValueError(f"DataFrame must contain {uuid_col} column")
    
    print(f"\nProcessing {entity_type} descriptions (target_dim={target_dim})...")
    print(f"Cache file: {cache_file.name}")
    
    # Process only rows with descriptions
    valid_descriptions = df[pd.notna(df['description'])]
    
    # Prepare fast vectorized storage for embeddings
    n_rows = len(df)
    # preallocate matrix aligned to df index; use float32 to match cache dtype
    # We first load/compute full 384d embeddings
    full_embedding_matrix = np.full((n_rows, original_dim), np.nan, dtype=np.float32)

    # Build mapping from UUID to dataframe row index (only for rows that have descriptions)
    valid_idx = np.where(pd.notna(df['description']))[0]
    uuid_to_row = {df.iloc[int(i)][uuid_col]: int(i) for i in valid_idx}

    # Load existing cache if it exists
    cached_embeddings = {}
    if cache_file.exists() and cache_file.stat().st_size > 0:
        print("Loading embeddings from cache...")
        try:
            cached_df = pd.read_csv(cache_file, index_col='uuid')
        except pd.errors.EmptyDataError:
            cached_df = None

        if cached_df is not None:
            # Verify cached dimension matches expected
            embedding_cols = [col for col in cached_df.columns if col.startswith('emb_')]
            cached_dim = len(embedding_cols)

            if cached_dim != original_dim:
                print(f"WARNING: Cached dimension ({cached_dim}) doesn't match original ({original_dim})!")
                print("Regenerating...")
                cached_embeddings = {}
            else:
                # Only consider cached UUIDs that are present in this DF
                relevant_cached_uuids = [uuid for uuid in cached_df.index if uuid in uuid_to_row]

                print(f"Loading {len(relevant_cached_uuids)} relevant embeddings from cache...")

                # Load embeddings into matrix
                for uuid in relevant_cached_uuids:
                    row = uuid_to_row.get(uuid)
                    if row is not None:
                        # Extract embedding columns
                        embedding_cols = [f'emb_{i}' for i in range(original_dim)]
                        embedding_values = cached_df.loc[uuid, embedding_cols].values.astype(np.float32)
                        full_embedding_matrix[row, :] = embedding_values
                        cached_embeddings[uuid] = embedding_values

    # Identify which UUIDs still need to be processed (no embedding loaded)
    to_process = {}
    for uuid, row in uuid_to_row.items():
        # check first element for nan to decide if embedding present
        if np.isnan(full_embedding_matrix[row, 0]):
            to_process[uuid] = df.iloc[row]['description']
    
    print(f"Found {len(cached_embeddings)} cached embeddings")
    print(f"Need to process {len(to_process)} new descriptions")
    
    # If we have new descriptions to process
    if to_process:
        print("Processing new descriptions...")
        # Initialize base model
        base_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Process new descriptions
        descriptions = list(to_process.values())
        uuids = list(to_process.keys())
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = base_model.encode(
                descriptions,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True
            )
        
        # Convert to numpy
        embedding_array = embeddings.cpu().numpy()
        
        # Add new embeddings to matrix
        for idx_, uuid in enumerate(uuids):
            row = uuid_to_row.get(uuid)
            if row is not None:
                full_embedding_matrix[row, :] = embedding_array[idx_]
        
        # Create DataFrame with new embeddings for saving
        new_embeddings_data = {}
        new_embeddings_data['uuid'] = uuids
        for i in range(original_dim):
            new_embeddings_data[f'emb_{i}'] = embedding_array[:, i]
        
        new_embeddings_df = pd.DataFrame(new_embeddings_data)
        new_embeddings_df.set_index('uuid', inplace=True)
        
        # Append/update CSV cache
        if cache_file.exists() and cache_file.stat().st_size > 0:
            # Load existing cache and append new embeddings
            try:
                existing_df = pd.read_csv(cache_file, index_col='uuid')
            except pd.errors.EmptyDataError:
                existing_df = None
            
            if existing_df is not None:
                # Update existing or add new embeddings
                combined_df = pd.concat([existing_df, new_embeddings_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df.to_csv(cache_file)
            else:
                # Cache was empty/corrupt, write fresh
                new_embeddings_df.to_csv(cache_file)
        else:
            # Create new cache file
            new_embeddings_df.to_csv(cache_file)
            
        print(f"Cached {len(uuids)} new embeddings to CSV")
    
    # Now perform dimensionality reduction if needed
    # We collect all valid embeddings (ignoring NaNs from rows without description)
    valid_mask = ~np.isnan(full_embedding_matrix[:, 0])
    X = full_embedding_matrix[valid_mask]
    
    if len(X) == 0:
        print("No descriptions to encode.")
        return df

    final_embedding_matrix = full_embedding_matrix # Default to full if no reduction
    
    final_embedding_matrix = full_embedding_matrix # Default to full if no reduction
    
    if target_dim != original_dim:
        # Determine cache filename for reduced embeddings
        if isinstance(target_dim, float):
            dim_str = f"var_{target_dim}"
        else:
            dim_str = f"{target_dim}d"
            
        reduced_cache_file = cache_dir / f"{entity_type}_embeddings_{dim_str}.csv"
        
        # Try to load from cache first
        reduced_loaded = False
        if reduced_cache_file.exists():
            print(f"Loading reduced embeddings from cache: {reduced_cache_file.name}")
            try:
                reduced_df = pd.read_csv(reduced_cache_file, index_col='uuid')
                
                # Check if we have all needed UUIDs
                # We need embeddings for all valid_idx
                needed_uuids = [df.iloc[i][uuid_col] for i in valid_idx]
                missing_uuids = [u for u in needed_uuids if u not in reduced_df.index]
                
                if not missing_uuids:
                    # All present! Load them into final_embedding_matrix
                    # We need to know the dimension from the file
                    reduced_cols = [c for c in reduced_df.columns if c.startswith('emb_')]
                    actual_dim = len(reduced_cols)
                    
                    final_embedding_matrix = np.full((n_rows, actual_dim), np.nan, dtype=np.float32)
                    
                    # Fill matrix; filter reduced_df to needed uuids
                    reduced_subset = reduced_df.loc[needed_uuids]
                    
                    # We need to map back to row indices. 
                    # uuid_to_row has the mapping
                    row_indices = [uuid_to_row[u] for u in reduced_subset.index]
                    final_embedding_matrix[row_indices] = reduced_subset[reduced_cols].values
                    
                    embedding_dim = actual_dim
                    reduced_loaded = True
                    print(f"Successfully loaded {len(needed_uuids)} reduced embeddings.")
                else:
                    print(f"Cache missing {len(missing_uuids)} embeddings. Regenerating reduction...")
            except Exception as e:
                print(f"Failed to load reduced cache: {e}. Regenerating...")

        if not reduced_loaded:
            print(f"Applying PCA reduction...")
            
            # Determine n_components
            if isinstance(target_dim, float) and 0.0 < target_dim < 1.0:
                n_components = target_dim
                print(f"Target explained variance: {target_dim:.1%}")
            elif isinstance(target_dim, int) and target_dim > 0:
                n_components = min(target_dim, len(X), original_dim)
                print(f"Target fixed dimensions: {n_components}")
            else:
                raise ValueError(f"Invalid target_dim: {target_dim}")
                
            pca = PCA(n_components=n_components)
            nan_mask = np.isnan(X)
            if nan_mask.any():
                nan_count = nan_mask.any(axis=1).sum()
                print(f"Warning: {nan_count} rows contain NaN values; filling with 0 before PCA.")
                X = np.nan_to_num(X, nan=0.0)
            X_reduced = pca.fit_transform(X)
            
            actual_dim = X_reduced.shape[1]
            print(f"PCA reduced dimensions: {original_dim} -> {actual_dim}")
            print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
                
            # Create a new matrix for reduced embeddings
            final_embedding_matrix = np.full((n_rows, actual_dim), np.nan, dtype=np.float32)
            final_embedding_matrix[valid_mask] = X_reduced
            
            embedding_dim = actual_dim
            
            # Save to cache
            print(f"Saving reduced embeddings to {reduced_cache_file.name}...")
            # Reconstruct a DataFrame for saving; valid_idx = rows where description is
            # not null, which X corresponds to.
            
            # Get UUIDs for the valid rows
            # valid_idx is numpy array of indices
            valid_uuids = [df.iloc[i][uuid_col] for i in valid_idx]
            
            save_data = {}
            save_data['uuid'] = valid_uuids
            for i in range(actual_dim):
                save_data[f'emb_{i}'] = X_reduced[:, i]
                
            save_df = pd.DataFrame(save_data)
            save_df.set_index('uuid', inplace=True)
            save_df.to_csv(reduced_cache_file)
            
    else:
        print(f"Using original dimensions: {original_dim}")
        embedding_dim = original_dim

    # Create all embedding columns at once using pd.concat to avoid fragmentation
    print("Adding embeddings to dataframe...")
    embedding_cols = {f'desc_emb_{i}': final_embedding_matrix[:, i] for i in range(embedding_dim)}
    embedding_df = pd.DataFrame(embedding_cols, index=df.index)
    
    # Drop original description column first, then concat embedding columns
    df = df.drop(columns=['description'])
    df = pd.concat([df, embedding_df], axis=1)
    
    print(f"Description processing complete! Using {embedding_dim} dimensions.")
    
    # Visualize if enabled
    if config["data_processing"].get("visualize_embeddings", False):
        print("Visualizing embeddings...")
        output_dir = Path("outputs/embeddings")
        
        # Extract labels if available
        labels = None
        target_mode = config["data_processing"]["target_mode"]
        
        # Determine label column based on target_mode
        label_col = None
        if target_mode == "multi_prediction":
            label_col = config["data_processing"].get("multi_column")
        elif target_mode == "binary_prediction":
            label_col = config["data_processing"].get("binary_column")
        elif target_mode == "multi_task":
            # Just use multi_column for visualization as it's more informative
            label_col = config["data_processing"].get("multi_column")
            
        if label_col and label_col in df.columns:
            labels = df[label_col].values
            print(f"Using labels from '{label_col}' for visualization")
        
        # Plot full embeddings (384d)
        plot_embeddings(
            full_embedding_matrix, 
            f"{entity_type.capitalize()} Embeddings (Original {original_dim}d)", 
            output_dir / f"{entity_type}_embeddings_original.png",
            labels=labels
        )
        
        # Plot reduced embeddings if they are different
        if target_dim != original_dim:
            plot_embeddings(
                final_embedding_matrix, 
                f"{entity_type.capitalize()} Embeddings (Reduced {embedding_dim}d)", 
                output_dir / f"{entity_type}_embeddings_reduced.png",
                labels=labels
            )
            
    return df
    
def preprocess_startup_dates(df, params):
    start_date = pd.to_datetime(params.get("start_date", "2014-01-01"))
    end_date = pd.to_datetime(params.get("end_date", "2100-01-01"))
    column = params.get("column", "founded_on")

    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")
        df = df[(df[column] >= start_date) & (df[column] <= end_date)]

    return df

def remove_operating(startup_df):
    initial_count = len(startup_df)
    startup_df = startup_df[startup_df["dc_status"] != "operating"]
    startup_df = startup_df[startup_df["future_status"] != "operating"]
    final_count = len(startup_df)
    print(f"Removed {initial_count - final_count} 'operating' startups!")
    return startup_df


def filter_startups_without_connections(startup_df, startup_city_df, startup_sector_df, startup_founder_df):
    """
    Filter out startups that don't have any connections to cities, sectors, or founders.
    These startups would be isolated nodes in the graph with no meaningful relationships.
    """
    print("\n" + "=" * 50)
    print("FILTERING STARTUPS WITHOUT CONNECTIONS")
    print("=" * 50)
    
    initial_count = len(startup_df)
    startup_uuids = set(startup_df['startup_uuid'])
    
    # Find startups with connections
    city_connected = set(startup_city_df['startup_uuid']) if 'startup_uuid' in startup_city_df.columns else set()
    sector_connected = set(startup_sector_df['startup_uuid']) if 'startup_uuid' in startup_sector_df.columns else set()
    founder_connected = set(startup_founder_df['startup_uuid']) if 'startup_uuid' in startup_founder_df.columns else set()
    
    print(f"Initial startups: {initial_count}")
    print(f"Startups with city connections: {len(city_connected)}")
    print(f"Startups with sector connections: {len(sector_connected)}")
    print(f"Startups with founder connections: {len(founder_connected)}")
    
    # Count startups missing each type of connection
    missing_city = startup_uuids - city_connected
    missing_sector = startup_uuids - sector_connected
    missing_founder = startup_uuids - founder_connected
    
    print(f"Startups missing city connections: {len(missing_city)}")
    print(f"Startups missing sector connections: {len(missing_sector)}")
    print(f"Startups missing founder connections: {len(missing_founder)}")
    
    # Find startups missing ALL THREE types of connections (completely isolated)
    completely_isolated = missing_city & missing_sector & missing_founder
    
    # Find startups missing ANY TWO of the three connections (also problematic)
    missing_city_and_sector = missing_city & missing_sector
    missing_city_and_founder = missing_city & missing_founder
    missing_sector_and_founder = missing_sector & missing_founder
    
    # Startups with ALL THREE connections (city AND sector AND founder)
    fully_connected_startups = city_connected & sector_connected & founder_connected
    
    print(f"\nConnection analysis:")
    print(f"Completely isolated (no connections): {len(completely_isolated)}")
    print(f"Missing city AND sector: {len(missing_city_and_sector)}")
    print(f"Missing city AND founder: {len(missing_city_and_founder)}")
    print(f"Missing sector AND founder: {len(missing_sector_and_founder)}")
    print(f"Fully connected (all three): {len(fully_connected_startups)}")
    
    # Filter to keep only startups with ALL THREE connections
    # Check for case study override
    case_study_uuid = config.get("eval", {}).get("case_study_uuid")
    if case_study_uuid:
        if case_study_uuid in startup_uuids:
            if case_study_uuid not in fully_connected_startups:
                print(f"⚠️ Case Study Startup ({case_study_uuid}) is missing connections but will be KEPT by override.")
                fully_connected_startups.add(case_study_uuid)
            else:
                print(f"✅ Case Study Startup ({case_study_uuid}) is fully connected.")
        else:
             print(f"⚠️ Case Study Startup ({case_study_uuid}) was not found in the initial startup list!")

    filtered_startup_df = startup_df[startup_df['startup_uuid'].isin(fully_connected_startups)]
    
    final_count = len(filtered_startup_df)
    removed_count = initial_count - final_count
    
    print(f"\nFiltering results:")
    print(f"Startups removed: {removed_count}")
    print(f"Startups kept: {final_count}")
    print(f"Removal ratio: {removed_count/initial_count:.1%}")
    
    if removed_count > 0:
        print(f"\nRemoved {removed_count}/{initial_count} ({removed_count/initial_count:.1%}) startups missing ANY city/sector/founder connections")
    else:
        print("All startups have all three connections - no filtering needed")
    
    return filtered_startup_df


def create_identity_edges(df1, df2, id_col1, id_col2, uuid_col1, uuid_col2):
    """
    Creates edges between entities in df1 and df2 that share the same UUID.
    """
    # Find overlapping UUIDs
    common_uuids = set(df1[uuid_col1]).intersection(set(df2[uuid_col2]))
    
    if not common_uuids:
        return torch.empty((2, 0), dtype=torch.long)
        
    # Filter DFs to common UUIDs to speed up merge
    df1_common = df1[df1[uuid_col1].isin(common_uuids)][[uuid_col1, id_col1]]
    df2_common = df2[df2[uuid_col2].isin(common_uuids)][[uuid_col2, id_col2]]
    
    # Merge to align and handle duplicates (1:N, N:M)
    merged = pd.merge(
        df1_common, 
        df2_common, 
        left_on=uuid_col1, 
        right_on=uuid_col2
    )
    
    ids1 = merged[id_col1].values
    ids2 = merged[id_col2].values
    
    edge_index = torch.tensor([ids1, ids2], dtype=torch.long)
    return edge_index


def perform_preprocessing(
    startups_filename,
    investors_filename,
    founders_filename,
    cities_filename,
    university_filename,
    sectors_filename,
    startup_investor_filename,
    startup_city_filename,
    startup_founder_filename,
    startup_sector_filename,
    founder_university_filename,
    investor_city_filename,
    investor_sector_filename,
    university_city_filename,
    founder_investor_employment_filename=None,
    founder_coworking_filename=None,
    founder_investor_identity_filename=None,
    founder_co_study_filename=None,
    founder_board_filename=None,
    founder_startup_director_filename=None,
    founder_investor_director_filename=None,
    config=None,  # Added config parameter
):
    if config is None:
        config = load_config()

    # --- EDGE PREPROCESSING ---

    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    # --- Edge loading with enable_only override for ablation ---
    # If enable_only is set, only that edge type is loaded; all others disabled.
    edge_loading = config.get("data_processing", {}).get("edge_loading", {})
    enable_only = edge_loading.get("enable_only", None)
    _all_optional = ["founder_investor_employment", "founder_coworking",
                     "founder_investor_identity",
                     "founder_co_study", "founder_role_edges"]
    if enable_only == "none":
        edge_flags = {k: False for k in _all_optional}
        print(f"  [Edge Ablation] enable_only='none' → all optional edges disabled")
    elif enable_only:
        edge_flags = {k: (k == enable_only) for k in _all_optional}
        print(f"  [Edge Ablation] enable_only='{enable_only}' → {edge_flags}")
    else:
        edge_flags = {
            "founder_investor_employment": edge_loading.get("founder_investor_employment", False),
            "founder_coworking": edge_loading.get("founder_coworking", False),
            "founder_investor_identity": edge_loading.get("founder_investor_identity", False),
            "founder_co_study": edge_loading.get("founder_co_study", False),
            "founder_role_edges": edge_loading.get("founder_role_edges", True),
        }

    (
        startup_df,
        investor_df,
        founder_df,
        city_df,
        university_df,
        sector_df,
        startup_investor_df,
        startup_city_df,
        startup_founder_df,
        startup_sector_df,
        founder_university_df,
        investor_city_df,
        university_city_df,
        founder_investor_employment_df,
        founder_coworking_df,
        founder_investor_identity_df,
        founder_co_study_df,
        founder_board_df,
        founder_startup_director_df,
        founder_investor_director_df,
    ) = load_data(
        startups_filename=startups_filename,
        investors_filename=investors_filename,
        founders_filename=founders_filename,
        cities_filename=cities_filename,
        university_filename=university_filename,
        sectors_filename=sectors_filename,
        startup_investor_filename=startup_investor_filename,
        startup_city_filename=startup_city_filename,
        startup_founder_filename=startup_founder_filename,
        startup_sector_filename=startup_sector_filename,
        founder_university_filename=founder_university_filename,
        investor_city_filename=investor_city_filename,
        university_city_filename=university_city_filename,
        investor_sector_filename=investor_sector_filename,
        founder_investor_employment_filename=founder_investor_employment_filename if edge_flags["founder_investor_employment"] else None,
        founder_coworking_filename=founder_coworking_filename if edge_flags["founder_coworking"] else None,
        founder_investor_identity_filename=founder_investor_identity_filename if edge_flags["founder_investor_identity"] else None,
        founder_co_study_filename=founder_co_study_filename if edge_flags["founder_co_study"] else None,
        founder_board_filename=founder_board_filename if edge_flags["founder_role_edges"] else None,
        founder_startup_director_filename=founder_startup_director_filename if edge_flags["founder_role_edges"] else None,
        founder_investor_director_filename=founder_investor_director_filename if edge_flags["founder_role_edges"] else None,
    )

    # Remove unused target columns
    startup_df = drop_target_columns(startup_df)
    
    # Remove startups outside date range
    startup_df = preprocess_startup_dates(startup_df, config["data_processing"])
    
    # Remove 'operating' startups
    if config["data_processing"].get("remove_operating", False) and config["data_processing"]["target_mode"] == "multi_prediction":
        startup_df = remove_operating(startup_df)
    
    # Filter startups without critical connections
    startup_df = filter_startups_without_connections(
        startup_df, startup_city_df, startup_sector_df, startup_founder_df
    )

    print("\n" + "=" * 50)
    print("CREATING NODES AND EDGES")
    print("=" * 50)

    # --- GRAPH FEATURE ABLATION INDEX ---
    # Maps a single sweep index to individual graph feature flag combinations.
    # 0=none, 1=all, 2=louvain, 3=degree, 4=pagerank, 5=smart_money,
    # 6=louvain+degree+pagerank
    # Note: use_centrality_features (per-edge-type) disabled — causes OOM on A40.
    gf_idx = config["data_processing"].get("graph_feature_ablation_index")
    if gf_idx is not None:
        _GF_CONFIGS = [
            # (louvain, degree, pagerank, smart_money)
            (False, False, False, False),  # 0: none
            (True,  True,  True,  True),   # 1: all
            (True,  False, False, False),  # 2: louvain only
            (False, True,  False, False),  # 3: degree only
            (False, False, True,  False),  # 4: pagerank only
            (False, False, False, True),   # 5: smart_money only
            (True,  True,  True,  False),  # 6: louvain+degree+pagerank
        ]
        louv, deg, pr, sm = _GF_CONFIGS[int(gf_idx)]
        dp = config["data_processing"]
        dp["use_louvain_clusters"] = louv
        dp["use_degree_centrality"] = deg
        dp["use_pagerank_centrality"] = pr
        dp["use_centrality_features"] = False
        dp["use_smart_money_features"] = sm
        names = []
        if louv: names.append("louvain")
        if deg: names.append("degree")
        if pr: names.append("pagerank")
        if sm: names.append("smart_money")
        print(f"  GRAPH FEATURE ABLATION index={gf_idx}: {'+'.join(names) or 'none'}")

    # --- FEATURE INFORMATION LEVEL ---
    # "intrinsic" keeps only entity-intrinsic features (no relational/graph aggregations).
    # null/absent = full features (Level B, default).
    feature_level = config["data_processing"].get("ablation", {}).get("feature_information_level")
    if feature_level == "intrinsic":
        # "Raw" tier for the two-tier feature study: keep CB-native fields and
        # intrinsic temporal trajectories (temporal_intrinsic); drop all
        # aggregates the GNN can recompute via message passing.
        _relational_groups = ["team", "funding_rounds", "investor_aggregates", "graph_features"]
        existing_drops = list(config["data_processing"].get("ablation", {}).get("drop_feature_groups", []))
        for g in _relational_groups:
            if g not in existing_drops:
                existing_drops.append(g)
        config["data_processing"]["ablation"]["drop_feature_groups"] = existing_drops
        # Disable all graph-structural features
        for flag in ["use_louvain_clusters", "use_edge_counts", "use_degree_centrality",
                     "use_pagerank_centrality", "use_centrality_features", "use_smart_money_features"]:
            config["data_processing"][flag] = False
        print(f"  FEATURE LEVEL: intrinsic — dropping relational groups {_relational_groups} + graph features")

    # --- FEATURE GROUP ABLATION ---
    drop_feature_groups = config["data_processing"].get("ablation", {}).get("drop_feature_groups", [])
    startup_drop_cols = [
        "category_list", "edu_background", "country_code", "clustering",
        "name", "Unnamed: 0", "city", "category_groups_list", "dc_closed_on", "dc_status",
        "continent", "industries", "Solar", "Health Diagnostics",
        "Renewable Energy", "Environmental Engineering", "Waste Management",
        "Water", "Sustainability", "Recycling", "Energy Management",
        "Water Purification", "Secondary Education", "Natural Resources",
        "sustainable_group", "AgTech", "Energy Efficiency", "uuid",
        "description", "industry_groups",
    ]
    use_description = config["data_processing"].get("use_org_description", False)

    if drop_feature_groups:
        print(f"  FEATURE ABLATION: Dropping feature groups: {drop_feature_groups}")
        for group in drop_feature_groups:
            if group == "description":
                use_description = False
                startup_drop_cols.extend(STARTUP_FEATURE_GROUPS["description"])
            elif group == "graph_features":
                config["data_processing"]["use_louvain_clusters"] = False
                config["data_processing"]["use_edge_counts"] = False
                config["data_processing"]["use_degree_centrality"] = False
                config["data_processing"]["use_pagerank_centrality"] = False
                config["data_processing"]["use_centrality_features"] = False
                config["data_processing"]["use_smart_money_features"] = False
            elif group == "financial_aggregates":
                # Legacy alias (pre two-tier refactor): expand to both
                # successor groups so old configs still behave.
                startup_drop_cols.extend(STARTUP_FEATURE_GROUPS["temporal_intrinsic"])
                startup_drop_cols.extend(STARTUP_FEATURE_GROUPS["investor_aggregates"])
            elif group in STARTUP_FEATURE_GROUPS:
                startup_drop_cols.extend(STARTUP_FEATURE_GROUPS[group])
            else:
                print(f"  WARNING: Unknown feature group '{group}', ignoring.")

    # Preserve LLM-relevant columns before node_preprocessing drops them.
    # These columns are needed by PromptBuilder but not by GNN feature tensors.
    _llm_cols = ["name", "short_description", "category_list",
                 "category_groups_list", "city", "founded_on"]
    _llm_preserve = startup_df[["startup_uuid"] + [c for c in _llm_cols if c in startup_df.columns]].copy()

    (
        startup_df,
        startup_node_features,
        target,
        startup_name_df,
        startup_feature_names,
        founded_on_col,
        status_changed,
    ) = node_preprocessing(
        startup_df,
        node_type="startup",
        uuid_col="startup_uuid",
        name_col="name",
        drop_cols=startup_drop_cols,
        target_mode=config["data_processing"]["target_mode"],
        multi_column=config["data_processing"]["multi_column"],
        binary_column=config["data_processing"]["binary_column"],
        special_preprocessing_fn=encode_description if use_description else None,
        methods_params={"entity_type": "org"},
        config=config # Pass config
    )

    # Helper to handle potential None return (Ablation)
    def safe_node_process(func, *args, **kwargs):
        res = func(*args, **kwargs)
        if res is None:
            return None, None, None, None
        return res

    # If intrinsic tier requested, also strip the aggregate columns from the
    # non-startup feature sets. These features are either Crunchbase-native
    # pre-aggregates (e.g. investor.total_funding_usd, a sum over disclosed
    # investments) or ranking derivatives of those aggregates (is_top_10_investor).
    # A GNN should reconstruct these via message-passing over funding/founded_by
    # edges; prebaking them as node features short-circuits the graph signal.
    _intrinsic_tier = (
        config["data_processing"].get("ablation", {}).get("feature_information_level") == "intrinsic"
    )
    _investor_extra_drops = (
        INVESTOR_FEATURE_GROUPS["aggregates"] if _intrinsic_tier else []
    )
    _founder_extra_drops = (
        FOUNDER_FEATURE_GROUPS["aggregates"] if _intrinsic_tier else []
    )
    if _intrinsic_tier:
        print(f"  FEATURE LEVEL intrinsic: also dropping investor aggregates "
              f"{_investor_extra_drops} and founder aggregates {_founder_extra_drops}")

    investor_df, investor_node_features, investor_name_df, investor_feature_names = safe_node_process(
        node_preprocessing,
        investor_df,
        node_type="investor",
        uuid_col="investor_uuid",
        name_col="name",
        drop_cols=[
            "name",
            "country_code",
            "Unnamed: 0",
            "city",
            "city_uuid",
            "closed_on",
            *_investor_extra_drops,
        ],
        config=config # Pass config
    )

    founder_df, founder_node_features, founder_name_df, founder_feature_names = safe_node_process(
        node_preprocessing,
        founder_df,
        node_type="founder",
        uuid_col="founder_uuid",
        name_col="name",
        drop_cols=[
            "name",
            "description",
            "type",
            "permalink",
            "featured_job_organization_uuid",
            *_founder_extra_drops,
        ],
        special_preprocessing_fn=encode_description if config["data_processing"].get("use_people_description", False) else None,
        methods_params={"entity_type": "people"},
        config=config # Pass config
    )

    city_df, city_node_features, city_name_df, city_feature_names = safe_node_process(
        node_preprocessing,
        city_df,
        node_type="city",
        uuid_col="city_uuid",
        name_col="city",
        drop_cols=["city", "country_code", "Unnamed: 0"],
        config=config # Pass config
    )

    university_df, university_node_features, university_name_df, university_feature_names = safe_node_process(
        node_preprocessing,
        university_df,
        node_type="university",
        uuid_col="university_uuid",
        name_col="university_name",
        drop_cols=["university_name", "city", "city_uuid", "Unnamed: 0"],
        config=config # Pass config
    )

    sector_df, sector_node_features, sector_name_df, sector_feature_names = safe_node_process(
        node_preprocessing,
        sector_df,
        node_type="sector",
        uuid_col="sector_uuid",
        name_col="sector_name",
        drop_cols=["sector_name", "Unnamed: 0"],
        config=config # Pass config
    )

    # --- EDGE PREPROCESSING ---
    
    # 1. Split Funded By edges into stages
    print("Processing funded_by edges (splitting by stage)...")
    if "stage" in startup_investor_df.columns:
        early_df = startup_investor_df[startup_investor_df["stage"] == "early"]
        mid_df = startup_investor_df[startup_investor_df["stage"] == "mid"]
        late_df = startup_investor_df[startup_investor_df["stage"] == "late"]
        other_df = startup_investor_df[startup_investor_df["stage"] == "other"]
        
        startup_early_funded_edge_index, _ = edge_preprocessing(early_df, startup_df, investor_df, "startup_id", "investor_id", "startup_uuid", "investor_uuid")
        startup_mid_funded_edge_index, _ = edge_preprocessing(mid_df, startup_df, investor_df, "startup_id", "investor_id", "startup_uuid", "investor_uuid")
        startup_late_funded_edge_index, _ = edge_preprocessing(late_df, startup_df, investor_df, "startup_id", "investor_id", "startup_uuid", "investor_uuid")
        startup_other_funded_edge_index, _ = edge_preprocessing(other_df, startup_df, investor_df, "startup_id", "investor_id", "startup_uuid", "investor_uuid")
    else:
        # Fallback if stage column missing (e.g. old data)
        print("WARNING: 'stage' column missing in startup_investor_edges. Using generic funded_by.")
        startup_early_funded_edge_index, _ = edge_preprocessing(startup_investor_df, startup_df, investor_df, "startup_id", "investor_id", "startup_uuid", "investor_uuid")
        startup_mid_funded_edge_index = torch.empty((2, 0), dtype=torch.long)
        startup_late_funded_edge_index = torch.empty((2, 0), dtype=torch.long)
        startup_other_funded_edge_index = torch.empty((2, 0), dtype=torch.long)

    # 2. Standard edges
    startup_city_edge_index, startup_city_edge_attributes = edge_preprocessing(
        startup_city_df, startup_df, city_df, "startup_id", "city_id", "startup_uuid", "city_uuid"
    )
    startup_founder_edge_index, startup_founder_edge_attributes = edge_preprocessing(
        startup_founder_df, startup_df, founder_df, "startup_id", "founder_id", "startup_uuid", "founder_uuid"
    )
    startup_sector_edge_index, startup_sector_edge_attributes = edge_preprocessing(
        startup_sector_df, startup_df, sector_df, "startup_id", "sector_id", "startup_uuid", "sector_uuid"
    )
    founder_university_edge_index, founder_university_edge_attributes = edge_preprocessing(
        founder_university_df, founder_df, university_df, "founder_id", "university_id", "founder_uuid", "university_uuid"
    )
    investor_city_edge_index, investor_city_edge_attributes = edge_preprocessing(
        investor_city_df, investor_df, city_df, "investor_id", "city_id", "investor_uuid", "city_uuid"
    )
    university_city_edge_index, university_city_edge_attributes = edge_preprocessing(
        university_city_df, university_df, city_df, "university_id", "city_id", "university_uuid", "city_uuid"
    )

    # 3. New Edges
    print("Processing new edges (employment, coworking, identity)...")
    if not founder_investor_employment_df.empty:
        founder_investor_employment_edge_index, _ = edge_preprocessing(
            founder_investor_employment_df, founder_df, investor_df, "founder_id", "investor_id", "founder_uuid", "investor_uuid"
        )
    else:
        founder_investor_employment_edge_index = torch.empty((2, 0), dtype=torch.long)

    if not founder_coworking_df.empty:
        founder_coworking_edge_index, _ = edge_preprocessing(
            founder_coworking_df, 
            founder_df, 
            founder_df, 
            "founder_id", 
            "founder_id", 
            "founder_uuid", 
            "founder_uuid",
            edge_first_col="founder_uuid_1",
            edge_second_col="founder_uuid_2"
        )
    else:
        founder_coworking_edge_index = torch.empty((2, 0), dtype=torch.long)

    if not founder_investor_identity_df.empty:
        founder_investor_identity_edge_index, _ = edge_preprocessing(
            founder_investor_identity_df, founder_df, investor_df, "founder_id", "investor_id", "founder_uuid", "investor_uuid"
        )
    else:
        founder_investor_identity_edge_index = torch.empty((2, 0), dtype=torch.long)

    if not founder_co_study_df.empty:
        founder_co_study_edge_index, _ = edge_preprocessing(
            founder_co_study_df, 
            founder_df, 
            founder_df, 
            "founder_id", 
            "founder_id", 
            "founder_uuid", 
            "founder_uuid",
            edge_first_col="founder_uuid_1",
            edge_second_col="founder_uuid_2"
        )
    else:
        founder_co_study_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # --- Role Edges ---
    # 1. Board Member (Founder -> Startup)
    if not founder_board_df.empty:
        founder_board_edge_index, _ = edge_preprocessing(
            founder_board_df, founder_df, startup_df, 
            "founder_id", "startup_id", "founder_uuid", "startup_uuid",
            edge_first_col="founder_uuid", edge_second_col="startup_uuid"
        )
    else:
        founder_board_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # 2. Startup Director (Founder -> Startup)
    if not founder_startup_director_df.empty:
        founder_startup_director_edge_index, _ = edge_preprocessing(
            founder_startup_director_df, founder_df, startup_df,
            "founder_id", "startup_id", "founder_uuid", "startup_uuid",
            edge_first_col="founder_uuid", edge_second_col="startup_uuid"
        )
    else:
        founder_startup_director_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # 3. Investor Director (Founder -> Investor)
    if not founder_investor_director_df.empty:
        founder_investor_director_edge_index, _ = edge_preprocessing(
            founder_investor_director_df, founder_df, investor_df,
            "founder_id", "investor_id", "founder_uuid", "investor_uuid",
            edge_first_col="founder_uuid", edge_second_col="investor_uuid"
        )
    else:
        founder_investor_director_edge_index = torch.empty((2, 0), dtype=torch.long)

    print(f"\nThe success ratio is: {success_ratio(startup_df, config)}")

    print("\n" + "=" * 50)
    print("CREATING GRAPH")
    
    feature_names = {
        "startup": startup_feature_names,
        "investor": investor_feature_names,
        "founder": founder_feature_names,
        "city": city_feature_names,
        "university": university_feature_names,
        "sector": sector_feature_names,
    }

    node_names = {
        "startup": startup_name_df,
        "investor": investor_name_df,
        "founder": founder_name_df,
        "city": city_name_df,
        "university": university_name_df,
        "sector": sector_name_df,
    }

    graph_data, params = create_graph(
        startup_df,
        config=config,  # Pass config explicitly
        params={
            "startup_node_features": startup_node_features,
            "investor_node_features": investor_node_features,
            "founder_node_features": founder_node_features,
            "city_node_features": city_node_features,
            "university_node_features": university_node_features,
            "sector_node_features": sector_node_features,
            
            # Split funded_by edges
            "startup_early_funded_edge_index": startup_early_funded_edge_index,
            "startup_mid_funded_edge_index": startup_mid_funded_edge_index,
            "startup_late_funded_edge_index": startup_late_funded_edge_index,
            "startup_other_funded_edge_index": startup_other_funded_edge_index,
            
            "startup_city_edge_index": startup_city_edge_index,
            "startup_city_edge_attributes": startup_city_edge_attributes,
            "startup_founder_edge_index": startup_founder_edge_index,
            "startup_founder_edge_attributes": startup_founder_edge_attributes,
            "startup_sector_edge_index": startup_sector_edge_index,
            "startup_sector_edge_attributes": startup_sector_edge_attributes,
            "founder_university_edge_index": founder_university_edge_index,
            "founder_university_edge_attributes": founder_university_edge_attributes,
            "investor_city_edge_index": investor_city_edge_index,
            "investor_city_edge_attributes": investor_city_edge_attributes,
            "investor_sector_edge_index": None,
            "investor_sector_edge_attributes": None,
            "university_city_edge_index": university_city_edge_index,
            "university_city_edge_attributes": university_city_edge_attributes,
            
            # New edges
            "founder_investor_employment_edge_index": founder_investor_employment_edge_index,
            "founder_coworking_edge_index": founder_coworking_edge_index,
            "founder_investor_identity_edge_index": founder_investor_identity_edge_index,
            "founder_co_study_edge_index": founder_co_study_edge_index,
            "founder_board_edge_index": founder_board_edge_index,
            "founder_startup_director_edge_index": founder_startup_director_edge_index,
            "founder_investor_director_edge_index": founder_investor_director_edge_index,
            
            "startup_id": startup_df.startup_id,
            "investor_id": investor_df.investor_id if investor_df is not None else None,
            "target": target,
            "use_edge_attributes": False,
            "founded_on_col": founded_on_col,
            "feature_names": feature_names,
            "node_names": node_names,
            "status_changed": status_changed,
        },
    )
    
    # Store the filtered startup DataFrame on graph_data for LLM baseline access.
    # Row order matches graph node indices exactly (node i = startup_df.iloc[i]).
    # Merge back LLM-relevant columns that were dropped during node_preprocessing.
    _raw = startup_df.reset_index(drop=True)
    _llm_filtered = _llm_preserve[_llm_preserve["startup_uuid"].isin(_raw["startup_uuid"])]
    _llm_filtered = _llm_filtered.set_index("startup_uuid").reindex(_raw["startup_uuid"].values).reset_index()
    for col in _llm_filtered.columns:
        if col != "startup_uuid" and col not in _raw.columns:
            _raw[col] = _llm_filtered[col].values
    graph_data["startup"].raw_df = _raw

    print("\n" + "=" * 50)
    print("FEATURES USED")
    print("=" * 50)
    print(params['feature_names'])

    # Export startup degree dict for post-hoc degree-stratified evaluation
    try:
        num_startups = graph_data['startup'].x.shape[0]
        degree_counts = [0] * num_startups
        for edge_type in graph_data.edge_types:
            src_type, _, dst_type = edge_type
            ei = graph_data[edge_type].edge_index
            if src_type == 'startup':
                for idx in ei[0].tolist():
                    if idx < num_startups:
                        degree_counts[idx] += 1
            if dst_type == 'startup':
                for idx in ei[1].tolist():
                    if idx < num_startups:
                        degree_counts[idx] += 1
        # Map UUID → degree
        degree_dict = {}
        for i, row in _raw.iterrows():
            degree_dict[row['startup_uuid']] = degree_counts[i]
        output_dir = config.get("output_dir", "outputs")
        seed = config.get("seed", 0)
        degree_dir = os.path.join(output_dir, "degree_dicts")
        os.makedirs(degree_dir, exist_ok=True)
        degree_path = os.path.join(degree_dir, f"startup_degrees_seed{seed}.json")
        with open(degree_path, "w") as _f:
            _json.dump(degree_dict, _f)
        print(f"📁 Saved startup degree dict ({len(degree_dict)} nodes) to {degree_path}")
    except Exception as _e:
        print(f"⚠️ Failed to export degree dict: {_e}")

    # A4: optionally augment startup features with per-channel training-neighbor label means
    if config.get("data_processing", {}).get("use_neighbor_label_features", False):
        from src.ml.neighbor_label_features import add_neighbor_label_features
        graph_data = add_neighbor_label_features(graph_data, config)

    return graph_data, node_names
