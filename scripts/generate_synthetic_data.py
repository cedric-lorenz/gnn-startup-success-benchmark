"""
Generate synthetic data for the startup success prediction pipeline.

Creates realistic-looking fake data in data/graph/ so that the full training
pipeline can be run without access to proprietary Crunchbase data.

Usage:
    python scripts/generate_synthetic_data.py [--output-dir data/graph] [--n-startups 1000] [--seed 42]
"""

import argparse
import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path


def make_uuid():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COUNTRIES = ["USA", "GBR", "DEU", "FRA", "CAN", "IND", "ISR", "SGP", "AUS", "BRA"]
COUNTRY_WEIGHTS = [0.45, 0.10, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.03, 0.08]

CITIES = [
    ("San Francisco", "USA"), ("New York", "USA"), ("Boston", "USA"),
    ("Los Angeles", "USA"), ("Austin", "USA"), ("Seattle", "USA"),
    ("Chicago", "USA"), ("Denver", "USA"), ("Miami", "USA"),
    ("London", "GBR"), ("Cambridge", "GBR"),
    ("Berlin", "DEU"), ("Munich", "DEU"),
    ("Paris", "FRA"), ("Toronto", "CAN"), ("Vancouver", "CAN"),
    ("Bangalore", "IND"), ("Mumbai", "IND"),
    ("Tel Aviv", "ISR"), ("Singapore", "SGP"),
    ("Sydney", "AUS"), ("Melbourne", "AUS"),
    ("São Paulo", "BRA"),
    ("Atlanta", "USA"), ("Washington D.C.", "USA"),
    ("Amsterdam", "NLD"), ("Stockholm", "SWE"),
    ("Zurich", "CHE"), ("Dublin", "IRL"), ("Hong Kong", "HKG"),
]

CONTINENTS = {
    "USA": "north_america", "CAN": "north_america",
    "GBR": "europe", "DEU": "europe", "FRA": "europe",
    "NLD": "europe", "SWE": "europe", "CHE": "europe", "IRL": "europe",
    "IND": "asia", "ISR": "asia", "SGP": "asia", "HKG": "asia",
    "AUS": "oceania",
    "BRA": "south_america",
}

UNIVERSITIES = [
    ("Stanford University", "San Francisco"),
    ("MIT", "Boston"),
    ("Harvard University", "Boston"),
    ("University of Cambridge", "Cambridge"),
    ("Oxford University", "London"),
    ("ETH Zurich", "Zurich"),
    ("TU Munich", "Munich"),
    ("IIT Bombay", "Mumbai"),
    ("University of Toronto", "Toronto"),
    ("Tsinghua University", "Hong Kong"),
    ("Tel Aviv University", "Tel Aviv"),
    ("UC Berkeley", "San Francisco"),
    ("Columbia University", "New York"),
    ("University of Washington", "Seattle"),
    ("Georgia Tech", "Atlanta"),
    ("NYU", "New York"),
    ("Imperial College London", "London"),
    ("National University of Singapore", "Singapore"),
    ("University of Sydney", "Sydney"),
    ("Hasso Plattner Institute", "Berlin"),
]

SECTORS = [
    "Artificial Intelligence", "Cloud Computing", "Fintech",
    "Enterprise Software", "Healthcare", "E-Commerce",
    "Cybersecurity", "Blockchain", "EdTech", "CleanTech",
    "Biotechnology", "SaaS", "IoT", "Robotics",
    "Developer Tools", "Data Analytics", "Social Media",
    "MarketPlace", "Gaming", "Logistics",
]

EMPLOYEE_COUNTS = ["1-10", "11-50", "51-100", "101-250", "251-500",
                   "501-1000", "1001-5000", "5001-10000", "10000+", "unknown"]
EMPLOYEE_WEIGHTS = [0.35, 0.30, 0.10, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01, 0.07]

STAGES = ["early", "mid", "late", "other"]
STAGE_WEIGHTS = [0.40, 0.25, 0.15, 0.20]

DEGREE_TYPES = ["BS", "MS", "MBA", "PhD", "Other"]
DEGREE_WEIGHTS = [0.35, 0.30, 0.15, 0.10, 0.10]

TITLES = ["Co-Founder & CEO", "Co-Founder & CTO", "Founder & CEO",
          "Co-Founder", "Founder", "Co-Founder & COO"]

INVESTOR_TYPES_LIST = [
    "is_venture_capital", "is_angel", "is_accelerator",
    "is_corporate_venture_capital", "is_private_equity_firm",
    "is_micro_vc", "is_family_investment_office",
]


def generate_cities(rng):
    """Generate city nodes."""
    rows = []
    city_uuids = {}
    for city_name, country_code in CITIES:
        uid = make_uuid()
        city_uuids[(city_name, country_code)] = uid
        continent = CONTINENTS.get(country_code, "unknown")
        continent_cols = {f"continent_{c}": 0 for c in
                         ["north_america", "asia", "europe", "oceania",
                          "unknown", "africa", "south_america", "antarctica"]}
        continent_cols[f"continent_{continent}"] = 1
        rows.append({
            "city": city_name,
            "country_code": country_code,
            **continent_cols,
            "city_uuid": uid,
            "population": int(rng.lognormal(12, 1.5)),
            "latitude": rng.uniform(-40, 60),
            "longitude": rng.uniform(-120, 140),
            "has_city_data": 1,
        })
    return pd.DataFrame(rows), city_uuids


def generate_universities(city_uuids, rng):
    """Generate university nodes."""
    rows = []
    uni_uuids = {}
    for uni_name, city_name in UNIVERSITIES:
        uid = make_uuid()
        uni_uuids[uni_name] = uid
        # Find matching city UUID
        matching = [v for k, v in city_uuids.items() if k[0] == city_name]
        c_uid = matching[0] if matching else make_uuid()
        rows.append({
            "university_uuid": uid,
            "university_name": uni_name,
            "city": city_name,
            "city_uuid": c_uid,
        })
    return pd.DataFrame(rows), uni_uuids


def generate_sectors():
    """Generate sector nodes."""
    rows = []
    sector_uuids = {}
    for name in SECTORS:
        uid = str(rng.integers(10**16, 10**17))
        sector_uuids[name] = uid
        rows.append({"sector_name": name, "sector_uuid": uid})
    return pd.DataFrame(rows), sector_uuids


def generate_investors(n, city_uuids, rng):
    """Generate investor nodes."""
    city_keys = list(city_uuids.keys())
    rows = []
    inv_uuids = []
    for _ in range(n):
        uid = make_uuid()
        inv_uuids.append(uid)
        city_key = city_keys[rng.integers(len(city_keys))]
        # Random investor type flags
        type_flags = {t: 0 for t in [
            "is_angel_group", "is_micro_vc", "is_incubator", "is_hedge_fund",
            "is_university_program", "is_corporate_venture_capital",
            "is_investment_bank", "is_fund_of_funds", "is_angel",
            "is_co_working_space", "is_entrepreneurship_program",
            "is_private_equity_firm", "is_accelerator", "is_pension_funds",
            "is_investment_partner", "is_venture_debt", "is_startup_competition",
            "is_government_office", "is_family_investment_office",
            "is_venture_capital", "is_secondary_purchaser", "is_syndicate",
        ]}
        # Assign 1-2 types
        primary_type = rng.choice(INVESTOR_TYPES_LIST)
        type_flags[primary_type] = 1
        is_org = int(rng.random() > 0.3)
        inv_count = int(rng.lognormal(2, 1.5))

        year = int(rng.integers(1990, 2023))
        rows.append({
            "investor_uuid": uid,
            "name": f"Investor_{_}",
            "type": "organization" if is_org else "person",
            "created_at": f"{year}-01-15 10:00:00",
            "updated_at": f"{year+1}-06-15 10:00:00",
            "country_code": city_key[1],
            "city": city_key[0],
            "investment_count": inv_count,
            "total_funding_usd": float(rng.lognormal(16, 2)) if rng.random() > 0.3 else np.nan,
            "total_funding_currency_code": "USD",
            "founded_on": f"{year}-01-01" if rng.random() > 0.3 else np.nan,
            "closed_on": np.nan,
            "is_org": is_org,
            "is_top_10_investor": int(rng.random() < 0.05),
            "founded_on_year": year if rng.random() > 0.3 else np.nan,
            "has_domain": int(rng.random() > 0.2),
            "has_email": int(rng.random() > 0.5),
            "has_phone": int(rng.random() > 0.7),
            "has_facebook_url": int(rng.random() > 0.4),
            "has_linkedin_url": int(rng.random() > 0.3),
            "has_twitter_url": int(rng.random() > 0.4),
            "has_logo_url": int(rng.random() > 0.3),
            **type_flags,
            "city_uuid": city_uuids[city_key],
        })
    return pd.DataFrame(rows), inv_uuids


def generate_founders(n, rng):
    """Generate founder nodes."""
    rows = []
    founder_uuids = []
    for i in range(n):
        uid = make_uuid()
        founder_uuids.append(uid)
        is_serial = int(rng.random() < 0.15)
        num_ventures = int(rng.integers(1, 4)) if is_serial else 1
        is_female = int(rng.random() < 0.2)
        has_desc = int(rng.random() > 0.4)
        rows.append({
            "founder_uuid": uid,
            "name": f"Founder_{i}",
            "type": "person",
            "description": f"Experienced entrepreneur in tech." if has_desc else np.nan,
            "has_description": has_desc,
            "description_length": rng.integers(50, 500) if has_desc else 0,
            "num_ventures": num_ventures,
            "is_serial_founder": is_serial,
            "ivy_league_plus_count": int(rng.random() < 0.1),
            "bachelor_count": int(rng.random() < 0.7),
            "business_degree_count": int(rng.random() < 0.2),
            "it_degree_count": int(rng.random() < 0.3),
            "law_degree_count": int(rng.random() < 0.05),
            "master_count": int(rng.random() < 0.35),
            "other_edu_count": int(rng.random() < 0.15),
            "phd_count": int(rng.random() < 0.08),
            "stem_degree_count": int(rng.random() < 0.4),
            "is_female": is_female,
        })
    return pd.DataFrame(rows), founder_uuids


def generate_startups(n, founder_uuids, inv_uuids, rng, feature_noise=1.2):
    """Generate startup nodes with planted signal for targets.

    Also returns the per-startup latent quality so the edge generators can
    form quality-assortative (homophilic) neighborhoods, giving the
    startup-to-startup meta-paths real label signal that GNNs can exploit.
    """
    rows = []
    startup_uuids = []
    qualities = []

    for i in range(n):
        uid = make_uuid()
        startup_uuids.append(uid)

        # --- Latent quality ---
        # `quality` is the TRUE latent that drives the labels and the
        # edge assortativity (so neighbors share it). `fq` is a NOISY readout
        # of it that drives the node's OWN features. With feature_noise > 0 the
        # node's features only partly reveal `quality`, so a feature-only model
        # is limited, while a GNN can average over assortative neighbors (who
        # share the true `quality`) to denoise the estimate — mirroring the
        # real Crunchbase regime where structure adds signal beyond node
        # features.
        quality = rng.normal(0, 1)
        qualities.append(quality)
        fq = quality + rng.normal(0, feature_noise)

        founded_year = int(rng.integers(2005, 2023))
        founded_on = f"{founded_year}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"
        # fq → more rounds, more funding (noisy readout of quality)
        num_rounds = max(0, int(rng.poisson(1.5 + max(0, fq * 1.5))))
        total_funding = float(rng.lognormal(13 + fq * 2.5, 1.2)) if num_rounds > 0 else 0.0
        # Employee count correlated with fq
        emp_idx = min(len(EMPLOYEE_COUNTS) - 2, max(0, int(3 + fq * 2.0 + rng.normal(0, 0.5))))
        emp = EMPLOYEE_COUNTS[emp_idx]

        # Founder aggregates — fq influences team composition
        def _clamp_p(p):
            return max(0.01, min(0.99, p))

        founder_count = max(1, int(rng.poisson(1.8 + max(0, fq * 0.5))))
        female_count = int(rng.binomial(founder_count, 0.2))
        serial_count = int(rng.binomial(founder_count, _clamp_p(0.15 + fq * 0.25)))
        ivy_count = int(rng.binomial(founder_count, _clamp_p(0.08 + fq * 0.20)))
        bachelor_count = int(rng.binomial(founder_count, _clamp_p(0.65 + fq * 0.10)))
        master_count = int(rng.binomial(founder_count, _clamp_p(0.30 + fq * 0.20)))
        phd_count = int(rng.binomial(founder_count, _clamp_p(0.07 + fq * 0.10)))
        business_count = int(rng.binomial(founder_count, _clamp_p(0.20 + fq * 0.10)))
        it_count = int(rng.binomial(founder_count, _clamp_p(0.30 + fq * 0.10)))
        law_count = int(rng.binomial(founder_count, 0.04))
        stem_count = int(rng.binomial(founder_count, _clamp_p(0.35 + fq * 0.15)))
        other_edu = int(rng.binomial(founder_count, 0.1))

        # Funding stage features
        stage_rounds = {}
        round_names = ["angel", "pre_seed", "seed", "series_a", "series_b",
                        "series_c", "series_d", "series_e", "series_f",
                        "series_g", "series_h", "series_i", "series_j",
                        "private_equity", "venture_round", "other_rounds"]
        for rn in round_names:
            has_round = int(rng.random() < _clamp_p(0.1 + fq * 0.15)) if num_rounds > 0 else 0
            stage_rounds[f"money_{rn}"] = float(rng.lognormal(12 + fq * 1.5, 1.2)) if has_round else 0.0
            stage_rounds[f"investors_{rn}"] = max(1, int(rng.integers(1, 6) + fq * 1.0)) if has_round else 0
            stage_rounds[f"{rn}_round"] = has_round

        total_investors = sum(stage_rounds[f"investors_{rn}"] for rn in round_names)
        top10_inv = int(rng.binomial(max(1, total_investors), _clamp_p(0.05 + fq * 0.15)))
        top50_inv = int(rng.binomial(max(1, total_investors), _clamp_p(0.15 + fq * 0.20)))

        days_first_fund = max(1, int(rng.lognormal(6 - fq * 0.8, 0.8))) if num_rounds > 0 else 0
        accel_count = int(rng.random() < _clamp_p(0.1 + fq * 0.15))
        inv_type = rng.choice(["unknown", "private", "public", "hybrid"],
                              p=[0.7, 0.25, 0.03, 0.02])
        # Higher fq → later stage
        stage_probs = np.array([0.5 - fq * 0.05, 0.1, 0.1 + fq * 0.03,
                                0.08 + fq * 0.02, 0.07, 0.05, 0.05, 0.05])
        stage_probs = np.clip(stage_probs, 0.01, None)
        stage_probs /= stage_probs.sum()
        last_stage = int(rng.choice([0, 1, 3, 4, 5, 6, 7, 14], p=stage_probs))
        avg_days_between = float(rng.lognormal(5.5 - fq * 0.3, 0.8)) if num_rounds > 1 else np.nan
        funding_growth = float(rng.normal(0.5 + fq * 0.8, 0.8)) if num_rounds > 1 else np.nan
        months_since = float(rng.lognormal(3 - fq * 0.3, 0.8)) if num_rounds > 0 else np.nan
        burn = float(rng.lognormal(10 + fq * 0.5, 1.5)) if total_funding > 0 else np.nan
        has_down = int(rng.random() < _clamp_p(0.05 - fq * 0.03)) if num_rounds > 1 else 0
        follow_on = float(rng.beta(2 + max(0, fq), 5)) if total_investors > 0 else np.nan

        has_desc = int(rng.random() < _clamp_p(0.5 + fq * 0.20))
        desc = "An innovative tech startup." if has_desc else np.nan
        desc_len = int(rng.integers(100, 500) + fq * 50) if has_desc else 0

        # Last funding date
        last_fund_year = min(2024, founded_year + max(0, int(rng.exponential(3))))
        last_funding_on = f"{last_fund_year}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}" if num_rounds > 0 else np.nan

        # --- Target variables (planted signal) ---
        # Steep quality coupling so the label is a sharp (low-noise) function
        # of quality. Combined with quality-assortative edges, this gives the
        # startup-to-startup meta-paths real label homophily (well above the
        # class-imbalance baseline) that GNNs can exploit. Intercepts keep the
        # base rates near the real-data targets (~15% seriesA, ~13% NFR, ~4% exit).
        seriesA = int(rng.random() < _sigmoid(quality * 2.2 - 2.4))
        # acquired: ~3.5% base rate (acquired+ipo should be ~4.2%)
        acquired = int(rng.random() < _sigmoid(quality * 2.0 - 4.7))
        # ipo: ~0.7% base rate
        ipo = int(rng.random() < _sigmoid(quality * 2.0 - 6.0))
        # new_funding_round: ~13% base rate
        new_funding_round = int(rng.random() < _sigmoid(quality * 2.2 - 2.9))
        # future_status
        if ipo:
            future_status = "ipo"
        elif acquired:
            future_status = "acquired"
        elif rng.random() < 0.08:
            future_status = "closed"
        else:
            future_status = "operating"

        rows.append({
            "startup_uuid": uid,
            "name": f"Startup_{i}",
            "created_at": f"{founded_year}-01-01 00:00:00",
            "updated_at": f"{founded_year+2}-06-15 00:00:00",
            "country_code": rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS),
            "city": rng.choice([c[0] for c in CITIES]),
            "dc_status": "operating",
            "num_funding_rounds": num_rounds,
            "total_funding_usd": total_funding if total_funding > 0 else np.nan,
            "total_funding_currency_code": "USD",
            "founded_on": founded_on,
            "last_funding_on": last_funding_on,
            "dc_closed_on": np.nan,
            "employee_count": emp,
            "primary_role": "company",
            "num_exits": int(rng.random() < _clamp_p(0.02 + fq * 0.03)),
            "founded_on_year": founded_year,
            "industries": str(rng.choice(SECTORS, size=rng.integers(1, 4), replace=False).tolist()),
            "industry_groups": str(rng.choice(["Software", "Information Technology", "Internet Services"], size=rng.integers(1, 3), replace=False).tolist()),
            "is_sustainable": int(rng.random() < 0.05),
            "has_domain": int(rng.random() < _clamp_p(0.7 + fq * 0.15)),
            "has_email": int(rng.random() < _clamp_p(0.5 + fq * 0.20)),
            "has_phone": int(rng.random() < _clamp_p(0.3 + fq * 0.20)),
            "has_facebook_url": int(rng.random() < _clamp_p(0.4 + fq * 0.15)),
            "has_linkedin_url": int(rng.random() < _clamp_p(0.6 + fq * 0.15)),
            "has_twitter_url": int(rng.random() < _clamp_p(0.5 + fq * 0.15)),
            "has_logo_url": int(rng.random() < _clamp_p(0.7 + fq * 0.10)),
            "type_organization": int(rng.random() > 0.05),
            "founder_count": founder_count,
            "female_count": female_count,
            "serial_founder_count": serial_count,
            "featured_job_organization_uuid_x": make_uuid() if rng.random() < 0.3 else np.nan,
            "ivy_league_plus_count": ivy_count,
            "featured_job_organization_uuid_y": make_uuid() if rng.random() < 0.3 else np.nan,
            "bachelor_count": bachelor_count,
            "business_degree_count": business_count,
            "it_degree_count": it_count,
            "law_degree_count": law_count,
            "master_count": master_count,
            "other_edu_count": other_edu,
            "phd_count": phd_count,
            "stem_degree_count": stem_count,
            "female_ratio": female_count / founder_count if founder_count > 0 else 0,
            "serial_founder_ratio": serial_count / founder_count if founder_count > 0 else 0,
            "ivy_league_ratio": ivy_count / founder_count if founder_count > 0 else 0,
            "bachelor_ratio": bachelor_count / founder_count if founder_count > 0 else 0,
            "master_ratio": master_count / founder_count if founder_count > 0 else 0,
            "phd_ratio": phd_count / founder_count if founder_count > 0 else 0,
            "stem_degree_ratio": stem_count / founder_count if founder_count > 0 else 0,
            "business_degree_ratio": business_count / founder_count if founder_count > 0 else 0,
            "it_degree_ratio": it_count / founder_count if founder_count > 0 else 0,
            "has_tech_and_biz": int(it_count > 0 and business_count > 0),
            **stage_rounds,
            "total_investors": total_investors,
            "top_10_percent_investor_count": top10_inv,
            "top_50_percent_investor_count": top50_inv,
            "days_until_first_funding": days_first_fund,
            "accelerator_participation_count": accel_count,
            "investor_type": inv_type,
            "avg_days_between_rounds": avg_days_between,
            "funding_growth_rate": funding_growth,
            "months_since_last_funding": months_since,
            "implied_monthly_burn": burn,
            "has_down_round": has_down,
            "investor_follow_on_ratio": follow_on,
            "last_funding_stage": last_stage,
            "description": desc,
            "has_description": has_desc,
            "description_length": desc_len,
            "seriesA": seriesA,
            "acquired": acquired,
            "ipo": ipo,
            "closed": int(future_status == "closed"),
            "operating": int(future_status == "operating"),
            "acquirer": int(rng.random() < 0.01),
            "future_status": future_status,
            "new_funding_round": new_funding_round,
            "new_acquired": acquired,
            "new_ipo": ipo,
            "acq_ipo_funding": int(acquired or ipo or new_funding_round),
        })

    return pd.DataFrame(rows), startup_uuids, np.array(qualities)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _assortative_indices(target_q, entity_qs, size, homophily, rng, sigma=0.2):
    """Pick `size` distinct entity indices. With probability `homophily` the
    pick is biased toward entities whose latent quality is near `target_q`
    (Gaussian weight, width `sigma`); otherwise it is uniform. This makes
    startups that share an investor / founder / city tend to have similar
    quality, hence similar labels, hence homophilic meta-paths.
    """
    n = len(entity_qs)
    size = min(size, n)
    if homophily > 0.0 and rng.random() < homophily:
        w = np.exp(-0.5 * ((entity_qs - target_q) / sigma) ** 2)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return rng.choice(n, size=size, replace=False)
        return rng.choice(n, size=size, replace=False, p=w / s)
    return rng.choice(n, size=size, replace=False)


# ---------------------------------------------------------------------------
# Edge generators
# ---------------------------------------------------------------------------

def generate_startup_investor_edges(startup_uuids, inv_uuids, rng,
                                    startup_qualities=None, inv_qualities=None,
                                    homophily=0.0):
    """Power-law: most startups have 1-3 investors, few have many. When
    `homophily` > 0, investors are picked quality-assortatively so startups
    sharing an investor tend to share labels."""
    rows = []
    for s_idx, s_uid in enumerate(startup_uuids):
        n_inv = max(1, int(rng.lognormal(0.5, 0.8)))
        n_inv = min(n_inv, len(inv_uuids))
        if startup_qualities is not None and inv_qualities is not None:
            idx = _assortative_indices(startup_qualities[s_idx], inv_qualities,
                                       n_inv, homophily, rng)
            chosen = [inv_uuids[i] for i in idx]
        else:
            chosen = rng.choice(inv_uuids, size=n_inv, replace=False)
        for inv_uid in chosen:
            rows.append({
                "startup_uuid": s_uid,
                "investor_uuid": inv_uid,
                "stage": rng.choice(STAGES, p=STAGE_WEIGHTS),
            })
    return pd.DataFrame(rows)


def generate_startup_city_edges(startup_uuids, city_uuids, rng,
                                startup_qualities=None, city_qualities=None,
                                homophily=0.0):
    """Each startup has exactly 1 city (quality-assortative when homophily>0)."""
    city_uid_list = list(city_uuids.values())
    rows = []
    for s_idx, s_uid in enumerate(startup_uuids):
        if startup_qualities is not None and city_qualities is not None:
            idx = _assortative_indices(startup_qualities[s_idx], city_qualities,
                                       1, homophily, rng)[0]
        else:
            idx = rng.integers(len(city_uid_list))
        rows.append({"city_uuid": city_uid_list[idx], "startup_uuid": s_uid})
    return pd.DataFrame(rows)


def generate_startup_founder_edges(startup_uuids, founder_uuids, rng,
                                   startup_qualities=None, founder_qualities=None,
                                   homophily=0.0):
    """1-3 founders per startup, ensuring all startups have at least 1. When
    `homophily` > 0, founders are picked quality-assortatively."""
    rows = []
    available = list(founder_uuids)
    rng.shuffle(available)
    idx = 0

    for s_idx, s_uid in enumerate(startup_uuids):
        n_founders = max(1, min(3, int(rng.poisson(1.5))))
        if startup_qualities is not None and founder_qualities is not None:
            chosen_idx = _assortative_indices(startup_qualities[s_idx],
                                              founder_qualities, n_founders,
                                              homophily, rng)
            chosen = [founder_uuids[i] for i in chosen_idx]
        else:
            chosen = []
            for _ in range(n_founders):
                if idx < len(available):
                    chosen.append(available[idx])
                    idx += 1
                else:
                    chosen.append(rng.choice(founder_uuids))
        for f_uid in chosen:
            year = int(rng.integers(2005, 2023))
            rows.append({
                "startup_uuid": s_uid,
                "founder_uuid": f_uid,
                "title": rng.choice(TITLES),
                "started_on": f"{year}-01-01",
                "ended_on": np.nan if rng.random() > 0.2 else f"{year + int(rng.integers(1,6))}-01-01",
                "days_of_employment": float(rng.integers(365, 3650)) if rng.random() > 0.3 else np.nan,
            })
    return pd.DataFrame(rows)


def generate_startup_sector_edges(startup_uuids, sector_uuids, rng):
    """1-3 sectors per startup."""
    sector_uid_list = list(sector_uuids.values())
    rows = []
    for s_uid in startup_uuids:
        n = max(1, int(rng.poisson(1.5)))
        n = min(n, len(sector_uid_list))
        chosen = rng.choice(sector_uid_list, size=n, replace=False)
        for sec_uid in chosen:
            rows.append({"startup_uuid": s_uid, "sector_uuid": sec_uid})
    return pd.DataFrame(rows)


def generate_founder_university_edges(founder_uuids, uni_uuids, rng):
    """~60% of founders have a university, some have 2."""
    uni_uid_list = list(uni_uuids.values())
    rows = []
    for f_uid in founder_uuids:
        if rng.random() < 0.6:
            n = 1 if rng.random() > 0.15 else 2
            chosen = rng.choice(uni_uid_list, size=n, replace=False)
            for u_uid in chosen:
                year = int(rng.integers(1990, 2020))
                rows.append({
                    "founder_uuid": f_uid,
                    "university_uuid": u_uid,
                    "degree_type": rng.choice(DEGREE_TYPES, p=DEGREE_WEIGHTS),
                    "subject": rng.choice(["Computer Science", "Business", "Engineering",
                                           "Economics", "Physics", "Mathematics", "Biology"]),
                    "started_on": f"{year}-09-01" if rng.random() > 0.4 else np.nan,
                    "completed_on": f"{year + int(rng.integers(2,6))}-06-01" if rng.random() > 0.3 else np.nan,
                    "is_completed": rng.choice([True, False], p=[0.85, 0.15]),
                    "days_of_study": float(rng.integers(700, 2200)) if rng.random() > 0.3 else np.nan,
                })
    return pd.DataFrame(rows)


def generate_investor_city_edges(inv_uuids, city_uuids, rng):
    """Each investor has 1 city."""
    city_uid_list = list(city_uuids.values())
    rows = []
    for inv_uid in inv_uuids:
        rows.append({
            "city_uuid": rng.choice(city_uid_list),
            "investor_uuid": inv_uid,
        })
    return pd.DataFrame(rows)


def generate_university_city_edges(uni_uuids_dict, city_uuids):
    """Each university has 1 city (from UNIVERSITIES constant)."""
    rows = []
    for uni_name, uni_uid in uni_uuids_dict.items():
        # Find the city for this university
        city_name = None
        for name, cname in UNIVERSITIES:
            if name == uni_name:
                city_name = cname
                break
        if city_name:
            matching = [v for k, v in city_uuids.items() if k[0] == city_name]
            if matching:
                rows.append({"city_uuid": matching[0], "university_uuid": uni_uid})
    return pd.DataFrame(rows)


def generate_investor_sector_edges(inv_uuids, sector_uuids, rng):
    """Optional: investor-sector edges."""
    sector_uid_list = list(sector_uuids.values())
    rows = []
    for inv_uid in inv_uuids:
        if rng.random() < 0.5:
            n = max(1, int(rng.poisson(2)))
            n = min(n, len(sector_uid_list))
            chosen = rng.choice(sector_uid_list, size=n, replace=False)
            for sec_uid in chosen:
                rows.append({"startup_uuid": inv_uid, "investor_uuid": inv_uid, "stage": "other"})
    # This file doesn't exist in real data; return empty.
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Optional edge generators
# ---------------------------------------------------------------------------

def generate_founder_investor_employment_edges(founder_uuids, inv_uuids, rng):
    """~5% of founders worked at an investor."""
    rows = []
    for f_uid in founder_uuids:
        if rng.random() < 0.05:
            inv_uid = rng.choice(inv_uuids)
            year = int(rng.integers(2000, 2020))
            rows.append({
                "founder_uuid": f_uid,
                "investor_uuid": inv_uid,
                "started_on": f"{year}-01-01",
                "ended_on": f"{year + int(rng.integers(1,5))}-01-01",
                "days_of_employment": float(rng.integers(180, 2500)),
            })
    return pd.DataFrame(rows)


def generate_founder_coworking_edges(startup_founder_df, rng):
    """Founders who share a startup."""
    rows = []
    grouped = startup_founder_df.groupby("startup_uuid")["founder_uuid"].apply(list)
    for s_uid, founders in grouped.items():
        if len(founders) > 1:
            for i in range(len(founders)):
                for j in range(i + 1, len(founders)):
                    rows.append({
                        "founder_uuid_1": founders[i],
                        "founder_uuid_2": founders[j],
                        "startup_uuid": s_uid,
                        "overlap_days": float(rng.integers(100, 3000)),
                    })
    return pd.DataFrame(rows)


def generate_founder_investor_identity_edges(founder_uuids, inv_uuids, rng):
    """~2% of founders are also investors."""
    rows = []
    for f_uid in founder_uuids:
        if rng.random() < 0.02:
            rows.append({
                "founder_uuid": f_uid,
                "investor_uuid": rng.choice(inv_uuids),
            })
    return pd.DataFrame(rows)


def generate_similarity_edges(uuids, prefix, rng, frac=0.02):
    """Generate similarity edges between a fraction of entity pairs."""
    rows = []
    n = len(uuids)
    n_edges = int(n * n * frac / 2)
    n_edges = min(n_edges, n * 5)  # Cap at 5 per node on average
    for _ in range(n_edges):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            sim = float(rng.beta(2, 5))
            rows.append({
                f"{prefix}_uuid_1": uuids[i],
                f"{prefix}_uuid_2": uuids[j],
                "similarity": sim,
                "rank_similarity": sim * rng.uniform(0.8, 1.2),
            })
    return pd.DataFrame(rows)


def generate_founder_co_study_edges(founder_uni_df, rng):
    """Founders who studied at the same university."""
    rows = []
    grouped = founder_uni_df.groupby("university_uuid")["founder_uuid"].apply(list)
    for u_uid, founders in grouped.items():
        if len(founders) > 1:
            # Sample a few pairs, not all
            n_pairs = min(len(founders) * (len(founders) - 1) // 2, 5)
            for _ in range(n_pairs):
                i, j = rng.integers(0, len(founders), size=2)
                if i != j:
                    rows.append({
                        "founder_uuid_1": founders[i],
                        "founder_uuid_2": founders[j],
                        "overlap_days": float(rng.integers(0, 1500)),
                    })
    return pd.DataFrame(rows)


def generate_founder_board_edges(founder_uuids, startup_uuids, rng):
    """~3% of founders serve on boards."""
    rows = []
    for f_uid in founder_uuids:
        if rng.random() < 0.03:
            rows.append({
                "founder_uuid": f_uid,
                "startup_uuid": rng.choice(startup_uuids),
            })
    return pd.DataFrame(rows)


def generate_founder_director_edges(founder_uuids, entity_uuids, rng, entity_col, frac=0.02):
    """Founders who are directors at startups or investors."""
    rows = []
    for f_uid in founder_uuids:
        if rng.random() < frac:
            rows.append({
                "founder_uuid": f_uid,
                entity_col: rng.choice(entity_uuids),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir, n_startups, seed, homophily=0.9, feature_noise=1.2):
    global rng
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    n_investors = max(50, n_startups // 5)
    n_founders = max(100, n_startups // 2)

    print(f"Generating synthetic data: {n_startups} startups, {n_investors} investors, "
          f"{n_founders} founders, {len(CITIES)} cities, {len(UNIVERSITIES)} universities, "
          f"{len(SECTORS)} sectors")
    print(f"Output directory: {output_dir}")
    print()

    # --- Nodes ---
    print("Generating nodes...")
    city_df, city_uuids = generate_cities(rng)
    uni_df, uni_uuids = generate_universities(city_uuids, rng)
    sector_df, sector_uuids = generate_sectors()
    inv_df, inv_uuids = generate_investors(n_investors, city_uuids, rng)
    founder_df, founder_uuids = generate_founders(n_founders, rng)
    startup_df, startup_uuids, startup_qualities = generate_startups(
        n_startups, founder_uuids, inv_uuids, rng, feature_noise)

    # Latent quality for shared-neighbor entities, on the same N(0,1) scale as
    # startup quality, so quality-assortative edges plant startup-to-startup
    # label homophily on the investor / founder / city meta-paths.
    inv_qualities = rng.normal(0, 1, size=len(inv_uuids))
    founder_qualities = rng.normal(0, 1, size=len(founder_uuids))
    city_qualities = rng.normal(0, 1, size=len(city_uuids))
    print(f"Structural homophily strength: {homophily}; feature noise: {feature_noise}")

    # --- Required edges ---
    print("Generating edges...")
    si_edges = generate_startup_investor_edges(
        startup_uuids, inv_uuids, rng, startup_qualities, inv_qualities, homophily)
    sc_edges = generate_startup_city_edges(
        startup_uuids, city_uuids, rng, startup_qualities, city_qualities, homophily)
    sf_edges = generate_startup_founder_edges(
        startup_uuids, founder_uuids, rng, startup_qualities, founder_qualities, homophily)
    ss_edges = generate_startup_sector_edges(startup_uuids, sector_uuids, rng)
    fu_edges = generate_founder_university_edges(founder_uuids, uni_uuids, rng)
    ic_edges = generate_investor_city_edges(inv_uuids, city_uuids, rng)
    uc_edges = generate_university_city_edges(uni_uuids, city_uuids)

    # --- Optional edges ---
    print("Generating optional edges...")
    fie_edges = generate_founder_investor_employment_edges(founder_uuids, inv_uuids, rng)
    fcw_edges = generate_founder_coworking_edges(sf_edges, rng)
    fii_edges = generate_founder_investor_identity_edges(founder_uuids, inv_uuids, rng)
    s_sim_edges = generate_similarity_edges(startup_uuids, "startup", rng)
    f_sim_edges = generate_similarity_edges(founder_uuids, "founder", rng)
    fcs_edges = generate_founder_co_study_edges(fu_edges, rng)
    fb_edges = generate_founder_board_edges(founder_uuids, startup_uuids, rng)
    fsd_edges = generate_founder_director_edges(founder_uuids, startup_uuids, rng, "startup_uuid")
    fid_edges = generate_founder_director_edges(founder_uuids, inv_uuids, rng, "investor_uuid")

    # --- Write CSVs ---
    print("Writing CSVs...")
    files = {
        "startup_nodes.csv": startup_df,
        "investor_nodes.csv": inv_df,
        "founder_nodes.csv": founder_df,
        "city_nodes.csv": city_df,
        "university_nodes.csv": uni_df,
        "sector_nodes.csv": sector_df,
        "startup_investor_edges.csv": si_edges,
        "startup_city_edges.csv": sc_edges,
        "startup_founder_edges.csv": sf_edges,
        "startup_sector_edges.csv": ss_edges,
        "founder_university_edges.csv": fu_edges,
        "investor_city_edges.csv": ic_edges,
        "university_city_edges.csv": uc_edges,
        "founder_investor_employment_edges.csv": fie_edges,
        "founder_coworking_edges.csv": fcw_edges,
        "founder_investor_identity_edges.csv": fii_edges,
        "startup_descriptively_similar_edges.csv": s_sim_edges,
        "founder_descriptively_similar_edges.csv": f_sim_edges,
        "founder_co_study_edges.csv": fcs_edges,
        "founder_board_edges.csv": fb_edges,
        "founder_startup_director_edges.csv": fsd_edges,
        "founder_investor_director_edges.csv": fid_edges,
    }

    for fname, df in files.items():
        path = os.path.join(output_dir, fname)
        df.to_csv(path, index=False)
        print(f"  {fname}: {len(df)} rows")

    # --- Crunchbase-format snapshots ---
    # The downstream-analysis scripts (plot_aggregated_pk_curves, extract_cohort_pk,
    # plot_pk_by_year) load metadata from data/crunchbase/{2023,2025}/*.csv via
    # src/ml/downstream_analysis.py:DownstreamAnalyzer. Emit minimal-but-faithful
    # synthetic versions so those scripts produce non-empty artifacts on the
    # synthetic smoke too (otherwise they exit with KeyErrors / empty outputs).
    print()
    print("Writing Crunchbase-format snapshots (data/crunchbase/{2023,2025}/)...")
    crunchbase_root = os.path.join(os.path.dirname(output_dir.rstrip("/")), "crunchbase")
    cb_2023 = os.path.join(crunchbase_root, "2023")
    cb_2025 = os.path.join(crunchbase_root, "2025")
    os.makedirs(cb_2023, exist_ok=True)
    os.makedirs(cb_2025, exist_ok=True)

    # organizations.csv: project the 7 columns DownstreamAnalyzer reads. Already
    # present in startup_df (just rename startup_uuid->uuid and supply
    # category_list from industries).
    orgs_df = pd.DataFrame({
        "uuid": startup_df["startup_uuid"],
        "name": startup_df["name"],
        "category_list": startup_df.get("industries", ""),
        "num_funding_rounds": startup_df["num_funding_rounds"],
        "total_funding_usd": startup_df["total_funding_usd"],
        "country_code": startup_df["country_code"],
        "founded_on": startup_df["founded_on"],
    })
    orgs_path = os.path.join(cb_2023, "organizations.csv")
    orgs_df.to_csv(orgs_path, index=False)
    print(f"  2023/organizations.csv: {len(orgs_df)} rows")

    # funding_rounds.csv (2025 = full historical + future, 2023 = pre-2024 snapshot).
    # Per startup with num_funding_rounds > 0, synthesize that many rounds with a
    # plausible stage progression (seed -> series_a -> b -> c -> d) and dates
    # spread between founded_on and a near-present anchor. raised_amount_usd
    # splits total_funding_usd across rounds with growing weight; post-money
    # valuation is a ~10x multiple.
    _STAGES = ["seed", "series_a", "series_b", "series_c", "series_d",
               "series_e", "series_f", "series_g"]
    rounds_rows = []
    for _, row in startup_df.iterrows():
        n_rounds = int(row["num_funding_rounds"])
        if n_rounds <= 0:
            continue
        total = float(row["total_funding_usd"]) if row["total_funding_usd"] else 0.0
        try:
            founded = pd.to_datetime(row["founded_on"], errors="coerce")
        except Exception:
            founded = pd.NaT
        if pd.isna(founded):
            founded = pd.Timestamp("2014-01-01")
        # Span rounds from founded_on to founded_on + ~n_rounds years, capped at 2025-09.
        end_anchor = min(pd.Timestamp("2025-09-01"), founded + pd.Timedelta(days=365 * max(n_rounds, 2)))
        if end_anchor <= founded:
            end_anchor = founded + pd.Timedelta(days=180 * n_rounds)
        # Linear weights skewed to later rounds for raised_amount.
        weights = [(i + 1) for i in range(n_rounds)]
        wsum = sum(weights) or 1
        for idx in range(n_rounds):
            stage = _STAGES[min(idx, len(_STAGES) - 1)]
            # Distribute date evenly between founded and end_anchor
            t = (idx + 1) / (n_rounds + 1)
            announced = founded + (end_anchor - founded) * t
            raised = total * weights[idx] / wsum
            rounds_rows.append({
                "uuid": make_uuid(),
                "org_uuid": row["startup_uuid"],
                "announced_on": announced.date().isoformat(),
                "raised_amount_usd": round(raised, 2),
                "post_money_valuation_usd": round(raised * 10.0, 2),
                "investment_type": stage,
            })
    funding_df = pd.DataFrame(rounds_rows)
    funding_2025_path = os.path.join(cb_2025, "funding_rounds.csv")
    funding_df.to_csv(funding_2025_path, index=False)
    print(f"  2025/funding_rounds.csv: {len(funding_df)} rows")
    # 2023 snapshot: rounds with announced_on < 2024-01-01 (the "at prediction
    # time" view the analyzer uses to set roi_start_date).
    if len(funding_df) > 0:
        funding_2023 = funding_df[funding_df["announced_on"] < "2024-01-01"]
    else:
        funding_2023 = funding_df
    funding_2023_path = os.path.join(cb_2023, "funding_rounds.csv")
    funding_2023.to_csv(funding_2023_path, index=False)
    print(f"  2023/funding_rounds.csv: {len(funding_2023)} rows")

    # --- Summary ---
    print()
    print("Summary:")
    print(f"  Startups: {len(startup_df)}")
    print(f"  Investors: {len(inv_df)}")
    print(f"  Founders: {len(founder_df)}")
    print(f"  Cities: {len(city_df)}")
    print(f"  Universities: {len(uni_df)}")
    print(f"  Sectors: {len(sector_df)}")
    print(f"  Total edge files: {sum(1 for df in files.values() if len(df) > 0)}")
    print()

    # Target stats
    print("Target distributions:")
    for col in ["new_funding_round", "acquired", "ipo", "new_acquired", "new_ipo", "seriesA"]:
        rate = startup_df[col].mean()
        print(f"  {col}: {rate:.3f} ({startup_df[col].sum()}/{len(startup_df)})")

    # Invalidate any stale graph cache: the content-addressable cache in
    # outputs/graphs/ keys on config, not on the underlying CSV contents, so a
    # cache built from a previous data generation would be silently reused and
    # the new data ignored. Clearing it forces a rebuild on the next run.
    cache_dir = "outputs/graphs"
    if os.path.isdir(cache_dir):
        stale = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
        for f in stale:
            os.remove(os.path.join(cache_dir, f))
        if stale:
            print(f"Cleared {len(stale)} stale graph cache file(s) in {cache_dir}/ "
                  "so the new data is used.")

    print()
    print(f"Done! Data written to {output_dir}/")
    print("You can now run: python src/main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic startup data")
    parser.add_argument("--output-dir", type=str, default="data/graph",
                        help="Output directory for CSV files (default: data/graph)")
    parser.add_argument("--n-startups", type=int, default=1000,
                        help="Number of startups to generate (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--homophily", type=float, default=0.9,
                        help="Structural homophily strength in [0,1]: fraction "
                             "of startups whose investor/founder/city neighbors "
                             "are picked quality-assortatively, planting "
                             "startup-to-startup label homophily on the "
                             "meta-paths (default: 0.8). Set 0 for random "
                             "structure.")
    parser.add_argument("--feature-noise", type=float, default=1.2,
                        help="Std of Gaussian noise added to the latent quality "
                             "before it drives the node's OWN features (default "
                             "1.2). Higher = node features reveal less, so GNNs "
                             "gain more from neighbor aggregation. Labels + edge "
                             "assortativity use the noise-free quality.")
    args = parser.parse_args()
    main(args.output_dir, args.n_startups, args.seed, args.homophily,
         args.feature_noise)
