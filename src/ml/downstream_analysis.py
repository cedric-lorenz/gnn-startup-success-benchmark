"""Downstream analysis simulating VC portfolio strategies to evaluate model predictions via ROI and precision."""
import argparse

import yaml
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

# Add project root to path to allow imports if running as script
import sys
sys.path.append(os.getcwd())
from src.data_engineering.aux_pipeline import convert_to_continent
from src.ml.utils import get_maturity_mask

_THESIS_RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_thesis_style(ax):
    """Remove top/right spines and apply thesis grid style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


class DownstreamAnalyzer:
    """
    Performs downstream analysis on model predictions, including ROI, Sector, Round Type, Age, and Geography performance.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Downstream Analyzer.
        
        Args:
            config: Configuration dictionary containing paths.
        """
        self.config = config
        self.data_dir = config["paths"]["crunchbase_dir"]
        self.graph_dir = config["paths"]["graph_dir"]
        self.output_dir = config.get("output_dir", "outputs")
        
        print(f"📊 Initializing Downstream Analyzer...")
        
        # Load Data
        self._load_data()
    
        # Dilution Rates for Valuation Estimation (Radicle Model)
        self.DILUTION_RATES = {
            'angel': 0.10,
            'pre_seed': 0.08,
            'seed': 0.10,
            'series_a': 0.20,
            'series_b': 0.16,
            'series_c': 0.13,
            'series_d': 0.12,
            'series_e': 0.10,
            'series_f': 0.09,
            'series_g': 0.08,
            'series_h': 0.06,
            'series_i': 0.05,
            'private_equity': 0.15, # Assumed
            'debt_financing': 0.05, # Assumed low equity impact
            'convertible_note': 0.10, # Assumed similar to seed
            'grant': 0.0, # Non-dilutive
            'post_ipo_equity': 0.05 # Assumed
        }
        
        # Step-Up Multiple for Unrealized Gains (Paper Markups)
        # If a startup raises a new round, we assume our previous stake appreciates by this factor.
        self.STEP_UP_MULTIPLE = 3.0
        
        # Fixed Ticket Size for Simulation
        self.TICKET_SIZE = 1_000_000.0
        
        # Undisclosed Exit Multiple (Capital Returned)
        self.UNDISCLOSED_EXIT_MULTIPLE = 1.0

        # Tiered Step-Ups for Paper Gains (Valuation follows Stage)
        self.TIERED_STEP_UPS = {
            'angel': 3.0,
            'pre_seed': 3.0,
            'seed': 3.0,
            'series_a': 2.5,
            'series_b': 2.0,
            # Late Stage / Growth / Default
            'default': 1.3
        }

        # Historical Funding Multiple (Estimate)
        self.HISTORICAL_FUNDING_MULTIPLE = 1.5
        
        # Benchmark Investors
        self.BENCHMARK_INVESTORS = {
            'a16z': 'ce91bad7-b6d8-e56e-0f45-4763c6c5ca29',
            'Sequoia': '0c867fde-2b9a-df10-fdb9-66b74f355f91',
            'YC': '73633ee4-ea65-2967-6c5d-9b5fec7d2d5e',
            'Benchmark': 'fe2d1e8b-f607-3c9f-fad7-98fb8412f77e',
            'Accel': 'b08efc27-da40-505a-6f9d-c9e14247bf36'
        }






        # Strategy Definitions for Downstream Analysis
        self.STAGE_STRATEGIES = {
            'Angel/Pre-Seed': {'angel', 'pre_seed', 'equity_crowdfunding', 'convertible_note', 'grant'},
            'Seed': {'seed'},
            'Series A': {'series_a'},
            'Series B': {'series_b'},
            'Series C+': {'series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j', 'private_equity'},
            'Early Stage': {'pre_seed', 'seed', 'angel', 'series_a', 'equity_crowdfunding', 'convertible_note'},
            'Growth': {'series_b', 'series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j', 'private_equity'}
        }

        self.CONTINENT_STRATEGIES = {
            'North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'
        }
        
        self.FUNDING_SOURCE_STRATEGIES = {
            'Private Backed': {'private'},
            'Public (Gov) Backed': {'public'},
            'Hybrid Backed': {'hybrid'}
        }



    def _load_data(self):
        """
        Load necessary CSVs for analysis.
        
        Files Loaded:
        - Funding Rounds (2023 & 2025): For ROI calculation and target window definition.
        - Organizations (2023): For sector, geography, and founding year features.
        - IPOs & Acquisitions (2025): For exit value calculation (Ground Truth).
        - Investments (2023): For benchmarking against investor portfolios.
        """
        # 1. Funding Rounds (for ROI)
        funding_path = os.path.join(self.data_dir, "2025", "funding_rounds.csv")
        if os.path.exists(funding_path):
            usecols = ['uuid', 'org_uuid', 'announced_on', 'raised_amount_usd', 'post_money_valuation_usd', 'investment_type']
            self.funding_df = pd.read_csv(funding_path, usecols=usecols)
            self.funding_df.rename(columns={'uuid': 'funding_round_uuid'}, inplace=True)
            self.funding_df['announced_on'] = pd.to_datetime(self.funding_df['announced_on'], errors='coerce')
            
            # Determine End Date from 2025 snapshot
            self.roi_end_date = self.funding_df['announced_on'].max()
            print(f"   Determined ROI End Date (from 2025 snapshot): {self.roi_end_date.date()}")
            
            # Filter for target window
            start_date = pd.Timestamp('2023-01-01')
            end_date = self.roi_end_date
            self.target_window_df = self.funding_df[
                (self.funding_df['announced_on'] >= start_date) & 
                (self.funding_df['announced_on'] <= end_date)
            ].copy()
            print(f"   Loaded {len(self.target_window_df)} funding rounds (2023-2025)")
            
            # Compute Last Stage Map (Pre-2023)
            pre_2023_df = self.funding_df[self.funding_df['announced_on'] < start_date].copy()
            if not pre_2023_df.empty:
                pre_2023_df = pre_2023_df.sort_values('announced_on')
                # Group by org and take last investment_type
                self.last_stage_map = pre_2023_df.groupby('org_uuid')['investment_type'].last().to_dict()
                print(f"   Computed last stage for {len(self.last_stage_map)} organizations (Pre-2023)")
            else:
                self.last_stage_map = {}
        else:
            print(f"⚠️ Funding rounds file not found: {funding_path}")
            self.target_window_df = None
            self.last_stage_map = {}
            self.roi_end_date = '2025-12-31' # Fallback

        # 1.1 Load 2023 Funding Rounds (for Start Date)
        funding_2023_path = os.path.join(self.data_dir, "2023", "funding_rounds.csv")
        if os.path.exists(funding_2023_path):
            usecols = ['announced_on']
            funding_2023_df = pd.read_csv(funding_2023_path, usecols=usecols)
            funding_2023_df['announced_on'] = pd.to_datetime(funding_2023_df['announced_on'], errors='coerce')
            self.roi_start_date = funding_2023_df['announced_on'].max()
            print(f"   Determined ROI Start Date (from 2023 snapshot): {self.roi_start_date.date()}")
        else:
            print(f"⚠️ 2023 Funding rounds file not found: {funding_2023_path}")
            self.roi_start_date = pd.Timestamp('2023-01-01') # Fallback

        # Update target window with dynamic start date
        if self.target_window_df is not None:
                self.target_window_df = self.funding_df[
                (self.funding_df['announced_on'] >= self.roi_start_date) & 
                (self.funding_df['announced_on'] <= self.roi_end_date)
            ].copy()
                print(f"   Refined target window to {len(self.target_window_df)} rounds ({self.roi_start_date.date()} - {self.roi_end_date.date()})")

        # 2. Organizations (for Sector, Round Type, Funding Source)
        orgs_path = os.path.join(self.data_dir, "2023", "organizations.csv") # Use 2023 for "at prediction time" features
        
        # Check if we need to load from startup_nodes.csv for investor_type
        # Since investor_type is a calculated feature not in raw organizations.csv
        nodes_path = os.path.join(self.graph_dir, "startup_nodes.csv")
        
        if os.path.exists(orgs_path):
            # Load category_list, num_funding_rounds, total_funding_usd
            # Also try to load investor_type if it exists, otherwise we merge it from nodes
            usecols = ['uuid', 'name', 'category_list', 'num_funding_rounds', 'total_funding_usd', 'country_code', 'founded_on']
            self.orgs_df = pd.read_csv(orgs_path, usecols=usecols)
            self.orgs_df.rename(columns={'uuid': 'org_uuid'}, inplace=True)
            self.orgs_df['founded_on'] = pd.to_datetime(self.orgs_df['founded_on'], errors='coerce')
            print(f"   Loaded {len(self.orgs_df)} organizations (2023)")
            
            # Merge investor_type from startup_nodes.csv if available
            if os.path.exists(nodes_path):
                try:
                    nodes_df = pd.read_csv(nodes_path, usecols=['startup_uuid', 'investor_type'])
                    nodes_df.rename(columns={'startup_uuid': 'org_uuid'}, inplace=True)
                    # Merge left on orgs_df
                    self.orgs_df = self.orgs_df.merge(nodes_df, on='org_uuid', how='left')
                    print(f"   Merged investor_type from startup_nodes.csv for {len(nodes_df)} startups")
                except Exception as e:
                    print(f"   ⚠️ Failed to load investor_type from startup_nodes.csv: {e}")
        else:
            print(f"⚠️ Organizations file not found: {orgs_path}")
            self.orgs_df = None
            
        # 3. IPOs (Target)
        ipos_path = os.path.join(self.data_dir, "2025", "ipos.csv") # Use 2025 for future labels
        if os.path.exists(ipos_path):
            self.ipos_df = pd.read_csv(ipos_path, usecols=['org_uuid', 'valuation_price_usd', 'went_public_on'])
            self.ipos_df['went_public_on'] = pd.to_datetime(self.ipos_df['went_public_on'], errors='coerce')
            print(f"   Loaded {len(self.ipos_df)} IPOs (2025)")
        else:
            print(f"⚠️ IPOs file not found: {ipos_path}")
            self.ipos_df = None

        # 4. Acquisitions (Target)
        acq_path = os.path.join(self.data_dir, "2025", "acquisitions.csv") # Use 2025 for future labels
        if os.path.exists(acq_path):
            self.acq_df = pd.read_csv(acq_path, usecols=['acquiree_uuid', 'price_usd', 'acquired_on'])
            self.acq_df.rename(columns={'acquiree_uuid': 'org_uuid', 'acquired_on': 'announced_on'}, inplace=True)
            self.acq_df['announced_on'] = pd.to_datetime(self.acq_df['announced_on'], errors='coerce')
            print(f"   Loaded {len(self.acq_df)} acquisitions (2025)")
        else:
            print(f"⚠️ Acquisitions file not found: {acq_path}")
            self.acq_df = None

        # 5. Investments (for Benchmarking)
        investments_path = os.path.join(self.data_dir, "2023", "investments.csv")
        if os.path.exists(investments_path):
            usecols = ['investor_uuid', 'funding_round_uuid']
            self.investments_df = pd.read_csv(investments_path, usecols=usecols)
            print(f"   Loaded {len(self.investments_df)} investments (2023)")
        else:
            print(f"⚠️ Investments file not found: {investments_path}")
            self.investments_df = None


    def perform_downstream_analysis(self, predictions: List[Tuple[str, any, any]]):
        """
        Run all downstream analyses.
        Supports both single-task (float scores) and multi-label (dict scores).
        """
        if not predictions:
            print("⚠️ No predictions provided for analysis.")
            return

        # 1. Detect Mode
        sample_score = predictions[0][1]
        task_queue = []

        if isinstance(sample_score, dict):
            # Check for Masked Multi-Task (New Strategy-Aware Logic)
            if 'mom' in sample_score and 'liq' in sample_score:
                print("\n🔄 Detected Masked Multi-Task Predictions (Momentum & Liquidity).")
                
                # 1. Venture Fund: Universe=All, Signal=Momentum
                task_queue.append(('mom', 'Venture Fund', False))

                # 2. Liquidity Fund: Universe=Mature, Signal=Liquidity
                task_queue.append(('liq', 'Liquidity Fund', True))

                # 3. Momentum Fund (Mature Growth)
                # Universe: Mature, Signal: Momentum
                #task_queue.append(('mom', 'Momentum Fund (Mature)', True))

                # 4. Balanced Fund (Composite)
                # Universe: Mature, Signal: Average(Mom, Liq)
                #task_queue.append(('balanced', 'Balanced Fund (Composite)', True))
            
            # Check for Multi-Label (Legacy/Alternative)
            elif 'fund' in sample_score or 'acq' in sample_score or 'ipo' in sample_score:
                print("\n🔄 Detected Multi-Label Predictions. Running Combined Analysis...")
                # Add Combined Task
                if 'fund' in sample_score and 'acq' in sample_score and 'ipo' in sample_score:
                    task_queue.append(('combined', 'Combined', False))
                else:
                     print("⚠️ Missing keys for combined analysis (fund, acq, ipo required)")
            
            else:
                 print(f"⚠️ Unknown dictionary output keys: {sample_score.keys()}")
                 
        else:
            task_queue.append(('default', 'Standard', False))

        # 2. Run Analysis Loop
        for task_item in task_queue:
            # Unpack task item
            if len(task_item) == 3:
                task_key, task_name, use_mature_filter = task_item
            else:
                task_key, task_name = task_item
                use_mature_filter = False
            
            if task_key == 'default':
                task_preds = predictions
                suffix = ""
                title_suffix = ""
            else:
                # Extract specific task predictions
                # Handle potential missing keys gracefully (though they should exist)
                task_preds = []
                
                if task_key == 'combined':
                     # Retrieve weights from config or use defaults
                     # Config structure: self.config["data_processing"]["multi_label"]["combined_metric_weights"]
                     # Safety check for config path
                     dp_config = self.config.get("data_processing", {})
                     ml_config = dp_config.get("multi_label", {}) if isinstance(dp_config, dict) else {}
                     weights_config = ml_config.get("combined_metric_weights", {}) if isinstance(ml_config, dict) else {}
                     
                     w_fund = float(weights_config.get("funding", 0.2))
                     w_acq = float(weights_config.get("acquisition", 0.3))
                     w_ipo = float(weights_config.get("ipo", 0.5))
                     
                     for u, s, l in predictions:
                         if 'fund' in s and 'acq' in s and 'ipo' in s:
                             # Weighted Score
                             combined_score = (w_fund * s['fund']) + (w_acq * s['acq']) + (w_ipo * s['ipo'])
                             # For ROI analysis, the specific label (FUNDED/EXIT/FAIL) depends on ground truth lookup.
                             # But 'label' arg is used to flag 'FUNDED' (paper gain) if not exited.
                             combined_label = max(l.get('fund', 0), l.get('acq', 0), l.get('ipo', 0))
                             task_preds.append((u, combined_score, combined_label))
                else:
                    # Specific Task (or Balanced Composite)
                    for u, s, l in predictions:
                        if task_key == 'balanced':
                             # Composite Score: (Mom + Liq) / 2
                             if 'mom' in s and 'liq' in s:
                                 score = (s['mom'] + s['liq']) / 2.0
                                 # Label? Doesn't matter for ROI (ground truth), but for plotting AUC:
                                 label = max(l.get('mom', 0), l.get('liq', 0))
                                 task_preds.append((u, score, label))
                        elif task_key in s and task_key in l:
                            task_preds.append((u, s[task_key], l[task_key]))
                
                suffix = f"_{task_key}"
                title_suffix = f" ({task_name})"
                
            # print(f"\n🔍 STARTING DOWNSTREAM ANALYSIS{title_suffix}")
            # print(f"   Predictions count: {len(task_preds)}")
            
            if not task_preds:
                print(f"   ⚠️ No predictions found for task {task_name}")
                continue

            # Convert to DataFrame
            try:
                pred_df = pd.DataFrame(task_preds, columns=['org_uuid', 'score', 'gt_label'])
            except Exception as e:
                print(f"   ⚠️ Failed to convert predictions to DataFrame: {e}")
                continue
            
            # Merge with Metadata
            full_df = pred_df
            if self.orgs_df is not None:
                # print("   Merging with organizations metadata...") # Reduce verbosity
                full_df = pred_df.merge(self.orgs_df, on='org_uuid', how='left')
                
            # 1. Filter if needed (Liquidity Strategy)
            if use_mature_filter:
                print(f"   🛡️ Applying Maturity Filter (for {task_name})...")
                task_preds = self._filter_mature_startups(task_preds)
                if not task_preds:
                    print("   ⚠️ No mature startups found in predictions. Skipping ROI.")
                    continue
            
            # 2. ROI Analysis
            roi_metrics = self.calculate_roi(
                task_preds, 
                filename_suffix=suffix, 
                title_suffix=title_suffix
            )
            
            # 1.1 Investor Benchmark
            benchmark_metrics = self.analyze_investor_benchmark()
            if benchmark_metrics and task_key == 'default': # Only print full benchmark once to avoid clutter
                print(f"\n🏆 Benchmark Results:")
                for name, metrics in benchmark_metrics.items():
                    print(f"   - {name}: Precision={metrics['precision']:.1%}, ROI={metrics['roi']:.1%}, k={metrics['k']}")

            # 1.2 Portfolio Comparison
            print(f"\n🔍 Portfolio Comparison (Model{title_suffix} vs Investors)")
            for investor_name in ['Sequoia', 'a16z']:
                if investor_name in self.BENCHMARK_INVESTORS:
                    self.compare_portfolios(task_preds, investor_name)
                
            # 1.3 Stage-Based Strategy Analysis
            strategy_results = {}
            for strategy_name, stages in self.STAGE_STRATEGIES.items():
                print(f"   🔬 Analyzing Strategy (Stage): {strategy_name}...")
                strat_preds = self.analyze_portfolio_by_stage(task_preds, strategy_name, stages, filename_suffix=suffix, verbose=False, plot_charts=False)
                if strat_preds:
                    strategy_results[strategy_name] = strat_preds

            # 1.4 Comparative Precision Plot (Strategies)
            if strategy_results:
                # Reorder: Global first, then stages, so legend reads naturally
                ordered_results = {'All Stages': task_preds}
                ordered_results.update(strategy_results)
                self._plot_comparative_precision_at_k(ordered_results, benchmark_metrics, filename_suffix=suffix)

            # 1.5 Geography-Based Strategy Analysis
            if 'country_code' in full_df.columns:
                print(f"\n   🌍 Analyzing Geography Strategies...")
                # Create Content Map on the fly - Ensure continent column exists
                if 'continent' not in full_df.columns:
                     full_df['continent'] = full_df['country_code'].apply(convert_to_continent)
                
                # Get available continents
                available_continents = full_df['continent'].dropna().unique()
                
                geo_results = {}
                print(f"\n   🌍 Analyzing Geography Strategies ({len(available_continents)} Continents)...")
                for continent in available_continents:
                    if continent == 'Unknown': continue
                    
                    # print(f"   📍 Region: {continent}...") # Silenced
                    cont_preds = self.analyze_portfolio_by_continent(task_preds, continent, full_df, filename_suffix=suffix, verbose=False, plot_charts=False)
                    if cont_preds:
                        geo_results[continent] = cont_preds

                
            # 3. Round Type Analysis
            if 'num_funding_rounds' in full_df.columns:
                self.analyze_funding_stage(full_df, top_k=None, filename_suffix=suffix)
                
            # 4. Founding Year Analysis
            if 'founded_on' in full_df.columns:
                self.analyze_founding_year(full_df, top_k=None, filename_suffix=suffix)
                
                # 1.6 Comparative Precision Plot (Geography)
                if geo_results:
                    # Add Standard for comparison
                    geo_results['All Regions'] = task_preds
                    self._plot_comparative_precision_at_k(geo_results, benchmark_metrics=None, filename_suffix=f"{suffix}_geography")

            # 1.7 Funding Source Strategy Analysis
            if 'investor_type' in full_df.columns:
                print(f"\n   🏛️ Analyzing Funding Source Strategies...")
                source_results = {}
                for strategy_name, allowed_types in self.FUNDING_SOURCE_STRATEGIES.items():
                    # print(f"   Using Strategy: {strategy_name}...") # Silenced
                    try:
                        source_preds = self.analyze_portfolio_by_funding_source(task_preds, strategy_name, allowed_types, full_df, filename_suffix=suffix, verbose=False, plot_charts=False)
                        if source_preds:
                            source_results[strategy_name] = source_preds
                    except Exception as e:
                        print(f"   ⚠️ Failed {strategy_name}: {e}")
                
                # Comparative Plot
                if source_results:
                   source_results['All Sources'] = task_preds
                   self._plot_comparative_precision_at_k(source_results, benchmark_metrics=None, filename_suffix=f"{suffix}_funding_source")

            # 2. Sector Analysis
            if 'category_list' in full_df.columns:
                self.analyze_sectors(full_df, top_k=None, filename_suffix=suffix)
                
            # 3. Round Type Analysis
            if 'num_funding_rounds' in full_df.columns:
                self.analyze_funding_stage(full_df, top_k=None, filename_suffix=suffix)
                
            # 4. Founding Year Analysis
            if 'founded_on' in full_df.columns:
                self.analyze_founding_year(full_df, top_k=None, filename_suffix=suffix)
                
            # 5. Legacy Geography Analysis (Country/Continent Distribution)
            if 'country_code' in full_df.columns:
                self.analyze_geography(full_df, top_k=None, filename_suffix=suffix)
                self.analyze_continents(full_df, top_k=None, filename_suffix=suffix)


    def _estimate_valuation(self, raised_amount: float, round_type: str) -> float:
        """
        Estimate Post-Money Valuation using Dilution Model.
        Formula: Post-Valuation = Raised Amount / Dilution Rate
        """
        if pd.isna(raised_amount) or raised_amount <= 0:
            return 0.0
            
        # Normalize round_type
        if pd.isna(round_type):
            round_type = 'seed' # Default
        else:
            round_type = str(round_type).lower().replace(' ', '_').replace('-', '_')
            
        dilution = self.DILUTION_RATES.get(round_type, 0.15) # Default to 15% if unknown
        
        if dilution <= 0: # Handle non-dilutive grants
            return raised_amount # Conservative: value = cash raised
            
        return raised_amount / dilution

    def _plot_net_profit_curve(self, ranks: np.ndarray, costs: np.ndarray, values: np.ndarray, labels: np.ndarray, filename_suffix: str = ""):
        """
        Generate Net Profit Curve (Cumulative Value - Cumulative Cost) for ALL predictions.
        Also includes Recall @ k on secondary axis.
        """
        if len(ranks) == 0:
            return

        profits = np.nan_to_num(values - costs)
        total_positives = np.sum(labels)
        recall_at_k = np.cumsum(labels) / total_positives if total_positives > 0 else np.zeros_like(labels)

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

                line1 = ax1.plot(ranks, profits, label='Net Profit', color='#1f77b4', linewidth=1.5)
                ax1.set_xlabel('Portfolio Rank (Number of Startups)')
                ax1.set_ylabel('Net Profit (USD)', color='#1f77b4')
                ax1.tick_params(axis='y', labelcolor='#1f77b4')

                peak_idx = np.argmax(profits)
                peak_rank = ranks[peak_idx]
                peak_profit = profits[peak_idx]
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

                def currency_formatter(x, pos):
                    if abs(x) >= 1e9: return f'${x/1e9:.1f}B'
                    if abs(x) >= 1e6: return f'${x/1e6:.0f}M'
                    return f'${x:.0f}'
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))

                ax2 = ax1.twinx()
                line2 = ax2.plot(ranks, recall_at_k, label='Recall @ k', color='#ff7f0e', linewidth=1.5, linestyle='--')
                ax2.set_ylabel('Recall / Precision', color='0.4')
                ax2.tick_params(axis='y', labelcolor='0.4')
                ax2.set_ylim(0, 1.05)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

                precision_at_k = np.cumsum(labels) / ranks
                line3 = ax2.plot(ranks, precision_at_k, label='Precision @ k', color='#2ca02c', linewidth=1.5, linestyle='--')

                ax1.scatter([peak_rank], [peak_profit], color='#d62728', s=60, zorder=5, label='Peak Profit')
                ax1.annotate(f'${peak_profit/1e6:.0f}M (k={peak_rank})',
                             (peak_rank, peak_profit),
                             xytext=(peak_rank + len(ranks)*0.05, peak_profit),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                             fontsize=8)

                lines = line1 + line2 + line3 + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=6, label='Peak Profit')]
                labs = [l.get_label() for l in lines]
                ax1.legend(lines, labs, loc='center right', fontsize=8)

                ax1.spines["top"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)
                fig.tight_layout()

                filename = f'roi_net_profit{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/roi_net_profit{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Net Profit Curve: {e}")

    def _plot_precision_at_k(self, ranks: np.ndarray, labels: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None, global_base_rate: Optional[float] = None):
        """
        Generate Precision @ k Curve (Cumulative Precision vs Rank).
        """
        if len(ranks) == 0:
            return

        # Calculate Cumulative Precision
        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks

        # Base Rate: use global if provided, else compute from labels
        base_rate = global_base_rate if global_base_rate is not None else np.mean(labels)

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.plot(ranks, precision_at_k, color='#9467bd', linewidth=1.5, label='Precision @ k')

                # Base Rate Line
                ax.axhline(y=base_rate, color='black', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(ranks[-1] * 0.85, base_rate + 0.02, f'Base Rate ({base_rate:.1%})',
                        fontsize=8, ha='right', va='bottom', color='0.3')

                # Benchmark
                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        color = bm_colors[i % len(bm_colors)]
                        marker = bm_markers[i % len(bm_markers)]
                        ax.scatter([bm_k], [bm_prec], color=color, marker=marker, s=80, zorder=10,
                                   edgecolors='white', linewidths=0.5, label=f"{name} ({bm_prec:.1%})")

                ax.set_xlabel('Portfolio Rank (k)')
                ax.set_ylabel('Precision')
                ax.legend(fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'precision_at_k{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_at_k{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Precision @ k Curve: {e}")

    def _plot_comparative_precision_at_k(self, strategy_predictions: Dict[str, List[Tuple[str, float, int]]], 
                                         benchmark_metrics: Optional[Dict] = None,
                                         filename_suffix: str = ""):
        """
        Generate Comparative Precision @ k Curve for multiple strategies.
        Args:
            strategy_predictions: Dict mapping Strategy Name -> List of Predictions
        """
        if not strategy_predictions:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 5.0))

                # Color palette — reserve consistent style for Global line
                GLOBAL_COLOR = '#1f3d7a'  # dark blue
                non_global = [(k, v) for k, v in strategy_predictions.items() if not k.startswith('All ')]
                n_non_global = len(non_global)
                palette = plt.cm.tab10(np.linspace(0, 0.9, min(n_non_global, 10))) if n_non_global <= 10 else plt.cm.tab20(np.linspace(0, 0.95, n_non_global))

                palette_idx = 0
                for strat_name, preds in strategy_predictions.items():
                    if not preds: continue
                    sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
                    labs = np.array([p[2] for p in sorted_preds])
                    if len(labs) == 0: continue
                    rnk = np.arange(1, len(labs) + 1)
                    prec = np.cumsum(labs) / rnk
                    if strat_name.startswith('All '):
                        ax.plot(rnk, prec, label=strat_name, color=GLOBAL_COLOR, linewidth=2.5, alpha=1.0, zorder=5)
                    else:
                        ax.plot(rnk, prec, label=strat_name, color=palette[palette_idx], linewidth=1.5, alpha=0.85)
                        palette_idx += 1

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    bm_sizes = [80, 80, 80, 80, 150]  # star needs larger s to match visual size
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        ax.scatter([bm_k], [bm_prec], color=bm_colors[i % len(bm_colors)],
                                   marker=bm_markers[i % len(bm_markers)],
                                   s=bm_sizes[i % len(bm_sizes)], zorder=10,
                                   edgecolors='white', linewidths=0.5, label=f"{name} ({bm_prec:.1%})")

                ax.set_xlabel('Portfolio Rank (k)')
                ax.set_ylabel('Precision')
                n_items = len(strategy_predictions) + (len(benchmark_metrics) if benchmark_metrics else 0)
                ncol = 4 if n_items <= 8 else 5
                legend_frac = 0.18  # fixed: 3 rows worth, ensures aligned axes across all plots
                ax.legend(fontsize=7, bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc='lower left',
                         mode='expand', ncol=ncol, framealpha=0.9, borderaxespad=0.,
                         columnspacing=1.0, handletextpad=0.5)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

                max_k = 1000
                if benchmark_metrics:
                    max_bm_k = max([m['k'] for m in benchmark_metrics.values()])
                    max_k = max(1000, max_bm_k + 100)
                ax.set_xlim(0, max_k)
                _apply_thesis_style(ax)
                fig.tight_layout(rect=[0, 0, 1, 1.0 - legend_frac])

                filename = f'precision_at_k_comparative_strategies{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_at_k_comparative_strategies{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Comparative Precision Strategies: {e}")

    def _plot_precision_vs_roi(self, ranks: np.ndarray, labels: np.ndarray, costs: np.ndarray, values: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Precision vs ROI Scatter Plot.
        """
        if len(ranks) == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        cum_costs = np.cumsum(costs)
        cum_values = np.cumsum(values)
        roi_at_k = (cum_values - cum_costs) / cum_costs

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                sc = ax.scatter(precision_at_k, roi_at_k, c=ranks, cmap='viridis', s=15, alpha=0.6)
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label('Portfolio Size (k)', fontsize=9)
                ax.plot(precision_at_k, roi_at_k, color='gray', linewidth=0.5, alpha=0.4)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        ax.scatter([metrics['precision']], [metrics['roi']],
                                   color=bm_colors[i % len(bm_colors)],
                                   marker=bm_markers[i % len(bm_markers)],
                                   s=100, zorder=10, edgecolors='white', linewidths=0.5, label=name)

                k_to_annotate = [10, 50, 100, 500, 1000, len(ranks)]
                for k in k_to_annotate:
                    if k <= len(ranks):
                        idx = k - 1
                        ax.annotate(f'k={k}', (precision_at_k[idx], roi_at_k[idx]),
                                    xytext=(5, 5), textcoords='offset points', fontsize=7)
                        ax.scatter([precision_at_k[idx]], [roi_at_k[idx]], color='#d62728', s=30, zorder=5)

                ax.set_xlabel('Precision @ k')
                ax.set_ylabel('ROI @ k (Multiple)')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
                if benchmark_metrics:
                    ax.legend(fontsize=8)
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'precision_vs_roi{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_vs_roi{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Precision vs ROI: {e}")

    def _plot_k_vs_precision_and_roi(self, ranks: np.ndarray, labels: np.ndarray, costs: np.ndarray, values: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Dual-Axis Plot: k vs Precision and ROI.
        """
        if len(ranks) == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        cum_costs = np.cumsum(costs)
        cum_values = np.cumsum(values)
        roi_at_k = (cum_values - cum_costs) / cum_costs

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

                color_prec = '#9467bd'
                ax1.set_xlabel('Portfolio Size (k)')
                ax1.set_ylabel('Precision', color=color_prec)
                ax1.plot(ranks, precision_at_k, color=color_prec, linewidth=1.5, label='Precision')
                ax1.tick_params(axis='y', labelcolor=color_prec)
                ax1.set_ylim(0, 1.05)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax1.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)

                ax2 = ax1.twinx()
                color_roi = '#2ca02c'
                ax2.set_ylabel('ROI (Multiple)', color=color_roi)
                ax2.plot(ranks, roi_at_k, color=color_roi, linewidth=1.5, linestyle='--', label='ROI')
                ax2.tick_params(axis='y', labelcolor=color_roi)
                ax2.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.3)
                ax2.spines["top"].set_visible(False)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#d62728', '#e377c2', '#8c564b', '#bcbd22']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        c = bm_colors[i % len(bm_colors)]
                        m = bm_markers[i % len(bm_markers)]
                        ax1.scatter([bm_k], [metrics['precision']], color=c, marker=m, s=80, zorder=10,
                                    edgecolors='white', linewidths=0.5, label=f"{name}")

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
                fig.tight_layout()

                filename = f'k_vs_precision_and_roi{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/k_vs_precision_and_roi{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot k vs Precision & ROI: {e}")

    def _plot_k_vs_precision_and_recall(self, ranks: np.ndarray, labels: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Dual-Axis Plot: k vs Precision and Recall.
        X-Axis: k (Rank)
        Left Y-Axis: Precision
        Right Y-Axis: Recall
        """
        if len(ranks) == 0:
            return

        total_positives = labels.sum()
        if total_positives == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        recall_at_k = cum_successes / total_positives

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 5.0))

                color_prec = '#9467bd'
                ax1.set_xlabel('Portfolio Size (k)')
                ax1.set_ylabel('Precision', color=color_prec)
                ax1.plot(ranks, precision_at_k, color=color_prec, linewidth=1.5, label='Precision')
                ax1.tick_params(axis='y', labelcolor=color_prec)
                ax1.set_ylim(0, 1.05)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax1.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)

                ax2 = ax1.twinx()
                color_rec = '#1f77b4'
                ax2.set_ylabel('Recall', color=color_rec)
                ax2.plot(ranks, recall_at_k, color=color_rec, linewidth=1.5, linestyle='--', label='Recall')
                ax2.tick_params(axis='y', labelcolor=color_rec)
                ax2.set_ylim(0, 1.05)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax2.spines["top"].set_visible(False)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        ax1.scatter([bm_k], [bm_prec], color=bm_colors[i % len(bm_colors)],
                                    marker=bm_markers[i % len(bm_markers)], s=80, zorder=10,
                                    edgecolors='white', linewidths=0.5, label=f"{name}")

                # No legend — y-axis labels already identify Precision (left) and Recall (right)
                # Keep same spacing as P@K plots for axis alignment in 2x2 figures
                fig.tight_layout(rect=[0, 0, 1, 0.82])

                filename = f'k_vs_precision_and_recall{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                try:
                    wandb.log({f"analysis/k_vs_precision_and_recall{filename_suffix}": wandb.Image(plot_path)})
                except Exception:
                    pass

        except Exception as e:
            print(f"Failed to plot k vs Precision & Recall: {e}")

    def _filter_mature_startups(self, predictions: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        """
        Filter predictions to only include Mature Startups using get_maturity_mask logic.
        """
        if self.orgs_df is None:
            print("   ⚠️ Organization metadata missing. Cannot filter for maturity.")
            return predictions

        # Calculate mask on orgs_df
        mature_mask = get_maturity_mask(self.orgs_df, self.config)
        
        if mature_mask is None:
            print("   ⚠️ Maturity mask generation failed (check config). Using all startups.")
            return predictions
            
        # Get Set of Mature UUIDs
        # orgs_df needs to be aligned with the mask
        mature_uuids = set(self.orgs_df[mature_mask == 1]['org_uuid'])
        
        filtered_preds = []
        for uuid, score, label in predictions:
            if uuid in mature_uuids:
                filtered_preds.append((uuid, score, label))
                
        print(f"   Filtered: {len(predictions)} -> {len(filtered_preds)} mature startups ({len(filtered_preds)/len(predictions):.1%})")
        
        return filtered_preds

    def analyze_portfolio_by_continent(self, predictions: List[Tuple[str, float, int]],
                                       continent_name: str,
                                       full_df: pd.DataFrame,
                                       filename_suffix: str = "",
                                       verbose: bool = True,
                                       plot_charts: bool = True,
                                       compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to a specific continent.

        Set ``compute_roi=False`` to skip the (heavy) ROI computation when only
        the filtered prediction list is needed (e.g. for aggregated P@k curves).
        """
        # Create a set of UUIDs belonging to this continent
        continent_uuids = set(full_df[full_df['continent'] == continent_name]['org_uuid'])

        constrained_preds = []
        for uuid, score, label in predictions:
            if uuid in continent_uuids:
                constrained_preds.append((uuid, score, label))

        if len(constrained_preds) == 0:
            if verbose:
                print(f"      Scanning... 0 startups found.")
            return []

        if verbose:
            print(f"      Scanning... {len(constrained_preds)} startups found.")

        if compute_roi:
            clean_name = continent_name.replace(" ", "_").replace("/", "_")
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}",
                               title_suffix=f" ({continent_name})", verbose=verbose, plot_charts=plot_charts)

        return constrained_preds

    
    def analyze_portfolio_by_founded_year(self, predictions: List[Tuple[str, float, int]],
                                           year: int,
                                           full_df: pd.DataFrame,
                                           filename_suffix: str = "",
                                           verbose: bool = True,
                                           plot_charts: bool = True,
                                           compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to startups founded in a specific year.

        Mirrors :meth:`analyze_portfolio_by_continent`: relies on a precomputed
        integer ``founded_year`` column on ``full_df`` and returns the
        predictions whose UUID falls in that year. ``compute_roi=False`` skips
        the ROI chart for the aggregated P@k pipeline.
        """
        year_uuids = set(full_df[full_df['founded_year'] == year]['org_uuid'])

        constrained_preds = []
        for uuid, score, label in predictions:
            if uuid in year_uuids:
                constrained_preds.append((uuid, score, label))

        if len(constrained_preds) == 0:
            if verbose:
                print(f"      Scanning... 0 startups found.")
            return []

        if verbose:
            print(f"      Scanning... {len(constrained_preds)} startups found.")

        if compute_roi:
            clean_name = str(int(year))
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}",
                               title_suffix=f" ({int(year)})", verbose=verbose, plot_charts=plot_charts)

        return constrained_preds


    def analyze_portfolio_by_stage(self, predictions: List[Tuple[str, float, int]],
                                   strategy_name: str,
                                   target_stages: List[str],
                                   filename_suffix: str = "",
                                   verbose: bool = True,
                                   plot_charts: bool = True,
                                   compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to a stage-strategy bucket (see STAGE_STRATEGIES).

        Set ``compute_roi=False`` to skip the (heavy) ROI computation when only
        the filtered prediction list is needed (e.g. for aggregated P@k curves).
        """
        constrained_preds = []
        for uuid, score, label in predictions:
            # Check stage at prediction time (Pre-2023)
            stage = self.last_stage_map.get(uuid, 'unknown')
            stage_norm = str(stage).lower().replace(' ', '_').replace('-', '_')

            if stage_norm in target_stages:
                constrained_preds.append((uuid, score, label))

        if len(constrained_preds) == 0:
            if verbose:
                print(f"   ⚠️ No startups found for {strategy_name}.")
            return []

        if verbose:
            print(f"   {strategy_name}: {len(constrained_preds)} startups found.")

        if compute_roi:
            clean_name = strategy_name.replace(" ", "_").replace("/", "_").replace("+", "Plus")
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}",
                               title_suffix=f" ({strategy_name})", verbose=verbose, plot_charts=plot_charts)

        return constrained_preds


    def analyze_portfolio_by_funding_source(self, predictions: List[Tuple[str, float, int]],
                                            strategy_name: str,
                                            allowed_types: List[str],
                                            full_df: pd.DataFrame,
                                            filename_suffix: str = "",
                                            verbose: bool = True,
                                            plot_charts: bool = True,
                                            compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to a funding-source-strategy bucket.

        Set ``compute_roi=False`` to skip the (heavy) ROI computation when only
        the filtered prediction list is needed (e.g. for aggregated P@k curves).
        """
        valid_uuids = set(full_df[full_df['investor_type'].isin(allowed_types)]['org_uuid'])

        constrained_preds = []
        for uuid, score, label in predictions:
            if uuid in valid_uuids:
                constrained_preds.append((uuid, score, label))

        if len(constrained_preds) == 0:
            if verbose:
                print(f"      Scanning... 0 startups found.")
            return []

        if verbose:
            print(f"      Scanning... {len(constrained_preds)} startups found.")

        if compute_roi:
            clean_name = strategy_name.replace(" ", "_").replace("(", "").replace(")", "")
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}",
                               title_suffix=f" ({strategy_name})", verbose=verbose, plot_charts=plot_charts)

        return constrained_preds


    def analyze_portfolio_by_sector(self, predictions: List[Tuple[str, float, int]],
                                    sector_name: str,
                                    full_df: pd.DataFrame,
                                    filename_suffix: str = "",
                                    verbose: bool = True,
                                    plot_charts: bool = True,
                                    compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to startups whose `category_list` contains `sector_name`.

        Crunchbase `category_list` is a comma-separated string of category tags
        (e.g. "SaaS,Enterprise Software"). Membership is checked case-insensitively
        as a substring; pass disambiguating sector names if collisions matter
        (e.g. "Enterprise Software" rather than "Software").

        Set ``compute_roi=False`` to skip the (heavy) ROI computation when only
        the filtered prediction list is needed (e.g. for aggregated P@k curves).
        """
        if 'category_list' not in full_df.columns:
            return []
        match = full_df['category_list'].fillna('').str.contains(
            sector_name, case=False, regex=False
        )
        sector_uuids = set(full_df.loc[match, 'org_uuid'])
        constrained_preds = [(u, s, l) for u, s, l in predictions if u in sector_uuids]
        if not constrained_preds:
            if verbose:
                print(f"      Sector '{sector_name}': 0 startups found.")
            return []
        if verbose:
            print(f"      Sector '{sector_name}': {len(constrained_preds)} startups found.")
        if compute_roi:
            clean_name = sector_name.replace(" ", "_").replace("/", "_")
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}",
                               title_suffix=f" ({sector_name})", verbose=verbose, plot_charts=plot_charts)
        return constrained_preds


    def analyze_portfolio_by_country(self, predictions: List[Tuple[str, float, int]],
                                     country_code: str,
                                     full_df: pd.DataFrame,
                                     filename_suffix: str = "",
                                     verbose: bool = True,
                                     plot_charts: bool = True,
                                     compute_roi: bool = True) -> List[Tuple[str, float, int]]:
        """Filter predictions to startups in `country_code` (ISO 3-letter).

        Set ``compute_roi=False`` to skip the (heavy) ROI computation when only
        the filtered prediction list is needed (e.g. for aggregated P@k curves).
        """
        if 'country_code' not in full_df.columns:
            return []
        country_uuids = set(full_df.loc[full_df['country_code'] == country_code, 'org_uuid'])
        constrained_preds = [(u, s, l) for u, s, l in predictions if u in country_uuids]
        if not constrained_preds:
            if verbose:
                print(f"      Country '{country_code}': 0 startups found.")
            return []
        if verbose:
            print(f"      Country '{country_code}': {len(constrained_preds)} startups found.")
        if compute_roi:
            self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{country_code}",
                               title_suffix=f" ({country_code})", verbose=verbose, plot_charts=plot_charts)
        return constrained_preds


    def analyze_investor_benchmark(self) -> Dict[str, Dict]:
        """
        Analyze the performance of benchmark investors during the target period.
        Period: Jan 2022 - June 2023 (18 months prior to prediction date).
        """
        if self.investments_df is None or self.funding_df is None:
            return {}
            
        #print(f"\n🏆 Benchmarking against Top Investors...")
        results = {}
        
        # Pre-calculate common maps to avoid re-doing it for every investor
        # Funding after June 2023
        future_funding = self.funding_df[self.funding_df['announced_on'] > self.roi_start_date]
        funded_orgs = set(future_funding['org_uuid'].unique())
        
        # IPOs
        ipo_orgs = set()
        if self.ipos_df is not None:
            ipo_orgs = set(self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date]['org_uuid'])
            
        # Acquisitions
        acq_orgs = set()
        if self.acq_df is not None:
            acq_orgs = set(self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date]['org_uuid'])
            
        # Maps for Value
        future_funding_map = future_funding.groupby('org_uuid').agg({
            'raised_amount_usd': 'sum',
            'post_money_valuation_usd': 'max',
            'investment_type': 'last'
        }).to_dict('index')
        
        ipo_map = {}
        if self.ipos_df is not None:
            ipo_map = self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date].set_index('org_uuid')['valuation_price_usd'].to_dict()
            
        acq_map = {}
        if self.acq_df is not None:
            acq_map = self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date].set_index('org_uuid')['price_usd'].to_dict()
            
        total_funding_map = {}
        if self.orgs_df is not None:
            total_funding_map = self.orgs_df.set_index('org_uuid')['total_funding_usd'].to_dict()

        successful_uuids = funded_orgs.union(ipo_orgs).union(acq_orgs)
        
        for investor_name, investor_uuid in self.BENCHMARK_INVESTORS.items():
            # 1. Filter Investments by Investor
            inv_investments = self.investments_df[self.investments_df['investor_uuid'] == investor_uuid].copy()
            
            if inv_investments.empty:
                print(f"   ⚠️ No investments found for {investor_name}.")
                continue
                
            merged_df = inv_investments.merge(self.funding_df, on='funding_round_uuid', how='inner')
            
            # 3. Filter by Date Range (Jan 2022 - June 2023)
            start_date = pd.Timestamp('2022-01-01')
            end_date = self.roi_start_date # June 2023
            
            benchmark_df = merged_df[
                (merged_df['announced_on'] >= start_date) & 
                (merged_df['announced_on'] <= end_date)
            ].copy()
            
            if benchmark_df.empty:
                print(f"   ⚠️ No {investor_name} investments found in benchmark period.")
                continue
                
            unique_orgs = benchmark_df['org_uuid'].unique()
            k = len(unique_orgs)
            
            # Calculate Precision
            successes = 0
            for uuid in unique_orgs:
                if uuid in successful_uuids:
                    successes += 1
            precision = successes / k if k > 0 else 0
            
            # Calculate ROI
            total_cost = k * self.TICKET_SIZE
            total_value = 0.0
            
            for uuid in unique_orgs:
                cost = self.TICKET_SIZE
                
                # Entry Val Logic
                data = future_funding_map.get(uuid, {})
                raised = data.get('raised_amount_usd', 0)
                if pd.isna(raised): raised = 0
                
                round_type = data.get('investment_type', None)
                if pd.isna(round_type):
                    dilution_rate = 0.10
                else:
                    round_type_str = str(round_type).lower().replace(' ', '_').replace('-', '_')
                    dilution_rate = self.DILUTION_RATES.get(round_type_str, 0.15)
                    
                current_post_val = raised / dilution_rate if dilution_rate > 0 else raised
                current_pre_val = current_post_val - raised
                entry_val = current_pre_val / self.STEP_UP_MULTIPLE if self.STEP_UP_MULTIPLE > 0 else current_pre_val
                
                if raised == 0 or entry_val <= 100_000:
                     total_funding = total_funding_map.get(uuid, 0)
                     if pd.notna(total_funding) and total_funding > 0:
                         entry_val = total_funding * self.HISTORICAL_FUNDING_MULTIPLE
                     else:
                         entry_val = 5_000_000
                         
                ownership = self.TICKET_SIZE / entry_val
                if ownership > 0.20: ownership = 0.20
                
                # Exit Value
                exit_val = 0.0
                status = 'FAIL'
                
                if uuid in ipo_map:
                    exit_val = ipo_map[uuid] * ownership
                    status = 'EXIT'
                elif uuid in acq_map:
                    exit_val = acq_map[uuid] * ownership
                    status = 'EXIT'
                    
                if status == 'EXIT' and (pd.isna(exit_val) or exit_val == 0):
                     total_funding = total_funding_map.get(uuid, 0)
                     if pd.isna(total_funding): total_funding = 0
                     if total_funding > 0:
                         exit_val = (total_funding * self.UNDISCLOSED_EXIT_MULTIPLE) * ownership
                     else:
                         exit_val = cost
                
                if status != 'EXIT':
                    if uuid in funded_orgs:
                        exit_val = cost * self.STEP_UP_MULTIPLE # Paper Gain
                        status = 'FUNDED'
                    else:
                        status = 'FAIL'
                
                total_value += exit_val
                
            roi = (total_value - total_cost) / total_cost if total_cost > 0 else 0
            
            results[investor_name] = {
            'precision': precision,
            'roi': roi,
            'k': k
        }
            
        return results


    def compare_portfolios(self, predictions: List[Tuple[str, float, int]], investor_name: str):
        """
        Compare the model's top predictions against a specific investor's portfolio.
        Analyzes Sector, Stage, and Missed Winners.
        """
        if investor_name not in self.BENCHMARK_INVESTORS:
            return
        if self.investments_df is None or self.funding_df is None:
            return

        investor_uuid = self.BENCHMARK_INVESTORS[investor_name]
        print(f"\n👉 Comparing vs {investor_name}...")
        
        # 1. Get Investor Portfolio (Jan 2022 - June 2023)
        inv_investments = self.investments_df[self.investments_df['investor_uuid'] == investor_uuid]
        merged_df = inv_investments.merge(self.funding_df, on='funding_round_uuid', how='inner')
        
        start_date = pd.Timestamp('2022-01-01')
        end_date = self.roi_start_date
        
        benchmark_df = merged_df[
            (merged_df['announced_on'] >= start_date) & 
            (merged_df['announced_on'] <= end_date)
        ].copy()
        
        investor_uuids = set(benchmark_df['org_uuid'].unique())
        k = len(investor_uuids)
        
        if k == 0:
            print(f"   No investments found for {investor_name} in target period.")
            return

        # 2. Get Model Portfolio (Top k)
        # Sort predictions by score
        sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        model_top_k = sorted_preds[:k]
        model_uuids = set([p[0] for p in model_top_k])
        
        # 3. Overlap
        overlap = investor_uuids.intersection(model_uuids)
        jaccard = len(overlap) / len(investor_uuids.union(model_uuids))
        #print(f"   Portfolio Size (k): {k}")
        #print(f"   Overlap: {len(overlap)} startups ({len(overlap)/k:.1%})")
        
        # 3.1 Test Set Coverage (New)
        all_test_uuids = set([p[0] for p in predictions])
        investor_in_test_set = investor_uuids.intersection(all_test_uuids)
        
        #print(f"   \n   🔍 Test Set Coverage:")
        #print(f"   Total Investor Bets: {len(investor_uuids)}")
        #print(f"   Bets in Test Set: {len(investor_in_test_set)} ({len(investor_in_test_set)/len(investor_uuids) if len(investor_uuids) > 0 else 0:.1%})")
        
        if len(investor_in_test_set) > 0:
            # Analyze Rank distribution of these bets
            ranks = []
            scores = []
            for uuid in investor_in_test_set:
                 for i, p in enumerate(sorted_preds):
                     if p[0] == uuid:
                         ranks.append(i+1)
                         scores.append(p[1])
                         break
            
            ranks = np.array(ranks)
            # print(f"   \n   Model Performance on Addressable Investor Bets:")
            # print(f"   Median Rank: {np.median(ranks):.0f} / {len(predictions)}")
            # print(f"   Mean Score: {np.mean(scores):.4f}")
            # print(f"   % in Top 100: {np.sum(ranks <= 100) / len(ranks):.1%}")
            # print(f"   % in Top 10%: {np.sum(ranks <= len(predictions)*0.1) / len(ranks):.1%}")
        
        # 4. Sector Analysis

        # Helper to get sectors
        def get_sectors(uuids):
            sectors = []
            if self.orgs_df is not None:
                subset = self.orgs_df[self.orgs_df['org_uuid'].isin(uuids)]
                for cats in subset['category_list'].dropna():
                    # Split by ',' or '|' (Crunchbase uses either)
                    parts = str(cats).split(',')
                    if '|' in str(cats): parts = str(cats).split('|')

                    sectors.extend([p.strip() for p in parts])
            return pd.Series(sectors).value_counts(normalize=True).head(5)

        # print(f"   \n   Top 5 Sectors ({investor_name}):")
        # print(get_sectors(investor_uuids).to_string())
        # print(f"   \n   Top 5 Sectors (Model):")
        # print(get_sectors(model_uuids).to_string())
        
        # 5. Stage Analysis (at time of investment)
        # For investor: use the investment_type from the round they invested in
        # For model: use the last_stage_map (stage at prediction time)
        
        inv_stages = benchmark_df['investment_type'].value_counts(normalize=True).head(5)
        
        model_stages = []
        for uuid in model_uuids:
            stage = self.last_stage_map.get(uuid, 'Unknown')
            model_stages.append(stage)
        model_stage_counts = pd.Series(model_stages).value_counts(normalize=True).head(5)
        
        # print(f"   \n   Top 5 Stages ({investor_name}):")
        # print(inv_stages.to_string())
        # print(f"   \n   Top 5 Stages (Model):")
        # print(model_stage_counts.to_string())
        
        # 6. Missed Winners (High Conviction Investor, Low Model Score)
        # Find startups in Investor Portfolio that were SUCCESSFUL but NOT in Model Top k
        # Only consider those IN THE TEST SET (Addressable)
        
        # Re-identify success (simplified)
        future_funding = self.funding_df[self.funding_df['announced_on'] > self.roi_start_date]
        funded_orgs = set(future_funding['org_uuid'].unique())
        ipo_orgs = set(self.ipos_df['org_uuid']) if self.ipos_df is not None else set()
        acq_orgs = set(self.acq_df['org_uuid']) if self.acq_df is not None else set()
        successful_uuids = funded_orgs.union(ipo_orgs).union(acq_orgs)
        
        missed_winners = []
        for uuid in investor_in_test_set: # Only check addressable ones
            if uuid in successful_uuids and uuid not in model_uuids:
                # Find rank in model
                rank = -1
                score = 0.0
                for i, p in enumerate(sorted_preds):
                    if p[0] == uuid:
                        rank = i + 1
                        score = p[1]
                        break
                
                # Get Name
                name = "Unknown"
                if self.orgs_df is not None:
                    name_row = self.orgs_df[self.orgs_df['org_uuid'] == uuid]
                    if not name_row.empty:
                        name = name_row.iloc[0]['name']
                        
                missed_winners.append((name, rank, score))
        
        # Sort by Rank (closest to being picked)
        missed_winners.sort(key=lambda x: x[1])
        
        # print(f"   \n   Top 5 Addressable Missed Winners (In Test Set, Investor picked, Model ranked low):")
        # if not missed_winners:
        #     print("     None found.")
        # for name, rank, score in missed_winners[:5]:
        #     print(f"     - {name}: Rank={rank}, Score={score:.4f}")
    

    def calculate_roi(self, predictions: List[Tuple[str, float, int]], top_k: int = 100, 
                      filename_suffix: str = "", title_suffix: str = "", verbose: bool = True, plot_charts: bool = True):
        """
        Calculate Investor-Centric ROI for the predictions.

        Methodology (Strategy-Aware):
        -----------------------------
        1.  **Fixed Ticket Size**: 
            - We simulate investing a fixed $1,000,000 ticket into each predicted startup.

        2.  **Entry Valuation & Ownership**:
            - **Entry Valuation**: Estimated using the formula: `(Raised / Dilution) / Step_Up`.
            - **Step-Up Multiple**: NOW VARIABLE.
                - Venture Strategy: 3.0x (Aggressive)
                - Liquidity Strategy: 1.5x (Conservative)
            
        3.  **Exit Value Calculation (DECOUPLED FROM PREDICTION)**:
            - **IPO**: `Market_Cap * Ownership`.
            - **Acquisition**: `Acquisition_Price * Ownership`.
            - **Funded (Paper Gain)**: 
                - Condition: Did it ACTUALLY raise money in the target window? (Ground Truth)
                - Value: `Cost * Step_Up`.
                - Note: This credits the asset regardless of whether we predicted 'Exit' or 'Funding'. 
                  Crucially, for Liquidity Strategy, the Step_Up is lower (1.5x), accurately reflecting a "Liquidity Trap" (Good asset, but not the Exit we wanted).
            - **Fail**: $0.

        Args:
            predictions (list): List of (uuid, score, label) tuples.
            filename_suffix (str): Suffix for saved plot files.
            title_suffix (str): Suffix for plot titles.
            verbose (bool): If True, print detailed output.
            plot_charts (bool): If True, generate and save plots.
        """
        if self.target_window_df is None:
            return {}
            
        if verbose:
            print(f"\n💰 ROI Analysis{title_suffix}")
        
        metrics = {}
        
        # Group funding by org
        org_funding = self.target_window_df.groupby('org_uuid').agg({
            'raised_amount_usd': 'sum',
            'post_money_valuation_usd': 'max',
            'investment_type': 'last'
        }).reset_index()
        
        successful_orgs = set(org_funding['org_uuid'].unique())
        funding_map = org_funding.set_index('org_uuid').to_dict('index')
        
        # Sort predictions by score
        sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # --- Pre-calculate Costs and Values for ALL predictions for plotting ---
        all_costs = []
        all_values = []
        full_data = [] # Store all data for filtering
        
        # Load Total Funding for Valuation Fallback
        total_funding_map = {}
        if self.orgs_df is not None:
            total_funding_map = self.orgs_df.set_index('org_uuid')['total_funding_usd'].to_dict()
        
        # Load IPOs and Acquisitions for Exit Value lookup
        # Filter for Exits occurring AFTER ROI Start Date (Future Exits)
        ipo_map = {}
        if self.ipos_df is not None:
            # Filter by date
            future_ipos = self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date]
            ipo_map = future_ipos.set_index('org_uuid')['valuation_price_usd'].to_dict()
            
        acq_map = {}
        if self.acq_df is not None:
            # Filter by date
            future_acqs = self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date]
            acq_map = future_acqs.set_index('org_uuid')['price_usd'].to_dict()

        # Load Names for Export
        name_map = {}
        if self.orgs_df is not None:
            name_map = self.orgs_df.set_index('org_uuid')['name'].to_dict()

        for uuid, score, label in sorted_preds:
            # Get Funding Data
            data = funding_map.get(uuid, {})
            raised = data.get('raised_amount_usd', 0)
            if pd.isna(raised): raised = 0
            
            # Determine Dilution Rate (Ownership Share)
            round_type = data.get('investment_type', None)
            
            if pd.isna(round_type):
                round_type_str = None # Export as None/NaN
                dilution_rate = 0.10 # Default to Seed rate for safety if needed
            else:
                round_type_str = str(round_type).lower().replace(' ', '_').replace('-', '_')
                dilution_rate = self.DILUTION_RATES.get(round_type_str, 0.15)
            
            # --- Fixed Ticket Size Simulation ---
            # Cost: Fixed Ticket Size
            cost = self.TICKET_SIZE
            all_costs.append(cost)
            
            # Calculate Entry Valuation (Pre-Money of the round we "invested" in)
            # We assume we invested in the PREVIOUS round.
            # Current Post-Val = Raised / Dilution
            # Current Pre-Val = Post-Val - Raised
            # Entry Val = Current Pre-Val / Step_Up (Approximate)
            
            current_post_val = raised / dilution_rate if dilution_rate > 0 else raised
            current_pre_val = current_post_val - raised
            
            # Use Tiered Step Up to estimate entry valuation backward
            # (Approximation: We invert the step up logic)
            # Default to 2.0x for backward estimation if unknown
            entry_step_divisor = 2.0 
            entry_val = current_pre_val / entry_step_divisor
            
            # Default Valuation if data missing or weird
            if raised == 0 or entry_val <= 100_000: 
                # Check Historical Funding
                total_funding = total_funding_map.get(uuid, 0)
                if pd.notna(total_funding) and total_funding > 0:
                     entry_val = total_funding * self.HISTORICAL_FUNDING_MULTIPLE
                else:
                     entry_val = 5_000_000 # Default Seed Pre-Money
            

            # Ownership: Ticket / Entry Val (Capped at 20%)
            ownership = self.TICKET_SIZE / entry_val
            if ownership > 0.20:
                ownership = 0.20
            
            # --- Exit Value Calculation ---
            exit_val = 0.0
            status = 'FAIL'
            
            # Check for IPO
            if uuid in ipo_map:
                # IPO Value = Market Cap * Ownership
                market_cap = ipo_map[uuid]
                exit_val = market_cap * ownership
                status = 'EXIT'
                
            # Check for Acquisition
            elif uuid in acq_map:
                # Acquisition Value = Price * Ownership
                acq_price = acq_map[uuid]
                exit_val = acq_price * ownership
                status = 'EXIT'
            
            # Check for Undisclosed Exit (Status=EXIT but Value=0)
            if status == 'EXIT' and (pd.isna(exit_val) or exit_val == 0):
                 # Fallback Estimation for Undisclosed Exit
                 total_funding = total_funding_map.get(uuid, 0)
                 if pd.isna(total_funding): total_funding = 0
                 
                 if total_funding > 0:
                     # Estimate: Modest Win (Capital Returned)
                     estimated_exit_valuation = total_funding * self.UNDISCLOSED_EXIT_MULTIPLE
                     exit_val = estimated_exit_valuation * ownership
                 else:
                     # Fallback: Break-even (1x Cost)
                     exit_val = cost
            
            # Check for Funding (Paper Gain) if not Exited
            if status != 'EXIT':
                # Did it raise money? (ground-truth check)
                if raised > 0:
                    # Determine Step-Up based on Stage of the NEW Round
                    # "Valuation follows Stage"
                    if round_type_str in self.TIERED_STEP_UPS:
                        current_step_up = self.TIERED_STEP_UPS[round_type_str]
                    else:
                        current_step_up = self.TIERED_STEP_UPS['default']
                        
                    # Paper Gain = Cost * current_step_up
                    exit_val = cost * current_step_up
                    status = 'FUNDED'
                else:
                    status = 'FAIL'
            
            all_values.append(exit_val)
            
            # Collect Data for Export
            profit = exit_val - cost
            roi_mult = exit_val / cost if cost > 0 else 0

            name = name_map.get(uuid, "Unknown")
            
            full_data.append({
                'UUID': uuid,
                'Name': name,
                'Score': score,
                'Status': status,
                'Cost': cost,
                'Value': exit_val,
                'Profit': profit,
                'ROI_Multiple': roi_mult,
                'Raised': raised,
                'Entry_Val': entry_val,
                'Ownership': ownership,
                'Stage': round_type_str,
                'Dilution': dilution_rate
            })

        all_costs = np.array(all_costs)
        all_values = np.array(all_values)
        
        # --- Process Top 100 Lists ---
        df_full = pd.DataFrame(full_data)
        
        # 1. Standard Top 100 (All Predictions)
        df_standard = df_full.head(100).copy()
        df_standard['Rank'] = range(1, len(df_standard) + 1)
        
        export_path = os.path.join(self.output_dir, f'top_100_investments{filename_suffix}.csv')
        df_standard.to_csv(export_path, index=False)
        # print(f"   Saved Standard Top 100 to {export_path}")
        
        # Calculate Standard Metrics
        std_cost = df_standard['Cost'].sum()
        std_val = df_standard['Value'].sum()
        std_roi = (std_val - std_cost) / std_cost if std_cost > 0 else 0
        std_prec = len(df_standard[df_standard['Status'] != 'FAIL']) / len(df_standard)
        
        if verbose:
            print(f"   Top-100 Portfolio (Standard){title_suffix}:")
            print(f"      Precision: {std_prec:.2%}")
            print(f"      ROI: {std_roi:.1%} ({std_val/std_cost:.2f}x)")
            print(f"      Total Capital: ${std_cost/1e6:.1f}M")
            print(f"      Total Value: ${std_val/1e6:.1f}M")

        if wandb.run is not None:
            wandb.log({
                f"analysis/top_100_precision{filename_suffix}": std_prec,
                f"analysis/top_100_roi{filename_suffix}": std_roi,
                f"analysis/top_100_capital{filename_suffix}": std_cost,
                f"analysis/top_100_value{filename_suffix}": std_val
            })



        # Calculate Cumulative Arrays for Plotting (Standard)
        cum_costs = np.cumsum(all_costs)
        cum_values = np.cumsum(all_values)
        ranks = np.arange(1, len(sorted_preds) + 1)
        sorted_labels = np.array([p[2] for p in sorted_preds]) # Labels in rank order
        
        # 1.1 Investor Benchmark (a16z)
        # We need to call this BEFORE plotting to pass the metrics
        benchmark_metrics = self.analyze_investor_benchmark()
        
        # --- Plotting (Cumulative) ---
        if plot_charts:
            # self._plot_roi_j_curve(ranks, sorted_costs_cum, sorted_values_cum, top_k, filename_suffix=filename_suffix) # Disabled
            self._plot_net_profit_curve(ranks, cum_costs, cum_values, labels=sorted_labels, filename_suffix=filename_suffix)
            
            # Plot Precision @ k (Standard) - NO BENCHMARK
            # self._plot_precision_at_k(ranks, sorted_labels, filename_suffix=filename_suffix)
            
            # Plot Precision vs ROI (Standard)
            # self._plot_precision_vs_roi(ranks, sorted_labels, sorted_costs_cum, sorted_values_cum, filename_suffix=filename_suffix)
        
        # Plot Precision @ k (Standard) - NO BENCHMARK
        # We need labels in rank order
        # sorted_labels = np.array([p[2] for p in sorted_preds])
        if plot_charts:
            self._plot_precision_at_k(ranks, sorted_labels, filename_suffix=filename_suffix)

            # Plot Precision vs ROI (Standard) - WITH BENCHMARK
            self._plot_precision_vs_roi(ranks, sorted_labels, all_costs, all_values, filename_suffix=filename_suffix, benchmark_metrics=benchmark_metrics)

            # Plot k vs Precision & ROI (Standard) - NO BENCHMARK
            self._plot_k_vs_precision_and_roi(ranks, sorted_labels, all_costs, all_values, filename_suffix=filename_suffix)

            # Plot k vs Precision & Recall (Standard) - NO BENCHMARK
            self._plot_k_vs_precision_and_recall(ranks, sorted_labels, filename_suffix=filename_suffix)
        
        # --- Zoomed Plots (k=1000) ---
        zoom_k = 1000
        if len(ranks) > 0 and plot_charts:
            # Slice data
            zoom_idx = min(len(ranks), zoom_k)
            zoom_ranks = ranks[:zoom_idx]
            zoom_labels = sorted_labels[:zoom_idx]
            zoom_costs = all_costs[:zoom_idx]
            zoom_values = all_values[:zoom_idx]
            
            # Plot Precision @ k (Zoomed) - WITH BENCHMARK, global base rate
            self._plot_precision_at_k(zoom_ranks, zoom_labels, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics, global_base_rate=np.mean(sorted_labels))
            
            # Plot k vs Precision & ROI (Zoomed) - WITH BENCHMARK
            self._plot_k_vs_precision_and_roi(zoom_ranks, zoom_labels, zoom_costs, zoom_values, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics)

            # Plot k vs Precision & Recall (Zoomed) - WITH BENCHMARK
            self._plot_k_vs_precision_and_recall(zoom_ranks, zoom_labels, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics)

        

        
        k_values = [10, 50, 100, 500, 1000]
        for k in k_values:
            if k > len(sorted_preds):
                continue
                
            top_k_preds = sorted_preds[:k]
            top_k_uuids = [p[0] for p in top_k_preds]
            
            # 1. Success Rate (Precision)
            successes = [uuid for uuid in top_k_uuids if uuid in successful_orgs]
            num_successes = len(successes)
            precision = num_successes / k
            
            # 2. Portfolio ROI (Dilution Model)
            # Sum of Costs and Values for top k
            k_cost = np.sum(all_costs[:k])
            k_value = np.sum(all_values[:k])
            
            roi_multiple = k_value / k_cost if k_cost > 0 else 0.0
            roi_percentage = (roi_multiple - 1) * 100
            
            # Metrics
            metrics[f'roi_precision_at_{k}'] = precision
            metrics[f'roi_total_raised_at_{k}'] = k_cost
            metrics[f'roi_percentage_at_{k}'] = roi_percentage
            metrics[f'roi_multiple_at_{k}'] = roi_multiple
            
            if k == 100:
                print(f"   Top-100 Portfolio:")
                print(f"      Precision: {precision:.2%}")
                print(f"      ROI: {roi_percentage:.1f}% ({roi_multiple:.2f}x)")
                print(f"      Total Capital Deployed: ${k_cost/1e6:.1f}M")
                print(f"      Total Portfolio Value: ${k_value/1e6:.1f}M")
                
                # Plot J-Curve for Top 100
                # self._plot_roi_j_curve(ranks, cum_costs, cum_values, top_k=100, filename_suffix=filename_suffix) # Disabled

        # Log to W&B
        if wandb.run is not None:
            wandb.log({f"roi_{k}": v for k, v in metrics.items()})
            
        # Add Curves to Metrics for external plotting
        metrics['curves'] = (ranks, cum_costs, cum_values, sorted_labels)

        return metrics

    def _calculate_group_metrics(self, df: pd.DataFrame, group_col: str, min_support: int = 10) -> pd.DataFrame:
        """
        Calculate comprehensive metrics for each group in the dataframe.
        Metrics: AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy, Support.
        """
        metrics_list = []
        
        # Get unique groups
        groups = df[group_col].unique()
        
        for group in groups:
            group_df = df[df[group_col] == group]
            
            # Skip if support is too low
            if len(group_df) < min_support:
                continue
                
            y_true = group_df['gt_label'].values
            y_score = group_df['score'].values
            y_pred = (y_score >= 0.5).astype(int) # Default threshold
            
            # Skip if only one class present (cannot calc AUC)
            if len(np.unique(y_true)) < 2:
                auc_roc = np.nan
                auc_pr = np.nan
            else:
                auc_roc = roc_auc_score(y_true, y_score)
                auc_pr = average_precision_score(y_true, y_score)
                
            metrics = {
                group_col: group,
                'support': len(group_df),
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred),
                'mean_score': np.mean(y_score),
                'success_rate': np.mean(y_true)
            }
            metrics_list.append(metrics)
            
        if not metrics_list:
            return pd.DataFrame()
            
        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df.sort_values('auc_pr', ascending=False)

    def _plot_analysis(self, stats_df: pd.DataFrame, raw_df: pd.DataFrame, group_col: str,
                      metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = "", color: str = '#1f77b4',
                      order: Optional[List[str]] = None):
        """
        Generate Bar Plot for Metric and Box Plot for Scores.
        """
        if stats_df.empty:
            return

        if order is None:
            order = stats_df[group_col].tolist()
        raw_df_filtered = raw_df[raw_df[group_col].isin(stats_df[group_col])].copy()

        # 1. Metric Bar Plot
        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                sns.barplot(data=stats_df, x=group_col, y=metric, order=order, color=color, ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                _apply_thesis_style(ax)
                fig.tight_layout()
                plot_path = os.path.join(self.output_dir, f'{group_col}_performance{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)
            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_performance_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} metric: {e}")

        # 2. Score Box Plot
        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                sns.boxplot(data=raw_df_filtered, x=group_col, y='score', order=order, color=color, ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                _apply_thesis_style(ax)
                fig.tight_layout()
                plot_path = os.path.join(self.output_dir, f'{group_col}_scores_boxplot{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)
            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_scores_boxplot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} boxplot: {e}")

    def _plot_correlation(self, stats_df: pd.DataFrame, group_col: str, metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = ""):
        """
        Generate Scatter Plot correlating Support with Metric.
        """
        if stats_df.empty:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.scatter(stats_df['support'], stats_df[metric], alpha=0.7, color='#1f77b4',
                           edgecolors='white', linewidths=0.5, s=50)

                if len(stats_df) < 50:
                    for _, row in stats_df.iterrows():
                        ax.annotate(str(row[group_col]), (row['support'], row[metric]),
                                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords='offset points')

                ax.set_xlabel('Support (Number of Samples)')
                ax.set_ylabel(metric.upper())
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_correlation{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_correlation_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} correlation: {e}")

    def _plot_success_correlation(self, stats_df: pd.DataFrame, group_col: str, metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = ""):
        """
        Generate Scatter Plot correlating Success Rate with Metric.
        """
        if stats_df.empty:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                min_s, max_s = stats_df['support'].min(), stats_df['support'].max()
                if max_s > min_s:
                    norm = (stats_df['support'] - min_s) / (max_s - min_s)
                else:
                    norm = pd.Series([0.5] * len(stats_df))
                sizes = 50 + norm * 300

                ax.scatter(stats_df['success_rate'], stats_df[metric], s=sizes, alpha=0.7,
                           color='#1f77b4', edgecolors='white', linewidths=0.5)

                if len(stats_df) > 2:
                    sns.regplot(data=stats_df, x='success_rate', y=metric, scatter=False,
                                color='#d62728', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                top_metric = stats_df.nlargest(5, metric).index
                top_success = stats_df.nlargest(5, 'success_rate').index
                for idx in set(top_metric).union(set(top_success)):
                    row = stats_df.loc[idx]
                    ax.annotate(str(row[group_col]), (row['success_rate'], row[metric]),
                                fontsize=7, alpha=0.8, xytext=(4, 2), textcoords='offset points')

                ax.set_xlabel('Success Rate (Mean Ground Truth)')
                ax.set_ylabel(metric.upper())
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_success_correlation{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_success_correlation_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} success correlation: {e}")

    def _plot_dual_metric_correlation(self, stats_df: pd.DataFrame, group_col: str,
                                       title_suffix: str = "", filename_suffix: str = ""):
        """
        Plot Success Rate vs both AUC-PR and AUC-ROC, showing that AUC-ROC is stable
        while AUC-PR varies with base rate. Bubble size = support. All points labelled.
        """
        if stats_df.empty or 'auc_pr' not in stats_df.columns or 'auc_roc' not in stats_df.columns:
            return

        try:
            from adjustText import adjust_text
        except ImportError:
            adjust_text = None

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                df = stats_df.sort_values('success_rate').copy()
                point_labels = df[group_col].astype(str).values

                min_s, max_s = df['support'].min(), df['support'].max()
                if max_s > min_s:
                    norm_support = (df['support'] - min_s) / (max_s - min_s)
                else:
                    norm_support = pd.Series([0.5] * len(df))
                sizes = 60 + norm_support * 300

                # AUC-PR scatter + trendline
                ax.scatter(df['success_rate'], df['auc_pr'], s=sizes, alpha=0.8,
                          color='#1f77b4', edgecolors='white', linewidths=0.5, zorder=3, label='AUC-PR')
                if len(df) > 2:
                    sns.regplot(data=df, x='success_rate', y='auc_pr', scatter=False,
                               color='#1f77b4', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                # AUC-ROC scatter + trendline
                ax.scatter(df['success_rate'], df['auc_roc'], s=sizes, alpha=0.8,
                          color='#ff7f0e', edgecolors='white', linewidths=0.5, zorder=3, label='AUC-ROC')
                if len(df) > 2:
                    sns.regplot(data=df, x='success_rate', y='auc_roc', scatter=False,
                               color='#ff7f0e', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                # Label all points (use adjustText to prevent overlap)
                texts = []
                for i, label in enumerate(point_labels):
                    row = df.iloc[i]
                    texts.append(ax.text(row['success_rate'], row['auc_pr'], label,
                                         fontsize=7, color='#1f77b4', alpha=0.9))
                    texts.append(ax.text(row['success_rate'], row['auc_roc'], label,
                                         fontsize=7, color='#ff7f0e', alpha=0.9))

                if adjust_text is not None:
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', alpha=0.4, lw=0.5),
                                expand=(1.3, 1.5), force_text=(0.5, 0.8))

                ax.set_xlabel('Success Rate (Base Rate)')
                ax.set_ylabel('Metric Value')

                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                           markersize=8, label='AUC-PR'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                           markersize=8, label='AUC-ROC'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', alpha=0.4,
                           markersize=5, label=f'n={int(min_s)}'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', alpha=0.4,
                           markersize=11, label=f'n={int(max_s)}'),
                    Line2D([0], [0], color='grey', linestyle='--', alpha=0.5, linewidth=1,
                           label='Trend'),
                    Patch(facecolor='grey', alpha=0.15, label='95% CI'),
                ]
                ax.legend(handles=legend_elements, fontsize=8, loc='best', framealpha=0.9)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_dual_metric{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_dual_metric{filename_suffix}": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} dual metric correlation: {e}")

    def analyze_sectors(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by sector (category_list)."""
        k_str = f"Top {top_k}" if top_k else "All"
        # print(f"\n🏭 Sector Analysis ({k_str})")
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        top_df['category_list'] = top_df['category_list'].fillna('Unknown').astype(str)
        top_df['sector'] = top_df['category_list'].apply(lambda x: [s.strip() for s in x.split(',')])
        exploded_df = top_df.explode('sector')
        
        # Calculate Metrics
        sector_stats = self._calculate_group_metrics(exploded_df, 'sector', min_support=20)
        
        if sector_stats.empty:
            print("   No sectors with sufficient support found.")
            return

        # print(f"   Top 5 Performing Sectors (by AUC-PR):")
        # print(sector_stats.head(5)[['sector', 'support', 'auc_pr', 'auc_roc', 'precision']])
        
        output_path = os.path.join(self.output_dir, f'sector_performance{filename_suffix}.csv')
        sector_stats.to_csv(output_path, index=False)
        # print(f"   Saved sector analysis to {output_path}")
        
        # Plotting (Top 20 by Support, then sorted by AUC-PR)
        # Select top 20 by support
        top_20_sectors = sector_stats.sort_values('support', ascending=False).head(20)
        # Sort by AUC-PR for display
        top_20_sectors = top_20_sectors.sort_values('auc_pr', ascending=False)
        
        self._plot_analysis(top_20_sectors, exploded_df, 'sector', metric='auc_pr', 
                           title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plot (All Sectors)
        self._plot_correlation(sector_stats, 'sector', metric='auc_pr', title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(sector_stats, 'sector', metric='auc_pr', title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        # Dual metric plot (top 20 by support for readability)
        top_sectors = sector_stats.sort_values('support', ascending=False).head(20)
        self._plot_dual_metric_correlation(top_sectors, 'sector', title_suffix=f"Sector (Top 20)", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/sector_performance{filename_suffix}": wandb.Table(dataframe=sector_stats)})

        # Comparative Precision @ k (Top 10 Sectors)
        # Select top 10 by support for comparative plots
        top_10_sectors_df = sector_stats.sort_values('support', ascending=False).head(10)
        top_10_sectors_list = top_10_sectors_df['sector'].values
        
        preds_by_sector = {}
        
        # --- 1. Global Baseline ---
        # Global uses the unique startups from top_df (before explode)
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_sector['All Sectors'] = global_preds_list
        
        # --- 2. Top 10 Combined ---
        # Union of startups in top 10 sectors
        top_10_combined_df = exploded_df[exploded_df['sector'].isin(top_10_sectors_list)]
        top_10_combined_unique = top_10_combined_df.drop_duplicates(subset='org_uuid')
        
        combined_preds_list = list(zip(top_10_combined_unique['org_uuid'], top_10_combined_unique['score'], top_10_combined_unique['gt_label']))
        preds_by_sector['Top 10 Combined'] = combined_preds_list

        # --- 3. Individual Top 10 Sectors ---
        for sec in top_10_sectors_list:
             sec_preds_df = exploded_df[exploded_df['sector'] == sec]
             preds_list = list(zip(sec_preds_df['org_uuid'], sec_preds_df['score'], sec_preds_df['gt_label']))
             preds_by_sector[sec] = preds_list
             
        self._plot_comparative_precision_at_k(preds_by_sector, filename_suffix=f"{filename_suffix}_Sector_Top10")

    def analyze_funding_stage(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by granular funding stage (Series A, B, etc.)."""
        k_str = f"Top {top_k}" if top_k else "All"
        # print(f"\n💸 Funding Stage Analysis ({k_str})")
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        # Map to Granular Stage using self.last_stage_map
        if hasattr(self, 'last_stage_map') and self.last_stage_map:
            top_df['stage'] = top_df['org_uuid'].map(self.last_stage_map).fillna('Unknown')
        # else:
        #     # Fallback to num_funding_rounds proxy if map not available
        #     print("   ⚠️ Last stage map not available, using num_funding_rounds proxy.")
        #     def get_stage(n):
        #         if pd.isna(n): return 'Unknown'
        #         if n <= 1: return 'Seed/Angel'
        #         if n <= 3: return 'Early Stage (Series A/B)'
        #         return 'Late Stage (Series C+)'
        #     top_df['stage'] = top_df['num_funding_rounds'].apply(get_stage)
            
        # Clean up stage names if needed (e.g., normalize)
        # Custom Grouping and Ordering
        desired_stages = {
            'angel', 'pre_seed', 'seed', 
            'series_a', 'series_b', 'series_c', 'series_d', 'series_e', 'series_f'
        }
        
        # Group everything else into 'other'
        top_df['stage'] = top_df['stage'].apply(lambda s: s if s in desired_stages else 'other')
        
        # Calculate Metrics
        stage_stats = self._calculate_group_metrics(top_df, 'stage', min_support=5)
        
        if stage_stats.empty:
            print("   No stages with sufficient support found.")
            return

        # print(stage_stats[['stage', 'support', 'auc_pr', 'auc_roc']])
        
        output_path = os.path.join(self.output_dir, f'stage_performance{filename_suffix}.csv')
        stage_stats.to_csv(output_path, index=False)
        
        # Explicit Stage Order
        stage_order = [
            'angel', 'pre_seed', 'seed', 
            'series_a', 'series_b', 'series_c', 'series_d', 'series_e', 'series_f', 
            'other'
        ]
        
        # Filter order to only include present stages
        final_order = [s for s in stage_order if s in stage_stats['stage'].values]
        
        # Use AUC-PR explicitly
        analysis_metric = 'auc_pr' 
        
        self._plot_analysis(stage_stats, top_df, 'stage', metric=analysis_metric, 
                           title_suffix=f"Funding Stage ({k_str})", order=final_order, filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(stage_stats, 'stage', metric=analysis_metric, title_suffix=f"Funding Stage ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(stage_stats, 'stage', metric=analysis_metric, title_suffix=f"Funding Stage ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(stage_stats, 'stage', title_suffix=f"Funding Stage", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/stage_performance{filename_suffix}": wandb.Table(dataframe=stage_stats)})

        # Comparative Precision @ k (All Stages in Order)
        preds_by_stage = {}
        # Global baseline first
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_stage['All Stages'] = global_preds_list
        for stage in final_order:
            stage_preds = top_df[top_df['stage'] == stage]
            preds_list = list(zip(stage_preds['org_uuid'], stage_preds['score'], stage_preds['gt_label']))
            preds_by_stage[stage] = preds_list

        self._plot_comparative_precision_at_k(preds_by_stage, filename_suffix=f"{filename_suffix}_Funding_Stage")

    def analyze_founding_year(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by founding year."""
        k_str = f"Top {top_k}" if top_k else "All"
        # print(f"\n📅 Founding Year Analysis ({k_str})")
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        top_df['founded_year'] = top_df['founded_on'].dt.year
        
        # Filter for reasonable years (e.g., >= 2000) to keep plot readable
        top_df = top_df[top_df['founded_year'] >= 2000]
        top_df['founded_year'] = top_df['founded_year'].astype(int)
        
        # Calculate Metrics
        year_stats = self._calculate_group_metrics(top_df, 'founded_year', min_support=10)
        
        if year_stats.empty:
            print(f"   ⚠️ Not enough data for Founding Year analysis (min_support=10)")
            return
        
        # print(year_stats[['founded_year', 'support', 'auc_pr', 'auc_roc']])
        
        output_path = os.path.join(self.output_dir, f'founding_year_performance{filename_suffix}.csv')
        year_stats.to_csv(output_path, index=False)
        
        # Plotting (Ordered by Year)
        year_order = sorted(year_stats['founded_year'].unique())
        
        self._plot_analysis(year_stats, top_df, 'founded_year', metric='auc_pr', 
                           title_suffix=f"Founding Year ({k_str})", order=year_order, filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(year_stats, 'founded_year', metric='auc_pr', title_suffix=f"Founding Year ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(year_stats, 'founded_year', metric='auc_pr', title_suffix=f"Founding Year ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(year_stats, 'founded_year', title_suffix=f"Founding Year", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/founding_year_performance{filename_suffix}": wandb.Table(dataframe=year_stats)})

        # Comparative Precision @ k by founding year cohort
        preds_by_year = {}

        # Global baseline
        global_preds = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_year['All Years'] = global_preds

        # Individual years (only those with sufficient support)
        valid_years = year_stats[year_stats['support'] >= 20]['founded_year'].values
        for year in sorted(valid_years):
            year_preds = top_df[top_df['founded_year'] == year]
            preds_list = list(zip(year_preds['org_uuid'], year_preds['score'], year_preds['gt_label']))
            preds_by_year[str(int(year))] = preds_list

        self._plot_comparative_precision_at_k(preds_by_year, filename_suffix=f"{filename_suffix}_Founding_Year")

    def analyze_geography(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by country."""
        k_str = f"Top {top_k}" if top_k else "All"
        # print(f"\n🌍 Geography Analysis ({k_str})")
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        # Calculate Metrics
        geo_stats = self._calculate_group_metrics(top_df, 'country_code', min_support=20)
        
        if geo_stats.empty:
             print(f"   ⚠️ Not enough data for Geography analysis (min_support=20)")
             return
        
        # print(f"   Top 5 Countries (by AUC-PR):")
        # print(geo_stats.head(5)[['country_code', 'support', 'auc_pr', 'auc_roc']])
        
        output_path = os.path.join(self.output_dir, f'geo_performance{filename_suffix}.csv')
        geo_stats.to_csv(output_path, index=False)
        
        # Plotting (Top 20 by Support, then sorted by AUC-PR)
        top_20_geo = geo_stats.sort_values('support', ascending=False).head(20)
        top_20_geo = top_20_geo.sort_values('auc_pr', ascending=False)
        
        self._plot_analysis(top_20_geo, top_df, 'country_code', metric='auc_pr', 
                           title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plot (All Countries)
        self._plot_correlation(geo_stats, 'country_code', metric='auc_pr', title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(geo_stats, 'country_code', metric='auc_pr', title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        
        if wandb.run is not None:
            wandb.log({f"analysis/geo_performance{filename_suffix}": wandb.Table(dataframe=geo_stats)})

        # Comparative Precision @ k (Top 10 Countries)
        # Select top 10 by support
        top_10_geo_df = geo_stats.sort_values('support', ascending=False).head(10)
        top_10_countries = top_10_geo_df['country_code'].values
        
        preds_by_country = {}
        
        # --- 1. Global Baseline ---
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_country['All Countries'] = global_preds_list
        
        # --- 2. Top 10 Combined ---
        top_10_combined_df = top_df[top_df['country_code'].isin(top_10_countries)]
        combined_preds_list = list(zip(top_10_combined_df['org_uuid'], top_10_combined_df['score'], top_10_combined_df['gt_label']))
        preds_by_country['Top 10 Combined'] = combined_preds_list
        
        # --- 3. Individual Top 10 Countries ---
        for country in top_10_countries:
            country_preds = top_df[top_df['country_code'] == country]
            preds_list = list(zip(country_preds['org_uuid'], country_preds['score'], country_preds['gt_label']))
            preds_by_country[country] = preds_list
            
        self._plot_comparative_precision_at_k(preds_by_country, filename_suffix=f"{filename_suffix}_Country_Top10")

    def analyze_continents(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by continent."""
        k_str = f"Top {top_k}" if top_k else "All"
        # print(f"\n🗺️ Continent Analysis ({k_str})")
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
            
        # Map Country to Continent
        top_df['continent'] = top_df['country_code'].apply(convert_to_continent)
        
        # Calculate Metrics
        cont_stats = self._calculate_group_metrics(top_df, 'continent', min_support=10)
        
        if cont_stats.empty:
             print(f"   ⚠️ Not enough data for Continent analysis (min_support=10)")
             return
        
        # print(cont_stats[['continent', 'support', 'auc_pr', 'auc_roc']])
        
        output_path = os.path.join(self.output_dir, f'continent_performance{filename_suffix}.csv')
        cont_stats.to_csv(output_path, index=False)
        
        # Plotting
        self._plot_analysis(cont_stats, top_df, 'continent', metric='auc_pr', 
                           title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(cont_stats, 'continent', metric='auc_pr', title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(cont_stats, 'continent', metric='auc_pr', title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(cont_stats, 'continent', title_suffix=f"Continent", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/continent_performance{filename_suffix}": wandb.Table(dataframe=cont_stats)})

        # Comparative Precision @ k (All Continents)
        preds_by_continent = {}
        # Global baseline first
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_continent['All Continents'] = global_preds_list
        # Sort by AUC-PR order for legend
        sorted_continents = cont_stats.sort_values('auc_pr', ascending=False)['continent'].values

        for cont in sorted_continents:
            cont_preds = top_df[top_df['continent'] == cont]
            preds_list = list(zip(cont_preds['org_uuid'], cont_preds['score'], cont_preds['gt_label']))
            preds_by_continent[cont] = preds_list

        self._plot_comparative_precision_at_k(preds_by_continent, filename_suffix=f"{filename_suffix}_Continent")

    def _plot_comparative_net_profit(self, group_data: Dict[str, Tuple], filename_suffix: str = ""):
        """
        Plot multiple Net Profit curves on the same chart.
        group_data: { 'Group Name': (ranks, costs, values, labels) }
        """
        if not group_data:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                n = len(group_data)
                palette = plt.cm.tab10(np.linspace(0, 0.9, min(n, 10))) if n <= 10 else plt.cm.tab20(np.linspace(0, 0.95, n))

                for i, (name, (ranks, costs, values, labels)) in enumerate(group_data.items()):
                    if len(ranks) == 0: continue
                    profits = np.nan_to_num(values - costs)
                    ax.plot(ranks, profits, label=name, color=palette[i], linewidth=1.5, alpha=0.85)
                    peak_idx = np.argmax(profits)
                    ax.scatter(ranks[peak_idx], profits[peak_idx], color=palette[i], s=30, marker='o', zorder=5)

                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
                ax.set_xlabel('Portfolio Rank (Number of Startups)')
                ax.set_ylabel('Net Profit (USD)')
                n_items = len(strategy_predictions)
                ncol = 4 if n_items <= 8 else 5
                legend_frac = 0.18  # fixed: matches P@K plots for axis alignment
                ax.legend(fontsize=7, bbox_to_anchor=(0.0, 1.02, 1.0, 0.2), loc='lower left',
                         mode='expand', ncol=ncol, framealpha=0.9, borderaxespad=0.,
                         columnspacing=1.0, handletextpad=0.5)

                def currency_formatter(x, pos):
                    if abs(x) >= 1e9: return f'${x/1e9:.1f}B'
                    if abs(x) >= 1e6: return f'${x/1e6:.0f}M'
                    return f'${x:.0f}'
                ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
                _apply_thesis_style(ax)
                fig.tight_layout(rect=[0, 0, 1, 1.0 - legend_frac])

                filename = f'roi_comparative_net_profit{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/roi_comparative_net_profit{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Comparative Net Profit: {e}")

    # =========================================================================
    # Multi-seed aggregated analysis with bootstrap confidence intervals.
    # Used by scripts/plot_aggregated_pk_curves.py for paper figures: produces
    # one P@k vs VCs PDF per sub-category (stages / continents / sectors /
    # countries / funding sources), with shaded CI bands from N seed-replicas
    # and bootstrap error bars on each VC's portfolio precision.
    # =========================================================================

    @staticmethod
    def _compute_pk_curve(predictions: List[Tuple[str, float, float]],
                          max_k: int) -> np.ndarray:
        """Cumulative P@k for k=1..max_k from one seed's prediction list.

        Predictions are sorted by score descending. For k beyond the available
        portfolio (i.e. k > len(predictions)), the curve is padded with NaN so
        downstream nan-aware aggregation handles seeds with shorter sub-category
        portfolios correctly.
        """
        if not predictions:
            return np.full(max_k, np.nan)
        sorted_preds = sorted(predictions, key=lambda p: p[1], reverse=True)
        eff_k = min(max_k, len(sorted_preds))
        labels = np.fromiter((float(p[2]) for p in sorted_preds[:eff_k]),
                             dtype=float, count=eff_k)
        cum_precision = np.cumsum(labels) / np.arange(1, eff_k + 1)
        out = np.full(max_k, np.nan)
        out[:eff_k] = cum_precision
        return out

    @staticmethod
    def _bootstrap_pk_ci(pk_per_seed: np.ndarray,
                         n_bootstrap: int = 10000,
                         alpha: float = 0.05,
                         random_state: int = 42,
                         batch_size: int = 500
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap percentile CI on the mean cumulative P@k across seeds.

        Args:
            pk_per_seed: shape ``(n_seeds, max_k)`` array of per-seed P@k curves
                (NaN-padded where a seed's portfolio is shorter than ``max_k``).
            n_bootstrap: number of bootstrap resamples of the seed dimension.
            alpha: two-sided significance (CI width = 1 - alpha).
            random_state: seed for reproducibility.
            batch_size: rows of bootstrap samples to materialise per chunk; keeps
                peak memory at ``batch_size * n_seeds * max_k * 8`` bytes.

        Returns:
            ``(mean, ci_lower, ci_upper)`` each shape ``(max_k,)``. NaN-aware.
        """
        n_seeds, max_k = pk_per_seed.shape
        rng = np.random.default_rng(random_state)
        boot_means = np.empty((n_bootstrap, max_k), dtype=float)
        for start in range(0, n_bootstrap, batch_size):
            end = min(start + batch_size, n_bootstrap)
            idx = rng.integers(0, n_seeds, size=(end - start, n_seeds))
            boot_means[start:end] = np.nanmean(pk_per_seed[idx], axis=1)
        lo = np.nanpercentile(boot_means, 100 * (alpha / 2), axis=0)
        hi = np.nanpercentile(boot_means, 100 * (1 - alpha / 2), axis=0)
        mean = np.nanmean(pk_per_seed, axis=0)
        return mean, lo, hi

    @staticmethod
    def _bootstrap_vc_precision_ci(n_picks: int, n_successes: int,
                                   n_bootstrap: int = 10000,
                                   alpha: float = 0.05,
                                   random_state: int = 42
                                   ) -> Tuple[float, float, float]:
        """Bootstrap percentile CI on a VC's portfolio precision.

        Models each pick as a Bernoulli trial with the empirical hit rate and
        resamples ``n_picks`` trials with replacement to capture the uncertainty
        in the VC's true success rate given their finite portfolio.

        Returns:
            ``(point_estimate, ci_lower, ci_upper)``.
        """
        if n_picks <= 0:
            return 0.0, 0.0, 0.0
        base = np.zeros(n_picks, dtype=float)
        base[:n_successes] = 1.0
        rng = np.random.default_rng(random_state)
        boot_idx = rng.integers(0, n_picks, size=(n_bootstrap, n_picks))
        boot_precs = base[boot_idx].mean(axis=1)
        lo = float(np.percentile(boot_precs, 100 * (alpha / 2)))
        hi = float(np.percentile(boot_precs, 100 * (1 - alpha / 2)))
        return n_successes / n_picks, lo, hi

    def _extract_mtl_task(self,
                          predictions: List[Tuple[str, dict, dict]],
                          task_key: str,
                          use_mature_filter: bool
                          ) -> List[Tuple[str, float, float]]:
        """Project MTL ``(uuid, score_dict, label_dict)`` to single-task tuples.

        Returns ``(uuid, score_dict[task_key], label_dict[task_key])`` for every
        entry where both dicts contain ``task_key``. Optionally restricts the
        result to mature startups via ``_filter_mature_startups`` (used for the
        Liquidity / Exit task per the maturity-mask thesis convention).

        For the Momentum (NFR) task we also drop entries where the Liquidity
        label is positive. This mirrors ``mask_mom = 1 - y_liq`` in
        preprocessing and keeps the aggregated PK figure aligned with the
        Table I headline metric, which is computed on the same masked subset.
        """
        if task_key == 'mom':
            projected = [
                (u, float(s[task_key]), float(l[task_key]))
                for u, s, l in predictions
                if isinstance(s, dict) and isinstance(l, dict)
                and task_key in s and task_key in l
                and float(l.get('liq', 0.0)) == 0.0
            ]
        else:
            projected = [
                (u, float(s[task_key]), float(l[task_key]))
                for u, s, l in predictions
                if isinstance(s, dict) and isinstance(l, dict)
                and task_key in s and task_key in l
            ]
        if use_mature_filter:
            projected = self._filter_mature_startups(projected)
        return projected

    @staticmethod
    def _build_full_df(predictions: List[Tuple[str, float, float]],
                       orgs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Materialise the per-startup metadata DataFrame used by the filters."""
        pred_df = pd.DataFrame(predictions, columns=['org_uuid', 'score', 'gt_label'])
        if orgs_df is None:
            return pred_df
        return pred_df.merge(orgs_df, on='org_uuid', how='left')

    def _plot_comparative_precision_at_k_with_ci(
        self,
        strategy_curves: Dict[str, Dict[str, np.ndarray]],
        benchmark_with_ci: Optional[Dict[str, Dict]] = None,
        filename_suffix: str = "",
        title: Optional[str] = None,
    ) -> None:
        """Plot mean P@k lines with shaded CI bands per strategy + VC scatter.

        Mirrors the styling of ``_plot_comparative_precision_at_k`` (palette,
        legend, axis), but renders aggregated curves with confidence bands and
        VC error bars instead of point lines/dots.

        Args:
            strategy_curves: ``{strategy_name: {k_values, mean, ci_lower,
                ci_upper, n_seeds}}``. ``k_values`` is shape ``(max_k,)`` and the
                others are aligned. Strategies whose names start with ``"All "``
                are rendered in a reserved global colour and bold linewidth.
            benchmark_with_ci: ``{vc_name: {k, precision, precision_lo,
                precision_hi}}`` (optional). Drawn as scatter points with
                vertical error bars.
            filename_suffix: appended to the output filename.
            title: optional figure title.
        """
        if not strategy_curves:
            return
        # Mirror the type scale of scripts/plot_pk_by_year.py (the cohort
        # figure) on this wider canvas: bump axis label, tick, and legend
        # sizes one to two points above the thesis defaults so the figure
        # reads at the same prominence as the single-column cohort plot.
        _local_rc = {
            **_THESIS_RCPARAMS,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
        try:
            with plt.rc_context(_local_rc):
                fig, ax = plt.subplots(figsize=(7.5, 5.0))

                global_color = '#1f3d7a'
                non_global_strats = [s for s in strategy_curves if not s.startswith('All ')]
                palette = (plt.cm.tab10(np.linspace(0, 0.9, max(1, min(len(non_global_strats), 10))))
                           if len(non_global_strats) <= 10
                           else plt.cm.tab20(np.linspace(0, 0.95, len(non_global_strats))))

                p_idx = 0
                for strat_name, curve in strategy_curves.items():
                    k = curve['k_values']
                    if k is None or len(k) == 0:
                        continue
                    mean = curve['mean']
                    lo = curve['ci_lower']
                    hi = curve['ci_upper']
                    is_global = strat_name.startswith('All ')
                    color = global_color if is_global else palette[p_idx]
                    lw = 2.5 if is_global else 1.5
                    alpha_line = 1.0 if is_global else 0.85
                    alpha_band = 0.18 if is_global else 0.15
                    ax.plot(k, mean, label=strat_name, color=color,
                            linewidth=lw, alpha=alpha_line, zorder=5 if is_global else 3)
                    ax.fill_between(k, lo, hi, color=color, alpha=alpha_band,
                                    linewidth=0, zorder=2 if is_global else 1)
                    if not is_global:
                        p_idx += 1

                if benchmark_with_ci:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    # Order matches BENCHMARK_INVESTORS insertion: a16z, Sequoia, YC, Benchmark, Accel.
                    # Sequoia and YC swapped vs default so YC is rendered as a square (closer to its
                    # rectangular portfolio interpretation) and Sequoia takes the triangle.
                    bm_markers = ['D', '^', 's', 'v', '*']
                    bm_sizes = [80, 80, 80, 80, 150]
                    for i, (name, m) in enumerate(benchmark_with_ci.items()):
                        c = bm_colors[i % len(bm_colors)]
                        marker = bm_markers[i % len(bm_markers)]
                        size = bm_sizes[i % len(bm_sizes)]
                        ax.errorbar([m['k']], [m['precision']],
                                    yerr=[[m['precision'] - m['precision_lo']],
                                          [m['precision_hi'] - m['precision']]],
                                    fmt='none', ecolor=c, elinewidth=1.0, capsize=3, zorder=9)
                        ax.scatter([m['k']], [m['precision']], color=c, marker=marker,
                                   s=size, zorder=10, edgecolors='white', linewidths=0.5,
                                   label=f"{name} ({m['precision']:.1%})")

                ax.set_xlabel('Portfolio Rank (K)')
                ax.set_ylabel('Precision (%)')

                # Dummy proxy handles so readers know what the visual cues mean.
                # Hard-coded to 95% because every band/error bar in this plot is built
                # at alpha=0.05 by perform_downstream_analysis_aggregated.
                handles, labels = ax.get_legend_handles_labels()
                model_ci_handle = mpatches.Patch(facecolor='grey', edgecolor='none',
                                                 alpha=0.25, label='Model 95% CI')
                handles.append(model_ci_handle)
                labels.append('Model 95% CI')
                if benchmark_with_ci:
                    # Vertical bar marker mimics the per-VC errorbar drawn above.
                    vc_ci_handle = Line2D([], [], color='grey', marker='|',
                                          markersize=10, markeredgewidth=1.4,
                                          linestyle='None', label='VC 95% CI')
                    handles.append(vc_ci_handle)
                    labels.append('VC 95% CI')

                # Use four columns always so each column has enough room for
                # the longest label (Angel/Pre-Seed). 14-item legends (stage +
                # benchmarks) wrap to four rows; six- to eight-item legends
                # (continent / sector axes) would otherwise wrap to two rows
                # and produce a shorter total legend block, which in turn
                # gives a taller plot area on those axes after tight_layout.
                # Pad to exactly 4 rows × 4 cols = 16 entries so every P@K
                # subfigure renders the same legend-block height and tight
                # savefig crops to the same canvas. Matplotlib lays out
                # legends column-major, so we have to place the blanks
                # column-by-column to land them in the right cells:
                #   - When the legend is underfilled by >= one full row
                #     (continent case, 8 visibles / 8 blanks): put
                #     `blanks_per_col` blanks at the TOP of every column,
                #     so the visible items pack into the bottom rows
                #     across all columns.
                #   - When the legend is almost full (stage case, 15
                #     visibles / 1 blank): leave the visible items in
                #     their original column-major order and append the
                #     blank at the end (col 4 row 4 cell).
                ncol = 4
                target_rows = 4
                target_n = target_rows * ncol
                n_visible = len(handles)
                n_blanks = target_n - n_visible
                blank_h = lambda: Line2D([], [], color='none')
                if n_blanks > 0:
                    if n_blanks >= ncol:
                        # Distribute blanks evenly to the TOP of each column.
                        blanks_per_col = n_blanks // ncol
                        items_per_col = (n_visible + ncol - 1) // ncol
                        new_h, new_l, idx = [], [], 0
                        for col in range(ncol):
                            for _ in range(blanks_per_col):
                                new_h.append(blank_h())
                                new_l.append(' ')
                            for _ in range(items_per_col):
                                if idx < n_visible:
                                    new_h.append(handles[idx])
                                    new_l.append(labels[idx])
                                    idx += 1
                        handles, labels = new_h, new_l
                    else:
                        # Append blanks at the end (fills last cells of
                        # the last column in column-major layout).
                        for _ in range(n_blanks):
                            handles.append(blank_h())
                            labels.append(' ')
                n_items = len(handles)
                # Extend the bbox past the axis bounds so the legend uses the
                # y-label margin on the left and a small right-margin reserve
                # on the right.
                legend_frac = 0.22
                # bbox in axes coords; extend 6% left to use the y-label
                # margin but cap the right edge at the right axis spine so
                # no label is clipped when savefig.bbox is 'standard'.
                ax.legend(handles, labels,
                          fontsize=10, bbox_to_anchor=(-0.06, 1.02, 1.06, 0.2),
                          loc='lower left', mode='expand', ncol=ncol,
                          frameon=False, borderaxespad=0.,
                          columnspacing=1.0, handletextpad=0.5)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}'))

                # Common K range across all P@K subfigures (continent / stage /
                # sector / country / funding-source). Set to 1100 so the
                # widest-portfolio VC benchmark (YC at K~1013) is comfortably
                # inside the axis on the stage figure, and the continent and
                # other axes that carry no benchmark scatter use the same
                # x-range for visual alignment with the stage figure.
                max_k_axis = 1100
                ax.set_xlim(0, max_k_axis)
                if title:
                    ax.set_title(title, pad=12)
                _apply_thesis_style(ax)
                fig.tight_layout(rect=[0, 0, 1, 1.0 - legend_frac])

                filename = f'precision_at_k_with_ci{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_at_k_with_ci{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Aggregated Precision Strategies: {e}")

    def _aggregate_strategy_curves(self,
                                   strategy_preds_per_seed: Dict[str, Dict[int, List[Tuple]]],
                                   max_k: int,
                                   n_bootstrap: int,
                                   alpha: float
                                   ) -> Dict[str, Dict[str, np.ndarray]]:
        """Convert ``{strategy: {seed: [preds]}}`` to per-strategy mean+CI curves."""
        out: Dict[str, Dict[str, np.ndarray]] = {}
        k_axis = np.arange(1, max_k + 1)
        for strat, by_seed in strategy_preds_per_seed.items():
            seeds = sorted(by_seed.keys())
            if not seeds:
                continue
            stacked = np.vstack([self._compute_pk_curve(by_seed[s], max_k) for s in seeds])
            mean, lo, hi = self._bootstrap_pk_ci(stacked, n_bootstrap=n_bootstrap, alpha=alpha)
            out[strat] = {
                'k_values': k_axis,
                'mean': mean,
                'ci_lower': lo,
                'ci_upper': hi,
                'n_seeds': len(seeds),
            }
        return out

    def _compute_vc_benchmarks_with_ci(self,
                                       n_bootstrap: int,
                                       alpha: float
                                       ) -> Dict[str, Dict]:
        """Wrap ``analyze_investor_benchmark`` with bootstrap CIs on precision."""
        bm = self.analyze_investor_benchmark()
        if not bm:
            return {}
        out = {}
        for name, m in bm.items():
            k = int(m['k'])
            n_succ = int(round(m['precision'] * k))
            point, lo, hi = self._bootstrap_vc_precision_ci(
                k, n_succ, n_bootstrap=n_bootstrap, alpha=alpha,
            )
            out[name] = {
                'k': k,
                'precision': point,
                'precision_lo': lo,
                'precision_hi': hi,
            }
        return out

    def _top_categorical_values(self, full_df: pd.DataFrame,
                                column: str, top_n: int,
                                multi_value_separator: Optional[str] = None
                                ) -> List[str]:
        """Return the top-N most frequent non-null values in ``column``.

        If ``multi_value_separator`` is given (e.g. ``,`` for ``category_list``),
        each cell is split and individual values counted separately.
        """
        if column not in full_df.columns:
            return []
        series = full_df[column].dropna()
        if multi_value_separator:
            tokens = (series.astype(str)
                            .str.split(multi_value_separator)
                            .explode()
                            .str.strip())
            counts = tokens[tokens.astype(bool)].value_counts()
        else:
            counts = series.value_counts()
        return counts.head(top_n).index.tolist()

    def perform_downstream_analysis_aggregated(
        self,
        predictions_per_seed: Dict[int, List[Tuple[str, dict, dict]]],
        output_subdir: Optional[str] = None,
        max_k: int = 1000,
        n_bootstrap: int = 10000,
        alpha: float = 0.05,
        top_n_sectors: int = 8,
        top_n_countries: int = 8,
    ) -> None:
        """Multi-seed P@k vs VCs analysis with bootstrap CI bands per sub-category.

        For each MTL task in the input (currently 'mom' / 'liq'), produces one
        PDF per sub-category axis (stages / continents / sectors / countries /
        funding sources) that shows mean cumulative P@k curves with shaded 95 %
        CI bands across seeds, plus VC benchmark scatter with bootstrap error
        bars. Other strategies (legacy distribution plots) are not re-run here.

        Args:
            predictions_per_seed: ``{seed: [(uuid, score_dict, label_dict), ...]}``
                where score and label dicts contain ``'mom'`` and ``'liq'`` keys.
            output_subdir: optional subdirectory under ``self.output_dir`` for
                the PDFs. The analyzer's working ``output_dir`` is temporarily
                redirected when set.
            max_k: highest portfolio rank to plot.
            n_bootstrap: bootstrap resamples for both seed-CI and VC-CI.
            alpha: two-sided significance level (0.05 → 95 % CIs).
            top_n_sectors: number of top categories from ``category_list`` to
                include as sector strategies.
            top_n_countries: number of top ISO country codes to include.
        """
        if not predictions_per_seed:
            print("⚠️ perform_downstream_analysis_aggregated: no predictions provided.")
            return
        sample = next(iter(predictions_per_seed.values()))
        if not sample or not isinstance(sample[0][1], dict):
            raise ValueError("Aggregated analysis requires MTL predictions "
                             "(score must be dict with 'mom'/'liq' keys).")

        original_output_dir = self.output_dir
        if output_subdir:
            self.output_dir = os.path.join(self.output_dir, output_subdir)
            os.makedirs(self.output_dir, exist_ok=True)

        try:
            tasks = [
                ('mom', 'Venture Fund (NFR)', False),
                ('liq', 'Liquidity Fund (Exit)', True),
            ]
            for task_key, task_name, use_mature_filter in tasks:
                self._run_aggregated_for_task(
                    predictions_per_seed, task_key, task_name, use_mature_filter,
                    max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                    top_n_sectors=top_n_sectors, top_n_countries=top_n_countries,
                )
        finally:
            self.output_dir = original_output_dir

    def _run_aggregated_for_task(
        self,
        predictions_per_seed: Dict[int, List[Tuple[str, dict, dict]]],
        task_key: str,
        task_name: str,
        use_mature_filter: bool,
        *,
        max_k: int,
        n_bootstrap: int,
        alpha: float,
        top_n_sectors: int,
        top_n_countries: int,
    ) -> None:
        """One task pass producing one PDF per sub-category axis."""
        print(f"\n🚀 Aggregated downstream analysis — {task_name}")

        # 1. Project each seed to single-task tuples and apply maturity filter.
        task_preds_per_seed: Dict[int, List[Tuple[str, float, float]]] = {
            seed: self._extract_mtl_task(preds, task_key, use_mature_filter)
            for seed, preds in predictions_per_seed.items()
        }
        task_preds_per_seed = {s: p for s, p in task_preds_per_seed.items() if p}
        if not task_preds_per_seed:
            print(f"   ⚠️ No predictions remain after extracting task '{task_key}'.")
            return
        print(f"   Seeds with predictions: {len(task_preds_per_seed)}")

        # 2. Build the metadata-joined DataFrame from the union of UUIDs.
        union_preds = [p for seed_preds in task_preds_per_seed.values() for p in seed_preds]
        full_df = self._build_full_df(union_preds, self.orgs_df)
        if 'country_code' in full_df.columns and 'continent' not in full_df.columns:
            full_df['continent'] = full_df['country_code'].apply(convert_to_continent)
        if 'founded_on' in full_df.columns and 'founded_year' not in full_df.columns:
            full_df['founded_year'] = pd.to_datetime(
                full_df['founded_on'], errors='coerce'
            ).dt.year

        # 3. VC benchmarks with bootstrap CI — same for every sub-category axis.
        vc_bench = self._compute_vc_benchmarks_with_ci(n_bootstrap=n_bootstrap, alpha=alpha)

        suffix = f"_{task_key}"

        # The aggregated path only needs the filtered prediction lists; the
        # per-strategy ROI compute inside each analyze_portfolio_by_* would run
        # ~ N_seeds × N_strategies × N_axes times, which dominates the runtime.
        # Pass ``compute_roi=False`` to skip it.
        common = dict(verbose=False, plot_charts=False, compute_roi=False)

        # 4. Stage axis (uses self.STAGE_STRATEGIES + 'All Stages').
        self._render_axis(
            axis_label='stages',
            global_label='All Stages',
            strategy_to_filter={
                name: lambda preds, name=name, stages=stages: self.analyze_portfolio_by_stage(
                    preds, name, stages, **common)
                for name, stages in self.STAGE_STRATEGIES.items()
            },
            task_preds_per_seed=task_preds_per_seed,
            full_df=full_df,
            vc_bench=vc_bench,
            suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
            include_vcs=True,
        )

        # 5. Continent axis.
        if 'continent' in full_df.columns:
            continents = [c for c in full_df['continent'].dropna().unique() if c != 'Unknown']
            self._render_axis(
                axis_label='continents',
                global_label='All Regions',
                strategy_to_filter={
                    cont: lambda preds, cont=cont: self.analyze_portfolio_by_continent(
                        preds, cont, full_df, **common)
                    for cont in continents
                },
                task_preds_per_seed=task_preds_per_seed,
                full_df=full_df,
                vc_bench=vc_bench,
                suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                include_vcs=False,
            )

        # 5b. Founded-year axis (one curve per year with enough support).
        if 'founded_year' in full_df.columns:
            year_counts = (full_df['founded_year']
                           .dropna()
                           .astype(int)
                           .value_counts())
            # Keep years inside the data-funnel range (2014-2023) with non-trivial
            # support. The 50-startup floor drops only the tail (e.g. 2023 n~64)
            # without forcing a coarser cohort scheme.
            min_support_per_year = 50
            year_range = (2014, 2023)
            founded_years = sorted([
                int(y) for y in year_counts.index
                if year_range[0] <= int(y) <= year_range[1]
                and year_counts[y] >= min_support_per_year
            ])
            if founded_years:
                self._render_axis(
                    axis_label='founded_year',
                    global_label='All Years',
                    strategy_to_filter={
                        str(year): lambda preds, year=year: self.analyze_portfolio_by_founded_year(
                            preds, year, full_df, **common)
                        for year in founded_years
                    },
                    task_preds_per_seed=task_preds_per_seed,
                    full_df=full_df,
                    vc_bench=vc_bench,
                    suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                    include_vcs=False,
                )

        # 6. Sector axis (top-N tags from category_list).
        sectors = self._top_categorical_values(full_df, 'category_list', top_n_sectors,
                                               multi_value_separator=',')
        if sectors:
            self._render_axis(
                axis_label='sectors',
                global_label='All Sectors',
                strategy_to_filter={
                    sec: lambda preds, sec=sec: self.analyze_portfolio_by_sector(
                        preds, sec, full_df, **common)
                    for sec in sectors
                },
                task_preds_per_seed=task_preds_per_seed,
                full_df=full_df,
                vc_bench=vc_bench,
                suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                include_vcs=False,
            )

        # 7. Country axis (top-N).
        countries = self._top_categorical_values(full_df, 'country_code', top_n_countries)
        if countries:
            self._render_axis(
                axis_label='countries',
                global_label='All Countries',
                strategy_to_filter={
                    cc: lambda preds, cc=cc: self.analyze_portfolio_by_country(
                        preds, cc, full_df, **common)
                    for cc in countries
                },
                task_preds_per_seed=task_preds_per_seed,
                full_df=full_df,
                vc_bench=vc_bench,
                suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                include_vcs=False,
            )

        # 8. Funding source axis.
        if 'investor_type' in full_df.columns:
            self._render_axis(
                axis_label='funding_sources',
                global_label='All Sources',
                strategy_to_filter={
                    name: lambda preds, name=name, types=types: self.analyze_portfolio_by_funding_source(
                        preds, name, types, full_df, **common)
                    for name, types in self.FUNDING_SOURCE_STRATEGIES.items()
                },
                task_preds_per_seed=task_preds_per_seed,
                full_df=full_df,
                vc_bench=vc_bench,
                suffix=suffix, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
                include_vcs=False,
            )

    def _render_axis(self, *,
                     axis_label: str,
                     global_label: str,
                     strategy_to_filter,
                     task_preds_per_seed: Dict[int, List[Tuple]],
                     full_df: pd.DataFrame,
                     vc_bench: Dict[str, Dict],
                     suffix: str,
                     max_k: int,
                     n_bootstrap: int,
                     alpha: float,
                     include_vcs: bool) -> None:
        """Build per-strategy curves for one sub-category axis and render the PDF."""
        print(f"   📊 Axis '{axis_label}': {len(strategy_to_filter)} strategies")
        strategy_preds: Dict[str, Dict[int, List[Tuple]]] = {}
        for strat_name, filter_fn in strategy_to_filter.items():
            per_seed = {}
            for seed, preds in task_preds_per_seed.items():
                filtered = filter_fn(preds)
                if filtered:
                    per_seed[seed] = filtered
            if per_seed:
                strategy_preds[strat_name] = per_seed

        if not strategy_preds:
            print(f"      ⚠️ No strategy yielded predictions on this axis. Skipping.")
            return

        # Add the global "All …" line over the unfiltered task predictions.
        strategy_preds = {global_label: task_preds_per_seed, **strategy_preds}

        curves = self._aggregate_strategy_curves(
            strategy_preds, max_k=max_k, n_bootstrap=n_bootstrap, alpha=alpha,
        )
        self._plot_comparative_precision_at_k_with_ci(
            curves,
            benchmark_with_ci=vc_bench if include_vcs else None,
            filename_suffix=f"{suffix}_{axis_label}",
            title=None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Downstream Analysis on Predictions CSV")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load Predictions
    print(f"📂 Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)
    
    # Process Predictions
    import ast
    
    print(f"   Parsing {len(pred_df)} predictions...")
    
    score_col = 'prediction' if 'prediction' in pred_df.columns else 'score'
    uuid_col = 'org_uuid' if 'org_uuid' in pred_df.columns else 'uuid'
    
    # Safe Eval Helper
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return float(x) if x else 0.0

    # Parse Columns
    pred_df[score_col] = pred_df[score_col].apply(safe_eval)
    pred_df['gt_label'] = pred_df['gt_label'].apply(safe_eval).fillna(0.0)

    predictions = list(zip(pred_df[uuid_col], pred_df[score_col], pred_df['gt_label']))

    print(f"✅ Successfully parsed {len(predictions)} predictions.")

    # Initialize Analyzer
    analyzer = DownstreamAnalyzer(config)
    
    # Run Analysis
    analyzer.perform_downstream_analysis(predictions)
