"""
delay_throughput_analyzer.py

Comprehensive delay and throughput analysis for 5G handover optimization.

Provides:
  - End-to-end handover delay modeling (measurement, decision, signaling)
  - Service interruption analysis during handovers
  - Throughput computation considering SINR-to-spectral-efficiency mapping
  - Load-dependent interference modeling
  - Comparative analysis vs. baseline 3GPP procedures
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from scipy import interpolate


class HandoverDelayModel:
    """
    Models end-to-end handover delay from trigger to completion.
    
    Delay breakdown:
    1. Measurement delay: L3 filtering accumulation (depends on WINDOW_SIZE)
    2. Physical layer: Cell measurement reporting (~50 ms)
    3. Decision: RRC HO decision processing
    4. Signaling: RRC Reconfiguration (~80 ms)
    5. Execution: Cell search and synchronization (~30 ms)
    6. Context transfer: Security context transfer (~60 ms)
    7. Data plane: Bearer re-establishment (~25 ms)
    """
    
    # Delay components in milliseconds (3GPP/LTE-M assumptions)
    DELAY_COMPONENTS = {
        'L3_FILTER': {
            'value_ms': 0,  # Dynamic: depends on WINDOW_SIZE
            'description': 'RSRP/SINR L3 filtering accumulation',
            'policy': 'both'  # Both baseline and AI
        },
        'PHYSICAL_LAYER': {
            'value_ms': 50,
            'description': 'Physical layer measurement reporting',
            'policy': 'both'
        },
        'DECISION_3GPP': {
            'value_ms': 20,
            'description': '3GPP A3 event processing (hysteresis + TTT)',
            'policy': 'baseline'
        },
        'DECISION_AI': {
            'value_ms': 15,
            'description': 'AI model inference + decision (faster)',
            'policy': 'ai'
        },
        'RRC_RECONFIG': {
            'value_ms': 80,
            'description': 'RRC Connection Reconfiguration',
            'policy': 'both'
        },
        'CELL_SEARCH': {
            'value_ms': 30,
            'description': 'Cell search and synchronization',
            'policy': 'both'
        },
        'CONTEXT_TRANSFER': {
            'value_ms': 60,
            'description': 'Security context transfer to target',
            'policy': 'both'
        },
        'BEARER_REEST': {
            'value_ms': 25,
            'description': 'Data radio bearer re-establishment',
            'policy': 'both'
        }
    }
    
    # Service interruption penalties
    INTERRUPTION_MS = {
        'successful_ho': 30,      # Normal successful handover
        'hof_recovery': 100,      # Handover Failure (requires retry)
        'rlf_recovery': 2000,     # Radio Link Failure (full sync needed)
        'ping_pong': 50           # Oscillation (measurement uncertainty)
    }
    
    @staticmethod
    def compute_total_delay(
        policy: str = 'baseline',
        window_size_ts: int = 20,
        time_step_s: float = 0.5
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Compute total handover delay for a given policy.
        
        Args:
            policy: 'baseline' (3GPP A3) or 'ai' (ML-based)
            window_size_ts: L3 filter window size in time steps
            time_step_s: Simulation time step in seconds
        
        Returns:
            Tuple of (total_delay_ms, list of (component_name, delay_ms))
        """
        # L3 filter delay = time for window to fill
        l3_filter_ms = window_size_ts * time_step_s * 1000
        
        total_ms = l3_filter_ms
        components_used = [('L3_FILTER', l3_filter_ms)]
        
        for comp_name, comp_dict in HandoverDelayModel.DELAY_COMPONENTS.items():
            if comp_name == 'L3_FILTER':
                continue
            
            policy_key = comp_dict['policy']
            
            # Skip if this component not applicable to this policy
            if policy_key == 'baseline' and policy == 'ai':
                continue
            if policy_key == 'ai' and policy == 'baseline':
                continue
            
            total_ms += comp_dict['value_ms']
            components_used.append((comp_name, comp_dict['value_ms']))
        
        return total_ms, components_used
    
    @staticmethod
    def delay_impact_analysis(
        comparison_df: pd.DataFrame,
        policy_baseline_ho_col: str = 'HO_Base',
        policy_ai_ho_col: str = 'HO_AI',
        window_size_ts: int = 20,
        time_step_s: float = 0.5,
        include_hof_rlf: bool = True,
        hof_col: str = 'HOF_Base',
        rlf_col: str = 'RLF_Base'
    ) -> pd.DataFrame:
        """
        Analyze handover delay impact on each route.
        
        Args:
            comparison_df: DataFrame with baseline and AI KPI counts
            policy_baseline_ho_col: Column name for baseline HO count
            policy_ai_ho_col: Column name for AI HO count
            window_size_ts: L3 filter window size
            time_step_s: Simulation time step
            include_hof_rlf: Whether to account for HOF/RLF recovery delays
            hof_col: Column name for HOF count
            rlf_col: Column name for RLF count
        
        Returns:
            DataFrame with delay analysis per route
        """
        # Per-HO signaling delay ONLY (L3 filter is a one-time warmup, not per-HO)
        # Baseline per-HO: PHYSICAL_LAYER(50) + DECISION_3GPP(20) + RRC_RECONFIG(80)
        #                  + CELL_SEARCH(30) + CONTEXT_TRANSFER(60) + BEARER_REEST(25) = 265 ms
        # AI per-HO:       PHYSICAL_LAYER(50) + DECISION_AI(15) + RRC_RECONFIG(80)
        #                  + CELL_SEARCH(30) + CONTEXT_TRANSFER(60) + BEARER_REEST(25) = 260 ms
        per_ho_delay_baseline_ms = sum(
            v['value_ms'] for k, v in HandoverDelayModel.DELAY_COMPONENTS.items()
            if k != 'L3_FILTER' and v['policy'] in ('baseline', 'both')
        )
        per_ho_delay_ai_ms = sum(
            v['value_ms'] for k, v in HandoverDelayModel.DELAY_COMPONENTS.items()
            if k != 'L3_FILTER' and v['policy'] in ('ai', 'both')
        )

        # One-time L3 filter warmup delay per route (same for both policies)
        l3_filter_ms = window_size_ts * time_step_s * 1000  # e.g. 20 * 0.5 * 1000 = 10,000 ms

        delay_results = []

        for _, row in comparison_df.iterrows():
            ho_base = row[policy_baseline_ho_col]
            ho_ai   = row[policy_ai_ho_col]

            # Total delay = one-time L3 warmup + (per-HO signaling × HO count)
            cumulative_delay_base = l3_filter_ms + ho_base * per_ho_delay_baseline_ms
            cumulative_delay_ai   = l3_filter_ms + ho_ai   * per_ho_delay_ai_ms
            
            # Add HOF/RLF penalties if columns exist
            if include_hof_rlf and hof_col in row and rlf_col in row:
                hof_base = row[hof_col]
                rlf_base = row[rlf_col]
                
                # HOF requires retry (counts as extra HO + penalty)
                cumulative_delay_base += (
                    hof_base * HandoverDelayModel.INTERRUPTION_MS['hof_recovery']
                )
                # RLF requires 2-second recovery
                cumulative_delay_base += (
                    rlf_base * HandoverDelayModel.INTERRUPTION_MS['rlf_recovery']
                )
            
            delay_savings = cumulative_delay_base - cumulative_delay_ai
            delay_savings_pct = (
                (delay_savings / cumulative_delay_base * 100)
                if cumulative_delay_base > 0 else 0
            )
            
            delay_results.append({
                'Route_ID': row.get('UE_ID', len(delay_results) + 1),
                'HO_Baseline': int(ho_base),
                'HO_AI': int(ho_ai),
                'HO_Reduction': int(ho_base - ho_ai),
                'Cumulative_Delay_Baseline_ms': cumulative_delay_base,
                'Cumulative_Delay_AI_ms': cumulative_delay_ai,
                'Delay_Savings_ms': delay_savings,
                'Delay_Savings_Percent': delay_savings_pct
            })
        
        return pd.DataFrame(delay_results)
    
    @staticmethod
    def plot_delay_analysis(
        delay_df: pd.DataFrame,
        save_path: str = None,
        figsize: Tuple = (16, 12)
    ):
        """
        Comprehensive visualization of delay analysis.
        
        Args:
            delay_df: DataFrame with delay metrics per route
            save_path: Path to save figure
            figsize: Figure dimensions
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Delay per Route
        ax = axes[0, 0]
        x_pos = np.arange(min(30, len(delay_df)))
        width = 0.35
        
        ax.bar(x_pos - width/2, delay_df['Cumulative_Delay_Baseline_ms'].iloc[:30],
              width, label='3GPP Baseline', alpha=0.8, color='#3498db', edgecolor='black')
        ax.bar(x_pos + width/2, delay_df['Cumulative_Delay_AI_ms'].iloc[:30],
              width, label='AI Optimised', alpha=0.8, color='#e74c3c', edgecolor='black')
        
        ax.set_ylabel('Cumulative Delay (ms)', fontsize=11, weight='bold')
        ax.set_xlabel('Route ID', fontsize=11, weight='bold')
        ax.set_title('End-to-End Handover Delay per Route (First 30)', fontsize=12, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 2: Delay Savings Distribution
        ax = axes[0, 1]
        ax.hist(delay_df['Delay_Savings_ms'], bins=25, color='#2ecc71',
               alpha=0.7, edgecolor='black', linewidth=1.2)
        
        mean_saving = delay_df['Delay_Savings_ms'].mean()
        ax.axvline(mean_saving, color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {mean_saving:.0f} ms')
        ax.axvline(delay_df['Delay_Savings_ms'].median(), color='orange', linestyle='--',
                  linewidth=2, label=f"Median: {delay_df['Delay_Savings_ms'].median():.0f} ms")
        
        ax.set_xlabel('Delay Savings (ms)', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Distribution of Delay Savings', fontsize=12, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 3: Cumulative Delay Over Routes
        ax = axes[0, 2]
        delay_df_sorted = delay_df.sort_values('Delay_Savings_ms', ascending=False)
        
        cumul_base = delay_df_sorted['Cumulative_Delay_Baseline_ms'].cumsum()
        cumul_ai = delay_df_sorted['Cumulative_Delay_AI_ms'].cumsum()
        
        ax.plot(cumul_base.values, label='3GPP Baseline', linewidth=2.5,
               color='#3498db', marker='o', markersize=3, alpha=0.8)
        ax.plot(cumul_ai.values, label='AI Optimised', linewidth=2.5,
               color='#e74c3c', marker='s', markersize=3, alpha=0.8)
        
        total_savings = cumul_base.iloc[-1] - cumul_ai.iloc[-1]
        ax.fill_between(range(len(cumul_base)), cumul_base, cumul_ai,
                       alpha=0.3, color='#2ecc71',
                       label=f'Total Savings: {total_savings:.0f} ms')
        
        ax.set_xlabel('Route (sorted by savings)', fontsize=11, weight='bold')
        ax.set_ylabel('Cumulative Delay (ms)', fontsize=11, weight='bold')
        ax.set_title('Cumulative Delay Over All Routes', fontsize=12, weight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Delay Savings % vs HO Volume
        ax = axes[1, 0]
        scatter = ax.scatter(delay_df['HO_Baseline'], delay_df['Delay_Savings_Percent'],
                            s=100, alpha=0.6, c=delay_df['Delay_Savings_Percent'],
                            cmap='RdYlGn', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Number of Handovers (Baseline)', fontsize=11, weight='bold')
        ax.set_ylabel('Delay Savings (%)', fontsize=11, weight='bold')
        ax.set_title('Delay Reduction vs. Handover Volume', fontsize=12, weight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Savings (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 5: HO Reduction
        ax = axes[1, 1]
        delay_df_sorted_ho = delay_df.sort_values('HO_Reduction', ascending=True)
        
        colors_ho = ['#2ecc71' if x > 0 else '#e74c3c' for x in delay_df_sorted_ho['HO_Reduction']]
        ax.barh(range(min(20, len(delay_df_sorted_ho))),
               delay_df_sorted_ho['HO_Reduction'].head(20),
               color=colors_ho[:20], alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('HO Count Reduction', fontsize=11, weight='bold')
        ax.set_title('Handover Reduction per Route (Top 20 by Reduction)', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.axvline(0, color='black', linewidth=1)
        
        # Plot 6: Summary Stats
        ax = axes[1, 2]
        ax.axis('off')
        
        total_delay_base = delay_df['Cumulative_Delay_Baseline_ms'].sum()
        total_delay_ai = delay_df['Cumulative_Delay_AI_ms'].sum()
        total_savings = total_delay_base - total_delay_ai
        
        summary_text = f"""
        DELAY ANALYSIS SUMMARY
        
        Total Routes: {len(delay_df)}
        
        Cumulative Delay (All Routes):
          Baseline: {total_delay_base/1000:.1f} seconds
          AI: {total_delay_ai/1000:.1f} seconds
          Savings: {total_savings/1000:.1f} seconds
        
        Per-Route Delay:
          Mean Baseline: {delay_df['Cumulative_Delay_Baseline_ms'].mean():.0f} ms
          Mean AI: {delay_df['Cumulative_Delay_AI_ms'].mean():.0f} ms
          Mean Savings: {delay_df['Delay_Savings_ms'].mean():.0f} ms
        
        Handover Statistics:
          Baseline HO/Route: {delay_df['HO_Baseline'].mean():.1f}
          AI HO/Route: {delay_df['HO_AI'].mean():.1f}
          Reduction: {delay_df['HO_Reduction'].mean():.1f} per route
        
        Delay Savings:
          Absolute: {delay_df['Delay_Savings_ms'].sum()/1000:.1f} seconds
          Percentage: {(total_savings/total_delay_base*100):.1f}%
        
        Min/Max Savings:
          Min: {delay_df['Delay_Savings_ms'].min():.0f} ms
          Max: {delay_df['Delay_Savings_ms'].max():.0f} ms
        """
        
        ax.text(0.05, 0.95, summary_text, fontsize=9.5, verticalalignment='top',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Delay analysis plot saved: {save_path}")
        
        plt.close()


class ThroughputAnalyzer:
    """
    Computes throughput metrics considering:
    - SINR-to-spectral-efficiency mapping (5G NR MCS)
    - HO-induced service interruption
    - Load-dependent interference effects
    """
    
    # 5G NR Modulation and Coding Scheme (MCS) mapping
    # SINR ranges -> Spectral Efficiency (bits/s/Hz)
    MCS_MAPPING = [
        (-10, 0.0),    # Below threshold: no transmission
        (-8, 0.25),
        (-5, 0.5),
        (0, 1.0),      # QPSK
        (5, 2.0),
        (10, 3.5),     # 16-QAM
        (15, 5.0),
        (20, 6.5),     # 64-QAM
        (40, 7.5),     # 256-QAM (max)
    ]
    
    @staticmethod
    def sinr_to_spectral_efficiency(sinr_db: float) -> float:
        """
        Map SINR to spectral efficiency using 5G NR MCS tables.
        
        Args:
            sinr_db: SINR in dB
        
        Returns:
            Spectral efficiency in bits/s/Hz
        """
        # Linear interpolation between breakpoints
        sinr_vals = [x[0] for x in ThroughputAnalyzer.MCS_MAPPING]
        se_vals = [x[1] for x in ThroughputAnalyzer.MCS_MAPPING]
        
        se = np.interp(sinr_db, sinr_vals, se_vals)
        return float(se)
    
    @staticmethod
    def compute_throughput_trajectory(
        sinr_trajectory: np.ndarray,
        bandwidth_mhz: float = 20.0,
        time_step_s: float = 0.5
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute instantaneous and average throughput from SINR trajectory.
        
        Args:
            sinr_trajectory: SINR values over time
            bandwidth_mhz: Channel bandwidth in MHz
            time_step_s: Simulation time step in seconds
        
        Returns:
            Tuple of (throughput_bps, total_bits, avg_throughput_mbps)
        """
        throughput_bps = np.array([
            ThroughputAnalyzer.sinr_to_spectral_efficiency(sinr) * bandwidth_mhz * 1e6
            for sinr in sinr_trajectory
        ])
        
        total_bits = np.sum(throughput_bps) * time_step_s
        avg_throughput_mbps = np.mean(throughput_bps) / 1e6
        
        return throughput_bps, total_bits, avg_throughput_mbps
    
    @staticmethod
    def apply_ho_interruption(
        throughput_bps: np.ndarray,
        ho_indices: List[int],
        interruption_ms: float = 30.0,
        time_step_s: float = 0.5
    ) -> np.ndarray:
        """
        Apply service interruption penalty around handover events.
        
        Args:
            throughput_bps: Throughput array
            ho_indices: List of sample indices where HOs occur
            interruption_ms: Downtime per HO in milliseconds
            time_step_s: Simulation time step in seconds
        
        Returns:
            Modified throughput array with interruption penalties
        """
        effective_throughput = throughput_bps.copy()
        
        # Downtime as fraction of time step
        downtime_ratio = interruption_ms / (time_step_s * 1000)
        downtime_ratio = min(downtime_ratio, 1.0)  # Cap at 100%
        
        for ho_idx in ho_indices:
            # Apply penalty in window around HO event
            start = max(0, ho_idx - 1)
            end = min(len(effective_throughput), ho_idx + 3)
            
            for j in range(start, end):
                # Reduce throughput proportionally to distance from HO
                distance_from_ho = abs(j - ho_idx)
                penalty_factor = (1 - downtime_ratio / (distance_from_ho + 1))
                effective_throughput[j] *= penalty_factor
        
        return effective_throughput
    
    @staticmethod
    def plot_throughput_comparison(
        throughput_comparison_df: pd.DataFrame,
        baseline_df: pd.DataFrame = None,
        save_path: str = None,
        figsize: Tuple = (18, 12)
    ):
        """
        Comprehensive throughput visualization.
        
        Args:
            throughput_comparison_df: DataFrame with throughput metrics
            baseline_df: Optional baseline HO data for correlation
            save_path: Path to save figure
            figsize: Figure dimensions
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Throughput Comparison
        ax = axes[0, 0]
        x_pos = np.arange(min(25, len(throughput_comparison_df)))
        width = 0.35
        
        tp_base = throughput_comparison_df['Throughput_Baseline_Mbps'].iloc[:25]
        tp_ai = throughput_comparison_df['Throughput_AI_Mbps'].iloc[:25]
        
        ax.bar(x_pos - width/2, tp_base, width, label='3GPP Baseline',
              alpha=0.8, color='#3498db', edgecolor='black')
        ax.bar(x_pos + width/2, tp_ai, width, label='AI Optimised',
              alpha=0.8, color='#e74c3c', edgecolor='black')
        
        ax.set_ylabel('Throughput (Mbps)', fontsize=11, weight='bold')
        ax.set_xlabel('Route ID', fontsize=11, weight='bold')
        ax.set_title('Throughput Comparison (First 25 Routes)', fontsize=12, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 2: Throughput Improvement Distribution
        ax = axes[0, 1]
        improvements = throughput_comparison_df['Throughput_Improvement_Percent']
        
        ax.hist(improvements, bins=25, color='#2ecc71', alpha=0.7,
               edgecolor='black', linewidth=1.2)
        
        mean_imp = improvements.mean()
        ax.axvline(mean_imp, color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {mean_imp:.1f}%')
        ax.axvline(improvements.median(), color='orange', linestyle='--',
                  linewidth=2, label=f"Median: {improvements.median():.1f}%")
        
        ax.set_xlabel('Throughput Improvement (%)', fontsize=11, weight='bold')
        ax.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax.set_title('Distribution of Throughput Gains', fontsize=12, weight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Plot 3: HO Impact on Throughput
        ax = axes[0, 2]
        if baseline_df is not None and len(baseline_df) == len(throughput_comparison_df):
            ho_counts = baseline_df['HO_Count']
        else:
            ho_counts = np.arange(len(throughput_comparison_df))
        
        scatter = ax.scatter(ho_counts, improvements,
                            s=100, alpha=0.6, c=improvements,
                            cmap='RdYlGn', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Number of Handovers (Baseline)', fontsize=11, weight='bold')
        ax.set_ylabel('Throughput Improvement (%)', fontsize=11, weight='bold')
        ax.set_title('Throughput Gain vs. Handover Volume', fontsize=12, weight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Improvement (%)', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: SINR Impact
        ax = axes[1, 0]
        if 'SINR_Improvement_dB' in throughput_comparison_df.columns:
            sinr_imp = throughput_comparison_df['SINR_Improvement_dB']
            
            scatter2 = ax.scatter(throughput_comparison_df['SINR_Mean_Baseline_dB'],
                                 sinr_imp, s=100, alpha=0.6, c=sinr_imp,
                                 cmap='coolwarm', edgecolor='black', linewidth=1.5)
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Baseline SINR Mean (dB)', fontsize=11, weight='bold')
            ax.set_ylabel('SINR Improvement (dB)', fontsize=11, weight='bold')
            ax.set_title('SINR Quality Improvement', fontsize=12, weight='bold')
            cbar2 = plt.colorbar(scatter2, ax=ax)
            cbar2.set_label('SINR Gain (dB)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 5: Cumulative Throughput
        ax = axes[1, 1]
        cumul_base = throughput_comparison_df['Throughput_Baseline_Mbps'].cumsum()
        cumul_ai = throughput_comparison_df['Throughput_AI_Mbps'].cumsum()
        
        ax.plot(cumul_base.values, label='3GPP Baseline', linewidth=2.5,
               color='#3498db', marker='o', markersize=3, alpha=0.8)
        ax.plot(cumul_ai.values, label='AI Optimised', linewidth=2.5,
               color='#e74c3c', marker='s', markersize=3, alpha=0.8)
        
        total_tp_gain = cumul_ai.iloc[-1] - cumul_base.iloc[-1]
        ax.fill_between(range(len(cumul_base)), cumul_base, cumul_ai,
                       alpha=0.3, color='#2ecc71',
                       label=f'Total Gain: {total_tp_gain:.0f} Mbit')
        
        ax.set_xlabel('Route Index', fontsize=11, weight='bold')
        ax.set_ylabel('Cumulative Throughput (Mbps)', fontsize=11, weight='bold')
        ax.set_title('Cumulative Throughput Over All Routes', fontsize=12, weight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 6: Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        THROUGHPUT SUMMARY
        
        Baseline Metrics:
          Mean: {throughput_comparison_df['Throughput_Baseline_Mbps'].mean():.2f} Mbps
          Median: {throughput_comparison_df['Throughput_Baseline_Mbps'].median():.2f} Mbps
          Std: {throughput_comparison_df['Throughput_Baseline_Mbps'].std():.2f} Mbps
        
        AI Optimised Metrics:
          Mean: {throughput_comparison_df['Throughput_AI_Mbps'].mean():.2f} Mbps
          Median: {throughput_comparison_df['Throughput_AI_Mbps'].median():.2f} Mbps
          Std: {throughput_comparison_df['Throughput_AI_Mbps'].std():.2f} Mbps
        
        Improvement:
          Absolute: {(throughput_comparison_df['Throughput_AI_Mbps'].mean() - throughput_comparison_df['Throughput_Baseline_Mbps'].mean()):.2f} Mbps
          Relative: {improvements.mean():.1f}%
          Min: {improvements.min():.1f}%
          Max: {improvements.max():.1f}%
        
        SINR Improvement:
          Mean: {throughput_comparison_df.get('SINR_Improvement_dB', pd.Series([0])).mean():.2f} dB
          Median: {throughput_comparison_df.get('SINR_Improvement_dB', pd.Series([0])).median():.2f} dB
        
        Total Data (100 routes):
          Baseline: {cumul_base.iloc[-1]:.0f} Mbit
          AI: {cumul_ai.iloc[-1]:.0f} Mbit
          Gain: {total_tp_gain:.0f} Mbit ({(total_tp_gain/cumul_base.iloc[-1]*100):.1f}%)
        """
        
        ax.text(0.05, 0.95, summary_text, fontsize=9.5, verticalalignment='top',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Throughput analysis plot saved: {save_path}")
        
        plt.close()


if __name__ == '__main__':
    print("Delay and Throughput Analyzer Module")
    print("This module is intended for integration into the main comparison script.")
    print(f"\nHO Delay Components (Baseline): {sum(v['value_ms'] for k,v in HandoverDelayModel.DELAY_COMPONENTS.items() if v['policy'] in ('baseline', 'both')):.0f} ms")
    print(f"HO Delay Components (AI): {sum(v['value_ms'] for k,v in HandoverDelayModel.DELAY_COMPONENTS.items() if v['policy'] in ('ai', 'both')):.0f} ms")