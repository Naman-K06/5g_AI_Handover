"""
statistical_analyzer.py

Comprehensive statistical testing module for 5G handover research.
Provides:
  - Independent samples t-tests with confidence intervals
  - Mann-Whitney U non-parametric tests
  - Bootstrap confidence intervals
  - Effect size calculations (Cohen's d, rank-biserial)
  - Normality and homogeneity tests
  - Publication-quality visualizations
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents tkinter thread errors
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalAnalyzer:
    """Formal hypothesis testing with confidence intervals."""

    def __init__(self, confidence_level: float = 0.95, random_seed: int = 42):
        """
        Initialize analyzer.
        
        Args:
            confidence_level: Confidence level for CIs (default 0.95 for 95% CI)
            random_seed: Seed for reproducible bootstrap sampling
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)

    @staticmethod
    def normality_test(data: np.ndarray) -> Dict:
        """
        Shapiro-Wilk test for normality.
        
        Returns:
            Dictionary with test statistic and p-value.
            If p > 0.05, data are approximately normally distributed.
        """
        statistic, p_value = stats.shapiro(data)
        return {
            'test': 'Shapiro-Wilk',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_normal': p_value > 0.05,
            'interpretation': 'Normal' if p_value > 0.05 else 'Non-normal'
        }

    @staticmethod
    def homogeneity_test(data1: np.ndarray, data2: np.ndarray) -> Dict:
        """
        Levene's test for equality of variances.
        
        Returns:
            Dictionary with test results.
            If p > 0.05, variances are approximately equal.
        """
        statistic, p_value = stats.levene(data1, data2)
        return {
            'test': "Levene's",
            'statistic': float(statistic),
            'p_value': float(p_value),
            'equal_variance': p_value > 0.05,
            'interpretation': 'Equal variances' if p_value > 0.05 else 'Unequal variances'
        }

    def independent_ttest_with_ci(
        self,
        baseline: np.ndarray,
        optimized: np.ndarray,
        test_name: str = 'KPI'
    ) -> Dict:
        """
        Performs Welch's independent samples t-test with confidence intervals.
        
        Welch's t-test does not assume equal variances, making it more robust.
        
        Args:
            baseline: Array of baseline group measurements
            optimized: Array of optimized group measurements
            test_name: Name of the test (for reporting)
        
        Returns:
            Dictionary containing:
            - Descriptive statistics (mean, std, n)
            - Test results (t-statistic, p-value)
            - Confidence interval of mean difference
            - Effect size (Cohen's d)
            - Interpretation string
        """
        baseline = np.asarray(baseline, dtype=float)
        optimized = np.asarray(optimized, dtype=float)
        
        n1, n2 = len(baseline), len(optimized)
        mean1, mean2 = np.mean(baseline), np.mean(optimized)
        std1, std2 = np.std(baseline, ddof=1), np.std(optimized, ddof=1)
        
        mean_diff = mean2 - mean1
        
        # Welch's t-test (assumes unequal variances)
        t_stat, p_val = stats.ttest_ind(optimized, baseline, equal_var=False)
        
        # Standard error of difference
        se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
        
        # Welch-Satterthwaite degrees of freedom
        df = ((std1**2 / n1 + std2**2 / n2)**2) / (
            (std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1)
        )
        
        # Confidence interval
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        # Cohen's d effect size (pooled standard deviation)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size_interp = 'negligible'
        elif abs_d < 0.5:
            effect_size_interp = 'small'
        elif abs_d < 0.8:
            effect_size_interp = 'medium'
        else:
            effect_size_interp = 'large'
        
        # Significance interpretation
        if p_val < 0.001:
            sig_level = '***'
        elif p_val < 0.01:
            sig_level = '**'
        elif p_val < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
        
        return {
            'test_name': test_name,
            'test_type': "Welch's t-test (unequal variances)",
            'mean_baseline': float(mean1),
            'mean_optimized': float(mean2),
            'std_baseline': float(std1),
            'std_optimized': float(std2),
            'mean_difference': float(mean_diff),
            'se_difference': float(se_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'df': float(df),
            'p_value': float(p_val),
            'p_value_log': float(-np.log10(max(p_val, 1e-10))),
            'cohens_d': float(cohens_d),
            'effect_size': effect_size_interp,
            'significance': sig_level,
            'n_baseline': int(n1),
            'n_optimized': int(n2),
            'improvement_percent': float((mean_diff / mean1 * 100) if mean1 != 0 else 0),
            'is_significant': p_val < 0.05
        }

    def mann_whitney_u_test(
        self,
        baseline: np.ndarray,
        optimized: np.ndarray,
        test_name: str = 'KPI'
    ) -> Dict:
        """
        Non-parametric Mann-Whitney U test (alternative to t-test).
        
        Does not assume normality or equal variances. Suitable for
        ordinal data or non-normal distributions.
        
        Args:
            baseline: Array of baseline group measurements
            optimized: Array of optimized group measurements
            test_name: Name of the test (for reporting)
        
        Returns:
            Dictionary with test results and effect size (rank-biserial correlation).
        """
        baseline = np.asarray(baseline, dtype=float)
        optimized = np.asarray(optimized, dtype=float)
        
        u_stat, p_val = stats.mannwhitneyu(
            baseline, optimized, alternative='two-sided'
        )
        
        n1, n2 = len(baseline), len(optimized)
        
        # Rank-biserial correlation (effect size for Mann-Whitney)
        # Ranges from -1 to 1
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        
        # Effect size interpretation (Cohen's guidelines adapted for rank-biserial)
        abs_r = abs(r_rb)
        if abs_r < 0.1:
            effect_size_interp = 'negligible'
        elif abs_r < 0.3:
            effect_size_interp = 'small'
        elif abs_r < 0.5:
            effect_size_interp = 'medium'
        else:
            effect_size_interp = 'large'
        
        # Significance
        if p_val < 0.001:
            sig_level = '***'
        elif p_val < 0.01:
            sig_level = '**'
        elif p_val < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
        
        return {
            'test_name': test_name,
            'test_type': 'Mann-Whitney U (non-parametric)',
            'u_statistic': float(u_stat),
            'p_value': float(p_val),
            'rank_biserial_r': float(r_rb),
            'median_baseline': float(np.median(baseline)),
            'median_optimized': float(np.median(optimized)),
            'median_difference': float(np.median(optimized) - np.median(baseline)),
            'effect_size': effect_size_interp,
            'significance': sig_level,
            'n_baseline': int(n1),
            'n_optimized': int(n2),
            'is_significant': p_val < 0.05
        }

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_func=np.mean,
        n_bootstrap: int = 10000,
        test_name: str = 'Statistic'
    ) -> Dict:
        """
        Non-parametric bootstrap confidence interval.
        
        Robust to distribution assumptions. Resamples data with replacement
        and computes desired statistic on each sample.
        
        Args:
            data: Input array
            statistic_func: Function to compute on each bootstrap sample
            n_bootstrap: Number of bootstrap iterations
            test_name: Name of test (for reporting)
        
        Returns:
            Dictionary with bootstrap CI and point estimate.
        """
        data = np.asarray(data, dtype=float)
        np.random.seed(self.random_seed)
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        point_estimate = statistic_func(data)
        
        ci_lower = np.percentile(bootstrap_stats, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - self.alpha / 2) * 100)
        
        # Bootstrap SE (standard error)
        bootstrap_se = np.std(bootstrap_stats, ddof=1)
        
        return {
            'test_name': test_name,
            'method': f'Bootstrap (n={n_bootstrap})',
            'point_estimate': float(point_estimate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_width': float(ci_upper - ci_lower),
            'bootstrap_se': float(bootstrap_se),
            'n_data': len(data),
            'n_bootstrap': n_bootstrap
        }

    @staticmethod
    def paired_ttest_with_ci(
        before: np.ndarray,
        after: np.ndarray,
        test_name: str = 'Paired Comparison'
    ) -> Dict:
        """
        Paired samples t-test (same subjects measured twice).
        
        Useful when comparing the same routes under both policies.
        
        Args:
            before: Measurements before intervention
            after: Measurements after intervention
            test_name: Name of test (for reporting)
        
        Returns:
            Dictionary with paired test results and CI of mean difference.
        """
        before = np.asarray(before, dtype=float)
        after = np.asarray(after, dtype=float)
        
        differences = after - before
        n = len(differences)
        
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)
        
        t_stat = mean_diff / se_diff if se_diff > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))  # Two-tailed
        
        # CI — use module-level default alpha (0.05 → 95% CI)
        alpha = 0.05  # paired_ttest is a @staticmethod so uses fixed alpha
        t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
        
        # Cohen's d for paired data
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        if p_val < 0.001:
            sig_level = '***'
        elif p_val < 0.01:
            sig_level = '**'
        elif p_val < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
        
        return {
            'test_name': test_name,
            'test_type': 'Paired t-test',
            'mean_before': float(np.mean(before)),
            'mean_after': float(np.mean(after)),
            'mean_difference': float(mean_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            't_statistic': float(t_stat),
            'df': int(n - 1),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'significance': sig_level,
            'n_pairs': int(n),
            'is_significant': p_val < 0.05
        }

    def plot_ci_comparison(
        self,
        results_dict: Dict[str, Dict],
        save_path: str = None,
        figsize: Tuple = (14, 10)
    ):
        """
        Create publication-quality plot of KPI comparisons with confidence intervals.
        
        Args:
            results_dict: Dictionary of test results from independent_ttest_with_ci
            save_path: Path to save figure (if None, displays only)
            figsize: Figure size (width, height)
        """
        n_tests = len(results_dict)
        n_cols = 2
        n_rows = (n_tests + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_tests > 1 else [axes]
        
        for idx, (kpi_name, result) in enumerate(results_dict.items()):
            ax = axes[idx]
            
            means = [result['mean_baseline'], result['mean_optimized']]
            ci_lowers = [
                result['mean_baseline'] - 1.96 * result['std_baseline'] / np.sqrt(result['n_baseline']),
                result['mean_optimized'] - 1.96 * result['std_optimized'] / np.sqrt(result['n_optimized'])
            ]
            ci_uppers = [
                result['mean_baseline'] + 1.96 * result['std_baseline'] / np.sqrt(result['n_baseline']),
                result['mean_optimized'] + 1.96 * result['std_optimized'] / np.sqrt(result['n_optimized'])
            ]
            
            errors = [
                [means[i] - ci_lowers[i] for i in range(2)],
                [ci_uppers[i] - means[i] for i in range(2)]
            ]
            
            x_pos = [0, 1]
            colors = ['#3498db', '#e74c3c']
            ax.bar(x_pos, means, yerr=errors, capsize=12, color=colors, 
                   alpha=0.75, edgecolor='black', linewidth=1.5, width=0.6)
            
            # Annotate sample sizes on bars
            ax.text(0, means[0] * 0.05, f'n={result["n_baseline"]}',
                   ha='center', va='bottom', fontsize=9, color='white', weight='bold')
            ax.text(1, means[1] * 0.05, f'n={result["n_optimized"]}',
                   ha='center', va='bottom', fontsize=9, color='white', weight='bold')
            
            # Statistical annotation
            p_val = result['p_value']
            sig_text = f"p={p_val:.4f} {result['significance']}"
            cohens_d_text = f"Cohen's d={result['cohens_d']:.3f} ({result['effect_size']})"
            
            y_max = max(means) * 1.2
            ax.text(0.5, y_max * 0.95, sig_text, ha='center', fontsize=10, 
                   weight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            ax.text(0.5, y_max * 0.85, cohens_d_text, ha='center', fontsize=9)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['3GPP Baseline', 'AI Optimised'], fontsize=10)
            ax.set_ylabel(f'{kpi_name} Count', fontsize=11, weight='bold')
            ax.set_title(f'{kpi_name} Comparison with 95% CI\n'
                        f'Improvement: {result["improvement_percent"]:.1f}%',
                        fontsize=12, weight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_ylim(0, y_max)
        
        # Hide unused subplots
        for idx in range(n_tests, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confidence interval plot saved: {save_path}")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def create_statistical_summary_table(
        test_results: Dict[str, Dict],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Create a publication-ready summary table of all statistical tests.
        
        Args:
            test_results: Dictionary of test results
            save_path: Path to save CSV
        
        Returns:
            DataFrame with summary statistics
        """
        summary_rows = []
        
        for kpi_name, result in test_results.items():
            summary_rows.append({
                'KPI': kpi_name,
                'Baseline_Mean': f"{result['mean_baseline']:.2f}",
                'Baseline_Std': f"{result['std_baseline']:.2f}",
                'Optimized_Mean': f"{result['mean_optimized']:.2f}",
                'Optimized_Std': f"{result['std_optimized']:.2f}",
                'Mean_Difference': f"{result['mean_difference']:.2f}",
                'CI_95_Lower': f"{result['ci_lower']:.2f}",
                'CI_95_Upper': f"{result['ci_upper']:.2f}",
                't_statistic': f"{result['t_statistic']:.3f}",
                'p_value': f"{result['p_value']:.6f}",
                'Significance': result['significance'],
                "Cohen's_d": f"{result['cohens_d']:.3f}",
                'Effect_Size': result['effect_size'],
                'Improvement_%': f"{result['improvement_percent']:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        if save_path:
            summary_df.to_csv(save_path, index=False)
            print(f"✓ Statistical summary table saved: {save_path}")
        
        return summary_df


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Simulate baseline and optimized data
    baseline_kpi = np.random.normal(12.5, 2.5, 100)
    optimized_kpi = np.random.normal(8.3, 2.0, 100)
    
    analyzer = StatisticalAnalyzer(confidence_level=0.95)
    
    # Run tests
    t_test_result = analyzer.independent_ttest_with_ci(baseline_kpi, optimized_kpi, 'HO_Count')
    mw_test_result = analyzer.mann_whitney_u_test(baseline_kpi, optimized_kpi, 'HO_Count')
    
    print("T-Test Results:")
    print(f"  p-value: {t_test_result['p_value']:.6f}")
    print(f"  95% CI: [{t_test_result['ci_lower']:.2f}, {t_test_result['ci_upper']:.2f}]")
    print(f"  Cohen's d: {t_test_result['cohens_d']:.3f}")
    print("\nMann-Whitney U Results:")
    print(f"  p-value: {mw_test_result['p_value']:.6f}")
    print(f"  Rank-biserial r: {mw_test_result['rank_biserial_r']:.3f}")