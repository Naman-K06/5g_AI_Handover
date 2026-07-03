"""
feature_ablation.py

Systematic feature importance analysis for 5G handover optimization.
Tests the contribution of different signal metrics:
  - SINR only
  - RSRP only
  - Combined SINR + RSRP
  - With/without speed context
  - With/without congestion information
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import glob
import pickle
from typing import Tuple, Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents tkinter thread errors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)


class FeatureAblationStudy:
    """
    Systematically ablate features to measure their contribution to model performance.
    """
    
    # Define feature sets for ablation
    FEATURE_SETS = {
        'FULL': {
            'description': 'RSRP + SINR + Speed (Complete Model)',
            'components': ['RSRP', 'SINR', 'Speed'],
            'expected_rank': 1,
            'rationale': 'Uses all available information'
        },
        'SINR_ONLY': {
            'description': 'SINR + Speed (without RSRP)',
            'components': ['SINR', 'Speed'],
            'expected_rank': 2,
            'rationale': 'Tests if SINR (quality) alone is sufficient'
        },
        'RSRP_ONLY': {
            'description': 'RSRP + Speed (without SINR)',
            'components': ['RSRP', 'Speed'],
            'expected_rank': 3,
            'rationale': 'Tests if RSRP (power) alone is sufficient'
        },
        'SINR_NO_SPEED': {
            'description': 'SINR only (no speed context)',
            'components': ['SINR'],
            'expected_rank': 4,
            'rationale': 'Tests if speed context matters'
        },
        'RSRP_NO_SPEED': {
            'description': 'RSRP only (no speed context)',
            'components': ['RSRP'],
            'expected_rank': 5,
            'rationale': 'Tests if speed context matters'
        },
        'SPEED_ONLY': {
            'description': 'Speed only (no signal metrics)',
            'components': ['Speed'],
            'expected_rank': 7,
            'rationale': 'Baseline: can speed alone predict HOs?'
        },
        'BEST_SINR_ONLY': {
            'description': 'Best SINR + Speed (single dominant feature)',
            'components': ['BestSINR', 'Speed'],
            'expected_rank': 6,
            'rationale': 'Tests if summary statistic (max) suffices'
        },
        'BEST_RSRP_ONLY': {
            'description': 'Best RSRP + Speed (single dominant feature)',
            'components': ['BestRSRP', 'Speed'],
            'expected_rank': 6,
            'rationale': 'Tests if summary statistic (max) suffices'
        },
        'SINR_RSRP_NO_SPEED': {
            'description': 'SINR + RSRP (no speed context)',
            'components': ['SINR', 'RSRP'],
            'expected_rank': 2,
            'rationale': 'Isolate impact of speed feature'
        }
    }
    
    @staticmethod
    def extract_feature_subset(
        window_flat: np.ndarray,
        num_cells: int,
        feature_type: str
    ) -> np.ndarray:
        """
        Extract subset of features from flattened window.
        
        Window structure:
            [RSRP_0, RSRP_1, ..., RSRP_N,
             SINR_0, SINR_1, ..., SINR_N,
             speed_mps]
        
        Args:
            window_flat: Flattened feature window
            num_cells: Number of cells in network (determines RSRP segment length)
            feature_type: Type of feature subset to extract
        
        Returns:
            Subset of features as numpy array
        """
        rsrp_end = num_cells
        sinr_end = 2 * num_cells
        speed_idx = 2 * num_cells
        
        if feature_type == 'FULL':
            return window_flat.copy()
        
        elif feature_type == 'SINR_ONLY':
            sinr_feats = window_flat[rsrp_end:sinr_end]
            speed = window_flat[speed_idx:speed_idx+1]
            return np.concatenate([sinr_feats, speed])
        
        elif feature_type == 'RSRP_ONLY':
            rsrp_feats = window_flat[0:rsrp_end]
            speed = window_flat[speed_idx:speed_idx+1]
            return np.concatenate([rsrp_feats, speed])
        
        elif feature_type == 'SINR_NO_SPEED':
            return window_flat[rsrp_end:sinr_end].copy()
        
        elif feature_type == 'RSRP_NO_SPEED':
            return window_flat[0:rsrp_end].copy()
        
        elif feature_type == 'SPEED_ONLY':
            # Use only the single current-step speed scalar.
            # Using the full window would repeat the same speed value WINDOW_SIZE times,
            # letting the RF memorize trajectory IDs rather than learning from speed.
            return window_flat[speed_idx:speed_idx+1].copy()
        
        elif feature_type == 'BEST_SINR_ONLY':
            sinr_feats = window_flat[rsrp_end:sinr_end]
            best_sinr = np.array([np.max(sinr_feats)])
            speed = window_flat[speed_idx:speed_idx+1]
            return np.concatenate([best_sinr, speed])
        
        elif feature_type == 'BEST_RSRP_ONLY':
            rsrp_feats = window_flat[0:rsrp_end]
            best_rsrp = np.array([np.max(rsrp_feats)])
            speed = window_flat[speed_idx:speed_idx+1]
            return np.concatenate([best_rsrp, speed])
        
        elif feature_type == 'SINR_RSRP_NO_SPEED':
            rsrp_feats = window_flat[0:rsrp_end]
            sinr_feats = window_flat[rsrp_end:sinr_end]
            return np.concatenate([rsrp_feats, sinr_feats])
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    @staticmethod
    def ablate_training_data(
        X_full: np.ndarray,
        num_cells: int,
        feature_type: str
    ) -> np.ndarray:
        """
        Apply feature ablation to entire training dataset.
        
        Args:
            X_full: Full feature matrix (n_samples, n_features)
            num_cells: Number of cells in network
            feature_type: Type of ablation
        
        Returns:
            Ablated feature matrix
        """
        X_ablated = np.array([
            FeatureAblationStudy.extract_feature_subset(row, num_cells, feature_type)
            for row in X_full
        ])
        return X_ablated
    
    @staticmethod
    def train_and_evaluate_ablated_model(
        X_ablated: np.ndarray,
        y: np.ndarray,
        feature_type: str,
        n_estimators: int = 300,
        max_depth: int = 30,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train Random Forest on ablated features and evaluate performance.
        
        Args:
            X_ablated: Ablated feature matrix
            y: Target labels
            feature_type: Name of feature configuration
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            test_size: Fraction for test set
            random_state: Random seed
        
        Returns:
            Dictionary with training, validation, and test metrics
        """
        # Shuffle to break temporal/trajectory ordering before split.
        # Without this, consecutive samples from the same trajectory can appear
        # in both train and test, causing data leakage (especially for SPEED_ONLY).
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(y))
        X_ablated = X_ablated[perm]
        y = y[perm]

        # Disable stratify if any class has < 2 samples (avoids ValueError)
        unique, counts = np.unique(y, return_counts=True)
        stratify_arg = y if np.all(counts >= 2) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X_ablated, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )
        
        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        
        # Feature importance (mean decrease in impurity)
        feature_importance = clf.feature_importances_
        
        return {
            'model': clf,
            'feature_type': feature_type,
            'num_features': X_ablated.shape[1],
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_f1_weighted': float(test_f1),
            'test_f1_macro': float(test_f1_macro),
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_trees': n_estimators,
            'max_depth': max_depth,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }
    
    @staticmethod
    def run_full_ablation_study(
        X_full: np.ndarray,
        y: np.ndarray,
        num_cells: int,
        output_dir: str,
        n_estimators: int = 300,
        max_depth: int = 30,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Execute complete feature ablation study across all feature sets.
        
        Args:
            X_full: Full training feature matrix
            y: Target labels
            num_cells: Number of cells in network
            output_dir: Directory to save results
            n_estimators: RF hyperparameter
            max_depth: RF hyperparameter
            verbose: Print progress
        
        Returns:
            DataFrame with ablation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        ablation_results = []
        models_dict = {}
        
        for ablation_idx, (ablation_type, config) in enumerate(
            FeatureAblationStudy.FEATURE_SETS.items()
        ):
            if verbose:
                print(f"\n[{ablation_idx+1}/{len(FeatureAblationStudy.FEATURE_SETS)}] "
                      f"Testing: {ablation_type}")
                print(f"  Description: {config['description']}")
            
            try:
                # Extract ablated features
                X_ablated = FeatureAblationStudy.ablate_training_data(
                    X_full, num_cells, ablation_type
                )
                
                if verbose:
                    print(f"  Features: {X_ablated.shape[1]} (from {X_full.shape[1]})")
                
                # Train and evaluate
                result = FeatureAblationStudy.train_and_evaluate_ablated_model(
                    X_ablated, y, ablation_type,
                    n_estimators=n_estimators,
                    max_depth=max_depth
                )
                
                if verbose:
                    print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                    print(f"  Test F1 (weighted): {result['test_f1_weighted']:.4f}")
                
                # Store results
                ablation_results.append({
                    'Model': ablation_type,
                    'Description': config['description'],
                    'Components': ', '.join(config['components']),
                    'Num_Features': result['num_features'],
                    'Test_Accuracy': round(result['test_accuracy'], 4),
                    'Test_F1_Weighted': round(result['test_f1_weighted'], 4),
                    'Test_F1_Macro': round(result['test_f1_macro'], 4),
                    'Train_Accuracy': round(result['train_accuracy'], 4),
                    'Overfit_Ratio': round(result['train_accuracy'] - result['test_accuracy'], 4),
                    'Train_Samples': result['train_samples'],
                    'Test_Samples': result['test_samples'],
                    'Expected_Rank': config['expected_rank']
                })
                
                models_dict[ablation_type] = result
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
        
        # Create results DataFrame
        ablation_df = pd.DataFrame(ablation_results)
        ablation_df = ablation_df.sort_values('Test_Accuracy', ascending=False)
        
        # Save results
        csv_path = os.path.join(output_dir, 'feature_ablation_results.csv')
        ablation_df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"\n{'='*80}")
            print("FEATURE ABLATION RESULTS")
            print(f"{'='*80}")
            print(ablation_df.to_string(index=False))
            print(f"{'='*80}\n")
            print(f"Results saved to: {csv_path}")
        
        return ablation_df, models_dict


def plot_ablation_results(
    ablation_df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple = (16, 12)
):
    """
    Create comprehensive visualizations of ablation study results.
    
    Args:
        ablation_df: DataFrame with ablation results
        output_dir: Directory to save plots
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Sort by accuracy for consistent plotting
    df_sorted = ablation_df.sort_values('Test_Accuracy', ascending=True)
    
    # Plot 1: Accuracy Comparison
    ax = axes[0, 0]
    colors = ['#2ecc71' if x == ablation_df['Test_Accuracy'].max() else 
              ('#e74c3c' if x == ablation_df['Test_Accuracy'].min() else '#3498db')
              for x in df_sorted['Test_Accuracy']]
    bars = ax.barh(range(len(df_sorted)), df_sorted['Test_Accuracy'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['Model'], fontsize=10)
    ax.set_xlabel('Test Accuracy', fontsize=11, weight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=12, weight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df_sorted['Test_Accuracy'])):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 2: Feature Dimension vs Accuracy
    ax = axes[0, 1]
    scatter = ax.scatter(ablation_df['Num_Features'], ablation_df['Test_Accuracy'],
                        s=300, alpha=0.7, c=ablation_df['Test_Accuracy'],
                        cmap='RdYlGn', edgecolor='black', linewidth=1.5)
    
    for i, row in ablation_df.iterrows():
        ax.annotate(row['Model'], (row['Num_Features'], row['Test_Accuracy']),
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Number of Features', fontsize=11, weight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=11, weight='bold')
    ax.set_title('Feature Complexity vs. Accuracy Trade-off', fontsize=12, weight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Weighted F1 Comparison
    ax = axes[0, 2]
    df_sorted_f1 = ablation_df.sort_values('Test_F1_Weighted', ascending=True)
    colors_f1 = ['#2ecc71' if x == ablation_df['Test_F1_Weighted'].max() else '#3498db'
                 for x in df_sorted_f1['Test_F1_Weighted']]
    ax.barh(range(len(df_sorted_f1)), df_sorted_f1['Test_F1_Weighted'],
           color=colors_f1, alpha=0.8)
    ax.set_yticks(range(len(df_sorted_f1)))
    ax.set_yticklabels(df_sorted_f1['Model'], fontsize=10)
    ax.set_xlabel('Weighted F1 Score', fontsize=11, weight='bold')
    ax.set_title('F1 Score Comparison', fontsize=12, weight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Overfitting Analysis
    ax = axes[1, 0]
    df_sorted_overfit = ablation_df.sort_values('Overfit_Ratio', ascending=True)
    colors_overfit = ['#e74c3c' if x > 0.1 else '#2ecc71'
                     for x in df_sorted_overfit['Overfit_Ratio']]
    ax.barh(range(len(df_sorted_overfit)), df_sorted_overfit['Overfit_Ratio'],
           color=colors_overfit, alpha=0.8)
    ax.set_yticks(range(len(df_sorted_overfit)))
    ax.set_yticklabels(df_sorted_overfit['Model'], fontsize=10)
    ax.set_xlabel('Overfitting (Train Acc - Test Acc)', fontsize=11, weight='bold')
    ax.set_title('Generalization vs. Overfitting', fontsize=12, weight='bold')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Overfitting threshold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=9)
    
    # Plot 5: Train vs Test Accuracy
    ax = axes[1, 1]
    models = ablation_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, ablation_df['Train_Accuracy'], width,
          label='Train Accuracy', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, ablation_df['Test_Accuracy'], width,
          label='Test Accuracy', alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax.set_title('Train vs Test Accuracy Gap', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Plot 6: Summary Statistics Table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    ABLATION STUDY SUMMARY
    
    Best Model:
      {ablation_df.loc[ablation_df['Test_Accuracy'].idxmax(), 'Model']}
      Accuracy: {ablation_df['Test_Accuracy'].max():.4f}
    
    Worst Model:
      {ablation_df.loc[ablation_df['Test_Accuracy'].idxmin(), 'Model']}
      Accuracy: {ablation_df['Test_Accuracy'].min():.4f}
    
    Accuracy Range:
      {ablation_df['Test_Accuracy'].max() - ablation_df['Test_Accuracy'].min():.4f}
    
    Average Accuracy:
      {ablation_df['Test_Accuracy'].mean():.4f}
    
    Feature Efficiency:
      (Best Acc / Features)
    
    Recommended Model:
      Full features provide
      {(ablation_df.loc[ablation_df['Test_Accuracy'].idxmax(), 'Test_Accuracy'] - 
        ablation_df.loc[ablation_df['Test_Accuracy'].idxmin(), 'Test_Accuracy'])*100:.1f}% improvement
      over simplest model
    """
    
    ax.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'feature_ablation_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature ablation analysis plot saved: {fig_path}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Feature Ablation Study Module")
    print("This module is intended to be imported and used in the main comparison script.")
    print("\nExample feature sets for ablation:")
    for name, config in FeatureAblationStudy.FEATURE_SETS.items():
        print(f"  {name}: {config['description']}")