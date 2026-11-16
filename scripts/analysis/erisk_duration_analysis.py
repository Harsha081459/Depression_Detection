"""
Stage A1: Duration analysis for eRisk Task 2 users
Calculates per-user duration statistics for understanding the labelled set
"""

import pandas as pd
import json
from pathlib import Path

def load_data():
    """Load the eRisk data"""
    # Use relative path from script location
    script_dir = Path(__file__).parent.parent.parent
    base_path = script_dir
    
    # Load metadata
    meta_path = base_path / "data/processed/erisk/task2_target_meta.csv"
    meta_df = pd.read_csv(meta_path)
    
    # Load user features (for labels)
    features_path = base_path / "data/processed/erisk/task2_user_l2_features.csv"
    features_df = pd.read_csv(features_path)
    
    return meta_df, features_df, base_path

def calculate_duration_stats(meta_df, features_df):
    """Calculate duration statistics for each user"""
    
    # Convert date strings to datetime
    meta_df['date'] = pd.to_datetime(meta_df['date'])
    
    # Group by user and calculate stats
    duration_stats = meta_df.groupby('user_id').agg({
        'date': ['min', 'max', 'count'],
        'kind': lambda x: (x == 'submission').sum()  # count submissions
    }).reset_index()
    
    # Flatten column names
    duration_stats.columns = ['user_id', 'first_date', 'last_date', 'n_writings', 'n_submissions']
    
    # Calculate span in days (fractional to capture same-day bursts)
    duration_stats['span_days'] = (duration_stats['last_date'] - duration_stats['first_date']).dt.total_seconds() / 86400
    
    # Add comments count
    duration_stats['n_comments'] = duration_stats['n_writings'] - duration_stats['n_submissions']
    
    # Merge with labels from features_df
    duration_stats = duration_stats.merge(
        features_df[['user_id', 'label']], 
        on='user_id', 
        how='left'
    )
    
    # Sort by span_days
    duration_stats = duration_stats.sort_values('span_days')
    
    return duration_stats

def generate_summary(duration_stats):
    """Generate summary statistics by label and span categories"""
    
    summary = {
        'total_users': len(duration_stats),
        'total_positives': int(duration_stats['label'].sum()),
        'total_controls': int(len(duration_stats) - duration_stats['label'].sum()),
        'overall_median_span_days': float(duration_stats['span_days'].median()),
        'overall_median_n_writings': float(duration_stats['n_writings'].median()),
        'span_calculation': 'fractional days (total_seconds / 86400) - same-day bursts show as ~0.2 days, not 0'
    }
    
    # Span categories for positives
    positive_users = duration_stats[duration_stats['label'] == 1]
    control_users = duration_stats[duration_stats['label'] == 0]
    
    def span_category_counts(users):
        # Bins use the same thresholds as before; span is fractional days.
        return {
            'lt_60_days': int((users['span_days'] < 60).sum()),
            'lt_180_days': int((users['span_days'] < 180).sum()),
            'lt_365_days': int((users['span_days'] < 365).sum()),
            'gte_2_years': int((users['span_days'] >= 730).sum()),
            'median_span': float(users['span_days'].median()),
            'median_writings': float(users['n_writings'].median()),
        }
    
    summary['positives'] = span_category_counts(positive_users)
    summary['controls'] = span_category_counts(control_users)
    
    return summary

def main():
    """Main execution function"""
    print("Loading eRisk data...")
    meta_df, features_df, base_path = load_data()
    
    print("Calculating duration statistics...")
    duration_stats = calculate_duration_stats(meta_df, features_df)
    
    print("Generating summary...")
    summary = generate_summary(duration_stats)
    
    # Create output directory
    output_dir = base_path / "results/erisk_understand"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save duration CSV
    duration_csv_path = output_dir / "user_duration.csv"
    duration_stats.to_csv(duration_csv_path, index=False)
    print(f"Saved duration stats to {duration_csv_path}")
    
    # Save summary JSON
    summary_json_path = output_dir / "duration_summary.json"
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_json_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("DURATION ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total users: {summary['total_users']}")
    print(f"Positives: {summary['total_positives']} ({summary['total_positives']/summary['total_users']*100:.1f}%)")
    print(f"Controls: {summary['total_controls']} ({summary['total_controls']/summary['total_users']*100:.1f}%)")
    print(f"\nOverall median span: {summary['overall_median_span_days']:.1f} days")
    print(f"Overall median writings: {summary['overall_median_n_writings']:.0f}")
    
    print("\n" + "-"*50)
    print("POSITIVES (n={}):".format(summary['total_positives']))
    print("-"*50)
    pos = summary['positives']
    print(f"Span < 60 days: {pos['lt_60_days']} ({pos['lt_60_days']/summary['total_positives']*100:.1f}%)")
    print(f"Span < 180 days: {pos['lt_180_days']} ({pos['lt_180_days']/summary['total_positives']*100:.1f}%)")
    print(f"Span < 365 days: {pos['lt_365_days']} ({pos['lt_365_days']/summary['total_positives']*100:.1f}%)")
    print(f"Span >= 2 years: {pos['gte_2_years']} ({pos['gte_2_years']/summary['total_positives']*100:.1f}%)")
    print(f"Median span: {pos['median_span']:.1f} days")
    print(f"Median writings: {pos['median_writings']:.0f}")
    
    print("\n" + "-"*50)
    print("CONTROLS (n={}):".format(summary['total_controls']))
    print("-"*50)
    ctrl = summary['controls']
    print(f"Span < 60 days: {ctrl['lt_60_days']} ({ctrl['lt_60_days']/summary['total_controls']*100:.1f}%)")
    print(f"Span < 180 days: {ctrl['lt_180_days']} ({ctrl['lt_180_days']/summary['total_controls']*100:.1f}%)")
    print(f"Span < 365 days: {ctrl['lt_365_days']} ({ctrl['lt_365_days']/summary['total_controls']*100:.1f}%)")
    print(f"Span >= 2 years: {ctrl['gte_2_years']} ({ctrl['gte_2_years']/summary['total_controls']*100:.1f}%)")
    print(f"Median span: {ctrl['median_span']:.1f} days")
    print(f"Median writings: {ctrl['median_writings']:.0f}")
    
    return duration_stats, summary

if __name__ == "__main__":
    duration_stats, summary = main()