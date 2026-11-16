"""
Stage A4: Export target writings for the 9 selected positive users
Exports date, kind, title, body for team review (private folder, not committed)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def load_sample_ids():
    """Load the sampled user IDs"""
    script_dir = Path(__file__).parent.parent.parent
    base_path = script_dir
    sample_path = base_path / "results/erisk_understand/sample_positive_ids.json"
    
    with open(sample_path, 'r') as f:
        sample_data = json.load(f)
    
    return sample_data['all_ids'], base_path

def load_user_json(user_id, base_path):
    """Load the raw JSON file for a specific user"""
    raw_data_path = base_path / "data/raw/erisk26/task2-contextualized-depression-20260825T204631Z-1-001/task2-contextualized-depression/all_users_combined-released/all_users_combined"
    user_file_path = raw_data_path / f"{user_id}.json"
    
    if not user_file_path.exists():
        print(f"Warning: File not found for {user_id}")
        return None
    
    with open(user_file_path, 'r', encoding='utf-8') as f:
        user_data = json.load(f)
    
    return user_data

def extract_target_writings(user_data, user_id):
    """Extract only target writings from user data"""
    if user_data is None:
        return []
    
    target_writings = []
    
    # The data is structured as an array of thread objects
    # Each thread has a "submission" and "comments" array
    for thread in user_data:
        # Check the submission
        if 'submission' in thread:
            submission = thread['submission']
            if submission.get('user_id') == user_id and submission.get('target', False) == True:
                writing = {
                    'user_id': user_id,
                    'date': submission.get('created_utc', ''),
                    'kind': 'submission',
                    'title': submission.get('title', ''),
                    'body': submission.get('body', '')
                }
                target_writings.append(writing)
        
        # Check the comments
        if 'comments' in thread:
            for comment in thread['comments']:
                if comment.get('user_id') == user_id and comment.get('target', False) == True:
                    writing = {
                        'user_id': user_id,
                        'date': comment.get('created_utc', ''),
                        'kind': 'comment',
                        'title': '',  # Comments don't have titles
                        'body': comment.get('body', '')
                    }
                    target_writings.append(writing)
    
    # Sort by date
    target_writings.sort(key=lambda x: x['date'])
    
    return target_writings

def export_user_writings(user_id, target_writings, output_dir):
    """Export writings for a single user to a CSV file"""
    if not target_writings:
        print(f"No target writings found for {user_id}")
        return 0

    df = pd.DataFrame(target_writings)
    output_file = output_dir / f"{user_id}_writings.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Exported {len(target_writings)} writings for {user_id} to {output_file}")
    return len(target_writings)


def prune_extra_csvs(output_dir, keep_ids):
    """Remove leftover CSVs from a previous sample (not in all_ids)."""
    keep = {f"{uid}_writings.csv" for uid in keep_ids}
    for path in output_dir.glob("subject_*_writings.csv"):
        if path.name not in keep:
            path.unlink()
            print(f"Deleted leftover {path.name}")


def verify_against_duration(base_path, user_ids, export_summary):
    duration_path = base_path / "results/erisk_understand/user_duration.csv"
    duration = pd.read_csv(duration_path)
    mismatches = []
    for item in export_summary:
        uid = item["user_id"]
        row = duration.loc[duration["user_id"] == uid]
        if row.empty:
            mismatches.append((uid, "missing from duration table"))
            continue
        expected = int(row.iloc[0]["n_writings"])
        label = int(row.iloc[0]["label"])
        if label != 1:
            mismatches.append((uid, f"label={label}, expected positive"))
        if expected != item["n_writings"]:
            mismatches.append((uid, f"csv={item['n_writings']} duration={expected}"))
    return mismatches


def main():
    """Main execution function"""
    print("Loading sample user IDs...")
    user_ids, base_path = load_sample_ids()

    output_dir = base_path / "data/intake/human_review_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    prune_extra_csvs(output_dir, user_ids)

    print(f"Exporting writings for {len(user_ids)} users to {output_dir}")
    print("Note: This directory is gitignored and contains sensitive data for team review only.")
    print("Count rows in Excel or pandas, not Notepad (bodies contain newlines).")

    total_writings = 0
    export_summary = []

    for user_id in user_ids:
        print(f"\nProcessing {user_id}...")
        user_data = load_user_json(user_id, base_path)
        target_writings = extract_target_writings(user_data, user_id)
        n_writings = export_user_writings(user_id, target_writings, output_dir)
        total_writings += n_writings
        export_summary.append({"user_id": user_id, "n_writings": n_writings})

    mismatches = verify_against_duration(base_path, user_ids, export_summary)
    if mismatches:
        raise SystemExit(f"Export does not match duration table: {mismatches}")

    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "export_date": datetime.now().isoformat(),
                "total_users": len(user_ids),
                "total_writings": total_writings,
                "users": export_summary,
                "count_rows_with": "Excel or pandas, not Notepad line count",
            },
            f,
            indent=2,
        )
    
    print("\n" + "="*50)
    print("EXPORT SUMMARY")
    print("="*50)
    print(f"Total users processed: {len(user_ids)}")
    print(f"Total writings exported: {total_writings}")
    print(f"Output directory: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    print("\nExport details:")
    for item in export_summary:
        print(f"  {item['user_id']}: {item['n_writings']} writings")
    
    print("\nIMPORTANT: These files contain user data for research purposes only.")
    print("   Do not commit to version control or share outside the research team.")
    print("   Respect the eRisk Data Use Agreement.")
    print("   NOTE: When counting writings, use Excel row count or pandas len(df), not Notepad lines.")

if __name__ == "__main__":
    main()