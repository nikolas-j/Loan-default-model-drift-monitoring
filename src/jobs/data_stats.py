import pandas as pd
from src.logging_config import setup_logging

logger = setup_logging(__name__)

def run_statistics(data: pd.DataFrame) -> dict:
    """
    Compute basic statistics on the dataset.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        stats (dict): Dictionary containing basic statistics.
    """
    if data.empty:
        return {
            "num_rows": 0,
            "num_columns": 0,
            "missing_values": {},
            "column_types": {},
            "descriptive_stats": {}
        }

    stats = {
        "num_rows": data.shape[0],
        "num_columns": data.shape[1],
        "missing_values": data.isnull().sum().to_dict(),
        "column_types": data.dtypes.astype(str).to_dict(),
        "descriptive_stats": data.describe().to_dict()
    }
    return stats


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Compute basic statistics on a dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save statistics as JSON")
    
    args = parser.parse_args()
    
    logger.info(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    
    stats = run_statistics(data)
    logger.info(f"Computed statistics: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(json.dumps(stats, indent=2))
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {args.output}")