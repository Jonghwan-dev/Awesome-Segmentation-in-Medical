import numpy as np
import sys
import pandas as pd

def main():
    """
    Reads metric values from a temporary file, calculates mean and std,
    and prints them.
    """
    if len(sys.argv) < 2:
        print("Usage: python aggregate_results.py <results_file_path>")
        return

    results_file = sys.argv[1]
    
    try:
        # Load the collected results into a pandas DataFrame
        df = pd.read_csv(results_file)
        
        print("\n--- Aggregated Test Results (Mean ± Std) ---")
        for metric in df.columns:
            mean = df[metric].mean()
            std = df[metric].std()
            print(f"  {metric:<15}: {mean:.4f} ± {std:.4f}")
        print("---------------------------------------------")

    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
