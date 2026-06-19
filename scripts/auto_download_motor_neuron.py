"""
Automated script to search and download motor neuron datasets from CellxGene API.

This script:
1. Searches CellxGene API for motor neuron datasets
2. Shows you the top results
3. Lets you pick which one to download
4. Automatically downloads and processes it
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cellxgene_api import CellxGeneAPI, display_search_results
from src.data.cellxgene_loader import StemCellDataLoader
from src.utils import setup_logger

logger = setup_logger("auto_download", level="INFO")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("[AUTO] AUTOMATED MOTOR NEURON DATASET DOWNLOADER")
    print("="*80)

    # Step 1: Search for datasets
    print("\n[SEARCH] Step 1: Searching CellxGene API for motor neuron datasets...")
    print("-"*80)

    api = CellxGeneAPI()
    results = api.search_motor_neuron_datasets()

    if results.empty:
        print("\n[ERROR] No motor neuron datasets found")
        print("\nTrying broader search...")

        # Fallback: broader search
        results = api.search_datasets(
            keywords=['neuron', 'differentiation', 'iPSC'],
            min_cells=10000
        )

    if results.empty:
        print("\n[ERROR] No suitable datasets found")
        print("\nOptions:")
        print("1. Try manual search at: https://cellxgene.cziscience.com/datasets")
        print("2. Use synthetic data: python examples/hybrid_ml_demo.py")
        return

    # Step 2: Display results
    print(f"\n[SUCCESS] Found {len(results)} datasets!")
    display_search_results(results, top_n=5)

    # Step 3: Let user choose
    print("\n" + "="*80)
    print("[DOWNLOAD] Step 2: Choose a dataset to download")
    print("="*80)

    print("\nTop recommendations:")
    for i in range(min(3, len(results))):
        row = results.iloc[i]
        print(f"\n[{i+1}] {row['collection_name']}")
        print(f"    {row['cell_count']:,} cells")
        print(f"    {row['dataset_name']}")

    print("\n" + "-"*80)

    # Auto-select first one or let user choose
    try:
        choice = input(f"\nEnter dataset number (1-{min(3, len(results))}) or press Enter for #1: ").strip()

        if not choice:
            choice = 1
        else:
            choice = int(choice)

        if choice < 1 or choice > min(3, len(results)):
            print(f"Invalid choice. Using dataset #1")
            choice = 1

        selected = results.iloc[choice - 1]

    except (ValueError, KeyboardInterrupt):
        print("\nUsing dataset #1")
        selected = results.iloc[0]

    print(f"\n[SUCCESS] Selected: {selected['collection_name']}")
    print(f"   Dataset: {selected['dataset_name']}")
    print(f"   Cells: {selected['cell_count']:,}")

    # Step 4: Download
    print("\n" + "="*80)
    print("[DOWNLOADING]  Step 3: Downloading dataset")
    print("="*80)

    url = selected['h5ad_url']
    dataset_name = f"motor_neuron_{selected['cell_count']}_cells.h5ad"

    try:
        filepath = api.download_dataset(url, 'data/raw', dataset_name)
        print(f"\n[SUCCESS] Download complete: {filepath}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return

    # Step 5: Process dataset
    print("\n" + "="*80)
    print("[PROCESSING] Step 4: Processing dataset")
    print("="*80)

    try:
        loader = StemCellDataLoader()

        print("\nLoading h5ad file...")
        adata = loader.load_h5ad(filepath)

        print(f"\n[SUCCESS] Loaded successfully!")
        print(f"   Cells: {adata.n_obs:,}")
        print(f"   Genes: {adata.n_vars:,}")

        # Check metadata
        print(f"\nMetadata columns: {list(adata.obs.columns)}")

        # Find time column
        time_cols = ['day', 'timepoint', 'time', 'days', 'development_stage']
        time_col = None

        for col in time_cols:
            if col in adata.obs.columns:
                time_col = col
                print(f"\n[SUCCESS] Found time column: '{col}'")
                print(f"   Timepoints: {sorted(adata.obs[col].unique())}")
                break

        if not time_col:
            print("\n[WARNING]  No time column found, will use pseudotime")
            time_col = 'pseudotime'

        # Extract trajectories
        print(f"\n[PROCESSING] Extracting trajectories...")
        trajectories = loader.extract_trajectories(
            adata,
            time_column=time_col,
            n_trajectories=200
        )

        print(f"\n[SUCCESS] Extracted {len(trajectories)} trajectories")

        # Save
        output_path = 'data/processed/motor_neuron_trajectories.pkl'
        loader.save_processed_data(trajectories, output_path)

        print(f"[SUCCESS] Saved to: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        print("\nYou can process manually with:")
        print(f"python scripts/prepare_motor_neuron_data.py --file {filepath}")
        return

    # Step 6: Success!
    print("\n" + "="*80)
    print("[COMPLETE] SUCCESS! Dataset ready for training")
    print("="*80)

    print(f"\n[SUCCESS] Downloaded: {selected['collection_name']}")
    print(f"[SUCCESS] Processed: {len(trajectories)} trajectories")
    print(f"[SUCCESS] Ready for ML training")

    print("\n[INFO] Next steps:")
    print("\n1. Train LSTM model:")
    print("   python experiments/train_predictor.py \\")
    print("       --load_data data/processed/motor_neuron_trajectories.pkl \\")
    print("       --model lstm --epochs 100")

    print("\n2. Train Transformer model:")
    print("   python experiments/train_predictor.py \\")
    print("       --load_data data/processed/motor_neuron_trajectories.pkl \\")
    print("       --model transformer --epochs 100")

    print("\n3. Run complete hybrid demo:")
    print("   python examples/hybrid_ml_demo.py")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING]  Interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
