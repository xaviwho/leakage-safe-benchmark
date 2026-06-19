"""
Helper script to download and prepare CellxGene datasets for training.

Usage:
    python scripts/prepare_cellxgene_data.py --url <CELLXGENE_DOWNLOAD_URL>
    python scripts/prepare_cellxgene_data.py --search
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.data.cellxgene_loader import (
    CellxGeneDatasetFinder,
    StemCellDataLoader,
    print_cellxgene_instructions
)
from src.utils import setup_logger

logger = setup_logger("prepare_cellxgene_data", level="INFO")


def search_datasets(keywords):
    """Search for stem cell datasets."""
    logger.info("Searching CellxGene for stem cell datasets...")

    finder = CellxGeneDatasetFinder()
    results = finder.search_stem_cell_datasets(keywords=keywords)
    finder.print_search_results(results)

    if not results:
        print("\nTip: Try different keywords or browse manually at:")
        print("https://cellxgene.cziscience.com/datasets")


def download_and_process(
    url: str,
    time_column: str = 'timepoint',
    n_trajectories: int = 200
):
    """Download and process a dataset from CellxGene."""
    logger.info("="*80)
    logger.info("DOWNLOADING AND PROCESSING CELLXGENE DATASET")
    logger.info("="*80)

    # Create loader
    loader = StemCellDataLoader(data_dir='data/raw')

    # Download
    logger.info("\nStep 1: Downloading dataset...")
    try:
        filepath = loader.download_from_cellxgene(url)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("\nMake sure you have the direct download URL from CellxGene.")
        logger.info("It should end with .h5ad")
        return False

    # Load
    logger.info("\nStep 2: Loading h5ad file...")
    try:
        adata = loader.load_h5ad(filepath)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False

    # Show available metadata
    logger.info("\nStep 3: Analyzing dataset metadata...")
    logger.info(f"Available observation columns: {list(adata.obs.columns)}")

    # Try to identify time column
    potential_time_cols = ['timepoint', 'day', 'time', 'development_stage', 'age']
    time_col_found = None

    for col in potential_time_cols:
        if col in adata.obs.columns:
            time_col_found = col
            logger.info(f"Found potential time column: '{col}'")
            logger.info(f"Unique values: {adata.obs[col].unique()[:10]}")
            break

    if time_col_found is None:
        logger.warning("No time column found. Will use pseudotime.")
        time_col_found = 'dpt_pseudotime'  # Will be computed

    # Extract trajectories
    logger.info(f"\nStep 4: Extracting {n_trajectories} trajectories...")
    try:
        trajectories = loader.extract_trajectories(
            adata,
            time_column=time_col_found,
            n_trajectories=n_trajectories
        )

        if len(trajectories) == 0:
            logger.error("No trajectories extracted!")
            return False

        logger.info(f"Successfully extracted {len(trajectories)} trajectories")

        # Show trajectory statistics
        lengths = [len(t) for t in trajectories]
        logger.info(f"Trajectory lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

    except Exception as e:
        logger.error(f"Failed to extract trajectories: {e}")
        return False

    # Save processed data
    logger.info("\nStep 5: Saving processed data...")
    output_path = 'data/processed/cellxgene_trajectories.pkl'

    try:
        loader.save_processed_data(trajectories, output_path)
        logger.info(f"Saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        return False

    # Success!
    logger.info("\n" + "="*80)
    logger.info("SUCCESS! Data ready for training")
    logger.info("="*80)
    logger.info(f"\nProcessed {len(trajectories)} trajectories from real scRNA-seq data")
    logger.info("\nNext steps:")
    logger.info("1. Train LSTM model on real data:")
    logger.info(f"   python experiments/train_predictor.py --load_data {output_path}")
    logger.info("\n2. Or use in your own script:")
    logger.info("   ```python")
    logger.info("   from src.data.data_generator import load_dataset")
    logger.info(f"   trajectories = load_dataset('{output_path}')")
    logger.info("   ```")
    logger.info("="*80)

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and prepare CellxGene datasets for ML training"
    )

    parser.add_argument(
        '--search',
        action='store_true',
        help='Search for stem cell datasets on CellxGene'
    )
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=['stem cell', 'iPSC', 'differentiation'],
        help='Keywords to search for (default: stem cell, iPSC, differentiation)'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Direct download URL from CellxGene (must end with .h5ad)'
    )
    parser.add_argument(
        '--time-column',
        type=str,
        default='timepoint',
        help='Name of time/stage column in dataset (default: timepoint)'
    )
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=200,
        help='Number of trajectories to generate (default: 200)'
    )
    parser.add_argument(
        '--instructions',
        action='store_true',
        help='Print instructions for using CellxGene'
    )

    args = parser.parse_args()

    # Print instructions
    if args.instructions or (not args.search and not args.url):
        print_cellxgene_instructions()
        return

    # Search
    if args.search:
        search_datasets(args.keywords)
        return

    # Download and process
    if args.url:
        success = download_and_process(
            args.url,
            time_column=args.time_column,
            n_trajectories=args.n_trajectories
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
