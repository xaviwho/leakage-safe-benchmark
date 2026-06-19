"""
Universal Stem Cell Differentiation Data Loader.

This script works with ANY stem cell differentiation dataset:
- Motor neurons, cardiomyocytes, neurons, hepatocytes, etc.
- Automatically adapts to different cell types
- Flexible marker detection
- Works across all differentiation trajectories

Usage:
    python scripts/universal_stem_cell_loader.py --search "YOUR_CELL_TYPE"
    python scripts/universal_stem_cell_loader.py --auto
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
from src.data.cellxgene_api import CellxGeneAPI, display_search_results
from src.data.cellxgene_loader import StemCellDataLoader
from src.utils import setup_logger

logger = setup_logger("universal_loader", level="INFO")


# Universal marker gene database for different cell types
CELL_TYPE_MARKERS = {
    'motor_neuron': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG', 'LIN28A'],
        'differentiation': ['ISL1', 'ISL2', 'MNX1', 'CHAT', 'OLIG2', 'PAX6']
    },
    'cardiomyocyte': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG', 'LIN28A'],
        'differentiation': ['TNNT2', 'MYH6', 'NKX2-5', 'GATA4', 'T', 'MIXL1']
    },
    'neuron': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['MAP2', 'TUBB3', 'SYP', 'NEUROD1', 'PAX6', 'SOX1']
    },
    'hepatocyte': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['ALB', 'AFP', 'HNF4A', 'FOXA2', 'SOX17']
    },
    'endothelial': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['PECAM1', 'CDH5', 'KDR', 'FLT1', 'MIXL1']
    },
    'hematopoietic': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['CD34', 'CD45', 'RUNX1', 'TAL1', 'FLI1']
    },
    'pancreatic': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['INS', 'GCG', 'PDX1', 'NKX6-1', 'FOXA2']
    },
    'astrocyte': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'SOX9']
    },
    'oligodendrocyte': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG'],
        'differentiation': ['MBP', 'MOG', 'OLIG1', 'OLIG2', 'SOX10']
    },
    'generic': {
        'pluripotency': ['POU5F1', 'SOX2', 'NANOG', 'LIN28A', 'KLF4', 'MYC'],
        'differentiation': []  # Will be detected from data
    }
}


SEARCH_KEYWORDS = {
    'motor_neuron': ['motor neuron', 'spinal cord', 'MNX1', 'ISL1'],
    'cardiomyocyte': ['cardiac', 'cardiomyocyte', 'heart', 'TNNT2'],
    'neuron': ['neural', 'neuron', 'cortical', 'brain'],
    'hepatocyte': ['hepatocyte', 'liver', 'hepatic'],
    'endothelial': ['endothelial', 'vascular', 'blood vessel'],
    'hematopoietic': ['hematopoietic', 'blood', 'HSC'],
    'pancreatic': ['pancreatic', 'beta cell', 'islet'],
    'astrocyte': ['astrocyte', 'glial'],
    'oligodendrocyte': ['oligodendrocyte', 'myelin'],
}


def get_cell_type_from_user():
    """Interactive cell type selection."""
    print("\n" + "="*80)
    print("SELECT CELL TYPE FOR DIFFERENTIATION STUDY")
    print("="*80)

    cell_types = list(CELL_TYPE_MARKERS.keys())

    print("\nAvailable cell types:")
    for i, ct in enumerate(cell_types, 1):
        print(f"  [{i:2d}] {ct.replace('_', ' ').title()}")

    print("\n" + "-"*80)

    try:
        choice = input("\nEnter cell type number (or type name, or 'auto' for automatic): ").strip().lower()

        if choice == 'auto':
            return 'auto'

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(cell_types):
                return cell_types[idx]

        # Try matching by name
        for ct in cell_types:
            if choice in ct or ct in choice:
                return ct

        print(f"Invalid choice. Using 'generic' mode.")
        return 'generic'

    except (ValueError, KeyboardInterrupt):
        print("\nUsing 'generic' mode.")
        return 'generic'


def search_datasets(api: CellxGeneAPI, cell_type: str) -> pd.DataFrame:
    """
    Search for datasets based on cell type.

    Args:
        api: CellxGene API client
        cell_type: Type of cell to search for

    Returns:
        DataFrame with search results
    """
    print(f"\nSearching for {cell_type.replace('_', ' ')} differentiation datasets...")

    if cell_type == 'auto':
        # Broad search for any stem cell differentiation
        keywords = ['stem cell', 'differentiation', 'iPSC', 'development']
    elif cell_type in SEARCH_KEYWORDS:
        keywords = SEARCH_KEYWORDS[cell_type] + ['differentiation', 'iPSC']
    else:
        keywords = [cell_type, 'differentiation', 'stem cell']

    results = api.search_datasets(
        keywords=keywords,
        min_cells=5000
    )

    # Additional filtering for time-series
    if not results.empty:
        # Prefer datasets with "time", "day", "development" in description
        time_keywords = ['time', 'day', 'development', 'stage', 'longitudinal']
        results['has_timeseries'] = results['description'].apply(
            lambda x: any(kw in str(x).lower() for kw in time_keywords)
        )
        # Sort by timeseries presence and cell count
        results = results.sort_values(['has_timeseries', 'cell_count'], ascending=[False, False])

    return results


def auto_detect_cell_type(adata) -> str:
    """
    Automatically detect cell type from dataset.

    Args:
        adata: AnnData object

    Returns:
        Detected cell type
    """
    logger.info("Auto-detecting cell type from dataset...")

    available_genes = set(adata.var_names)

    # Check which cell type markers are most present
    best_match = 'generic'
    best_score = 0

    for cell_type, markers in CELL_TYPE_MARKERS.items():
        if cell_type == 'generic':
            continue

        all_markers = markers['differentiation']
        found = len([g for g in all_markers if g in available_genes])
        score = found / len(all_markers) if all_markers else 0

        if score > best_score:
            best_score = score
            best_match = cell_type

    logger.info(f"Detected cell type: {best_match} (confidence: {best_score:.2%})")

    return best_match


def process_dataset(
    filepath: str,
    cell_type: str = 'auto',
    n_trajectories: int = 200
) -> bool:
    """
    Process any stem cell differentiation dataset.

    Args:
        filepath: Path to h5ad file
        cell_type: Type of cell (auto-detect if 'auto')
        n_trajectories: Number of trajectories to generate

    Returns:
        Success status
    """
    logger.info("="*80)
    logger.info(f"PROCESSING STEM CELL DIFFERENTIATION DATASET")
    logger.info("="*80)

    loader = StemCellDataLoader()

    # Load data
    logger.info(f"\n1. Loading dataset: {filepath}")
    try:
        adata = loader.load_h5ad(filepath)
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return False

    logger.info(f"   Cells: {adata.n_obs:,}")
    logger.info(f"   Genes: {adata.n_vars:,}")

    # Auto-detect cell type if needed
    if cell_type == 'auto':
        cell_type = auto_detect_cell_type(adata)

    logger.info(f"\n2. Cell type: {cell_type.replace('_', ' ').title()}")

    # Get markers for this cell type
    if cell_type in CELL_TYPE_MARKERS:
        markers = CELL_TYPE_MARKERS[cell_type]
        pluripotency_genes = markers['pluripotency']
        differentiation_genes = markers['differentiation']
    else:
        # Generic markers
        pluripotency_genes = CELL_TYPE_MARKERS['generic']['pluripotency']
        differentiation_genes = None  # Will use top variable genes

    logger.info(f"   Pluripotency markers: {', '.join(pluripotency_genes[:5])}")
    if differentiation_genes:
        logger.info(f"   Differentiation markers: {', '.join(differentiation_genes[:5])}")

    # Find time column
    logger.info(f"\n3. Detecting temporal information...")
    time_cols = ['day', 'timepoint', 'time', 'days', 'development_stage', 'age']
    time_col = None

    for col in time_cols:
        if col in adata.obs.columns:
            time_col = col
            unique_vals = sorted(adata.obs[col].unique())
            logger.info(f"   [OK] Found: '{col}' with {len(unique_vals)} timepoints")
            logger.info(f"   Values: {unique_vals}")
            break

    if not time_col:
        logger.warning("   [WARNING] No time column found, will use pseudotime")

    # Extract trajectories
    logger.info(f"\n4. Extracting {n_trajectories} differentiation trajectories...")

    try:
        # Use custom markers
        marker_df = loader.extract_stem_cell_markers(
            adata,
            pluripotency_genes=pluripotency_genes,
            differentiation_genes=differentiation_genes
        )

        trajectories = loader.extract_trajectories(
            adata,
            time_column=time_col if time_col else 'dpt_pseudotime',
            n_trajectories=n_trajectories
        )

        logger.info(f"   [OK] Extracted {len(trajectories)} trajectories")

        # Show example
        if trajectories:
            ex = trajectories[0]
            logger.info(f"\n   Example trajectory:")
            logger.info(f"   Start: P={ex[0,0]:.3f}, D={ex[0,1]:.3f}, N={ex[0,2]:.0f}")
            logger.info(f"   End:   P={ex[-1,0]:.3f}, D={ex[-1,1]:.3f}, N={ex[-1,2]:.0f}")

    except Exception as e:
        logger.error(f"   [ERROR] Failed: {e}")
        return False

    # Save
    logger.info(f"\n5. Saving processed data...")
    output_name = f"{cell_type}_trajectories.pkl"
    output_path = f"data/processed/{output_name}"

    try:
        loader.save_processed_data(trajectories, output_path)
        logger.info(f"   [OK] Saved to: {output_path}")
    except Exception as e:
        logger.error(f"   [ERROR] Failed to save: {e}")
        return False

    # Success!
    logger.info("\n" + "="*80)
    logger.info("[SUCCESS] SUCCESS! Data ready for ML training")
    logger.info("="*80)

    logger.info(f"\n[DATA] Dataset: {cell_type.replace('_', ' ').title()}")
    logger.info(f"[FILE] Output: {output_path}")
    logger.info(f"[CHART] Trajectories: {len(trajectories)}")

    logger.info(f"\n[READY] Next: Train models on this data")
    logger.info(f"\n   python experiments/train_predictor.py \\")
    logger.info(f"       --load_data {output_path} \\")
    logger.info(f"       --model lstm --epochs 100")

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Universal stem cell differentiation data loader"
    )

    parser.add_argument(
        '--search',
        type=str,
        help='Cell type to search for (e.g., "motor neuron", "cardiomyocyte")'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Automatically search and download best dataset'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process existing h5ad file'
    )
    parser.add_argument(
        '--cell-type',
        type=str,
        default='auto',
        help='Cell type (auto-detect if not specified)'
    )
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=200,
        help='Number of trajectories (default: 200)'
    )

    args = parser.parse_args()

    # Process existing file
    if args.file:
        success = process_dataset(
            args.file,
            cell_type=args.cell_type,
            n_trajectories=args.n_trajectories
        )
        sys.exit(0 if success else 1)

    # Interactive mode
    api = CellxGeneAPI()

    # Get cell type
    if args.search:
        cell_type = args.search.lower().replace(' ', '_')
    elif args.auto:
        cell_type = 'auto'
    else:
        cell_type = get_cell_type_from_user()

    # Search
    results = search_datasets(api, cell_type)

    if results.empty:
        logger.error("No datasets found")
        sys.exit(1)

    # Display
    display_search_results(results, top_n=5)

    # Download
    print("\n" + "="*80)
    print("SELECT DATASET TO DOWNLOAD")
    print("="*80)

    try:
        choice = input(f"\nEnter dataset number (1-{min(5, len(results))}) or Enter for #1: ").strip()
        idx = int(choice) - 1 if choice else 0

        if idx < 0 or idx >= len(results):
            idx = 0

        selected = results.iloc[idx]

    except (ValueError, KeyboardInterrupt):
        selected = results.iloc[0]

    logger.info(f"\n[SUCCESS] Selected: {selected['collection_name']}")

    # Download
    url = selected['h5ad_url']
    filename = f"{cell_type}_{selected['cell_count']}_cells.h5ad"

    try:
        filepath = api.download_dataset(url, 'data/raw', filename)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # Process
    success = process_dataset(filepath, cell_type, args.n_trajectories)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING]️ Interrupted")
    except Exception as e:
        print(f"\n\n[FAILED] Error: {e}")
        import traceback
        traceback.print_exc()
