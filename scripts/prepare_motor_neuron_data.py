"""
Specialized script for motor neuron differentiation data from CellxGene.

This script is optimized for iPSC → Motor Neuron differentiation datasets.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from src.data.cellxgene_loader import StemCellDataLoader
from src.utils import setup_logger

logger = setup_logger("prepare_motor_neuron", level="INFO")


# Motor neuron differentiation markers
MOTOR_NEURON_MARKERS = {
    'pluripotency': [
        'POU5F1',   # OCT4 - pluripotency marker
        'SOX2',     # Pluripotency marker
        'NANOG',    # Pluripotency marker
        'LIN28A',   # Pluripotency marker
        'DPPA4',    # Pluripotency marker
    ],
    'neural_progenitor': [
        'PAX6',     # Neural progenitor marker
        'SOX1',     # Neural marker
        'NES',      # Nestin - neural stem cell
        'HES5',     # Notch signaling
    ],
    'motor_neuron_progenitor': [
        'OLIG2',    # Motor neuron progenitor
        'NKX6-1',   # Ventral spinal cord
        'DBX1',     # Ventral progenitor
        'NKX2-2',   # Ventral identity
    ],
    'mature_motor_neuron': [
        'ISL1',     # Motor neuron marker
        'ISL2',     # Motor neuron marker
        'MNX1',     # HB9 - motor neuron marker
        'CHAT',     # Choline acetyltransferase
        'SLC18A3',  # Vesicular acetylcholine transporter
        'PRPH',     # Peripherin - motor neuron
        'TUBB3',    # Beta-III tubulin - neuron
        'MAP2',     # Mature neuron marker
    ]
}


def print_motor_neuron_dataset_guide():
    """Print guide for finding motor neuron datasets."""
    print("\n" + "="*80)
    print("FINDING iPSC → MOTOR NEURON DATASETS ON CELLXGENE")
    print("="*80)

    print("\n📍 STEP 1: Go to CellxGene")
    print("   https://cellxgene.cziscience.com/datasets")

    print("\n🔍 STEP 2: Try these searches (in order):")
    print("   1. 'motor neuron differentiation'")
    print("   2. 'spinal cord organoid'")
    print("   3. 'MNX1 OLIG2 ISL1' (motor neuron markers)")
    print("   4. 'ALS iPSC' (disease studies often have controls)")

    print("\n✅ STEP 3: Look for datasets with:")
    print("   • Multiple timepoints (Day 0, 7, 14, 21, 28+)")
    print("   • >10,000 cells")
    print("   • 'differentiation' or 'development' in title")
    print("   • Time-series or longitudinal data")

    print("\n🎯 STEP 4: Good signs in dataset description:")
    print("   • 'Directed differentiation protocol'")
    print("   • 'Developmental time course'")
    print("   • 'Days 0-30' or similar")
    print("   • 'Spinal motor neurons'")

    print("\n📚 STEP 5: Recommended papers to find on CellxGene:")
    print("   • Wichterle lab - motor neuron pioneers")
    print("   • Eggan lab - ALS/motor neuron studies")
    print("   • Novitch lab - spinal cord development")

    print("\n💡 Pro Tips:")
    print("   • Start with the most recent datasets (2021+)")
    print("   • Look for 'protocol paper' datasets (more timepoints)")
    print("   • Check if marker genes are present in dataset")

    print("\n" + "="*80)


def process_motor_neuron_dataset(
    filepath: str,
    time_column: str = 'day',
    n_trajectories: int = 200
):
    """
    Process motor neuron differentiation dataset.

    Args:
        filepath: Path to h5ad file
        time_column: Name of time column
        n_trajectories: Number of trajectories to generate
    """
    logger.info("="*80)
    logger.info("PROCESSING MOTOR NEURON DIFFERENTIATION DATASET")
    logger.info("="*80)

    # Load data
    logger.info(f"\n1. Loading dataset from: {filepath}")
    loader = StemCellDataLoader()

    try:
        adata = loader.load_h5ad(filepath)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

    # Show metadata
    logger.info("\n2. Dataset information:")
    logger.info(f"   Cells: {adata.n_obs:,}")
    logger.info(f"   Genes: {adata.n_vars:,}")
    logger.info(f"   Metadata columns: {list(adata.obs.columns)[:10]}...")

    # Check for time column
    logger.info("\n3. Checking for time information...")
    potential_time_cols = ['day', 'timepoint', 'time', 'days', 'development_stage', 'age']

    time_col_found = None
    for col in potential_time_cols:
        if col in adata.obs.columns:
            logger.info(f"   ✓ Found time column: '{col}'")
            unique_vals = adata.obs[col].unique()
            logger.info(f"   Timepoints: {sorted(unique_vals)[:10]}")
            time_col_found = col
            break

    if time_col_found is None:
        logger.warning("   ⚠ No time column found, will use pseudotime")
        time_col_found = time_column

    # Check for motor neuron markers
    logger.info("\n4. Checking for motor neuron marker genes...")

    all_markers = []
    for stage, genes in MOTOR_NEURON_MARKERS.items():
        all_markers.extend(genes)

    found_markers = [g for g in all_markers if g in adata.var_names]
    logger.info(f"   Found {len(found_markers)}/{len(all_markers)} marker genes")

    # Show which stages have markers
    for stage, genes in MOTOR_NEURON_MARKERS.items():
        found = [g for g in genes if g in adata.var_names]
        logger.info(f"   {stage}: {len(found)}/{len(genes)} markers")
        if found:
            logger.info(f"      Found: {', '.join(found[:5])}")

    # Extract trajectories with motor neuron-specific markers
    logger.info(f"\n5. Extracting {n_trajectories} motor neuron trajectories...")

    # Combine all differentiation markers (not pluripotency)
    differentiation_markers = (
        MOTOR_NEURON_MARKERS['neural_progenitor'] +
        MOTOR_NEURON_MARKERS['motor_neuron_progenitor'] +
        MOTOR_NEURON_MARKERS['mature_motor_neuron']
    )

    try:
        trajectories = loader.extract_trajectories(
            adata,
            time_column=time_col_found,
            n_trajectories=n_trajectories
        )

        logger.info(f"   ✓ Extracted {len(trajectories)} trajectories")

        # Show statistics
        lengths = [len(t) for t in trajectories]
        logger.info(f"   Trajectory lengths: {min(lengths)}-{max(lengths)} (mean: {np.mean(lengths):.1f})")

        # Show example trajectory
        if len(trajectories) > 0:
            example = trajectories[0]
            logger.info(f"\n   Example trajectory shape: {example.shape}")
            logger.info(f"   Initial state: P={example[0,0]:.3f}, D={example[0,1]:.3f}, N={example[0,2]:.0f}")
            logger.info(f"   Final state:   P={example[-1,0]:.3f}, D={example[-1,1]:.3f}, N={example[-1,2]:.0f}")

    except Exception as e:
        logger.error(f"   ✗ Failed to extract trajectories: {e}")
        return False

    # Save processed data
    logger.info("\n6. Saving processed motor neuron data...")
    output_path = 'data/processed/motor_neuron_trajectories.pkl'

    try:
        loader.save_processed_data(trajectories, output_path)
        logger.info(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        logger.error(f"   ✗ Failed to save: {e}")
        return False

    # Success summary
    logger.info("\n" + "="*80)
    logger.info("✓ SUCCESS! Motor neuron data ready for training")
    logger.info("="*80)
    logger.info(f"\nProcessed {len(trajectories)} motor neuron differentiation trajectories")
    logger.info(f"Data saved to: {output_path}")

    logger.info("\n📊 Next steps:")
    logger.info("\n1. Train LSTM on motor neuron data:")
    logger.info(f"   python experiments/train_predictor.py \\")
    logger.info(f"       --load_data {output_path} \\")
    logger.info(f"       --model lstm \\")
    logger.info(f"       --epochs 100")

    logger.info("\n2. Compare with Transformer:")
    logger.info(f"   python experiments/train_predictor.py \\")
    logger.info(f"       --load_data {output_path} \\")
    logger.info(f"       --model transformer \\")
    logger.info(f"       --epochs 100")

    logger.info("\n3. Run hybrid demo with trained model:")
    logger.info("   python examples/hybrid_ml_demo.py")

    logger.info("\n" + "="*80)

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process motor neuron differentiation data from CellxGene"
    )

    parser.add_argument(
        '--guide',
        action='store_true',
        help='Print guide for finding motor neuron datasets'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to downloaded h5ad file'
    )
    parser.add_argument(
        '--time-column',
        type=str,
        default='day',
        help='Name of time column (default: day)'
    )
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=200,
        help='Number of trajectories to generate (default: 200)'
    )

    args = parser.parse_args()

    # Show guide
    if args.guide or not args.file:
        print_motor_neuron_dataset_guide()
        if not args.file:
            return

    # Process file
    if args.file:
        success = process_motor_neuron_dataset(
            args.file,
            time_column=args.time_column,
            n_trajectories=args.n_trajectories
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
