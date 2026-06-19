"""
Data loader for dopaminergic neuron differentiation dataset.

Loads and preprocesses the Jerber et al. (2021) dataset for digital twin training.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DopaminergicDataLoader:
    """
    Load and preprocess dopaminergic neuron differentiation data.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing downloaded data
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.timepoints = ["day0", "day11", "day30"]
        self.adata_dict = {}
        self.combined_adata = None

    def load_timepoint(self, timepoint: str) -> anndata.AnnData:
        """
        Load data for a single timepoint.

        Args:
            timepoint: Timepoint name (e.g., "day0", "day11", "day30")

        Returns:
            AnnData object
        """
        file_path = self.data_dir / f"{timepoint}_counts.h5ad"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please run: python src/data/download_data.py"
            )

        logger.info(f"Loading {timepoint} data from {file_path}")
        adata = sc.read_h5ad(file_path)

        # Add timepoint information
        adata.obs['timepoint'] = timepoint
        adata.obs['timepoint_numeric'] = int(timepoint.replace('day', ''))

        logger.info(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

        return adata

    def load_all_timepoints(self) -> Dict[str, anndata.AnnData]:
        """
        Load all timepoints.

        Returns:
            Dictionary mapping timepoint names to AnnData objects
        """
        logger.info("Loading all timepoints...")

        for tp in self.timepoints:
            try:
                self.adata_dict[tp] = self.load_timepoint(tp)
            except FileNotFoundError as e:
                logger.warning(f"Could not load {tp}: {e}")

        if not self.adata_dict:
            raise ValueError("No data files found. Please download data first.")

        logger.info(f"Loaded {len(self.adata_dict)} timepoints")
        return self.adata_dict

    def combine_timepoints(self) -> anndata.AnnData:
        """
        Combine all timepoints into single AnnData object.

        Returns:
            Combined AnnData object
        """
        if not self.adata_dict:
            self.load_all_timepoints()

        logger.info("Combining timepoints...")

        # Concatenate
        self.combined_adata = anndata.concat(
            self.adata_dict.values(),
            join='outer',
            label='timepoint',
            keys=self.adata_dict.keys()
        )

        logger.info(f"Combined data: {self.combined_adata.n_obs} cells, "
                   f"{self.combined_adata.n_vars} genes")

        return self.combined_adata

    def preprocess(
        self,
        min_genes: int = 200,
        min_cells: int = 3,
        n_top_genes: int = 2000,
        normalize: bool = True
    ) -> anndata.AnnData:
        """
        Preprocess the combined dataset.

        Args:
            min_genes: Minimum genes per cell
            min_cells: Minimum cells per gene
            n_top_genes: Number of highly variable genes to keep
            normalize: Whether to normalize and log-transform

        Returns:
            Preprocessed AnnData object
        """
        if self.combined_adata is None:
            self.combine_timepoints()

        adata = self.combined_adata.copy()

        logger.info("Preprocessing data...")

        # Quality control
        logger.info(f"Filtering cells with < {min_genes} genes")
        sc.pp.filter_cells(adata, min_genes=min_genes)

        logger.info(f"Filtering genes expressed in < {min_cells} cells")
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # Normalize and log-transform
        if normalize:
            logger.info("Normalizing and log-transforming")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        # Highly variable genes
        logger.info(f"Selecting {n_top_genes} highly variable genes")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        # Calculate QC metrics
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1

        logger.info(f"Preprocessed data: {adata.n_obs} cells, {adata.n_vars} genes")

        return adata

    def calculate_scores(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Calculate pluripotency and differentiation scores.

        Args:
            adata: AnnData object

        Returns:
            AnnData with scores added to .obs
        """
        logger.info("Calculating cell state scores...")

        # Pluripotency markers (iPSC markers)
        pluripotency_genes = ['POU5F1', 'NANOG', 'SOX2', 'UTF1', 'TDGF1']

        # Dopaminergic neuron markers
        differentiation_genes = ['TH', 'DDC', 'SLC6A3', 'DRD2', 'LMX1A', 'FOXA2']

        # Neural markers
        neural_genes = ['MAP2', 'TUBB3', 'DCX', 'NCAM1']

        # Check which genes are available
        available_pluri = [g for g in pluripotency_genes if g in adata.var_names]
        available_diff = [g for g in differentiation_genes if g in adata.var_names]
        available_neural = [g for g in neural_genes if g in adata.var_names]

        logger.info(f"  Pluripotency genes found: {len(available_pluri)}/{len(pluripotency_genes)}")
        logger.info(f"  Differentiation genes found: {len(available_diff)}/{len(differentiation_genes)}")
        logger.info(f"  Neural genes found: {len(available_neural)}/{len(neural_genes)}")

        # Calculate scores using scanpy's score_genes
        if available_pluri:
            sc.tl.score_genes(adata, available_pluri, score_name='pluripotency_score')

        if available_diff:
            sc.tl.score_genes(adata, available_diff, score_name='differentiation_score')

        if available_neural:
            sc.tl.score_genes(adata, available_neural, score_name='neural_score')

        return adata

    def save_processed(self, adata: anndata.AnnData, filename: str = "dopaminergic_processed.h5ad"):
        """
        Save processed data.

        Args:
            adata: Processed AnnData object
            filename: Output filename
        """
        output_path = self.processed_dir / filename
        logger.info(f"Saving processed data to {output_path}")
        adata.write_h5ad(output_path)
        logger.info("Data saved successfully")

    def load_processed(self, filename: str = "dopaminergic_processed.h5ad") -> anndata.AnnData:
        """
        Load previously processed data.

        Args:
            filename: Processed data filename

        Returns:
            Processed AnnData object
        """
        input_path = self.processed_dir / filename

        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")

        logger.info(f"Loading processed data from {input_path}")
        adata = sc.read_h5ad(input_path)
        logger.info(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

        return adata

    def get_trajectory_data(
        self,
        adata: anndata.AnnData,
        n_cells_per_timepoint: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trajectory data for digital twin training.

        Args:
            adata: AnnData object
            n_cells_per_timepoint: Subsample to this many cells per timepoint (optional)

        Returns:
            Tuple of (timepoints, features) arrays for training
        """
        logger.info("Extracting trajectory data...")

        if n_cells_per_timepoint:
            logger.info(f"Subsampling to {n_cells_per_timepoint} cells per timepoint")
            adata = adata.copy()
            sc.pp.subsample(adata, n_obs=n_cells_per_timepoint, by='timepoint')

        # Sort by timepoint
        adata = adata[adata.obs.sort_values('timepoint_numeric').index]

        # Extract features
        timepoints = adata.obs['timepoint_numeric'].values

        # Use scores if available, otherwise use gene expression
        if 'pluripotency_score' in adata.obs.columns:
            features = adata.obs[['pluripotency_score', 'differentiation_score']].values
        else:
            # Use PCA
            sc.tl.pca(adata, n_comps=50)
            features = adata.obsm['X_pca']

        logger.info(f"Trajectory data: {len(timepoints)} cells, {features.shape[1]} features")

        return timepoints, features


def main():
    """Main data loading pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and preprocess dopaminergic data")
    parser.add_argument("--data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--skip-download-check", action="store_true",
                       help="Skip checking if data is downloaded")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Initialize loader
        loader = DopaminergicDataLoader(data_dir=args.data_dir)

        # Load and process
        loader.load_all_timepoints()
        adata = loader.combine_timepoints()
        adata = loader.preprocess()
        adata = loader.calculate_scores(adata)

        # Save
        loader.save_processed(adata)

        logger.info("\n" + "=" * 60)
        logger.info("DATA PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed data saved to: data/processed/")
        logger.info(f"Ready for digital twin training!")

    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        logger.info("\nPlease download data first:")
        logger.info("  python src/data/download_data.py")


if __name__ == "__main__":
    main()
