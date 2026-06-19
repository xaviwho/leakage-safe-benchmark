"""
CellxGene Dataset Loader for Real scRNA-seq Data.

This module provides tools to:
1. Search and download datasets from CellxGene
2. Load and preprocess h5ad files
3. Extract differentiation trajectories
4. Convert to format compatible with ML models
"""

import requests
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class CellxGeneDatasetFinder:
    """
    Search and find relevant stem cell datasets on CellxGene.
    """

    def __init__(self):
        """Initialize CellxGene dataset finder."""
        self.api_base = "https://api.cellxgene.cziscience.com"
        self.datasets_endpoint = f"{self.api_base}/curation/v1/collections"

    def search_stem_cell_datasets(
        self,
        keywords: Optional[List[str]] = None,
        cell_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for stem cell differentiation datasets.

        Args:
            keywords: Keywords to search for (e.g., ['iPSC', 'differentiation', 'pluripotent'])
            cell_types: Cell types to filter (e.g., ['stem cell', 'iPSC'])

        Returns:
            List of matching datasets with metadata
        """
        if keywords is None:
            keywords = [
                'stem cell', 'pluripotent', 'iPSC', 'ESC',
                'differentiation', 'development', 'lineage'
            ]

        logger.info("Searching CellxGene for stem cell datasets...")
        logger.info(f"Keywords: {keywords}")

        # Note: CellxGene API structure - adjust based on actual API
        # This is a template that needs to be adapted to the actual API
        results = []

        try:
            response = requests.get(self.datasets_endpoint, timeout=30)
            response.raise_for_status()
            collections = response.json()

            # Filter collections based on keywords
            for collection in collections:
                name = collection.get('name', '').lower()
                description = collection.get('description', '').lower()
                text = name + ' ' + description

                # Check if any keyword matches
                if any(keyword.lower() in text for keyword in keywords):
                    results.append({
                        'id': collection.get('id'),
                        'name': collection.get('name'),
                        'description': collection.get('description'),
                        'datasets': collection.get('datasets', []),
                        'curator_name': collection.get('curator_name'),
                        'doi': collection.get('doi')
                    })

            logger.info(f"Found {len(results)} matching collections")

        except Exception as e:
            logger.error(f"Error searching CellxGene: {e}")
            logger.info("Tip: Visit https://cellxgene.cziscience.com/datasets to browse manually")

        return results

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_id: CellxGene dataset ID

        Returns:
            Dataset metadata
        """
        try:
            url = f"{self.api_base}/curation/v1/datasets/{dataset_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {}

    def print_search_results(self, results: List[Dict]):
        """Print search results in a readable format."""
        if not results:
            print("\nNo datasets found.")
            print("Visit https://cellxgene.cziscience.com/datasets to browse manually.")
            return

        print(f"\n{'='*80}")
        print(f"Found {len(results)} Stem Cell Dataset Collections")
        print(f"{'='*80}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['name']}")
            print(f"   ID: {result['id']}")
            if result.get('description'):
                desc = result['description'][:150]
                print(f"   Description: {desc}{'...' if len(result['description']) > 150 else ''}")
            print(f"   Number of datasets: {len(result.get('datasets', []))}")
            if result.get('doi'):
                print(f"   DOI: {result['doi']}")
            print()


class StemCellDataLoader:
    """
    Load and preprocess stem cell scRNA-seq data for digital twin.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_from_cellxgene(
        self,
        dataset_url: str,
        filename: Optional[str] = None
    ) -> Path:
        """
        Download dataset from CellxGene.

        Args:
            dataset_url: Direct download URL from CellxGene
            filename: Local filename (if None, extracts from URL)

        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = dataset_url.split('/')[-1]
            if not filename.endswith('.h5ad'):
                filename = filename + '.h5ad'

        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath

        logger.info(f"Downloading from CellxGene: {dataset_url}")

        try:
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Downloaded to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            raise

    def load_h5ad(self, filepath: str) -> sc.AnnData:
        """
        Load h5ad file using Scanpy.

        Args:
            filepath: Path to h5ad file

        Returns:
            AnnData object
        """
        logger.info(f"Loading h5ad file: {filepath}")
        adata = sc.read_h5ad(filepath)

        logger.info(f"Loaded dataset:")
        logger.info(f"  Cells: {adata.n_obs:,}")
        logger.info(f"  Genes: {adata.n_vars:,}")
        logger.info(f"  Observations: {list(adata.obs.columns)}")
        logger.info(f"  Variables: {list(adata.var.columns)}")

        return adata

    def extract_stem_cell_markers(
        self,
        adata: sc.AnnData,
        pluripotency_genes: Optional[List[str]] = None,
        differentiation_genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract pluripotency and differentiation marker expression.

        Args:
            adata: AnnData object
            pluripotency_genes: List of pluripotency marker genes
            differentiation_genes: List of differentiation marker genes

        Returns:
            DataFrame with marker scores per cell
        """
        if pluripotency_genes is None:
            # Default pluripotency markers
            pluripotency_genes = [
                'POU5F1', 'SOX2', 'NANOG', 'KLF4', 'MYC',  # Yamanaka factors
                'DNMT3B', 'TDGF1', 'LIN28A', 'UTF1', 'DPPA4'
            ]

        if differentiation_genes is None:
            # Default differentiation markers (broad)
            differentiation_genes = [
                'T', 'MIXL1', 'EOMES',  # Mesoderm
                'SOX1', 'PAX6', 'NES',  # Ectoderm
                'SOX17', 'FOXA2', 'GATA6'  # Endoderm
            ]

        logger.info("Computing marker gene scores...")

        # Find available genes
        available_pluri = [g for g in pluripotency_genes if g in adata.var_names]
        available_diff = [g for g in differentiation_genes if g in adata.var_names]

        logger.info(f"  Pluripotency markers found: {len(available_pluri)}/{len(pluripotency_genes)}")
        logger.info(f"  Differentiation markers found: {len(available_diff)}/{len(differentiation_genes)}")

        if len(available_pluri) == 0:
            logger.warning("No pluripotency markers found! Using PCA components as proxy.")
            # Fallback: use PCA
            if 'X_pca' not in adata.obsm:
                sc.pp.pca(adata)
            pluripotency_score = adata.obsm['X_pca'][:, 0]
        else:
            # Compute pluripotency score (mean expression)
            pluri_expr = adata[:, available_pluri].X
            if hasattr(pluri_expr, 'toarray'):
                pluri_expr = pluri_expr.toarray()
            pluripotency_score = np.mean(pluri_expr, axis=1)

        if len(available_diff) == 0:
            logger.warning("No differentiation markers found! Using inverse of pluripotency as proxy.")
            differentiation_score = 1 - pluripotency_score
        else:
            # Compute differentiation score
            diff_expr = adata[:, available_diff].X
            if hasattr(diff_expr, 'toarray'):
                diff_expr = diff_expr.toarray()
            differentiation_score = np.mean(diff_expr, axis=1)

        # Normalize scores to [0, 1]
        pluripotency_score = (pluripotency_score - pluripotency_score.min()) / \
                            (pluripotency_score.max() - pluripotency_score.min() + 1e-10)
        differentiation_score = (differentiation_score - differentiation_score.min()) / \
                               (differentiation_score.max() - differentiation_score.min() + 1e-10)

        # Create DataFrame
        marker_df = pd.DataFrame({
            'pluripotency': pluripotency_score,
            'differentiation': differentiation_score,
            'cell_id': adata.obs_names
        })

        # Add metadata if available
        for col in ['timepoint', 'day', 'time', 'development_stage', 'cell_type']:
            if col in adata.obs.columns:
                marker_df[col] = adata.obs[col].values

        return marker_df

    def extract_trajectories(
        self,
        adata: sc.AnnData,
        time_column: str = 'timepoint',
        group_column: Optional[str] = None,
        n_trajectories: int = 100
    ) -> List[np.ndarray]:
        """
        Extract cell state trajectories over time.

        Args:
            adata: AnnData object
            time_column: Column with time information
            group_column: Column for grouping (e.g., cell line, condition)
            n_trajectories: Number of pseudobulk trajectories to generate

        Returns:
            List of trajectory arrays (each of shape [timepoints, 3])
        """
        logger.info("Extracting trajectories...")

        # Get marker scores
        marker_df = self.extract_stem_cell_markers(adata)

        # Check if time information exists
        if time_column not in adata.obs.columns:
            logger.warning(f"Time column '{time_column}' not found. Using pseudotime or creating synthetic time.")
            # Try to use pseudotime if available
            if 'dpt_pseudotime' in adata.obs.columns:
                time_values = adata.obs['dpt_pseudotime'].values
            else:
                logger.info("Computing diffusion pseudotime...")
                sc.pp.neighbors(adata)
                sc.tl.diffmap(adata)
                sc.tl.dpt(adata)
                time_values = adata.obs['dpt_pseudotime'].values
        else:
            time_values = adata.obs[time_column].values

        marker_df['time'] = time_values

        # Group if requested
        if group_column and group_column in adata.obs.columns:
            marker_df['group'] = adata.obs[group_column].values
            groups = marker_df['group'].unique()
        else:
            marker_df['group'] = 'all'
            groups = ['all']

        logger.info(f"Generating {n_trajectories} trajectories from {len(groups)} groups...")

        trajectories = []

        for _ in tqdm(range(n_trajectories), desc="Generating trajectories"):
            # Sample a group
            group = np.random.choice(groups)
            group_data = marker_df[marker_df['group'] == group]

            # Sample cells at different timepoints
            time_points = np.linspace(group_data['time'].min(), group_data['time'].max(), 50)

            trajectory = []
            for t in time_points:
                # Find cells near this timepoint
                time_window = (group_data['time'] >= t - 0.1) & (group_data['time'] <= t + 0.1)
                cells_at_t = group_data[time_window]

                if len(cells_at_t) > 0:
                    # Average state at this timepoint
                    P = cells_at_t['pluripotency'].mean()
                    D = cells_at_t['differentiation'].mean()
                    N = len(cells_at_t) * 100  # Proxy for population

                    trajectory.append([P, D, N])

            if len(trajectory) > 10:  # Only keep if we have enough points
                trajectories.append(np.array(trajectory))

        logger.info(f"Extracted {len(trajectories)} trajectories")

        return trajectories

    def save_processed_data(
        self,
        trajectories: List[np.ndarray],
        output_path: str
    ):
        """
        Save processed trajectories for ML training.

        Args:
            trajectories: List of trajectory arrays
            output_path: Path to save file
        """
        import pickle

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(trajectories, f)

        logger.info(f"Saved {len(trajectories)} trajectories to {output_path}")


def print_cellxgene_instructions():
    """Print instructions for finding datasets on CellxGene."""
    print("\n" + "="*80)
    print("HOW TO FIND STEM CELL DATASETS ON CELLXGENE")
    print("="*80)
    print("\n1. Visit: https://cellxgene.cziscience.com/datasets")
    print("\n2. Search for relevant datasets using keywords:")
    print("   - 'stem cell'")
    print("   - 'pluripotent'")
    print("   - 'iPSC' or 'ESC'")
    print("   - 'differentiation'")
    print("   - Specific cell types (e.g., 'cardiomyocyte', 'neuron')")
    print("\n3. Look for datasets with:")
    print("   ✓ Time-series or developmental stages")
    print("   ✓ Multiple timepoints (Day 0, Day 3, Day 7, etc.)")
    print("   ✓ Clear differentiation progression")
    print("\n4. Click on a dataset and find the 'Download' button")
    print("   - Copy the direct download URL (ends with .h5ad)")
    print("\n5. Use the URL with our data loader:")
    print("   ```python")
    print("   loader = StemCellDataLoader()")
    print("   filepath = loader.download_from_cellxgene('YOUR_URL_HERE')")
    print("   ```")
    print("\n" + "="*80)
    print("\nRECOMMENDED DATASETS TO LOOK FOR:")
    print("="*80)
    print("- iPSC to cardiomyocyte differentiation")
    print("- iPSC to neural/neuron differentiation")
    print("- ESC developmental trajectories")
    print("- Organoid development time-series")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print instructions
    print_cellxgene_instructions()

    # Example usage
    print("\nEXAMPLE USAGE:")
    print("="*80)
    print("""
# 1. Search for datasets
finder = CellxGeneDatasetFinder()
results = finder.search_stem_cell_datasets(keywords=['iPSC', 'differentiation'])
finder.print_search_results(results)

# 2. Download a dataset
loader = StemCellDataLoader(data_dir='data/raw')
filepath = loader.download_from_cellxgene('DIRECT_DOWNLOAD_URL_FROM_CELLXGENE')

# 3. Load and process
adata = loader.load_h5ad(filepath)
trajectories = loader.extract_trajectories(adata, time_column='timepoint')
loader.save_processed_data(trajectories, 'data/processed/real_trajectories.pkl')

# 4. Train models on real data
# Use with experiments/train_predictor.py --load_data data/processed/real_trajectories.pkl
    """)
    print("="*80)
