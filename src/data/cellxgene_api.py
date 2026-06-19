"""
CellxGene API Client for programmatic dataset access.

Uses the official CellxGene API to search, filter, and download datasets.
"""

import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class CellxGeneAPI:
    """
    Official CellxGene API client for dataset discovery and download.

    API Documentation: https://api.cellxgene.cziscience.com/curation/v1/
    """

    def __init__(self):
        """Initialize CellxGene API client."""
        self.base_url = "https://api.cellxgene.cziscience.com/curation/v1"
        self.collections_url = f"{self.base_url}/collections"

    def get_all_collections(self) -> List[Dict]:
        """
        Get all available collections from CellxGene.

        Returns:
            List of collection metadata dictionaries
        """
        logger.info("Fetching all collections from CellxGene API...")

        try:
            response = requests.get(self.collections_url, timeout=30)
            response.raise_for_status()
            collections = response.json()

            logger.info(f"Found {len(collections)} total collections")
            return collections

        except Exception as e:
            logger.error(f"Failed to fetch collections: {e}")
            return []

    def search_motor_neuron_datasets(self) -> pd.DataFrame:
        """
        Search for motor neuron differentiation datasets.

        Returns:
            DataFrame with relevant datasets
        """
        logger.info("Searching for motor neuron datasets...")

        # Keywords for motor neuron differentiation
        keywords = [
            'motor neuron', 'spinal cord', 'mnx1', 'isl1', 'hb9',
            'olig2', 'motor', 'neuron', 'differentiation', 'ipsc',
            'pluripotent', 'stem cell'
        ]

        # Use the updated search_datasets method
        return self.search_datasets(keywords=keywords, min_cells=1000)

    def search_datasets(
        self,
        keywords: List[str],
        tissue: Optional[str] = None,
        cell_type: Optional[str] = None,
        disease: Optional[str] = None,
        organism: str = "Homo sapiens",
        min_cells: int = 1000
    ) -> pd.DataFrame:
        """
        Search for datasets with custom filters.

        Args:
            keywords: List of keywords to search for
            tissue: Tissue type filter
            cell_type: Cell type filter
            disease: Disease filter (use 'normal' for healthy)
            organism: Organism (default: human)
            min_cells: Minimum cell count

        Returns:
            DataFrame with matching datasets
        """
        logger.info(f"Searching with keywords: {keywords}")

        # Get basic collection list
        collections = self.get_all_collections()

        if not collections:
            return pd.DataFrame()

        matching_datasets = []

        # Filter collections by keywords first
        relevant_collections = []
        for collection in collections:
            name = collection.get('name', '').lower()
            description = str(collection.get('description', '')).lower()
            text = f"{name} {description}"

            # Check keywords
            if any(kw.lower() in text for kw in keywords):
                relevant_collections.append(collection)

        logger.info(f"Found {len(relevant_collections)} collections matching keywords")

        # Fetch detailed info for each relevant collection
        for i, coll in enumerate(relevant_collections):
            collection_id = coll.get('collection_id')

            try:
                # Fetch full collection details (includes cell_count and assets)
                url = f"{self.base_url}/collections/{collection_id}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                full_collection = response.json()

                # Extract datasets with full details
                for dataset in full_collection.get('datasets', []):
                    cell_count = dataset.get('cell_count', 0)

                    # Filter by cell count
                    if cell_count < min_cells:
                        continue

                    # Get download URL
                    h5ad_url = None
                    for asset in dataset.get('assets', []):
                        if asset.get('filetype') == 'H5AD':
                            h5ad_url = asset.get('url')
                            break

                    if not h5ad_url:
                        continue

                    matching_datasets.append({
                        'collection_id': full_collection.get('collection_id'),
                        'collection_name': full_collection.get('name'),
                        'dataset_id': dataset.get('dataset_id'),
                        'dataset_name': dataset.get('title', dataset.get('name', '')),
                        'cell_count': cell_count,
                        'h5ad_url': h5ad_url,
                        'description': full_collection.get('description', '')[:200],
                        'doi': full_collection.get('doi', ''),
                    })

            except Exception as e:
                logger.debug(f"Error fetching collection {collection_id}: {e}")
                continue

        df = pd.DataFrame(matching_datasets)

        if not df.empty:
            df = df.sort_values('cell_count', ascending=False)
            logger.info(f"Found {len(df)} matching datasets")
        else:
            logger.warning("No matching datasets found")

        return df

    def download_dataset(
        self,
        url: str,
        output_path: str,
        dataset_name: str = "dataset"
    ) -> Path:
        """
        Download dataset from CellxGene.

        Args:
            url: Direct download URL
            output_path: Directory to save file
            dataset_name: Name for the file

        Returns:
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename
        if not dataset_name.endswith('.h5ad'):
            dataset_name = dataset_name + '.h5ad'

        filepath = output_path / dataset_name

        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath

        logger.info(f"Downloading dataset: {dataset_name}")
        logger.info(f"URL: {url}")

        try:
            # Stream download with progress bar
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f, tqdm(
                desc=dataset_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Successfully downloaded to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            raise

    def get_dataset_info(self, dataset_id: str) -> Dict:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_id: CellxGene dataset ID

        Returns:
            Dataset metadata
        """
        url = f"{self.base_url}/datasets/{dataset_id}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return {}


def display_search_results(df: pd.DataFrame, top_n: int = 10):
    """
    Display search results in a readable format.

    Args:
        df: DataFrame with search results
        top_n: Number of top results to show
    """
    if df.empty:
        print("\n[NO RESULTS] No datasets found matching your criteria")
        return

    print(f"\n{'='*80}")
    print(f"Found {len(df)} datasets - Showing top {min(top_n, len(df))}")
    print(f"{'='*80}\n")

    for i, row in df.head(top_n).iterrows():
        print(f"Dataset #{i+1}")
        print(f"   Collection: {row['collection_name']}")
        print(f"   Dataset: {row['dataset_name']}")
        print(f"   Cells: {row['cell_count']:,}")

        if row['description']:
            desc = row['description'][:150]
            print(f"   Description: {desc}...")

        if row.get('matched_keywords'):
            print(f"   Matches: {row['matched_keywords']}")

        if row['doi']:
            print(f"   DOI: {row['doi']}")

        print(f"   Download URL: {row['h5ad_url'][:80]}...")
        print()


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("CellxGene API - Motor Neuron Dataset Search")
    print("="*80)

    # Create API client
    api = CellxGeneAPI()

    # Search for motor neuron datasets
    print("\nSearching for motor neuron differentiation datasets...")
    results = api.search_motor_neuron_datasets()

    # Display results
    display_search_results(results, top_n=10)

    if not results.empty:
        print("\n" + "="*80)
        print("To download a dataset:")
        print("="*80)
        print("""
# Option 1: In Python
from src.data.cellxgene_api import CellxGeneAPI

api = CellxGeneAPI()
results = api.search_motor_neuron_datasets()

# Download first result
url = results.iloc[0]['h5ad_url']
api.download_dataset(url, 'data/raw', 'motor_neuron_dataset.h5ad')

# Option 2: Use the automated script
python scripts/auto_download_motor_neuron.py
        """)
