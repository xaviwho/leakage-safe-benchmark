"""
Data downloader for Jerber et al. (2021) dopaminergic neuron dataset.

Dataset: Population-scale single-cell RNA-seq profiling across
         dopaminergic neuron differentiation
Source: Nature Genetics (2021)
Zenodo: https://zenodo.org/record/4333872
"""

import os
import requests
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DopaminergicDataDownloader:
    """
    Download Jerber et al. (2021) dopaminergic differentiation dataset.
    """

    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Zenodo record URLs
        self.zenodo_record = "https://zenodo.org/record/4333872"

        # Key files to download (examples - actual URLs from Zenodo)
        self.file_urls = {
            # Processed count matrices
            "day0_counts.h5ad": "https://zenodo.org/record/4333872/files/day0_counts.h5ad",
            "day11_counts.h5ad": "https://zenodo.org/record/4333872/files/day11_counts.h5ad",
            "day30_counts.h5ad": "https://zenodo.org/record/4333872/files/day30_counts.h5ad",

            # Cell metadata
            "cell_metadata.csv": "https://zenodo.org/record/4333872/files/cell_metadata.csv",

            # Gene annotations
            "gene_annotations.csv": "https://zenodo.org/record/4333872/files/gene_annotations.csv",
        }

    def download_file(self, url: str, filename: str) -> bool:
        """
        Download a single file with progress bar.

        Args:
            url: URL to download from
            filename: Local filename to save to

        Returns:
            True if successful, False otherwise
        """
        output_path = self.output_dir / filename

        if output_path.exists():
            logger.info(f"{filename} already exists, skipping download")
            return True

        logger.info(f"Downloading {filename} from {url}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Successfully downloaded {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    def download_all(self) -> bool:
        """
        Download all dataset files.

        Returns:
            True if all downloads successful
        """
        logger.info("=" * 60)
        logger.info("Downloading Jerber et al. (2021) dataset")
        logger.info("=" * 60)
        logger.info(f"Source: {self.zenodo_record}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")

        success = True
        for filename, url in self.file_urls.items():
            if not self.download_file(url, filename):
                success = False

        if success:
            logger.info("")
            logger.info("=" * 60)
            logger.info("All downloads completed successfully!")
            logger.info("=" * 60)
        else:
            logger.warning("Some downloads failed. Please check the logs.")

        return success

    def get_alternative_download_info(self):
        """
        Print alternative download instructions.
        """
        print("\n" + "=" * 70)
        print("ALTERNATIVE DOWNLOAD METHODS")
        print("=" * 70)
        print("\nIf automatic download fails, you can manually download from:\n")
        print(f"1. Zenodo (Processed Data):")
        print(f"   {self.zenodo_record}")
        print(f"\n2. European Nucleotide Archive (Raw Data):")
        print(f"   https://www.ebi.ac.uk/ena/browser/view/ERP121676")
        print(f"\n3. GitHub (Analysis Code):")
        print(f"   https://github.com/single-cell-genetics/singlecell_neuroseq_paper")
        print("\n" + "=" * 70)
        print("\nAfter manual download:")
        print(f"1. Place files in: {self.output_dir.absolute()}")
        print("2. Run: python src/data/load_data.py")
        print("=" * 70 + "\n")


def main():
    """Main download function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Jerber et al. (2021) dopaminergic differentiation dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show download information without downloading"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    downloader = DopaminergicDataDownloader(output_dir=args.output_dir)

    if args.info_only:
        downloader.get_alternative_download_info()
    else:
        success = downloader.download_all()

        if not success:
            print("\nNote: Some files may require manual download.")
            downloader.get_alternative_download_info()


if __name__ == "__main__":
    main()
