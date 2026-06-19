"""Data loading and preprocessing package."""
from .data_generator import SyntheticDataGenerator, load_dataset
from .cellxgene_loader import CellxGeneDatasetFinder, StemCellDataLoader
from .cellxgene_api import CellxGeneAPI

# Legacy imports (kept for compatibility)
try:
    from .load_data import DopaminergicDataLoader
    from .download_data import DopaminergicDataDownloader
    __all__ = [
        'SyntheticDataGenerator', 'load_dataset',
        'CellxGeneDatasetFinder', 'StemCellDataLoader', 'CellxGeneAPI',
        'DopaminergicDataLoader', 'DopaminergicDataDownloader'
    ]
except ImportError:
    __all__ = ['SyntheticDataGenerator', 'load_dataset', 'CellxGeneDatasetFinder', 'StemCellDataLoader', 'CellxGeneAPI']
