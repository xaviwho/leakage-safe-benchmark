"""
Setup script for iPSC Digital Twin package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="stem-cell-digital-twin",
    version="0.1.0",
    author="Xavier Kanu",
    author_email="",
    description="Hybrid Physics-ML Digital Twin for Predicting Stem Cell Differentiation Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xaviwho/stem-cell-digital-twin",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "scanpy>=1.9.0",
        "anndata>=0.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "streamlit>=1.25.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "digital-twin=src.cli:main",
        ],
    },
)
