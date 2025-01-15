# Clustering Algorithms

This repository contains Python implementations of various clustering algorithms, commonly used in unsupervised machine learning for grouping similar data points.

## Algorithms Implemented

- **K-Means Clustering**: Partitions data into K distinct clusters based on feature similarity.
- **Hierarchical Clustering**: Builds nested clusters by either merging or splitting them successively.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Forms clusters based on areas of high density, identifying outliers as noise.
- **Gaussian Mixture Models (GMM)**: Assumes data is generated from a mixture of several Gaussian distributions with unknown parameters.

## Repository Structure

- `k_means.py`: Implementation of the K-Means algorithm.
- `hierarchical_clustering.py`: Implementation of Hierarchical Clustering.
- `dbscan.py`: Implementation of the DBSCAN algorithm.
- `gmm.py`: Implementation of Gaussian Mixture Models.
- `utils.py`: Utility functions for data preprocessing and visualization.
- `datasets/`: Contains sample datasets used for demonstrating the algorithms.
- `notebooks/`: Jupyter notebooks with examples and visualizations of each algorithm.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following packages:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pandas`

You can install these packages using pip:

```bash
pip install numpy scipy scikit-learn matplotlib pandas
