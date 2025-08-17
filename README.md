
# Recommender System with SVD

## Introduction

This project implements a **recommender system** based on **bias-adjusted Singular Value Decomposition (SVD) with shrinkage**.  
It predicts missing ratings in a user–item matrix (e.g., movie ratings), uncovers latent factors representing genres and user preferences, and provides interpretable insights into user and item embeddings.

The system is built from scratch with **custom SVD via power iteration and deflation**, in addition to standard evaluation and visualization techniques.

Datasets:
[Training set](https://s33.picofile.com/file/8485574792/data.csv.html)
[Test set](https://s33.picofile.com/file/8485574800/test.csv.html)

## Table of Contents

-   [Introduction](#introduction)
    
-   [Features](#features)
    
-   [Installation](#installation)
    
-   [License](#license)

## Features

-   **Bias-aware rating prediction** using global, user, and item biases with shrinkage regularization.
    
-   **Custom SVD implementation** using power iteration and deflation.
    
-   **Grid search** for best hyperparameters (`k` = latent dimensions, `λ` = shrinkage factor).
    
-   **Evaluation with RMSE** on a held-out test set.
    
-   **Visualization tools**:
    
    -   Energy per component
        
    -   Cumulative variance explained
        
    -   RMSE vs. number of components
        
    -   Latent embeddings of movies and users
        
    -   User–movie proximity in latent space
        
    -   KMeans clustering of movies
        
-   **Interpretability**: inspection of movies and users at extreme ends of latent factors, cluster summaries.
    

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/azizi-zahra/recommender-system-with-svd.git
cd recommender-system-with-svd
pip install -r requirements.txt
```

### Requirements

-   Python 3.8+
    
-   NumPy
    
-   Pandas
    
-   Matplotlib
    
-   scikit-learn
    

## License

This project is licensed under the [MIT License](https://github.com/azizi-zahra/recommender-system-with-svd/blob/main/LICENSE).

