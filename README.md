<h1 align="center">Portfolio management </h1>

Portfolio optimization is the process of allocating capital to different asset
classes by selecting the best asset distribution across a selection of assets to maximize risk-adjusted returns. The traditional approach is based on Modern Portfolio Theory, which optimizes various factors such as returns, Sharpe ratio or risk.
 In this work, we extract latent factors using unsupervised machine learning techniques, and create a portfolio based on these factors. These latent factors can be employed aside from traditional factors such as value or size as an alternative and for diversification purposes.


# Objectives

- Our objective in this work is to use unsupervised machine learning methods on a dataset of stock returns to optimize the risk-adjusted returns of an equity portfolio.

- To assess the efficacy of the strategy, the generated portfolios' performance will be compared to a benchmark and backtested.

# Files
- `utils.py`: python file containing utils functions.
- `Portfolio_management.ipynb`: jupyter notebook with the modeling.
- `sp500.csv` dataset with SP500 components prices.

 
 # Data

The data consists of the adjusted closing price of 410 assets, which constitute the SP500. The data ranges from 2001-2014. The prices can be downloaded using the Yahoo Finance API.


 # Quick Preview


![pca_results](fig/peak.png)


# Conclusion

- We presented a systematic approach for constructing factor-free portfolio strategies based on unsupervised machine learning techniques.
- The presented approach is based on the eigendecomposition of the empirical covariances of stock returns into systematic components.
- Exploring linear and non-linear mapping enables the construction of different portfolios with higher adjusted returns compared to the market.
- The portfolios may be included in the traditional factor model portfolio to increase the diversification of the overall portfolio.
