#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from matplotlib import style
import sklearn as sklearn
from sklearn.decomposition import KernelPCA


def sharpe_ratio(ts_returns, periods_per_year=252):

    
    n_years = ts_returns.shape[0] / periods_per_year
    annualized_return = np.power(np.prod(1+ts_returns),(1/n_years))-1
    
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)

    annualized_sharpe = annualized_return / annualized_vol # R_f is considered zero.
    
    return annualized_return, annualized_vol, annualized_sharpe


def apply_kpca(cov_matrix, kernel):
    
        
    if kernel == 'commum_pca':
        pca = PCA()
        pca.fit(cov_matrix)
        pcs = pca.components_
        
        normalized_pcs = list()
        for eigen_vector in pcs:
            # scaling  values to sum to 1 
            normalized_values = eigen_vector / eigen_vector.sum()
            normalized_pcs.append(normalized_values)
            
    else:
        kpca = KernelPCA( kernel= kernel) 
        transformed_data = kpca.fit_transform(cov_matrix)    
        loadings = kpca.eigenvalues_* kpca.eigenvectors_
        normalized_pcs = list()

        total_variance = sum(kpca.eigenvalues_)
        for i in range(len(kpca.eigenvalues_)):
            normalized_values = (loadings[:,i] / total_variance) * kpca.eigenvectors_[:,i]
            weights_sum = np.abs(normalized_values.sum())
            weights = normalized_values / weights_sum
            normalized_pcs.append(weights)
            

        
    return normalized_pcs

def optimizedPortfolio(normalized_pcs, df_raw_test, stock_tickers, plot=False):
    
    n_portfolios = 20
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    id_highest_sharpe = 0 

    for ix in range(n_portfolios):

        pc_w = normalized_pcs[ix]
        eigen_prtf = pd.DataFrame(data ={'weights': pc_w}, index = stock_tickers)
        eigen_returns = np.dot(df_raw_test.loc[:, eigen_prtf.index], eigen_prtf )
        eigen_returns = pd.Series(eigen_returns.squeeze(), index=df_raw_test.index)

        annualized_ret[ix], annualized_vol[ix], sharpe_metric[ix] = sharpe_ratio( eigen_returns )


    sharpe = sharpe_metric[~np.isnan(sharpe_metric)]
    max_sharpe = np.amax(sharpe)
    index_max_sharpe = np.where(sharpe_metric == max_sharpe)
    id_highest_sharpe = index_max_sharpe[0]

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' % 
          (id_highest_sharpe,
           annualized_ret[id_highest_sharpe]*100, 
           annualized_vol[id_highest_sharpe]*100, 
           sharpe_metric[id_highest_sharpe]))
    results_highest_sharpe = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol, 'Sharpe': sharpe_metric})
    results_highest_sharpe.sort_values(by=['Sharpe'], ascending=False, inplace=True)

    results_highest_sharpe.dropna(inplace=True)
    if plot:
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 8)
        ax.plot(sharpe_metric, linewidth=3,marker='X',linestyle='-.')

        ax.set_title('Sharpe ratio of eigen-portfolios')
        ax.set_ylabel('Sharpe ratio')
        ax.set_xlabel('Portfolios')
        
    return results_highest_sharpe

def plot_best_portfolio(id_highest_sharpe, normalized_pcs, df_raw_test, stock_tickers, plot=False):
    pc_w = normalized_pcs[id_highest_sharpe]
    best_prtf = pd.DataFrame(data ={'weights': pc_w.squeeze()}, index = stock_tickers)

    best_prtf_returns = np.dot(df_raw_test.loc[:, best_prtf.index], best_prtf)
    best_prtf_returns = pd.Series(best_prtf_returns.squeeze(), index=df_raw_test.index)
    er, vol, sharpe = sharpe_ratio(best_prtf_returns)
    if plot:    
        print('Best eigen-portfolio'+ str(id_highest_sharpe)+ ':\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (er*100, vol*100, sharpe))
        df_plot = pd.DataFrame({'PC_best': best_prtf_returns, 'SPX': df_raw_test.loc[:, 'SPX']}, index=df_raw_test.index)
        np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. Best eigen-portfolio', 
                                 figsize=(18,8), linewidth=3)
    return best_prtf_returns
