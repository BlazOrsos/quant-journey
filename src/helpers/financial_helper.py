"""Financial Helper Functions

This module contains reusable functions for calculating financial metrics
such as returns, volatility, Sharpe ratio, Sortino ratio, CAGR, and drawdowns.
"""

import numpy as np
import pandas as pd
from typing import Optional


def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return from a series of returns.
    
    Args:
        returns: Series of returns (simple returns, not log returns)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return as a decimal (e.g., 0.15 for 15%)
    """
    if len(returns) == 0:
        return np.nan
    
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years <= 0:
        return np.nan
    
    annualized_return = (1 + total_return) ** (1 / years) - 1
    return annualized_return


def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility (standard deviation) from a series of returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized volatility as a decimal
    """
    if len(returns) == 0:
        return np.nan
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    ann_return = calculate_annualized_return(returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(returns, periods_per_year)
    
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    
    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio (like Sharpe but using downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return np.nan
    
    ann_return = calculate_annualized_return(returns, periods_per_year)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        # No downside, Sortino ratio is infinite (return a large number)
        return np.nan if np.isnan(ann_return) else 999.0
    
    downside_std = downside_returns.std()
    downside_vol = downside_std * np.sqrt(periods_per_year)
    
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan
    
    sortino = (ann_return - risk_free_rate) / downside_vol
    return sortino


def calculate_cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).
    
    Note: This is the same as annualized return for consistency.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        CAGR as a decimal
    """
    return calculate_annualized_return(returns, periods_per_year)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a series of returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as a decimal (negative value, e.g., -0.20 for -20%)
    """
    if len(returns) == 0:
        return np.nan
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown at each point
    drawdown = (cumulative - running_max) / running_max
    
    # Return the maximum drawdown (most negative value)
    max_dd = drawdown.min()
    
    return max_dd


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """Calculate all standard financial metrics.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'Annualized Return': calculate_annualized_return(returns, periods_per_year),
        'Annualized Volatility': calculate_annualized_volatility(returns, periods_per_year),
        'Sharpe Ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'Sortino Ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'CAGR': calculate_cagr(returns, periods_per_year),
        'Max Drawdown': calculate_max_drawdown(returns)
    }
    
    return metrics
