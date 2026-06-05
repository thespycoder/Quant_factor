# Quant Factor Research Engine

## Project Overview
<!-- LLM-assisted pipeline that discovers, backtests, and validates quantitative trading factors from SEC filings and price data. -->

## Architecture
<!-- Multi-agent system: signal-finder agent extracts factor hypotheses from filings; backtest engine validates them; report-writer agent summarises results. -->

## Setup
<!-- Copy .env.example to .env, fill in credentials, then `pip install -r requirements.txt`. -->

## Data Sources
<!-- SEC EDGAR (10-K/10-Q filings via sec-edgar-downloader) and daily equity prices via yfinance. -->

## Backtesting Methodology
<!-- Vectorized cross-sectional factor backtest: monthly rebalance, equal-weight long/short quintile portfolios. -->

## Factor Validation
<!-- Information Coefficient (IC), IC t-test, Fama-French 3-factor alpha regression (statsmodels). -->

## Limitations

**Survivorship bias.** The universe is sampled from the current S&P 500 constituent list and applied across 2015–2024. Companies that left the index during this period (via delisting, acquisition, or removal) are excluded, which biases results upward by omitting underperformers. A full correction requires point-in-time index-membership data, which is out of scope for this project. Reported factor performance should therefore be read as an upper bound, not a tradeable estimate.

## How to Run
<!-- `streamlit run dashboard/app.py` — or run individual pipeline stages via the CLI entry points in each package. -->

