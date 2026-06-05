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
<!-- Survivorship bias in universe construction; look-ahead bias risk in filing date alignment; single-country (US) coverage. -->

## How to Run
<!-- `streamlit run dashboard/app.py` — or run individual pipeline stages via the CLI entry points in each package. -->
