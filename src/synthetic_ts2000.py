"""Synthetic stand-in for the TS2000 financial-statement panel used in this project.

The original data (TS2000, Korea Listed Companies Association) is licensed and cannot be
redistributed. This module simulates a firm-year panel of externally audited Korean firms with
the same raw line items, so the full pipeline in the notebook runs end to end.

How it works
------------
Each firm has a latent operating profitability (operating income / assets) that follows a
mean-reverting AR(1) process around a firm-specific level, plus an economy-wide term that is
negative in the three shock periods (dot-com bust 2000-01, global financial crisis 2007-08,
European debt crisis 2011-12). All line items are derived from that latent state, firm traits
and noise, so financial ratios carry real (but simulated) information about next year's outcome.

The numbers produced here are NOT real and say nothing about actual Korean firms.
"""
import numpy as np
import pandas as pd

YEARS = np.arange(1998, 2019)

# economy-wide shift in operating profitability (fraction of assets)
MACRO = {2000: -0.025, 2001: -0.030, 2002: 0.010,
         2007: -0.015, 2008: -0.035, 2009: 0.005,
         2011: -0.020, 2012: -0.025, 2013: 0.005}
# economy-wide shift in borrowing rates
RATE_SHIFT = {1998: 0.04, 1999: 0.02, 2000: 0.02, 2001: 0.015, 2007: 0.01, 2008: 0.015}


def simulate_panel(n_firms=16000, seed=42):
    rng = np.random.default_rng(seed)
    n_years = len(YEARS)

    # firm traits
    mu = rng.normal(0.060, 0.035, n_firms)                 # long-run operating ROA
    rho = rng.uniform(0.45, 0.80, n_firms)                 # persistence of shocks
    shock_beta = rng.lognormal(0.0, 0.4, n_firms)          # sensitivity to macro shocks
    manufacturing = rng.random(n_firms) < 0.55
    turnover = rng.lognormal(np.log(1.0), 0.35, n_firms)   # sales / assets
    gross_margin = np.clip(rng.normal(0.22, 0.08, n_firms) + 0.6 * mu, 0.05, 0.7)
    lev_base = np.clip(rng.normal(0.58, 0.15, n_firms) - 1.5 * mu, 0.15, 0.95)
    borrow_share = rng.uniform(0.35, 0.75, n_firms)        # borrowings / liabilities
    rate_base = np.clip(rng.normal(0.050, 0.012, n_firms) - 0.15 * mu, 0.02, 0.12)
    ca_share = np.clip(rng.normal(0.45, 0.15, n_firms), 0.1, 0.9)
    inv_share = np.where(manufacturing, rng.uniform(0.2, 0.45, n_firms), rng.uniform(0.02, 0.25, n_firms))
    rec_share = rng.uniform(0.25, 0.55, n_firms)
    cash_share = rng.uniform(0.08, 0.35, n_firms)
    tangible_share = np.where(manufacturing, rng.uniform(0.5, 0.85, n_firms), rng.uniform(0.15, 0.6, n_firms))
    cl_share = rng.uniform(0.45, 0.85, n_firms)
    log_assets0 = rng.normal(23.0, 1.1, n_firms)           # ~ KRW 10bn median
    first_year = rng.choice(YEARS[:15], n_firms, p=np.linspace(2, 1, 15) / np.linspace(2, 1, 15).sum())

    rows = []
    p = mu + rng.normal(0, 0.03, n_firms)
    log_assets = log_assets0.copy()
    prev_sales = None
    for t, year in enumerate(YEARS):
        macro = MACRO.get(int(year), 0.0) * shock_beta
        p = mu + rho * (p - mu) + macro + rng.normal(0, 0.028, n_firms)
        growth = 0.04 + 0.6 * (p - 0.03) + rng.normal(0, 0.08, n_firms)
        log_assets = log_assets + growth
        assets = np.exp(log_assets)

        sales = assets * turnover * np.exp(rng.normal(0, 0.12, n_firms) + 1.5 * (p - mu))
        operating_income = p * assets + rng.normal(0, 0.004, n_firms) * assets
        gross_profit = sales * np.clip(gross_margin + rng.normal(0, 0.02, n_firms), 0.03, 0.8)
        gross_profit = np.maximum(gross_profit, operating_income + 0.02 * sales)
        cogs = sales - gross_profit
        sga = gross_profit - operating_income

        lev = np.clip(lev_base - 0.8 * (p - mu) + rng.normal(0, 0.03, n_firms), 0.08, 1.3)
        liabilities = lev * assets
        equity = assets - liabilities
        borrowings = liabilities * np.clip(borrow_share + rng.normal(0, 0.05, n_firms), 0.05, 0.95)
        rate = np.clip(rate_base + RATE_SHIFT.get(int(year), 0.0) - 0.1 * (p - mu)
                       + rng.normal(0, 0.006, n_firms), 0.01, 0.2)
        interest = borrowings * rate
        other = rng.normal(0, 0.01, n_firms) * assets
        pretax = operating_income - interest + other
        net_income = pretax - np.maximum(pretax, 0) * 0.22

        current_assets = assets * np.clip(ca_share + rng.normal(0, 0.03, n_firms), 0.05, 0.95)
        inventory = current_assets * inv_share
        receivables = current_assets * rec_share
        cash = current_assets * np.clip(cash_share + 0.5 * (p - mu) + rng.normal(0, 0.02, n_firms), 0.01, 0.6)
        tangible_assets = (assets - current_assets) * tangible_share
        current_liabilities = liabilities * cl_share
        operating_cash_flow = net_income + 0.04 * tangible_assets + rng.normal(0, 0.02, n_firms) * assets

        rows.append(pd.DataFrame({
            'firm_id': np.arange(n_firms), 'year': year,
            'total_assets': assets, 'current_assets': current_assets, 'cash': cash,
            'receivables': receivables, 'inventory': inventory, 'tangible_assets': tangible_assets,
            'total_liabilities': liabilities, 'current_liabilities': current_liabilities,
            'borrowings': borrowings, 'equity': equity,
            'sales': sales, 'cogs': cogs, 'sga': sga, 'operating_income': operating_income,
            'interest_expense': interest, 'net_income': net_income,
            'operating_cash_flow': operating_cash_flow,
            'manufacturing': manufacturing.astype(int),
            'financial_or_spc': (np.arange(n_firms) % 50 == 0).astype(int),
            'active': year >= first_year,
        }))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel.pop('active')].reset_index(drop=True)

    # mimic missing and unusable entries in the source database
    miss = rng.random((len(panel), 4)) < np.array([0.04, 0.03, 0.05, 0.02])
    for j, col in enumerate(['inventory', 'receivables', 'operating_cash_flow', 'tangible_assets']):
        panel.loc[miss[:, j], col] = np.nan
    panel.loc[rng.random(len(panel)) < 0.03, 'inventory'] = 0.0
    return panel


if __name__ == '__main__':
    df = simulate_panel()
    print(df.shape)
    print(df.groupby('year').size())
