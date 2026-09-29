# Data sources and scope

This repository uses **generated Black–Scholes prices as its main dataset**, plus a small set of **independently published numerical reference cases**. It does not use historical market prices as ground truth. 数据可以完全本地复现；这里验证的是神经网络对理论定价公式的近似误差，而不是对真实市场价格的预测能力。

## 1. Main training and evaluation data: generated locally

The first experiment concerns a non-dividend-paying European call, with fixed strike K = 100 and annual continuously compounded risk-free rate r = 0.02. The rate is an experimental constant, not a current market observation. Inputs vary over:

| Input | Domain | Unit |
|---|---|---|
| Moneyness S/K | 0.5 to 1.5 | dimensionless |
| Remaining maturity T | 1/365 to 1 | years; synthetic convention, no calendar required |
| Volatility sigma | 0.05 to 0.80 | annualized decimal |

The label is the Black–Scholes call price divided by K. A vectorized formula generates numerical labels; the neural network learns that mapping. Synthetic records have no observation date or market ticker, and the domain is a designed experimental range, not an empirical market distribution.

The approach has a research precedent: Liu, Oosterlee and Bohte (2019), [Pricing options and computing implied volatilities using neural networks](https://arxiv.org/abs/1901.08943), studies neural approximations of pricing and implied-volatility solvers, including Black–Scholes and Heston. This repo is a small validation experiment, not a reproduction of that paper or its speed-up claims.

**Why synthetic data is appropriate here:** errors against an analytic reference can be measured objectively, rare parameter combinations can be deliberately probed, samples are inexpensive, and no market-data license is needed. These advantages make it possible to isolate whether an agent chooses useful test points. They do not show that Black–Scholes is the correct model of a real option market. In this low-dimensional example, the formula is already fast; a trained neural network or an LLM is not automatically faster or cheaper.

### Splits and reproducibility

Generate training, validation and held-out audit data from separate seeded random draws. Record the generator configuration, split sizes, seeds, package versions and data hashes with every run. The model's training seed must also be recorded. Reusing only a seed without preserving code and software versions is not a complete reproducibility record.

- Training labels update model weights.
- Validation labels select the trained model/checkpoint.
- A fixed audit set reports held-out approximation errors after training. It does not drive the search agent.
- Search methods receive their own evaluated points and summaries under an explicit budget. A search agent's adaptively selected points are not an unbiased estimate of average market error.
- For a formal study, develop prompts on one network and reserve separately trained networks for the final comparison. A publicly packaged demo test set is a reproducibility example, not permanently unseen research evidence.

There is no time series in this synthetic experiment, so an independent random split is appropriate. Real historical option data would require date-aware splitting and controls for overlapping contracts.

## 2. Published independent reference fixtures: downloaded and bundled

File: `data/reference/black_scholes_reference.csv` (22 rows).

Source: QuantLib's [`testValues` European option tests](https://github.com/lballabio/QuantLib/blob/aaed737ba0d11a95b1df4c97c256c500275884df/test-suite/europeanoption.cpp), pinned to commit `aaed737ba0d11a95b1df4c97c256c500275884df` (2026-09-25). The source attributes its values to E. G. Haug, *Option pricing formulas*, McGraw-Hill, 1998. The CSV contains all 22 call rows from that test; put rows are excluded. Numerical values are retained, columns are renamed/reordered, and the original line number is recorded.

Use these fixtures **only to check the reference formula implementation**. They are not neural-network training data. Only one row has zero dividend yield; the other rows intentionally test the general formula with a continuous dividend yield. The main training domain still has q = 0. Expected prices are rounded to four decimals, with the original absolute tolerance of 1e-4 currency units. Do not require machine precision against these rounded numbers.

`maturity_years` retains the source's year fraction. The upstream code uses Actual360 and an integer day conversion; every selected maturity multiplied by 360 is an integer, so these values need no date adjustment. Do not reinterpret them as the main synthetic data's calendar-day convention.

Provenance, SHA-256 checksums, units, extraction rule and source links are in `data/reference/provenance.json`. The upstream QuantLib license is preserved in `third_party/QUANTLIB_LICENSE.txt`; upstream file copyright notices are retained in provenance. A preliminary check using Python's standard-library `math.erfc` normal CDF agreed with all published values within their stated tolerances. This provides external regression evidence, not a proof that the full parameter domain is numerically correct.

## 3. Real-market datasets reviewed for a later project

These links were checked on 2026-09-29 UTC. No account was created, purchase made, or market dataset bundled. They are optional future sources, not prerequisites for running the repo.

| Source | What is available | Why it is not used in the first experiment |
|---|---|---|
| [Cboe DataShop Option Quotes](https://datashop.cboe.com/option-quote-intervals) | Listed U.S. option quote intervals, including NBBO, volume and optional analytics; sample download and paid historical products | Commercial data terms, potentially large files, and restrictions on some index underlying-price fields. Market quotes are not exact Black–Scholes labels; discrepancies would combine neural approximation error, model assumptions and quote noise. |
| [OptionsDX](https://www.optionsdx.com/), [FAQ](https://www.optionsdx.com/faq/), [field definitions](https://www.optionsdx.com/option-chain-field-definitions/) | Option chains with strikes, expiry, bids/asks, underlying price and vendor-implied volatility | Free selections require account/checkout. Provider [disclaimer](https://www.optionsdx.com/disclaimer/) does not guarantee accuracy/completeness; redistribution permission and selected-product terms must be checked before committing data to a public repo. |

Before using real quotes, specify exercise style, dividend treatment, quote timestamp/time zone, settlement conventions and day-count basis. Use bid/ask and liquidity filters; last trade prices can be stale. Do not treat a model price as a market price.

In particular, implied volatility is usually inferred from an option price. Feeding a same-quote implied volatility back into the corresponding pricing formula can approximately reconstruct that quote by construction. That is not evidence of out-of-sample market-price prediction. A later market-data experiment must state what information was actually available at the prediction time and what independent target it predicts.

## What this demo can establish

It can establish that data generation, neural training, independent reference checks, budgeted search, error summaries and an agent feedback interface work together. It can measure error-search performance on the supplied frozen model. A single run cannot establish that LLM agents outperform established numerical search methods or that the system is suitable for production option pricing.
