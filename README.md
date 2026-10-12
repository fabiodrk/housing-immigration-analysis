# Housing and immigration: exploratory annual analysis

A Python statistical analysis adapted from an academic project investigating associations between immigration and a house-price index. The original analysis has been made portable, and a fictional dataset is included so anyone can run the workflow without redistributing the course data.

**All bundled observations and resulting statistics are synthetic. They are not evidence about Canada or any other housing market.**

## Run

Python 3.10 or newer; standard library only:

```sh
python analysis.py
```

To analyze your own appropriately licensed quarterly data:

```sh
python analysis.py --input path/to/quarterly.csv
```

Required columns: `Date` in `YYYYQ1` through `YYYYQ4` format, `Composite_HPI` as a positive index, and `immigration` as a quarterly count. Provide one observation per quarter, consecutive years, and enough complete years for the regressions, including after the 2020–2022 exclusions. The script expects clean input; it is not a general-purpose data validator.

## Method

1. Keep complete calendar years.
2. Sum quarterly immigration and retain the fourth-quarter HPI.
3. Calculate annual percentage changes.
4. Compare same-year and one-year-lagged associations using Pearson correlation and simple regression.
5. Repeat growth comparisons excluding observations involving 2020–2022.

The script prints six comparisons and writes `annual_hpi_immigration_results.csv`. The implementation calculates two-tailed p-values using a regularized incomplete beta function.

## Interpretation and limits

Correlation does not establish causation. These simple comparisons do not control for interest rates, supply, incomes, population composition, or other confounders. Time-series dependence, a small sample, multiple comparisons, and nonstationarity can make ordinary regression p-values misleading. Excluding selected years is a sensitivity check, not a causal identification strategy. Constant or perfectly correlated inputs are outside this implementation's supported cases.

The original course dataset and reports are omitted. The synthetic fixture is only a demonstration of the computation. This repository preserves the original statistical workflow while replacing its computer-specific input path with a command-line option.
