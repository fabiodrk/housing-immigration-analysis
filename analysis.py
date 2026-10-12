from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "data" / "synthetic_quarterly.csv"
OUTPUT = Path(__file__).resolve().with_name("annual_hpi_immigration_results.csv")


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be between 0 and 1")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def pearson_regression(x: list[float], y: list[float]) -> dict[str, float]:
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    ss_x = sum((value - mean_x) ** 2 for value in x)
    ss_y = sum((value - mean_y) ** 2 for value in y)
    cross = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    r = cross / math.sqrt(ss_x * ss_y)
    slope = cross / ss_x
    intercept = mean_y - slope * mean_x
    df = n - 2
    t_value = r * math.sqrt(df / (1.0 - r * r))
    p_value = regularized_incomplete_beta(
        df / 2.0, 0.5, df / (df + t_value * t_value)
    )
    return {
        "n": n,
        "r": r,
        "r_squared": r * r,
        "slope": slope,
        "intercept": intercept,
        "t": t_value,
        "df": df,
        "p_two_tailed": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Exploratory annual housing and immigration associations. Bundled data is synthetic.')
    parser.add_argument('--input', type=Path, default=SOURCE)
    args = parser.parse_args()
    print(f'Input: {args.input}. Bundled sample is synthetic; results are not empirical findings.')
    years: dict[int, list[dict[str, float | int]]] = {}
    with args.input.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            match = re.fullmatch(r"(\d{4})Q([1-4])", row["Date"].strip())
            if match is None:
                raise ValueError(f"Invalid quarterly date: {row['Date']!r}")
            year, quarter = map(int, match.groups())
            years.setdefault(year, []).append(
                {
                    "quarter": quarter,
                    "hpi": float(row["Composite_HPI"]),
                    "immigration": float(row["immigration"]),
                }
            )

    annual: list[dict[str, float | int | None]] = []
    for year, observations in sorted(years.items()):
        if {int(item["quarter"]) for item in observations} != {1, 2, 3, 4}:
            continue
        q4 = next(item for item in observations if item["quarter"] == 4)
        annual.append(
            {
                "year": year,
                "annual_immigration": sum(
                    float(item["immigration"]) for item in observations
                ),
                "q4_hpi": float(q4["hpi"]),
                "hpi_q4_yoy_pct": None,
                "immigration_yoy_pct": None,
            }
        )

    for previous, current in zip(annual, annual[1:]):
        current["hpi_q4_yoy_pct"] = 100.0 * (
            float(current["q4_hpi"]) / float(previous["q4_hpi"]) - 1.0
        )
        current["immigration_yoy_pct"] = 100.0 * (
            float(current["annual_immigration"])
            / float(previous["annual_immigration"])
            - 1.0
        )

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annual[0]))
        writer.writeheader()
        writer.writerows(annual)

    same_year = annual[1:]
    same_year_levels = pearson_regression(
        [float(row["annual_immigration"]) for row in same_year],
        [float(row["hpi_q4_yoy_pct"]) for row in same_year],
    )
    growth_to_growth = pearson_regression(
        [float(row["immigration_yoy_pct"]) for row in same_year],
        [float(row["hpi_q4_yoy_pct"]) for row in same_year],
    )
    prior_year_levels = pearson_regression(
        [float(row["annual_immigration"]) for row in annual[:-1]],
        [float(row["hpi_q4_yoy_pct"]) for row in annual[1:]],
    )
    prior_year_growth = pearson_regression(
        [float(row["immigration_yoy_pct"]) for row in annual[1:-1]],
        [float(row["hpi_q4_yoy_pct"]) for row in annual[2:]],
    )
    same_year_sensitivity_rows = [
        row for row in same_year if int(row["year"]) not in {2020, 2021, 2022}
    ]
    same_year_growth_sensitivity = pearson_regression(
        [float(row["immigration_yoy_pct"]) for row in same_year_sensitivity_rows],
        [float(row["hpi_q4_yoy_pct"]) for row in same_year_sensitivity_rows],
    )
    lagged_pairs = [
        (predictor, outcome)
        for predictor, outcome in zip(annual[1:-1], annual[2:])
        if int(predictor["year"]) not in {2020, 2021, 2022}
        and int(outcome["year"]) not in {2020, 2021, 2022}
    ]
    prior_year_growth_sensitivity = pearson_regression(
        [float(predictor["immigration_yoy_pct"]) for predictor, _ in lagged_pairs],
        [float(outcome["hpi_q4_yoy_pct"]) for _, outcome in lagged_pairs],
    )

    for name, result in (
        ("Same-year immigration total vs HPI annual change", same_year_levels),
        ("Same-year immigration growth vs HPI annual change", growth_to_growth),
        ("Prior-year immigration total vs next-year HPI annual change", prior_year_levels),
        ("Prior-year immigration growth vs next-year HPI annual change", prior_year_growth),
        ("Same-year growth excluding 2020-2022", same_year_growth_sensitivity),
        ("Lagged growth excluding pairs involving 2020-2022", prior_year_growth_sensitivity),
    ):
        print(name)
        print(
            "  N={n}, r={r:.4f}, R2={r_squared:.4f}, "
            "t({df})={t:.4f}, p={p_two_tailed:.4f}, slope={slope:.8f}".format(
                **result
            )
        )
    print(f"Annual data written to: {OUTPUT}")


if __name__ == "__main__":
    main()
