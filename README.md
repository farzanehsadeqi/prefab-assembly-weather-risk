# Prefab assembly under weather risk

Which crane crews should a project manager contract in advance when wind can stop work for days at a time?
This project allocates 10,000 prefabricated units across four crane crews with different costs and wind
limits, using deterministic, chance-constrained and two-stage stochastic models with an emergency crew and
an optional weather insurance. The wind simulator of the original case study is **calibrated on 31 years of
Bielefeld wind data** and the decisions are tested **against real historical 20-day periods**.

**Main finding:** the same project needs a different contractor mix depending on the season. In summer the
cheap but wind-sensitive crew C carries the work; in autumn the model shifts to the robust crew A at its
maximum. Moving construction from summer to autumn raises the expected cost by about 13% and the tail risk
(5% CVaR, historical) by about 40%.

## Key results

Calibrated on daily maximum wind at 100 m, Bielefeld, 1995–2025 (Open-Meteo / ERA5).

| | course simulator | summer (May–Aug) | autumn (Oct–Nov) |
|---|---|---|---|
| expected completion α, crews A / B / C / D | 0.95 / 0.90 / 0.80 / 0.85 | 0.97 / 0.92 / 0.77 / 0.85 | 0.87 / 0.75 / 0.53 / 0.64 |
| day-to-day persistence ρ | 0.65 | 0.45 | 0.59 |
| two-stage allocation A / B / C / D | 1000 / 5000 / 3647 / 1000 | 1000 / 5000 / 3882 / 1000 | 6000 / 3600 / 500 / 1000 |
| expected cost, simulated / historical (EUR) | 112,472 / – | 110,408 / 111,100 | 125,042 / 126,623 |
| 5% CVaR, simulated / historical (EUR) | 150,056 / – | 131,913 / 133,833 | 169,963 / 187,236 |
| value of the stochastic solution (EUR) | 2,697 | 1,545 | 4,158 |
| 95% chance constraint | feasible | feasible | **infeasible** |

1. **The course simulator describes a Bielefeld summer.** Its completion fractions match May–August almost
   exactly. In autumn the probability that crew C works on half the days or fewer is 52% in reality versus
   6% in the course simulator.
2. **Season reverses the allocation.** Crew A goes from its minimum (1000) in summer to its maximum (6000) in
   autumn, crew C the other way round.
3. **Uncertainty is worth more in autumn.** The value of the stochastic solution almost triples.
4. **A 95% no-penalty guarantee is impossible in autumn** with these four crews, even at full capacity.
5. **The calibrated simulator is slightly optimistic in the tail.** Evaluated on real history, the summer
   chance-constrained plan misses the target in 9.0% of periods instead of the planned 5%, and the autumn
   CVaR is about 10% higher than simulated.
6. **Weather insurance (EUR 4,000) is bought in every case**, partly because reimbursed penalties up to the
   EUR 20,000 cap cost nothing at the margin, so the model lets the first 500 missing units go to penalty
   instead of paying the emergency crew.

![Completion coefficient of crew C](results/figures/alpha_distribution.png)

![Two-stage allocation](results/figures/allocation_two_stage.png)

![Cost distribution](results/figures/cost_distribution.png)

## Method

**Wind data.** Daily maximum of hourly wind speed at 100 m (the working height of a tower crane boom) from
the Open-Meteo ERA5 archive. Wind at 10 m was too calm to create risk; 10 m gusts made the problem
infeasible even on average.

**Calibration.** The course simulator maps a Gaussian AR(1) series through a Weibull quantile function.
A Weibull marginal (with two or three parameters) missed the share of calm days at the crew thresholds by
3–6 percentage points, so the calibrated simulator keeps the AR(1) dependence but uses the empirical
distribution of each season as marginal (Gaussian copula). ρ is the lag-1 correlation of the normal scores
of consecutive days, estimated separately for each season.

**Models** (python-mip, CBC), each solved on 200 scenarios:
- *Deterministic:* allocate exactly 10,000 units using expected completion fractions.
- *Joint chance constraint:* cover 10,000 units without penalties in at least 95% of scenarios (big-M).
- *Two-stage:* first-stage contracts; second-stage emergency crew (EUR 16/unit, up to 4000) and penalties
  (EUR 40/unit).
- *Insurance:* the two-stage model plus a binary choice to buy a policy that reimburses penalties up to
  EUR 20,000.

**Evaluation.** Every decision is evaluated on 10,000 new simulated scenarios and on all historical 20-day
windows of the season (3,224 in summer, 1,302 in autumn). Expected completion fractions, optimisation
scenarios and validation scenarios use three different random seeds.

## Assumptions and limitations

- A 20-day working cycle is taken as 20 consecutive calendar days.
- A crew works on a day if the daily maximum 100 m wind is at or below its threshold; no partial days.
- AR(1) persistence fades faster than real weather regimes and ignores differences between years, which is
  why the simulator underestimates the tail compared with history.
- All models minimise expected cost. The two-stage plan has the lowest mean cost but not the lowest CVaR;
  a risk-averse objective (e.g. mean-CVaR) would be the natural next step.
- Historical windows overlap, so they are not independent observations.

## How to run

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_pipeline.py --refresh   # download wind data and run everything
python -m pytest -q                        # tests
```

All inputs (site, wind variable, seasons, crew data, prices, seeds) are in `config/params.yaml`.

## Project structure

```
config/params.yaml          all model inputs
src/prefabrisk/
    weather.py              Open-Meteo download (10 m wind, gusts, 100 m wind)
    calibration.py          AR(1) persistence and empirical marginal per season
    scenarios.py            wind simulator, completion coefficients, historical windows
    models.py               deterministic, chance-constrained, two-stage, insurance, VSS
    risk.py                 out-of-sample cost, shortfall probability, VaR, CVaR
    plots.py                figures
scripts/run_pipeline.py     end-to-end analysis
tests/                      sanity checks
results/                    summary.json and figures
```

## Background

The case is based on a homework assignment from the course *Combining OR and Data Science*
(Prof. Dr. Michael Römer, Bielefeld University, summer term 2026). This version calibrates the weather
simulator on real data, compares two construction seasons, validates every decision against historical
weather, and corrects the coverage constraint of the insurance model (an equality in the original
formulation capped the total allocation at 10,000 units).

Weather data: [Open-Meteo](https://open-meteo.com) (CC BY 4.0), based on ERA5 reanalysis.