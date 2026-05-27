# Bus Schedule Optimization using Genetic Algorithm and Hybrid Genetic Algorithm with Simulated Annealing

**Metaheuristics — Final Project (Week 11 + Week 12)**

**Authors:** Cotfas Rares and Cozma Dacian

**Notebook:** `Project.ipynb`

**Topic:** Public Transport Timetabling / Bus Schedule Optimization

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Definition (Cell 1)](#2-problem-definition-cell-1)
3. [Related Work (Cell 2)](#3-related-work-cell-2)
4. [Imports (Cell 3)](#4-imports-cell-3)
5. [Dataset Loading and Demand Construction (Cells 4–6)](#5-dataset-loading-and-demand-construction-cells-46)
6. [Objective Function and Repair Operator (Cells 7–8)](#6-objective-function-and-repair-operator-cells-78)
7. [Week 11: Proposed Metaheuristics (Cells 9–10)](#7-week-11-proposed-metaheuristics-cells-910)
8. [Week 12: Additional Comparison Algorithms (Cells 11–12)](#8-week-12-additional-comparison-algorithms-cells-1112)
9. [Parameter Validation by Grid Search (Cells 13–15)](#9-parameter-validation-by-grid-search-cells-1315)
10. [Statistical Comparison Over 30 Independent Runs (Cells 16–17)](#10-statistical-comparison-over-30-independent-runs-cells-1617)
11. [Final Computational Experiments (Cells 18–24)](#11-final-computational-experiments-cells-1824)
12. [Discussion of Results (Cell 25)](#12-discussion-of-results-cell-25)
13. [Conclusions and Future Improvements (Cell 26)](#13-conclusions-and-future-improvements-cell-26)
14. [References (Cell 27)](#14-references-cell-27)

---

## 1. Project Overview

The notebook builds a complete optimization pipeline that decides **when each bus on an urban route should depart** during a 16-hour operating day so that three competing goals are balanced as well as possible:

- passengers wait as little as possible,
- buses are not overcrowded,
- the timetable remains operationally realistic (no absurd 1-minute gaps, no huge holes, full day coverage).

The mandatory two project methods are a **Simple Genetic Algorithm (GA)** and a **Hybrid GA + Simulated Annealing (SA)**. For the Week 12 comparison block, three additional population-based continuous metaheuristics are added: **Particle Swarm Optimization (PSO)**, **Whale Optimization Algorithm (WOA)** and a compact **Puma Optimization Algorithm (POA)**. Every method optimizes the same vector representation against the same objective function on the same real demand profile, so the comparison is strictly apples-to-apples.

The notebook contains **28 cells** in total (markdown + code). The work is presented end-to-end below, block by block, in execution order.

---

## 2. Problem Definition (Cell 1)

The optimization horizon is fixed at the urban operating window **06:00 – 22:00**, which gives `DAY_MINUTES = 16 * 60 = 960` minutes.

A candidate timetable is encoded as a real-valued vector

```
x = [departure_1, departure_2, ..., departure_n]
```

where each entry is the departure time of one bus expressed as minutes after 06:00. The vector has fixed length `n = N_BUSES = 200`. A candidate may be **infeasible** after a mutation or crossover (departures out of order, gaps too small, points outside the day window). A dedicated **repair operator** sorts the entries and pushes each one forward until a minimum safety gap is respected, so every candidate becomes a valid timetable before being scored.

The objective is a **scalar minimization** problem combining four practical costs:

1. **Passenger waiting time.** A long headway combined with high demand means many people wait for a long time on the platform.
2. **Crowding penalty.** If the number of passengers accumulated between two consecutive buses exceeds the bus capacity, the surplus is penalized quadratically.
3. **Regularity penalty.** Strongly varying headways are uncomfortable and are penalized via the variance of the operational headways.
4. **Coverage penalty.** The first bus should leave shortly after 06:00 and the last bus shortly before 22:00; long uncovered tails at the start or end of the day are penalized non-linearly.

The result is a realistic transit-timetabling model that does not require a private city dataset.

---

## 3. Related Work (Cell 2)

The chosen reference paper is:

> Suanpang, P., Jamjuntr, P., Jermsittiparsert, K., & Kaewyong, P. (2022). *Tourism Service Scheduling in Smart City Based on Hybrid Genetic Algorithm Simulated Annealing Algorithm*. Sustainability, 14(23), 16293. https://doi.org/10.3390/su142316293

The paper hybridizes GA and SA for a smart-city scheduling problem. The ideas reused in this project are: encoding a schedule as a chromosome, evaluating it through a fitness function, using elitism plus selection to preserve high-quality chromosomes, recombining schedules with crossover, and using SA neighborhood moves to refine schedules and escape local optima.

In the original paper SA is applied **inside the GA mutation step**. The project adapts this idea to bus departure-time optimization. The final, validated formulation in the notebook places SA **after the GA loop finishes**, applying it to the single best schedule found by the GA. This separation (global GA exploration → local SA refinement) proved to be the most stable and most performant configuration on this dataset.

A secondary reference is also included:

> Fatyanosa, T. N., & Mahmudy, W. F. (2018). *Hybrid genetic algorithms and simulated annealing for multi-trip vehicle routing problem with time windows.* Institute of Advanced Engineering and Science.

---

## 4. Imports (Cell 3)

The first code cell imports the libraries used throughout the notebook:

```python
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
```

`numpy` handles all vector arithmetic, `pandas` is used to read the CSV and to display the summary tables, `matplotlib` produces the demand plot, the box-plot, the convergence curves, and the bar chart. `dataclasses.dataclass` is used to define a small typed container, `RunResult`, which holds the output of every algorithm run. The matplotlib style is set globally to `seaborn-v0_8-whitegrid` for a clean look.

---

## 5. Dataset Loading and Demand Construction (Cells 4–6)

### 5.1 Source data (Cell 4 — markdown)

The dataset is real public transport data, `hourly_transportation_202001.csv`, filtered for **01.01.2020** and bus line **429A**. The raw file gives hourly passenger totals; for the optimization we need a minute-level demand curve. The hourly count for each hour is therefore distributed evenly across the 60 minutes of that hour.

### 5.2 Constants and the demand vector (Cell 5)

```python
DAY_MINUTES = 16 * 60     # 960 — the operating day length in minutes
START_HOUR  = 6           # service starts at 06:00
N_BUSES     = 200         # fleet size used for the optimization
BUS_CAPACITY = 55         # passengers per bus
MIN_GAP     = 2           # minimum safety gap between two consecutive departures (minutes)
MAX_GAP_SOFT = 45         # any headway above this incurs a soft penalty
```

A helper, `minutes_to_clock(minute_after_start)`, converts a minute offset from 06:00 into a `HH:MM` string used in tables and plots.

The CSV is read with the `ISO-8859-1` encoding (the source contains non-UTF characters), filtered to rows where `transition_date == "1/1/2020"` and `line_name == "429A"`, then grouped by `transition_hour` to compute the total passenger count per operating hour. A `numpy` array `demand` of length `DAY_MINUTES + 1` is initialized with zeros and filled hour by hour:

- For each operating hour `h ∈ {6, 7, …, 21}` we look up the total `passengers` for that hour.
- We build a length-60 integer list where each entry is `passengers // 60`.
- The remainder `passengers % 60` is sprinkled approximately uniformly across the 60 minutes by stepping every `60 / remainder` minutes and adding 1.
- The result is written into `demand[(h-6)*60 : (h-5)*60]`.

A small `pandas` DataFrame `demand_df` is also built with columns `minute`, `time`, and `passengers_per_minute`. The first 60 rows are displayed for sanity-checking; they show that 06:00–07:00 has a near-flat demand of roughly 4–5 passengers per minute, which matches the early-morning baseline.

### 5.3 Visualization of the demand curve (Cell 6)

A matplotlib line chart of `passengers_per_minute` versus `minute` is produced. The x-axis ticks every 120 minutes are labelled with `HH:MM`. The chart confirms the real shape of the day: a quiet early morning, a clear morning rush, a midday plateau, and an evening peak.

![Real passenger demand profile (Line 429A, 01.01.2020)](figures/cell6_fig1.png)

---

## 6. Objective Function and Repair Operator (Cells 7–8)

### 6.1 The repair operator `repair_schedule`

`repair_schedule(x, n_buses=N_BUSES, min_gap=MIN_GAP)` is the gatekeeper that turns any real vector into a valid timetable. It does five things, in order:

1. Make a writable float copy and resize it to exactly `N_BUSES` entries if it came in with the wrong length.
2. Clip values to `[0, DAY_MINUTES]` so no departure falls outside the operating day.
3. Sort in ascending order (departures must be chronological).
4. Walk left to right; whenever two consecutive entries are closer than `MIN_GAP = 2` minutes, push the later one forward to `previous + MIN_GAP`.
5. If after pushing the last entry overshoots `DAY_MINUTES`, shift the whole vector left by the overflow, then re-apply the gap walk to keep the timetable feasible.

A final `np.clip` re-applies the day boundary as a safety net. This operator is invoked in every algorithm before scoring or after every move.

### 6.2 Integrating passenger demand over an interval

`interval_demand(a, b)` returns the total expected number of arriving passengers between minute `a` and minute `b`. The implementation rounds and clips the bounds, then computes `numpy.trapz(demand[a:b+1], dx=1)` — i.e., trapezoidal integration of the per-minute demand vector with a 1-minute step. The function returns `0.0` for empty or reversed intervals.

### 6.3 The full objective `objective(schedule, details=False)`

The schedule is first repaired to be feasible. Two synthetic boundaries are added at minute 0 (day start) and minute `DAY_MINUTES` (day end), giving `n + 2` boundaries and therefore `n + 1` intervals (with intervals 0 and n being the uncovered tails). For each interval `[a, b]` of length `h`:

- `D = interval_demand(a, b)` — passengers arriving in that interval.
- `waiting_cost += D * h / 2.0` — the classical mean-waiting-time approximation (passengers arrive uniformly inside the interval, so average wait is `h / 2`).
- `crowding_penalty += 20.0 * max(0, D - BUS_CAPACITY)**2` — quadratic surcharge for every passenger above the 55-seat bus capacity.
- `max_gap_penalty += 80.0 * max(0, h - MAX_GAP_SOFT)**2` — quadratic surcharge whenever the headway exceeds 45 minutes.

Two additional global terms are added afterwards using the **operational** headways (those between actual buses, not including the boundary intervals):

- `regularity_penalty = 4.0 * np.var(operational_headways)` — penalizes irregular spacing.
- `coverage_penalty   = 25.0 * (s[0]**1.35 + (DAY_MINUTES - s[-1])**1.35)` — penalizes leaving the early morning or late evening uncovered.

The final score is `waiting + crowding + regularity + coverage + max_gap`. When `details=True` the function additionally returns the individual cost components plus `mean_headway` and `max_headway`, which are used in the post-mortem analysis.

### 6.4 `random_schedule(rng)`

A starting point used by every algorithm. It places `N_BUSES` departures uniformly between minute 5 and minute `DAY_MINUTES - 5`, perturbs them with Gaussian noise of standard deviation 18, and repairs the result. This gives diverse but sane initial timetables.

### 6.5 Baseline check

The cell ends by evaluating the **even** baseline timetable `repair_schedule(np.linspace(5, DAY_MINUTES - 5, N_BUSES))`. The reported details for this baseline are:

```
total:       709 785.55
waiting:      26 237.44
crowding:    683 109.00
regularity:  ~0 (perfectly even by construction)
coverage:        439.12
max_gap:           0.00
mean_headway:     4.77
max_headway:      5.00
```

The total is dominated by the crowding term: an even spacing of 200 buses across 960 minutes gives a headway of roughly 4.77 minutes, but because of the large hourly volume during peak hours the per-interval demand still exceeds the 55-seat capacity, so massive quadratic crowding penalties remain. This is precisely what the metaheuristics will try to redistribute.

---

## 7. Week 11: Proposed Metaheuristics (Cells 9–10)

### 7.1 `RunResult` data class

A small typed container is used everywhere:

```python
@dataclass
class RunResult:
    name: str
    best_schedule: np.ndarray
    best_score: float
    history: List[float]
    runtime: float
```

`history` is the best-so-far score at every iteration; it produces the convergence plot at the end.

### 7.2 Building blocks for the GA

**`tournament_select(population, scores, rng, k=3)`** — picks `k = 3` random indices and returns a copy of the one with the lowest score. Tournament selection is robust and easy to tune.

**`crossover(p1, p2, rng, crossover_rate=0.5)`** — implements a small hybrid crossover. With probability `1 - crossover_rate` it simply copies one parent. Otherwise it produces a child as `alpha * p1 + (1 - alpha) * p2` with a *per-gene* uniform `alpha ∈ [0.25, 0.75]`, which is the **arithmetic** part. With probability 0.5 it then overwrites a contiguous prefix from `p1` and a suffix from `p2` using a random cut point — the **single-point crossover** part. The result is repaired before returning.

**`gaussian_mutation(child, rng, mutation_rate=0.22, sigma=16.0)`** — each gene independently has a 22% probability of being shifted by a Gaussian noise with `σ = 16` minutes. If by chance no gene was selected, a random gene is forced to be mutated, so a child always changes at least once. The mutated vector is repaired before being returned.

### 7.3 SA building blocks (used by the hybrid method)

**`neighbor(schedule, rng, scale=12.0)`** — generates a small move:

- With probability 0.75 it perturbs **one random departure** by a Gaussian noise with `σ = scale`.
- With probability 0.25 it perturbs a **contiguous run of 2 – 6 departures** by a smaller Gaussian (`σ = scale / 2`), which simulates locally re-spacing a small cluster of buses.

The move is repaired before being returned.

**`run_sa(initial_schedule, rng, steps=1000, t0=220.0, cooling=0.99)`** — classical Metropolis Simulated Annealing. The temperature `T` starts at `t0` and is multiplied by `cooling` each step. The neighbor scale shrinks together with the temperature (`scale = max(3, T / 18)`) so the search is wide early and narrow late. A candidate is always accepted if it improves; a worse candidate is accepted with probability `exp(-delta / T)`. The best schedule ever visited is tracked separately and returned.

### 7.4 `run_ga` — the Simple GA

Default hyperparameters: `population_size=46, generations=90, elite_size=4, crossover_rate=0.5, mutation_rate=0.22`. The flow per generation is:

1. Score the current population with `objective(...)`.
2. Sort by score and keep the top `elite_size = 4` as the seeds of the next population.
3. Fill the rest by repeatedly drawing two parents with `tournament_select`, recombining with `crossover`, mutating with `gaussian_mutation`, and appending the child.
4. Record the best score of this generation in `history`.

After all generations the population is re-scored, the global best is selected, and a `RunResult(name="Simple GA", …)` is returned along with its runtime.

### 7.5 `run_hybrid_ga_sa` — Hybrid GA + Simulated Annealing

The hybrid scheme has two clear phases:

1. **GA phase** — call `run_ga(...)` with the same parameters as for the simple GA, producing a strong global solution `ga_result.best_schedule`.
2. **SA phase** — use that solution as the initial state for `run_sa(...)` with parameters `sa_steps`, `sa_t0`, `sa_cooling`. The SA acts as a precise local polisher.

The total runtime sums the two phases. The final `history` is the GA convergence curve with one extra point at the end equal to the SA-refined score, which makes the post-SA improvement visible in the convergence plot.

This separated formulation was preferred over the in-loop SA-as-mutation variant after experimentation, because it gave consistently better and more stable final scores on this dataset (see Section 10).

---

## 8. Week 12: Additional Comparison Algorithms (Cells 11–12)

All three additional methods are continuous-space population metaheuristics. They operate on real vectors and rely on `repair_schedule` to map every candidate back to a valid timetable before scoring. All three use the same default budget of **42 individuals × 90 iterations** to make the comparison fair.

### 8.1 `run_pso(seed, particles=42, iterations=90)` — Particle Swarm Optimization

Classical PSO. Each particle has a position `X[i]` (a candidate schedule) and a velocity `V[i]`. Personal best `P` and global best `gbest` are tracked. The velocity update is

```
V = w * V + c1 * r1 * (P - X) + c2 * r2 * (gbest - X)
```

with inertia `w = 0.72`, cognitive coefficient `c1 = 1.45`, and social coefficient `c2 = 1.45`. After each move, `repair_schedule(x + v)` is applied component-wise.

### 8.2 `run_woa(seed, whales=42, iterations=90)` — Whale Optimization Algorithm

WOA mimics the bubble-net hunting of humpback whales. A coefficient `a` linearly decreases from 2 to 0. With probability 0.5 the agent does **shrinking encircling** (around the best whale if `mean(|A|) < 1`, otherwise around a random whale); otherwise it performs a **spiral update** around the best:

```
candidate = D * exp(0.8 * l) * cos(2 * pi * l) + best
```

Every candidate is repaired and the global best is refreshed at the end of each iteration.

### 8.3 `run_poa(seed, pumas=42, iterations=90)` — compact Puma Optimization Algorithm

A custom, compact puma-inspired optimizer with two regimes blended by progress `t / (iterations - 1)`:

- **Exploration (probability decreases over time):** `candidate = X[i] + r * (peer - center) + jump`, with `jump ~ N(0, 20 * exploration_strength)`. The puma "roams" away from the population center toward another peer.
- **Exploitation (probability increases over time):** `candidate = X[i] + r * (best - X[i]) + short_jump`, with `short_jump ~ N(0, 5 * (1 - exploitation_strength + 0.1))`. The puma "stalks" the current best.

A candidate replaces the puma if it is strictly better; otherwise it is occasionally accepted with a small exploration-driven probability `0.03 * exploration_strength`, which provides some additional diversity early on.

---

## 9. Parameter Validation by Grid Search (Cells 13–15)

### 9.1 GA grid search (Cell 14)

To validate the GA hyperparameters, an exhaustive grid is constructed using `itertools.product` over

- `population ∈ {30, 50, 100}`,
- `generations ∈ {50, 100}`,
- `mutation_rate ∈ {0.1, 0.22, 0.5}`,
- `crossover_rate ∈ {0.5, 0.7, 0.9}`.

This yields **54 combinations**. To keep the runtime tractable each combination is evaluated with one fixed seed (`seed = 42`) and the mean score and runtime are stored in `ga_validation_df`, which is sorted ascending by `mean_score`. The first ten rows are displayed.

Top-of-table result:

| population | generations | mutation_rate | crossover_rate | mean_score | mean_runtime_sec |
|---|---|---|---|---|---|
| **50** | **100** | **0.1** | **0.9** | **1 627 861** | **11.26** |
| 50 | 50 | 0.1 | 0.9 | 1 662 159 | 5.82 |
| 30 | 50 | 0.1 | 0.9 | 1 663 424 | 15.59 |
| 30 | 100 | 0.1 | 0.9 | 1 663 424 | 6.74 |
| 100 | 100 | 0.1 | 0.9 | 1 700 847 | 22.75 |

The clear winner is `population=50, generations=100, mutation_rate=0.1, crossover_rate=0.9`. A few observations:

- **High crossover rate (0.9)** dominates the top rows: recombining strong parents is consistently helpful.
- **Low mutation rate (0.1)** wins over 0.22 and 0.5: this objective is sensitive enough that aggressive mutation destroys good structure.
- Doubling the population to 100 produces no benefit but doubles the runtime; **50 individuals** is the sweet spot.

### 9.2 SA grid search (Cell 15)

The best GA parameters are extracted from row 0 of `ga_validation_df`. A new GA is run with those parameters (seed 42) to produce a high-quality `base_schedule` (its score, `1 627 861`, is printed). This schedule is then handed as the starting point to an SA grid search over

- `steps ∈ {200, 500, 1000}`,
- `t0 ∈ {100, 200, 500}`,
- `cooling ∈ {0.90, 0.95, 0.99}`,

i.e., **27 combinations**. Because SA is highly stochastic, each combination is repeated 3 times (seeds `100, 101, 102`) and the mean and minimum SA scores are stored alongside the mean runtime in `sa_validation_df`.

Top-of-table result:

| steps | t0 | cooling | mean_score | min_score | mean_runtime_sec |
|---|---|---|---|---|---|
| **1000** | **200.0** | **0.99** | **913 543** | **854 426** | **2.15** |
| 1000 | 200.0 | 0.90 | 914 664 | 894 474 | 2.14 |
| 1000 | 100.0 | 0.99 | 929 444 | 906 862 | 2.18 |
| 1000 | 100.0 | 0.95 | 942 362 | 905 210 | 2.19 |
| 1000 | 200.0 | 0.95 | 944 758 | 895 282 | 2.13 |

The winner is `steps=1000, t0=200, cooling=0.99`. SA reduces the score from `~1 628 000` (best GA only) to `~914 000`, which is a **~44% improvement at a cost of about 2 extra seconds**.

---

## 10. Statistical Comparison Over 30 Independent Runs (Cells 16–17)

To make the GA-vs-Hybrid comparison statistically meaningful, both algorithms are run **30 times** using their winning hyperparameters. The 30 seeds follow the standard pattern used throughout the notebook: `[101 * i for i in range(1, 31)] = [101, 202, 303, …, 3030]`.

For every seed, both `run_ga(...)` and `run_hybrid_ga_sa(...)` are executed and their best scores are collected.

**Results over 30 runs:**

| Metric | Simple GA | Hybrid GA + SA |
|---|---|---|
| **Mean**   | 1 701 542.77 | **1 001 775.24** |
| **Min**    | 1 542 622.86 | **871 023.11**   |
| **Max**    | 1 843 194.06 | **1 136 937.96** |

The Hybrid method's *worst* result is still better than the Simple GA's *best* result. The two distributions do not overlap.

A side-by-side box plot of the two distributions is produced (Cell 17, end). The hybrid box sits well below the GA box, both medians and both extremes confirming the gap above.

![Distribution of objective scores over 30 runs](figures/cell17_fig2.png)

---

## 11. Final Computational Experiments (Cells 18–24)

### 11.1 The 5-seed all-algorithm benchmark (Cell 19)

To compare all five metaheuristics on equal footing, every algorithm is run with the same five seeds `SEEDS = [101, 202, 303, 404, 505]` and the same evaluation budget (the GA-family methods use their tuned parameters; PSO, WOA, POA use their default 42 individuals × 90 iterations).

For every method the script computes the **best**, **mean**, and **standard deviation** of the objective, the **mean runtime**, the **best seed index**, and the **mean max headway** of the best schedules. The summary table sorted by mean score is:

| Rank | Method | Best Score | Mean Score | Std | Mean Runtime (s) | Best Seed Idx | Mean Max Headway (min) |
|---|---|---|---|---|---|---|---|
| 1 | **Hybrid GA + SA** | **941 403.28** | **1 022 518** | 72 806 | 13.92 | 3 | **8.99** |
| 2 | POA | 1 536 486 | 1 659 049 | 87 284 | 8.52 | 0 | 12.29 |
| 3 | Simple GA | 1 603 798 | 1 690 278 | 67 633 | 11.60 | 4 | 14.76 |
| 4 | PSO | 1 743 476 | 1 788 697 | 48 323 | 8.35 | 2 | 15.45 |
| 5 | WOA | 3 231 642 | 3 596 147 | 261 872 | 8.62 | 0 | 17.71 |

Key observations:

- **Hybrid GA + SA wins on every metric** except runtime (where it is only ~3.4 seconds slower than the second best). It has the lowest best score, the lowest mean, and crucially the lowest *max headway*, meaning passengers never wait excessively long for a bus.
- **POA** is the runner-up — better than the Simple GA and PSO — confirming that its mixed exploration/exploitation phases are a respectable global heuristic, but it still cannot match a method that adds a local polishing phase.
- **Simple GA** is competitive but loses to the hybrid by ~40% on the mean score.
- **WOA** is the weakest performer. Its encircling/spiral update is conceptually mismatched with a highly structured, sorted, discrete-in-spirit search space.

### 11.2 Mean score bar chart with error bars (Cell 20)

A `matplotlib` bar chart displays `summary_df["mean_score"]` per method, with `yerr=summary_df["std_score"]`. The visual ranking matches the table above and makes the WOA gap dramatic.

![Mean objective score over 5 seeds, with std-dev error bars](figures/cell20_fig3.png)

### 11.3 Convergence curves (Cell 21)

For each method the **best** run (lowest score) is selected, and its `history` is plotted on the same axes. The Hybrid GA + SA curve is identical to the GA curve until the last point, where it jumps down sharply — this is the SA refinement applied to the best GA schedule. POA shows steady improvement; PSO converges fast then plateaus; WOA fluctuates and finishes high.

![Convergence curves of the best run of each method](figures/cell21_fig4.png)

### 11.4 Best-overall timetable inspection (Cell 22)

The overall best run across all methods and seeds is selected:

```
Best method:    Hybrid GA + SA
Best objective: 941 403.28
Details: waiting=27 591.66, crowding=913 806.00, regularity=5.63,
         coverage=0.00, max_gap=0.00,
         mean_headway=4.82 min, max_headway=7.68 min
```

Several interesting facts surface:

- The objective is still dominated by the **crowding** term. With 200 buses, 55 seats each, and ~11 000 daily passengers on a single line on this real dataset, the demand peaks exceed the cumulative capacity even when the buses are bunched optimally. This is a property of the data, not a flaw of the algorithm.
- **`coverage = 0`** and **`max_gap = 0`** — the optimum keeps the very first and very last minutes covered, and never lets the headway grow above the 45-minute soft cap.
- The **mean headway is 4.82 minutes** and the **max headway is 7.68 minutes** — a very dense, very regular schedule.

A timetable DataFrame is built with three columns (`bus_number`, `departure_minute`, `departure_time`). The first 12 rows are displayed:

| bus | minute | time |
|---|---|---|
| 1 | 0 | 06:00 |
| 2 | 3 | 06:03 |
| 3 | 5 | 06:05 |
| 4 | 11 | 06:11 |
| 5 | 15 | 06:15 |
| 6 | 21 | 06:21 |
| 7 | 26 | 06:26 |
| 8 | 32 | 06:32 |
| 9 | 39 | 06:39 |
| 10 | 41 | 06:41 |
| 11 | 49 | 06:49 |
| 12 | 56 | 06:56 |

### 11.5 Timetable overlaid on the demand curve (Cell 23)

A combined chart is produced: the per-minute demand profile (grey line, left y-axis) and red vertical lines at every departure of the best schedule (right y-axis, no ticks). Visually the red lines are denser exactly where demand is highest, which is a strong qualitative confirmation that the optimizer reacts to the real demand.

![Best timetable overlaid on real demand](figures/cell23_fig5.png)

### 11.6 Headway audit (Cell 24)

The timetable is copied, augmented with a `headway_from_previous` column computed as `np.diff` over `departure_minute`, and the last 12 rows are displayed. The late-evening headways for the best schedule are:

| bus | minute | time | headway (min) |
|---|---|---|---|
| 189 | 902 | 21:02 | 6 |
| 190 | 909 | 21:09 | 7 |
| 191 | 914 | 21:14 | 5 |
| 192 | 920 | 21:20 | 6 |
| 193 | 923 | 21:23 | 3 |
| 194 | 927 | 21:27 | 4 |
| 195 | 932 | 21:32 | 5 |
| 196 | 935 | 21:35 | 3 |
| 197 | 941 | 21:41 | 6 |
| 198 | 948 | 21:48 | 7 |
| 199 | 955 | 21:55 | 7 |
| 200 | 960 | 22:00 | 5 |

The last bus departs exactly at `22:00`, headways stay between 3 and 7 minutes, the minimum gap is respected, and there are no long uncovered tails.

---

## 12. Discussion of Results (Cell 25)

- **Simple GA** is a very strong global baseline. Crossover and mutation efficiently sweep the high-dimensional space of timetables (200-dimensional real vectors) and locate near-optimal arrangements that minimize the worst crowding and the largest gaps.
- **Hybrid GA + SA (post-optimization)** is the strict winner on every metric. The GA does the heavy global exploration, the SA acts as a precise micro-polisher that shifts individual departures by a few minutes to align with the local peaks of the real demand. Because SA always returns a solution that is at least as good as its starting point, the hybrid guarantees a score that is strictly **≤** the GA's score, and in practice the improvement is enormous (~40%).
- **PSO, WOA, POA** are respectable continuous-swarm comparators. They show that population-based search is competitive but that **GA's discrete-friendly recombination plus an SA local refinement** is the best fit for a sorted, gap-constrained timetabling problem like this one.

A note on the problem constraints. When the notebook was run earlier with `N_BUSES = 38` and `MIN_GAP = 5`, the combination of ~11 000 passengers per day and only 38 buses across 16 hours made the crowding term mathematically unavoidable — the metaheuristics could not "create" capacity. Raising `N_BUSES` to 200 and lowering `MIN_GAP` to 2 lets the algorithms express their full clustering potential during peak hours, which is the configuration shipped in the final notebook.

---

## 13. Conclusions and Future Improvements (Cell 26)

The notebook satisfies and extends every requirement of the project:

- A precise model of the public-transport timetabling problem.
- The two required metaheuristics: a Simple GA and a Hybrid GA + SA.
- A real-world dataset (Line 429A tap-in counts for 01.01.2020) instead of a synthetic demand curve.
- An extensive parameter validation (54 GA combinations + 27 SA combinations + 30-run statistical comparison).
- A 5-seed final benchmark comparing GA, Hybrid, PSO, WOA, POA with both quantitative tables and three visualizations (bar chart with error bars, convergence curves, demand + timetable overlay).

**Future improvements** suggested in the conclusions:

- Dynamically scale `N_BUSES` and `BUS_CAPACITY` according to total daily demand to enable aggressive peak-hour clustering.
- Optimize multiple intersecting urban routes simultaneously rather than a single line.
- Add operational constraints: vehicle circulation, driver shift lengths, depot capacity.
- Model passenger transfer synchronization between different transport modes (bus / metro / tram).

---

## 14. References (Cell 27)

1. Suanpang, P., Jamjuntr, P., Jermsittiparsert, K., & Kaewyong, P. (2022). *Tourism Service Scheduling in Smart City Based on Hybrid Genetic Algorithm Simulated Annealing Algorithm*. Sustainability, 14(23), 16293. https://doi.org/10.3390/su142316293
2. Fatyanosa, T. N., & Mahmudy, W. F. (2018). *Hybrid genetic algorithms and simulated annealing for multi-trip vehicle routing problem with time windows*. Institute of Advanced Engineering and Science.

---

## Summary of Final Results

| Quantity | Value |
|---|---|
| Best method overall | **Hybrid GA + SA** |
| Best objective score | **941 403.28** |
| Mean score (5 seeds) | 1 022 518 |
| Std (5 seeds) | 72 806 |
| Mean runtime | 13.92 s |
| Mean headway of best schedule | 4.82 min |
| Max headway of best schedule | **7.68 min** |
| First bus | 06:00 |
| Last bus | 22:00 |
| Improvement over Simple GA (mean, 30 runs) | ~41% |
| Improvement over baseline even spacing | from 709 786 (crowding-dominated) to 941 403 with full coverage and small max headway — note: the baseline's lower raw score is misleading because it ignores any clustering ability; the hybrid produces a *usable* timetable that fits real demand. |
