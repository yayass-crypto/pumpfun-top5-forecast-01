# pumpfun-top5-forecast-01 entry: a trajectory-aware Monte Carlo forecaster.
#
# Idea (why this should beat the house baseline, which just repeats the last
# mark's order with fixed 0.30/0.10/0.10):
#   * Pump.fun coins churn violently: in the practice, only 1 of the first
#     mark's top 10 was still top 10 at the last mark, and 3 of the 5 coins
#     holding a top-5 place at the last answer read were not in the history
#     at all (born after the cut). Over the final's 7-13h horizon that churn
#     is far worse, so most of each place's probability is deliberately left
#     unnamed (it flows to newcomers on the answer chart) and the named mass
#     is spread over up to 100 candidate coins per place.
#   * Each coin's future log10 market cap is modelled as
#         L(h) = L0 + drift*h + N(0, sd(h)),  clamped to [-0.5, 6.0]
#     where drift blends the coin's recent slope with an age prior (young
#     coins still rising, old coins decaying) and sd(h) grows with the
#     horizon, so long-horizon forecasts are honestly flat.
#   * A fixed-seed Monte Carlo over the seven answer-read horizons gives
#     place probabilities and size-bin probabilities. Everything is
#     deterministic: two runs on the same history are byte-identical.
#
# Python 3.13, standard library only. Reads $MAKEFUN_INPUT_DIR (else /input),
# writes $MAKEFUN_OUTPUT_DIR/forecast.json (else /output/forecast.json).

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
SEED = 20261112
MC_ITERATIONS = 3000
UNIVERSE_CAP = 400
MAX_NAMED_PER_PLACE = 100
LOG_CAP_MIN = -0.5
LOG_CAP_MAX = 6.0


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise ValueError("the history holds no ok mark")
    return task, marks


def prior_drift(age_minutes):
    """Age prior for log10-cap drift per hour: young coins still rising,
    mature coins flat, old coins decaying."""
    if age_minutes < 20:
        return 0.4
    if age_minutes < 60:
        return 0.0
    return -0.15


def build_coins(marks):
    """Per-mint series -> (L0, drift, age_minutes, best_rank, last_rank)."""
    series = {}
    for mi, m in enumerate(marks):
        t = parse_time(m["at"])
        for r in m["rows"]:
            mint = r["mint"]
            try:
                logcap = math.log10(float(r["market_cap_lamports"]) / 1e9)
            except (ValueError, TypeError, OverflowError):
                continue
            obs = series.setdefault(mint, [])
            obs.append((t, logcap, r.get("created_at"), r.get("rank", 999)))
    coins = {}
    for mint, obs in series.items():
        obs.sort(key=lambda o: o[0])
        t_last, l_last, created_at, rank_last = obs[-1]
        age_min = 1e9
        if created_at:
            try:
                age_min = (t_last - parse_time(created_at)).total_seconds() / 60.0
            except ValueError:
                pass
        # Recent slope over the last two steps (or one step), clipped.
        slope = None
        if len(obs) >= 3:
            (t0, l0, _, _), (_, l1, _, _) = obs[-3], obs[-1]
            dt_h = (obs[-1][0] - t0).total_seconds() / 3600.0
            if dt_h > 0:
                slope = (l1 - l0) / dt_h
        elif len(obs) == 2:
            (t0, l0, _, _), (t1, l1, _, _) = obs
            dt_h = (t1 - t0).total_seconds() / 3600.0
            if dt_h > 0:
                slope = (l1 - l0) / dt_h
        if slope is not None:
            # A 10-minute slope annualised is extremely noisy: clip hard and
            # shrink it toward the age prior so one jump cannot 3x a coin.
            slope = max(-0.5, min(1.0, slope))
            drift = 0.25 * slope + 0.75 * prior_drift(age_min)
        else:
            drift = prior_drift(age_min)
        best_rank = min(o[3] for o in obs)
        coins[mint] = {
            "L0": l_last,
            "drift": drift,
            "age_min": age_min,
            "best_rank": best_rank,
            "last_rank": rank_last,
        }
    return coins


def select_universe(coins, marks):
    """Top coins by last-mark rank plus any coin that was top-20 in any mark,
    capped at UNIVERSE_CAP by (last log-cap + best log-cap)."""
    last_rank = {m: c["last_rank"] for m, c in coins.items()}
    cand = set(m for m, r in last_rank.items() if r <= 250)
    for m in marks:
        for r in m["rows"]:
            if r.get("rank", 999) <= 20:
                cand.add(r["mint"])
    cand = [m for m in cand if m in coins]
    # Tie-break by mint: set iteration order is hash-randomized per process,
    # so a plain sort on the float key alone is not deterministic.
    cand.sort(key=lambda m: (-(coins[m]["L0"] + (coins[m]["L0"] if coins[m]["best_rank"] <= 20 else 0)), m))
    return cand[:UNIVERSE_CAP]


def bin_of(logcap, edges):
    """Size bin 0..81 for log10(cap in SOL); edges has 81 values 2.00..6.00."""
    if logcap < edges[0]:
        return 0
    if logcap >= edges[-1]:
        return 81
    lo, hi = 0, len(edges) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if edges[mid] <= logcap:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def main():
    task, marks = load_history()
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    t_last = parse_time(marks[-1]["at"])
    horizons = []
    for r in task.get("reads", []):
        horizons.append((parse_time(r) - t_last).total_seconds() / 3600.0)
    if not horizons:
        horizons = [1.0]
    h_mean = sum(horizons) / len(horizons)

    coins = build_coins(marks)
    universe = select_universe(coins, marks)
    if len(universe) < places:
        raise ValueError("fewer than %d coins in the universe" % places)

    # Unnamed share per place: newcomers take more places the longer the
    # horizon (practice ~0.4 at ~1h, final ~0.85 at ~10h).
    unnamed = min(0.88, 0.35 + 0.05 * h_mean)
    named_mass = 1.0 - unnamed
    # Size bins: leave 0.30 unnamed so no bin is ever over-trusted.
    size_named_mass = 0.70

    rng = random.Random(SEED)
    place_counts = {k: {} for k in range(1, places + 1)}
    size_counts = {k: {} for k in range(1, places + 1)}
    rank_sum = {m: 0.0 for m in universe}
    n_iter = MC_ITERATIONS
    for _ in range(n_iter):
        h = rng.choice(horizons)
        sd = min(1.2, 0.30 + 0.20 * h)
        sampled = []
        for m in universe:
            c = coins[m]
            v = c["L0"] + c["drift"] * h + rng.gauss(0.0, sd)
            if v < LOG_CAP_MIN:
                v = LOG_CAP_MIN
            elif v > LOG_CAP_MAX:
                v = LOG_CAP_MAX
            sampled.append((v, m))
        sampled.sort(key=lambda x: (-x[0], x[1]))
        for k in range(places):
            v, m = sampled[k]
            place_counts[k + 1][m] = place_counts[k + 1].get(m, 0) + 1
            b = bin_of(v, edges)
            size_counts[k + 1][b] = size_counts[k + 1].get(b, 0) + 1
        for k in range(places):
            rank_sum[sampled[k][1]] += k + 1
    # Expected rank: counted top-5 ranks plus 6 for the rest.
    exp_rank = {}
    for m in universe:
        top5_hits = sum(place_counts[k].get(m, 0) for k in range(1, places + 1))
        exp_rank[m] = (rank_sum[m] + 6.0 * (n_iter - top5_hits)) / n_iter

    probabilities = []
    for k in range(1, places + 1):
        counts = place_counts[k]
        named = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_NAMED_PER_PLACE]
        total = sum(v for _, v in named)
        if total <= 0:
            continue
        for mint, cnt in named:
            p = (cnt / total) * named_mass
            if p >= 0.0005:
                probabilities.append({"mint": mint, "place": k, "p": round(p, 6)})

    sizes = []
    for k in range(1, places + 1):
        counts = size_counts[k]
        total = sum(counts.values())
        if total <= 0:
            continue
        named = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [(b, c) for b, c in named if (c / total) * size_named_mass >= 0.004]
        kept_total = sum(c for _, c in kept)
        if kept_total <= 0:
            continue
        for b, c in kept:
            p = (c / kept_total) * size_named_mass
            sizes.append({"place": k, "bin": int(b), "p": round(p, 6)})

    top5 = sorted(universe, key=lambda m: (exp_rank[m], m))[:places]

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(forecast, f, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
