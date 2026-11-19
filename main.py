"""Forecast GTA VI trophy earn rates at a case's target, in counts.

Reads every <id>.in in the input directory and writes <id>.out with one
row per trophy: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the
trophy's earn rate (percent of tracked players) at the target time.

A rate is a ratio, so this forecasts its two counts separately:

    rate(target) = EARNERS(target) / PLAYERS(target)

PLAYERS is Exophase's tracked-player count. It grows through launch
week as more players sync their profiles, and it grows in JUMPS: the
page sits unchanged for hours and then refreshes to a much larger
count. PLAYERS is therefore forecast in log-time (a least-squares fit
of log players against log hours since release), which smooths the
jumps into a trend.

EARNERS is forecast in two cohorts, because the two groups behave
differently:

  * the players already tracked at the last read keep progressing:
    their share moves toward the trophy's class ceiling in logit
    space, the same pull v1 used, over the horizon;
  * the players who join between the last read and the target earn
    from scratch: their share follows a tenure curve whose time
    constant is inferred from the trophy's own observed rate and the
    player base's age (a trophy already held by many players after a
    few hours is a fast trophy; one held by almost nobody is slow),
    over the new joiners' average exposure, half the horizon.

The stale-page rule: when every read in the history is byte-identical
(same tracked count, same rates), the page is not telling us nothing
is happening -- it is telling us it has not refreshed. A frozen page
means a refresh is coming, and the refresh reveals all the joining
that happened while it was frozen. The house documents exactly one
such refresh (competition.yaml): frozen 23:48-04:00Z on launch night,
then refreshed at 04:30Z with the tracked count up 638 -> 3,868, a
factor of 6.06 over about 4.7 hours of frozen page. The stale prior
below is calibrated on that documented refresh (log growth 0.383 per
hour over the frozen span plus the horizon to the next refresh, then
diminishing power-law growth after it) and it ONLY fires on provably
flat history. On the final case, whose history is full of refreshes
and real movement, the fitted trends dominate and the prior stays off.

The spread is asymmetric, because a refresh only dilutes rates:
players join faster than earners, so the lower tail is where the
risk lives and the upper tail is bounded by ceilings and
progression. The upside keeps the symmetric core spread. The
downside adds a dilution tail built from the counts themselves: how
far the tracked base could grow past its point forecast, and how low
earners could sit if the players who join earn nothing. The stale
prior's growth rate comes from ONE documented refresh, so it is
treated as uncertain by about half in log space, not exact -- that
uncertainty is what sizes the player leg of the tail.

One special case: a trophy at 100% (every tracked player holds it,
to the two decimals Exophase prints) is earned on arrival -- new
players hold it as fast as they appear in the count -- so its
earners stay equal to the tracked count and its rate stays at 100%.

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 100 / (1 + math.exp(-x))
    e = math.exp(x)
    return 100 * e / (1 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------- counts
#
# Everything below works in counts, not rates. A case's history is a
# list of reads; each read carries the tracked-player count and every
# trophy's rate (percent, to two decimals). Earners for trophy i in a
# read are rate/100 * players. Rates are quantised to 0.01 points, so
# earners are quantised too (at 638 tracked players, one hundredth of
# a point is 0.06 players); the model never trusts the last digit.


def parse_time(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def hours_between(later: datetime, earlier: datetime) -> float:
    """Hours from earlier to later, floored at 0.05: a read stamped
    just before the release date still counts as just after it."""
    return max(0.05, (later - earlier).total_seconds() / 3600.0)


def least_squares(xs: list, ys: list):
    """(intercept, slope) of the least-squares line, or None when the
    x values carry no variance."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y)
                for x, y in zip(xs, ys)) / den
    return mean_y - slope * mean_x, slope


def case_series(case: dict):
    """Split a case into its counts series.

    Returns (origin, h_target, times, players, earners) where origin
    is midnight UTC on the release date, h_target is the target's age
    in hours, times are each read's age in hours, players are the
    tracked counts, and earners[i] is the list of (age, earners) pairs
    for trophy i over the reads where its rate was present.
    """
    origin = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    h_target = hours_between(parse_time(case["target"]), origin)
    times = []
    players = []
    earners = [[] for _ in case["trophies"]]
    for read in case["history"]:
        age = hours_between(parse_time(read["at"]), origin)
        count = read["players_tracked"]
        times.append(age)
        players.append(count)
        for i, rate in enumerate(read["rates"]):
            if rate is not None:
                earners[i].append((age, rate / 100.0 * count))
    return origin, h_target, times, players, earners


def history_is_stale(case: dict) -> bool:
    """True when the page never moved: every read has the same tracked
    count and every trophy the same rate. That is the stale-page
    signature the house documents for launch night, not evidence of
    stasis -- the page was frozen, the players were not."""
    players = [r["players_tracked"] for r in case["history"]]
    if len(set(players)) != 1:
        return False
    for i in range(len(case["trophies"])):
        seen = {r["rates"][i] for r in case["history"]
                if r["rates"][i] is not None}
        if len(seen) > 1:
            return False
    return True


# The stale-page prior, calibrated on the one refresh the house
# documents (competition.yaml): the page frozen from 23:48Z, refreshed
# at 04:30Z, tracked count 638 -> 3,868 (x6.06) over about 4.7 hours
# of frozen page. Log growth over that window is ln(6.06)/4.7 = 0.383
# per hour. The prior says: the next refresh, expected within
# STALE_REFRESH_HOURS, reveals growth at that rate over the frozen
# span plus the time to the refresh; after the refresh, growth
# continues at the diminishing power-law pace of a launch week
# (doubling slows as the week goes on). Both legs are capped so a
# long horizon can never turn the prior into fantasy.
STALE_GAMMA = 0.383
STALE_REFRESH_HOURS = 6.0
STALE_JUMP_CAP = 12.0
STALE_TOTAL_CAP = 1000.0


def forecast_players(times: list, players: list, h_target: float,
                     horizon_hours: float, stale: bool,
                     freeze_hours: float):
    """Tracked players at the target, from the counts series.

    Returns (p_target, p_hi): the point forecast and a high leg.
    The growth leg is uncertain by about +-50% in log space -- the
    stale prior is calibrated on ONE documented refresh, and the
    log-time trend on one launch week -- so the high leg scales the
    forecast growth factor by 1.5 in log space, capped like the point
    forecast. p_hi exists so the rate spread can price the dilution
    risk: if players grow faster than the point forecast, every rate
    falls, and the lower tail must admit that.
    """
    p_last = float(max(players[-1], 1))
    if stale:
        refresh_h = min(horizon_hours, STALE_REFRESH_HOURS)
        jump = math.exp(STALE_GAMMA * (freeze_hours + refresh_h))
        jump = min(jump, STALE_JUMP_CAP)
        mult = jump
        rest = horizon_hours - refresh_h
        if rest > 0:
            mult *= (1.0 + rest / 24.0) ** 1.5
        p_target = p_last * min(mult, STALE_TOTAL_CAP)
        p_hi = p_last * min((p_target / p_last) ** 1.5, STALE_TOTAL_CAP)
        return p_target, p_hi
    # Log-time fit: log players against log hours since release, the
    # house curve baseline's own clock. A rising base stays rising;
    # a flat one stays flat; a falling fit is clipped to flat, because
    # a tracked count is cumulative in expectation.
    xs = [math.log1p(h) for h in times]
    ys = [math.log(max(p, 1)) for p in players]
    fit = least_squares(xs, ys)
    if fit is None:
        return p_last, p_last
    intercept, slope = fit
    slope = max(0.0, slope)
    p_target = math.exp(intercept + slope * math.log1p(h_target))
    p_target = max(p_last, min(p_target, p_last * STALE_TOTAL_CAP))
    p_hi = p_last * min((p_target / p_last) ** 1.5, STALE_TOTAL_CAP)
    return p_target, p_hi


# Progression classes. Each trophy is sorted into one class from its own
# tier, points and description text (never from a hardcoded trophy id).
# A class carries:
#   ceiling -- the share of tracked players who will ever earn the
#              trophy, once the game has been out long enough (percent).
#              It caps both cohorts: the existing players' share
#              pulls toward it, and the new joiners' tenure curve
#              saturates at it;
#   tau     -- time constant in days for the existing cohort's pull
#              toward the ceiling, and the fallback when a trophy's
#              own tenure curve cannot be inferred from its rate;
#   unc     -- extra spread for this class, in logit units per root day,
#              on top of the base spread: grinds are the least predictable.
#
# The ceilings are deliberately ordered by how much of the player base
# can reach them: an early story chapter is nearly universal, a final
# mission is a minority, a 250-stunt or 100%-completion grind is a few
# percent even at saturation. Story depth matters: chapter 1 is reached
# by far more players than chapter 4, and the final mission by fewer
# still, so chapters get their own ceilings instead of one story number.
CLASSES = {
    # The first story beats. Almost every player who keeps playing gets
    # here; the only question is how fast launch week carries them.
    "chapter_early": (78.0, 1.6, 0.10),
    "chapter_mid": (66.0, 2.0, 0.12),
    "chapter_late": (54.0, 2.4, 0.14),
    # Named story missions and the final mission. A mission is one
    # sitting's work, so these rise almost as fast as chapters, but the
    # final mission sits behind the whole story and stays rarer.
    "story_mission": (52.0, 2.2, 0.14),
    "story_final": (26.0, 3.0, 0.18),
    # Stranger / side-character mission strands: several missions each,
    # found off the main path, so slower and lower than story missions.
    "strand_final": (30.0, 3.6, 0.18),
    # One-off activities anyone can stumble into: an arcade game, a
    # bowling alley, a cocktail with a view, beating a rival at fishing.
    # Cheap to try, so they climb fast toward a middling ceiling.
    "activity_easy": (48.0, 1.8, 0.16),
    # Activities needing a little intent: winning races, discovering
    # vehicles, robbing shops and trucks, a wanted-level escape.
    "activity_mid": (22.0, 3.4, 0.20),
    # Long grinds: counts in the dozens or hundreds, every location, every
    # challenge, every vault. A week barely dents these; their ceilings
    # are low and their time constants are measured in weeks, not days.
    "grind_hard": (4.0, 21.0, 0.30),
    "grind_mid": (9.0, 12.0, 0.26),
    # The platinum: every other trophy first. Even at saturation only a
    # small slice of tracked players ever gets there, and in week one
    # almost nobody is close, however high the launch-night rate reads.
    "platinum": (1.6, 30.0, 0.30),
}

# Phrases that mark a grind, checked before the easier classes. The
# numbers in a description are the giveaway: a trophy that asks for 250
# of something, 100 of something, or "every" / "all" of a set is a
# grind no matter how friendly the rest of the sentence sounds.
_GRIND_WORDS = (
    "100% completion", "250 stunts", "gold medal", "100 locations",
    "50 challenges", "collectible card sets", "8 mission strands",
    "20 different races", "100 vehicles", "5 bank vaults",
    "every spot", "every location", "all of leonida",
    "laundering marked bills", "all garage properties",
    "10 vehicles to the", "3 shops back-to-back",
)


def classify(trophy: dict) -> str:
    """Sort one trophy into a progression class from its own fields."""
    tier = str(trophy.get("tier", "")).lower()
    text = str(trophy.get("description", "")).lower()
    name = str(trophy.get("name", "")).lower()
    if tier == "platinum":
        return "platinum"
    if any(word in text for word in _GRIND_WORDS):
        # The very longest grinds get the hardest class; the rest the mid one.
        if any(w in text for w in ("100% completion", "250 stunts", "gold medal")):
            return "grind_hard"
        return "grind_mid"
    if "chapter 1" in text:
        return "chapter_early"
    if "chapter 2" in text:
        return "chapter_early"
    if "chapter 3" in text:
        return "chapter_mid"
    if "chapter 4" in text:
        return "chapter_late"
    if "final mission" in text and "stranger" not in text:
        # The story's final mission, not a stranger strand's.
        if "andy" in text or "kenton" in text or "billy" in text \
                or "randy" in text or "tyrone" in text:
            return "strand_final"
        return "story_final"
    if "final mission for" in text:
        return "strand_final"
    if text.startswith("complete ") and ("mission" in text or "run" in text
                                         or "heist" in text or "play" in text
                                         or "takedown" in text or "operation" in text
                                         or "headliners" in text or "dwnfall" in text):
        return "story_mission"
    if text.startswith("complete "):
        # Any other "Complete ..." trophy is a named mission beat.
        return "story_mission"
    if any(word in text for word in ("win ", "beat ", "bowl", "arcade game",
                                     "cocktail", "drive-by", "melee")):
        return "activity_easy"
    if any(word in text for word in ("rob ", "escape", "wanted level",
                                     "discover", "explore", "earn $",
                                     "spend $", "purchase", "register",
                                     "take down", "hunt", "meditate",
                                     "smash", "enjoy", "follow-through")):
        return "activity_mid"
    # Unknown phrasing: a bronze one-off is the safest middle guess.
    if tier == "gold":
        return "story_final"
    if tier == "silver":
        return "activity_mid"
    return "activity_easy"


def tenure_tau(rate_pct: float, ceiling: float, age_hours: float):
    """Time constant (days) for a trophy's tenure curve, inferred
    from its own observed rate.

    Assume a player who has been around t hours holds the trophy
    with probability ceiling * (1 - exp(-t/tau)) -- a saturating
    tenure curve -- and that the tracked population's tenures are
    spread roughly evenly over [0, age], so the observed rate is the
    curve at the average tenure, age/2. Solving for tau:

        tau = -(age/2) / ln(1 - rate/ceiling)

    A trophy already held by many players after a few hours is fast
    (small tau); one held by almost nobody after the same time is
    slow (large tau). Returns None when the rate is already at or
    above the ceiling (the class time constant is used instead), and
    infinity when the rate is zero (nobody has it yet, so the new
    cohort is not expected to either).
    """
    if rate_pct <= 0:
        return float("inf")
    if rate_pct >= ceiling:
        return None
    share = rate_pct / ceiling
    tau_hours = -(age_hours / 2.0) / math.log(1.0 - share)
    return tau_hours / 24.0


def forecast_counts(trophy: dict, e_last: float, p_last: float,
                    p_target: float, age_hours: float,
                    horizon_days: float, horizon_hours: float):
    """Median earners at the target, in counts, from two cohorts.

    Existing cohort (the players already tracked): their share moves
    toward the class ceiling in logit space over the horizon -- the
    v1 pull, kept because it is gentle at low rates and strong near
    the ceiling. New cohort (the players who join between the last
    read and the target): their share follows the tenure curve over
    their average exposure, half the horizon, with the time constant
    inferred from the trophy's own observed rate. Earners never
    decrease: the result is floored at the last observed earners.
    A trophy at 100% is universal -- its earners stay equal to the
    tracked count, so its rate stays at 100%.

    Returns (earners_at_target, class_uncertainty, earners_existing):
    earners_existing is the existing cohort alone -- the earner floor
    if the players who join between the last read and the target earn
    nothing at all. It is the pessimistic earner leg the rate spread
    prices against the optimistic player leg.
    """
    ceiling, tau, unc = CLASSES[classify(trophy)]
    rate0 = 100.0 * e_last / p_last if p_last > 0 else 0.0
    if rate0 >= 99.99:
        return p_target, unc, p_target
    if rate0 > ceiling:
        ceiling = max(ceiling, min(0.92 * rate0, 95.0))
    # Existing cohort: logit pull toward the ceiling.
    start = logit(rate0)
    end = logit(ceiling)
    frac = 1.0 - math.exp(-horizon_days / tau)
    centre = end + (start - end) * (1.0 - frac)
    e_existing = p_last * sigmoid(centre) / 100.0
    # New cohort: tenure curve over half the horizon of exposure.
    # The inferred time constant is shrunk toward the class constant
    # by the earner count: with only a handful of earners, the
    # observed rate is mostly quantisation noise (at 638 tracked
    # players, 0.63% is four players -- four launch-night players
    # cannot show that a 100%-completion grind is fast), so a thin
    # history must not override the class prior. The weight is
    # e/(e+20) on the inferred value, in log space.
    tau_tenure = tenure_tau(rate0, ceiling, age_hours)
    if tau_tenure is None or math.isinf(tau_tenure):
        tau_tenure = tau
    else:
        weight = e_last / (e_last + 20.0)
        tau_tenure = math.exp(weight * math.log(tau_tenure)
                              + (1.0 - weight) * math.log(tau))
    exposure_days = (horizon_hours / 2.0) / 24.0
    share_new = ceiling * (1.0 - math.exp(-exposure_days / tau_tenure))
    e_new = max(0.0, p_target - p_last) * share_new / 100.0
    return max(e_existing + e_new, e_last), unc, max(e_existing, e_last)


def forecast_one(trophy: dict, e_last, p_last, p_target, p_hi,
                 age_hours, horizon_days, horizon_hours,
                 stale: bool) -> list:
    """Five quantiles for one trophy at the target horizon.

    The spread is ASYMMETRIC, because a refresh only dilutes rates:
    players join faster than earners, so the rate's lower tail is
    where the risk lives and the upper tail is bounded by ceilings
    and progression. The upside keeps v2's symmetric core spread.
    The downside adds a dilution tail, in logit units, built from
    the counts themselves:

        down_extra = log(p_hi / p_target) + log(e_target / e_floor)

    The first term is the player-growth uncertainty: if the tracked
    base grows to its high leg instead of its point forecast, every
    rate falls by that factor. The second term is the earner floor:
    if the players who join earn nothing, earners sit at the
    existing cohort alone. Both legs are pessimistic corners of the
    same event (a big refresh with slow earning), which is exactly
    what a 5% tail should price. The tail loads mostly on q05 -- a
    refresh is a jump event, not a gradual drift -- and half as much
    on q25; q75/q95 are untouched. down_extra is capped at 3.0 logit
    units (about a 20x combined dilution), past which the tail is
    already far beyond any plausible refresh. A universal trophy
    (median 100%) takes no dilution tail: its earners equal the
    tracked count by definition, whatever the count does.
    """
    e_target, unc, e_floor = forecast_counts(
        trophy, e_last, p_last, p_target,
        age_hours, horizon_days, horizon_hours)
    median = 100.0 * e_target / p_target if p_target > 0 else 0.01
    centre = logit(median)
    # Core spread, as in v2: widens with the horizon like a random
    # walk, plus the class uncertainty, plus a floor so even a
    # 90-minute forecast is not absurdly certain. A stale page adds
    # refresh uncertainty: the next refresh can move the counts a
    # long way at once.
    spread = 0.12 + (0.30 + unc) * math.sqrt(horizon_days)
    if stale:
        spread += 0.25
    spread = min(spread, 2.2)
    down_extra = 0.0
    if median < 99.99:
        down_extra = math.log(max(p_hi, p_target) / p_target)
        down_extra += math.log(max(e_target, e_floor) / max(e_floor, 1e-9))
        down_extra = min(down_extra, 3.0)
    offsets = [-1.645 * spread - down_extra,
               -0.674 * spread - 0.5 * down_extra,
               0.0,
               0.674 * spread,
               1.645 * spread]
    out = []
    prev = -1.0
    for off in offsets:
        value = round4(sigmoid(centre + off))
        value = max(prev, min(99.99, max(0.01, float(value))))
        out.append(value)
        prev = value
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    _, h_target, times, players, earners = case_series(case)
    last_read = parse_time(case["history"][-1]["at"])
    first_read = parse_time(case["history"][0]["at"])
    origin = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    # Age of the tracked population at the last read, and how long
    # the page has sat frozen: the two clocks the counts model needs.
    age_hours = hours_between(last_read, origin)
    freeze_hours = max(0.0, (last_read - first_read).total_seconds() / 3600.0)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    stale = history_is_stale(case)
    p_target, p_hi = forecast_players(times, players, h_target,
                                      horizon_hours, stale, freeze_hours)
    p_last = float(max(players[-1], 1))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        series = earners[i]
        if series:
            e_last = series[-1][1]
        else:
            # Never observed: start from a low prior for the class.
            ceiling = CLASSES[classify(trophy)][0]
            e_last = (0.5 if ceiling < 10 else 2.0) / 100.0 * p_last
        rows.append({"id": trophy["id"], "q": forecast_one(
            trophy, e_last, p_last, p_target, p_hi, age_hours,
            horizon_days, horizon_hours, stale)})
    return {"trophies": rows}


def validate(forecast: dict, case: dict, name: str) -> None:
    """Refuse to write an output the grader would score as invalid."""
    rows = forecast["trophies"]
    want = [t["id"] for t in case["trophies"]]
    got = [r["id"] for r in rows]
    if got != want:
        raise ValueError(f"{name}: rows {got} do not match trophies {want}")
    for row in rows:
        qs = row["q"]
        if len(qs) != 5 or set(row) != {"id", "q"}:
            raise ValueError(f"{name}: bad row {row}")
        for q in qs:
            if not isinstance(q, (int, float)) or not math.isfinite(q):
                raise ValueError(f"{name}: non-finite quantile {q}")
            if q < 0 or q > 100:
                raise ValueError(f"{name}: quantile out of range {q}")
        if any(b < a for a, b in zip(qs, qs[1:])):
            raise ValueError(f"{name}: quantiles not non-decreasing {qs}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        validate(forecast, case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: output over the 65,536 byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
