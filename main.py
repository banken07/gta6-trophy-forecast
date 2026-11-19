"""Forecast GTA VI trophy earn rates at a case's target, horizon-aware.

Reads every <id>.in in the input directory and writes <id>.out with one
row per trophy: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the
trophy's earn rate (percent of tracked players) at the target time.

Why not persistence: the house baseline forecasts the last observed rate
forever. That is fine over 90 minutes, but the final case spans about
4.5 days of launch week, over which three things move every rate:

  * players progress through the story and activities, so rare trophies
    climb toward a trophy-specific ceiling;
  * the tracked-player base explodes as launch week proceeds, diluting
    rates that start near 100% (a trophy every early player has is not
    one every later player has yet);
  * grind trophies (250 stunts, 100% completion, gold medals everywhere)
    barely move at all in a week, however fast the story trophies rise.

So each trophy is classified from its own tier, points and description
text into a progression class with a ceiling rate and a time constant,
and its median is pulled from the last observed rate toward that ceiling
in logit space, by a fraction that grows with the horizon. The spread
widens with the horizon and with how uncertain the class is. History
movement, when the case has any, nudges the drift; flat history (as in
the practice cases) leaves the class model in charge.

Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
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


# Progression classes. Each trophy is sorted into one class from its own
# tier, points and description text (never from a hardcoded trophy id).
# A class carries:
#   ceiling -- the share of tracked players who will ever earn the
#              trophy, once the game has been out long enough (percent);
#   tau     -- time constant in days: how fast the rate closes on the
#              ceiling while players are still in their first weeks;
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


def last_rate(history: list, i: int):
    """Last non-null rate for trophy i, or None when never observed."""
    for read in reversed(history):
        rate = read["rates"][i]
        if rate is not None:
            return rate
    return None


def history_drift(history: list, i: int) -> float:
    """Observed logit drift per day for trophy i, shrunk toward zero.

    Fits a line through the logit of the observed rates against read
    time. With flat history (every read identical, as in the practice
    cases) this is exactly zero. With real movement it is shrunk by a
    factor of four and capped, because a launch-night slope measured
    over hours is mostly noise and early-adopter rush, not a rate that
    holds for four more days.
    """
    points = []
    for read in history:
        rate = read["rates"][i]
        if rate is None:
            continue
        points.append((read["at"], logit(rate)))
    if len(points) < 2:
        return 0.0
    if len({t for t, _ in points}) < 2:
        return 0.0
    # Read times are ISO strings; sort order is chronological order,
    # and the spacing between reads is near-uniform, so index time is
    # a fair clock. Convert index steps to days from the read count and
    # the case's own span instead of parsing timestamps.
    n = len(points)
    mean_x = (n - 1) / 2.0
    mean_y = sum(y for _, y in points) / n
    num = sum((k - mean_x) * (y - mean_y) for k, (_, y) in enumerate(points))
    den = sum((k - mean_x) ** 2 for k in range(n))
    if den == 0:
        return 0.0
    # Slope per read step; a step is roughly half an hour in the early
    # recording, so scale to per-day and shrink hard.
    slope_per_day = (num / den) * 48.0
    shrunk = 0.25 * slope_per_day
    return max(-0.6, min(0.6, shrunk))


def forecast_one(trophy: dict, rate0, drift: float, horizon_days: float) -> list:
    """Five quantiles for one trophy at the target horizon."""
    ceiling, tau, unc = CLASSES[classify(trophy)]
    if rate0 is None:
        # Never observed: start from a low prior for the class.
        rate0 = 0.5 if ceiling < 10 else 2.0
    # A high observed rate is itself evidence the trophy is early or
    # easy: a trophy already above its class ceiling on launch night
    # (the tutorial-style first mission every early player has) keeps
    # a high ceiling, only diluted a little by late joiners. The cap
    # keeps this from ever claiming a trophy stays at 100%.
    if rate0 > ceiling:
        ceiling = max(ceiling, min(0.92 * rate0, 95.0))
    start = logit(rate0)
    end = logit(ceiling)
    # Fraction of the way to the ceiling covered by this horizon.
    # Short horizons barely move; a 4.5-day horizon covers most of the
    # way for story and easy activities, little for grinds. The same
    # formula pulls a rate above its ceiling back down: that is the
    # dilution of early 100%-style rates as the player base grows.
    frac = 1.0 - math.exp(-horizon_days / tau)
    centre = end + (start - end) * (1.0 - frac)
    # History drift, applied in logit space, capped so a noisy slope
    # can never drag the forecast far from the class model.
    centre += max(-0.5, min(0.5, drift * horizon_days))
    centre = max(logit(0.01), min(logit(99.99), centre))
    # Spread widens with the horizon like a random walk, plus the
    # class uncertainty, plus a floor so even a 90-minute forecast is
    # not absurdly certain. Grinds and platinum widen fastest.
    spread = 0.12 + (0.30 + unc) * math.sqrt(horizon_days)
    spread = min(spread, 2.2)
    out = []
    prev = -1.0
    for z in Z:
        value = round4(sigmoid(centre + z * spread))
        value = max(prev, min(99.99, max(0.01, float(value))))
        out.append(value)
        prev = value
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate0 = last_rate(case["history"], i)
        drift = history_drift(case["history"], i)
        rows.append({"id": trophy["id"], "q": forecast_one(trophy, rate0, drift, horizon_days)})
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
