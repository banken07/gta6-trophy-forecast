# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/banken07/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/banken07/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Horizon-aware forecast of how many GTA VI players earn each trophy in week one. Instead of repeating the last observed earn rate, each trophy is classified from its own tier, points and description text into a progression class (story chapters, story missions, stranger strands, easy and mid activities, hard and mid grinds, platinum), each with a ceiling rate and a time constant. The forecast median moves from the last observed rate toward that ceiling in logit space by a fraction that grows with the horizon, so a 90-minute forecast barely moves while a 4.5-day forecast lets story trophies climb, grind trophies stay low, and near-100% early rates dilute as the tracked-player base grows. Quantile spreads widen with the horizon and with class uncertainty. Pure Python 3.13 standard library, offline, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=xe3m7rxf&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
