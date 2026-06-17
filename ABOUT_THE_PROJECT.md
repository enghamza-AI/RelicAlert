# 🏛 About This Project — Explained like you are a non tech person

This file exists for ONE reason: so you can understand, in plain English,
what we're building and why — before you open any code.

---

## What are we actually building?

Imagine a museum with alarm sensors (motion detectors, pressure pads under
the floor, glass-break sensors, infrared beams). Every time a sensor trips,
it logs an "alarm event". Most of these events are **nothing** — a draft of
air, a humidity change, a cleaner walking past. But occasionally, one of
these events is a **real theft in progress**.

I am building a small AI model that looks at an alarm event (which
sensors fired, what time it was, etc.) and predicts: **"Is this a real
theft, or just noise?"**

That's it. No fancy infrastructure, no live camera feeds, no real museum
data (none exists publicly) — just a clean, small, understandable example
of a classification problem that mirrors a real one.

---

## Why does this matter / what's the "problem" I'm solving?

Two extreme strategies are both bad:

- **Alarm goes off for EVERYTHING** → guards get exhausted, ignore alerts,
  trust in the system collapses. (Too many false positives.)
- **Alarm barely ever goes off** → looks great on paper ("we rarely have
  false alarms!") but a real thief strolls out with a painting. (Missed
  real thefts — the worst outcome.)

The real skill isn't "build a model" — it's **deciding how sensitive the
alarm should be**, and that decision should be backed by numbers, not gut
feeling. That's what this project teaches.

---

## What will I actually learn? (the concepts)

| Concept | In plain English |
|---|---|
| **Class imbalance** | Real thefts are rare (~5% of events). A model that always says "not a theft" looks 95% accurate but is useless. |
| **Confusion matrix** | A 2x2 scoreboard of how the model's predictions compare to reality: correct catches, false alarms, missed thefts, correct "all clears". |
| **Precision** | Of all the times we sounded the alarm, how often were we right? |
| **Recall** | Of all the real thefts that happened, how many did we actually catch? |
| **F1 score** | One number that punishes a model for being lopsided (great recall but terrible precision, or vice versa). |
| **Threshold tuning** | The model gives a probability ("73% chance this is a theft"). We choose the cutoff point where we decide to sound the alarm. Move that cutoff, and precision/recall trade off against each other. |
| **Operational cost** | F1 treats every mistake as equally bad. Real life doesn't — missing a theft is way more costly than one extra false alarm. We translate the confusion matrix into actual £ cost and find the threshold that saves the most money, which is often a DIFFERENT threshold than the one that maximizes F1. |

---

## What does each file do?

```
relic-alert/
├── README.md                  → Setup + how-to-run instructions (the "quick start")
├── ABOUT_THE_PROJECT.md       → You are here. Plain-English explainer.
├── requirements.txt           → List of Python packages to install
├── data/
│   └── generate_data.py       → Creates the fake (synthetic) alarm dataset
│   └── alarm_logs.csv         → The dataset itself (created when you run the script above)
├── build_notebook.py          → Internal helper that generated the notebook below (you don't need to run this)
└── relic_alert.ipynb          → THE MAIN FILE. Open this and work through it top to bottom.
```

### `data/generate_data.py`
There's no public "museum theft alarm logs" dataset anywhere — so this
script **invents one**. It creates ~2,000 fake alarm events, deliberately
designed so that real thefts happen mostly at night and trip multiple
sensors at once, while false alarms happen mostly during the day from a
single twitchy sensor. This mirrors how a real system would behave, without
needing any real museum's actual security data.

### `relic_alert.ipynb`
This is the file you actually open and work through. It's a Jupyter
notebook split into 6 sections:
1. Load the data, see how rare real thefts are
2. Train a simple model (Logistic Regression)
3. Read its confusion matrix, calculate precision/recall/F1
4. Tune the alarm's sensitivity threshold and watch the tradeoffs
5. Attach real £ costs to mistakes and find the cheapest threshold
6. Reflection questions to lock in what you learned

### `requirements.txt`
The handful of Python libraries needed (pandas, numpy, scikit-learn,
matplotlib, seaborn, jupyter). Install them with one command — see
`README.md`.

---
