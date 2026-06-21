# 🏛 RelicAlert

*"Fewer false alarms, zero missing masterpieces — tuned with F1."*

A tiny machine learning mini-project: optimize a simulated museum theft
alarm system to catch real thefts while minimizing false alerts — and
learn **confusion matrices**, **F1 score**, and **operational cost
tradeoffs** along the way.

> 👋 New here? Read **[ABOUT_THE_PROJECT.md](ABOUT_THE_PROJECT.md)** first —
> it explains what this project is and what you'll learn in plain English.

---

## What's inside

```
relic-alert/
├── README.md                 ← you are here
├── ABOUT_THE_PROJECT.md      ← plain-English explainer (read this first!)
├── requirements.txt          ← Python dependencies
├── data/
│   ├── generate_data.py      ← generates the synthetic dataset
│   └── alarm_logs.csv        ← the dataset (created after you run the script)
├── build_notebook.py  SKIPPED       ← internal script that built the notebook (no need to run)
└── relic_alert.ipynb RENAME - training.py        ← 🎯 the main notebook — open and work through this
```

---



## What I'll learn

1. **Confusion matrix** — TP / FP / FN / TN, and what each means for a
   museum alarm (a missed theft vs. a wasted guard dispatch)
2. **Precision & Recall** — the tradeoff between "don't cry wolf" and
   "don't miss a real theft"
3. **F1 score** — why it's a better headline number than accuracy when
   one class (theft) is rare
4. **Threshold tuning** — sliding the alarm's sensitivity and watching
   precision/recall/F1 shift
5. **Operational cost modeling** — converting the confusion matrix into
   real £ cost and finding the threshold that minimizes cost (which is
   often a *different* threshold than the one that maximizes F1 — this is
   the key "roadmap reinforcement" insight of the project)

---

## Dataset

There's no public museum-theft-alarm dataset, so `data/generate_data.py`
generates a small synthetic one (~2,000 rows) with a realistic ~95/5
false-alarm/theft split. Real thefts are simulated to correlate with
after-hours timing and multiple sensors firing together; false alarms
correlate with daytime hours and humidity-related sensor noise. 

---

## Notes

- The model is intentionally simple (Logistic Regression) — the point of
  this project is the **evaluation and decision-making layer** (confusion
  matrix → F1 → cost), not building a fancy model.
