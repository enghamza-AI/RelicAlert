# generate_data.py

import numpy as np
import pandas as pd
RNG = np.random.default_rng(42)
N_SAMPLES = 2000      
THEFT_RATE = 0.05     

N_THEFTS = int(N_SAMPLES * THEFT_RATE)
N_FALSE_ALARMS = N_SAMPLES - N_THEFTS


def generate_group(n, is_theft):
    
    if is_theft:
        hour_weights = np.array([
            5 if (h < 6 or h >= 22) else 0.3 for h in range(24)
        ])
        hour_weights = hour_weights / hour_weights.sum()
        hour_of_day = RNG.choice(np.arange(24), size=n, p=hour_weights)

        sensor_p = 0.75

        humidity_mean, humidity_std = 20, 8

    else:
      
        hour_weights = np.array([
            1 if (h < 6 or h >= 22) else 2 for h in range(24)
        ])
        hour_weights = hour_weights / hour_weights.sum()
        hour_of_day = RNG.choice(np.arange(24), size=n, p=hour_weights)

        sensor_p = 0.12

        humidity_mean, humidity_std = 45, 15

    after_hours = ((hour_of_day < 6) | (hour_of_day >= 22)).astype(int)

    
    motion_sensor = RNG.binomial(1, sensor_p, size=n)
    pressure_pad = RNG.binomial(1, sensor_p, size=n)
    glass_break = RNG.binomial(1, sensor_p, size=n)
    infrared_beam = RNG.binomial(1, sensor_p, size=n)

    humidity_spike = RNG.normal(humidity_mean, humidity_std, size=n)
    humidity_spike = np.clip(humidity_spike, 0, None)

    label = np.full(n, 1 if is_theft else 0)

    return pd.DataFrame({
        "motion_sensor": motion_sensor,
        "pressure_pad": pressure_pad,
        "glass_break": glass_break,
        "infrared_beam": infrared_beam,
        "hour_of_day": hour_of_day,
        "after_hours": after_hours,
        "humidity_spike": humidity_spike.round(1),
        "label": label,
    })



thefts = generate_group(N_THEFTS, is_theft=True)
false_alarms = generate_group(N_FALSE_ALARMS, is_theft=False)

df = pd.concat([thefts, false_alarms], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the final dataset
df.to_csv("data/alarm_logs.csv", index=False)

print(f"Saved {len(df)} rows to data/alarm_logs.csv")
print(f"Theft rate: {df['label'].mean():.2%}")
print(df.head())
