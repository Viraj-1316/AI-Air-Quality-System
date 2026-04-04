# Viva Preparation: AI Air Quality System Feature Engineering

This document outlines the core intelligent feature engineering steps used in `model_script.py` to prepare raw sensor data for the machine learning model.

## 1. Cyclical Time Encoding (Sine & Cosine)
**The Problem:** Normal time is linear (a straight line of numbers). If we just feed the raw hours (0 to 23) to a Tree-Based model (like Random Forest), it mathematically calculates that 23 (11 PM) and 0 (Midnight) are 23 hours apart. This artificial gap causes the model to output aggressive, failed predictions ("spikes") exactly at midnight.
**The Solution:** Instead of a digital straight line, we map the time to a round clock face by calculating both its Sine and Cosine:
* **Sine Feature (X Coordinate):** `sin(2 * π * hour / 24)`
* **Cosine Feature (Y Coordinate):** `cos(2 * π * hour / 24)`
**The Benefit:** 11 PM and 1 AM are now plotted extremely close to each other on the top of the circle. The model understands smooth time transition, leading to highly accurate, uninterrupted predictions when the day rolls over.

## 2. 1-Hour Semantic Block (Top 95% Mean)
**What It Is:** The script aggregates the last 60 minutes of data into a semantic block (`summarize_window`). However, instead of taking a blanket average, it throws away the top 5% highest readings and averages the remaining 95%.
**The Benefit:** Removes artificial physical noise. For example, if a massive diesel truck idles next to the sensor for 10 seconds, the PM2.5 shoots to 500, but general city air quality is still fine. By removing the top 5%, we guarantee the AI receives a smooth, true representation of the hour, preventing false panic predictions.

## 3. Standard Deviation (`PM2.5_std_w15`)
**What It Is:** Calculates how sharply the data is bouncing up and down over a moving 15-minute window.
**The Benefit:** It measures **Volatility**. It tells the AI whether the environment is currently stable or completely chaotic (like a storm front or shifting wind patterns), directly impacting its prediction.

## 4. Rate of Change (`PM2.5_roc_5min`)
**What It Is:** Calculates the mathematical slope (momentum) over the *very last* 5 minutes of the hour.
**The Benefit:** It provides the model with **Current Trajectory**. 
For example, the 1-hour average might be a perfectly clean and low AQI. But if the 5-minute Rate of Change shows a steep positive slope at the exact end of the hour, the model knows an intense pollution event is starting *right now*. The 1-Hour Average gives the "baseline," while the 5-Minute RoC gives the "direction we are heading."

## 5. Interaction Terms (`pm_temp = PM2.5 * TEMP`)
**What It Is:** Multiplying the PM2.5 value by the Temperature value into a unified new feature.
**The Benefit:** Trees process data one column at a time. In the real physical world, pollution behaves much worse when it is very hot (sunlight bakes chemicals into smog). By multiplying them, we mathematically force the AI to directly recognize this combined heat-pollution relationship, drastically speeding up training and accuracy.

## 6. Lagged Memory (`AQI_lag1`)
**What It Is:** Using `prediction_state.json` to save the model's final prediction for the current hour, and feeding it back into the model exactly one hour later.
**The Benefit:** Gives the model "short-term memory." Air pollution acts essentially as a slow-moving cloud. Because it does not teleport randomly, knowing the precise AQI from 1 hour ago is the single strongest indicator of what the AQI will be right now.
