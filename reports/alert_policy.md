# Alert Policy Report

- Threshold: **0.9760**  (manual override)
- Cooldown: **15 min**
- Precision target (config): **0.80**

## Validation
- PR-AUC: **0.1070**
- Precision: **0.326**  |  Recall: **0.103**  |  Alerts: **227**

## Test (policy view)
- Precision: **0.111**  |  Recall: **0.002**  |  Alerts fired: **9**

## Alerts per day (TEST)

| date       |   alerts |
|:-----------|---------:|
| 2025-08-08 |        0 |
| 2025-08-09 |        2 |
| 2025-08-10 |        2 |
| 2025-08-11 |        5 |

## Top 50 alerts (TEST)

| timestamp                 | service_id   |    score | y_label_15min   |
|:--------------------------|:-------------|---------:|:----------------|
| 2025-08-09 07:17:00+00:00 | svc_046      | 0.987816 | False           |
| 2025-08-10 03:08:00+00:00 | svc_044      | 0.982329 | False           |
| 2025-08-11 17:49:00+00:00 | svc_025      | 0.980861 | True            |
| 2025-08-10 19:29:00+00:00 | svc_002      | 0.980365 | False           |
| 2025-08-11 03:32:00+00:00 | svc_007      | 0.978355 | False           |
| 2025-08-11 03:48:00+00:00 | svc_007      | 0.977273 | False           |
| 2025-08-11 06:22:00+00:00 | svc_039      | 0.977152 | False           |
| 2025-08-11 03:43:00+00:00 | svc_022      | 0.977046 | False           |
| 2025-08-09 04:29:00+00:00 | svc_012      | 0.976588 | False           |
