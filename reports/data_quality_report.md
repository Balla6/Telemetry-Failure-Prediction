# Data Quality Report

**Total rows scanned:** 2,159,995
**Good rows to bronze:** 2,051,972
**Quarantined rows:** 108,023

## Quarantine reasons

| reason | count |
|---|---:|
| counts_mismatch | 108,020 |
| error_rate_mismatch | 88,294 |
| disk_io_mb_s_negative | 3 |

## Continuity summary (gaps/duplicates)

| partition (date/service) | gaps | duplicates |
|---|---:|---:|
| 2025-07-30/svc_009 | 1 | 0 |
| 2025-07-30/svc_013 | 1 | 0 |
| 2025-07-30/svc_030 | 1 | 0 |
| 2025-07-30/svc_033 | 1 | 0 |
| 2025-07-30/svc_047 | 1 | 0 |

## Labels & incidents

- Overall positive label rate (`y_label_15min`): **0.1535%**
- Incident starts per service (top 10):
  - svc_041: 10
  - svc_039: 9
  - svc_020: 8
  - svc_045: 7
  - svc_050: 7
  - svc_003: 6
  - svc_007: 6
  - svc_009: 6
  - svc_027: 6
  - svc_046: 6
