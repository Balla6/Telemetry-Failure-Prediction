# Drift Report

- PSI thresholds: WARN ≥ **0.1**, ALERT ≥ **0.25**

**Score PSI (VAL→TEST)**: **0.0001**

**Feature drift summary**: ALERT=1  |  WARN=10  |  Total checked=94

## Top drifting features (VAL→TEST)

| feature                  |    psi | level   |
|:-------------------------|-------:|:--------|
| latency_p95_ms_mean_5m   | 0.0913 | ok      |
| latency_p95_ms           | 0.0809 | ok      |
| latency_p95_ms_mean_1m   | 0.0809 | ok      |
| latency_p95_ms_p95_1m    | 0.0809 | ok      |
| latency_p95_ms_p95_5m    | 0.0683 | ok      |
| latency_p95_ms_p95_30m   | 0.0645 | ok      |
| latency_p95_ms_p95_15m   | 0.0471 | ok      |
| latency_p50_ms_std_60m   | 0.0243 | ok      |
| request_count_mean_60m   | 0.011  | ok      |
| request_count_mean_30m   | 0.0109 | ok      |
| request_count_mean_15m   | 0.0107 | ok      |
| request_count_mean_5m    | 0.0106 | ok      |
| request_count            | 0.0103 | ok      |
| request_count_mean_1m    | 0.0103 | ok      |
| request_count_std_60m    | 0.0094 | ok      |
| latency_p50_ms_std_30m   | 0.0084 | ok      |
| error_rate_std_60m       | 0.0068 | ok      |
| latency_p95_ms_slope_60m | 0.0064 | ok      |
| request_count_std_30m    | 0.0057 | ok      |
| error_rate_std_30m       | 0.0044 | ok      |
| error_rate_p95_60m       | 0.0039 | ok      |
| cpu_pct_std_30m          | 0.0037 | ok      |
| cpu_pct_std_60m          | 0.0036 | ok      |
| error_rate_std_15m       | 0.0035 | ok      |
| request_count_std_15m    | 0.0033 | ok      |

## All features (sorted by severity)

| feature                  |     psi | level   |
|:-------------------------|--------:|:--------|
| latency_p95_ms_mean_5m   |  0.0913 | ok      |
| latency_p95_ms           |  0.0809 | ok      |
| latency_p95_ms_mean_1m   |  0.0809 | ok      |
| latency_p95_ms_p95_1m    |  0.0809 | ok      |
| latency_p95_ms_p95_5m    |  0.0683 | ok      |
| latency_p95_ms_p95_30m   |  0.0645 | ok      |
| latency_p95_ms_p95_15m   |  0.0471 | ok      |
| latency_p50_ms_std_60m   |  0.0243 | ok      |
| request_count_mean_60m   |  0.011  | ok      |
| request_count_mean_30m   |  0.0109 | ok      |
| request_count_mean_15m   |  0.0107 | ok      |
| request_count_mean_5m    |  0.0106 | ok      |
| request_count            |  0.0103 | ok      |
| request_count_mean_1m    |  0.0103 | ok      |
| request_count_std_60m    |  0.0094 | ok      |
| latency_p50_ms_std_30m   |  0.0084 | ok      |
| error_rate_std_60m       |  0.0068 | ok      |
| latency_p95_ms_slope_60m |  0.0064 | ok      |
| request_count_std_30m    |  0.0057 | ok      |
| error_rate_std_30m       |  0.0044 | ok      |
| error_rate_p95_60m       |  0.0039 | ok      |
| cpu_pct_std_30m          |  0.0037 | ok      |
| cpu_pct_std_60m          |  0.0036 | ok      |
| error_rate_std_15m       |  0.0035 | ok      |
| request_count_std_15m    |  0.0033 | ok      |
| error_rate_p95_30m       |  0.0032 | ok      |
| latency_p95_ms_std_60m   |  0.0032 | ok      |
| mem_pct_mean_60m         |  0.0027 | ok      |
| latency_p50_ms_std_15m   |  0.0022 | ok      |
| cpu_pct_std_15m          |  0.0022 | ok      |
| error_rate_p95_15m       |  0.0021 | ok      |
| latency_p95_ms_slope_30m |  0.002  | ok      |
| error_rate_mean_60m      |  0.0019 | ok      |
| error_rate_std_5m        |  0.0018 | ok      |
| net_out_mb_s             |  0.0017 | ok      |
| net_in_mb_s              |  0.0015 | ok      |
| disk_io_mb_s             |  0.0013 | ok      |
| mem_pct_std_60m          |  0.0012 | ok      |
| mem_pct_mean_30m         |  0.0011 | ok      |
| error_rate               |  0.001  | ok      |
| error_rate_mean_1m       |  0.001  | ok      |
| error_rate_p95_1m        |  0.001  | ok      |
| request_count_std_5m     |  0.0009 | ok      |
| error_rate_p95_5m        |  0.0009 | ok      |
| error_rate_delta_1m      |  0.0007 | ok      |
| cpu_pct_std_5m           |  0.0007 | ok      |
| latency_p95_ms_std_30m   |  0.0007 | ok      |
| cpu_pct_mean_60m         |  0.0007 | ok      |
| error_rate_slope_5m      |  0.0006 | ok      |
| error_rate_mean_30m      |  0.0006 | ok      |
| latency_p50_ms_std_5m    |  0.0005 | ok      |
| mem_pct_mean_15m         |  0.0005 | ok      |
| mem_pct_std_30m          |  0.0005 | ok      |
| cpu_pct_mean_15m         |  0.0004 | ok      |
| mem_pct_std_15m          |  0.0004 | ok      |
| error_rate_mean_5m       |  0.0003 | ok      |
| latency_p95_ms_std_5m    |  0.0003 | ok      |
| latency_p95_ms_std_15m   |  0.0003 | ok      |
| error_rate_slope_15m     |  0.0003 | ok      |
| cpu_pct_mean_30m         |  0.0003 | ok      |
| error_rate_slope_30m     |  0.0003 | ok      |
| cpu_pct                  |  0.0002 | ok      |
| cpu_pct_mean_1m          |  0.0002 | ok      |
| cpu_pct_delta_1m         |  0.0002 | ok      |
| cpu_pct_mean_5m          |  0.0002 | ok      |
| mem_pct_mean_5m          |  0.0002 | ok      |
| mem_pct_std_5m           |  0.0002 | ok      |
| latency_p95_ms_slope_15m |  0.0002 | ok      |
| mem_pct                  |  0.0001 | ok      |
| mem_pct_mean_1m          |  0.0001 | ok      |
| latency_p95_ms_delta_1m  |  0.0001 | ok      |
| request_count_delta_1m   |  0.0001 | ok      |
| latency_p95_ms_slope_5m  |  0.0001 | ok      |
| error_rate_mean_15m      |  0.0001 | ok      |
| error_rate_slope_60m     |  0.0001 | ok      |
| latency_p95_ms_std_1m    |  0      | ok      |
| latency_p50_ms_std_1m    |  0      | ok      |
| error_rate_std_1m        |  0      | ok      |
| cpu_pct_std_1m           |  0      | ok      |
| mem_pct_std_1m           |  0      | ok      |
| request_count_std_1m     |  0      | ok      |
| latency_p95_ms_slope_1m  |  0      | ok      |
| error_rate_slope_1m      |  0      | ok      |
| latency_p50_ms_mean_60m  |  0.1583 | WARN    |
| latency_p95_ms_mean_60m  |  0.1517 | WARN    |
| latency_p50_ms_mean_30m  |  0.1507 | WARN    |
| latency_p50_ms_mean_5m   |  0.1501 | WARN    |
| latency_p50_ms_mean_15m  |  0.1501 | WARN    |
| latency_p95_ms_mean_30m  |  0.1457 | WARN    |
| latency_p95_ms_mean_15m  |  0.1403 | WARN    |
| latency_p95_ms_p95_60m   |  0.1145 | WARN    |
| latency_p50_ms           |  0.1014 | WARN    |
| latency_p50_ms_mean_1m   |  0.1014 | WARN    |
| day_idx                  | 12.206  | ALERT   |
