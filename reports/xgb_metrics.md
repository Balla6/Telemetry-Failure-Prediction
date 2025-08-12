# XGBoost Metrics

- Features used: **98**
- Precision target (config): **0.80**

### Validation
- PR-AUC: **0.1070**  |  ROC-AUC: **0.6843**  |  thr: **0.9914**

```
              precision    recall  f1-score   support

           0      0.998     1.000     0.999    359281
           1      0.833     0.007     0.014       719

    accuracy                          0.998    360000
   macro avg      0.916     0.503     0.506    360000
weighted avg      0.998     0.998     0.997    360000

```

### Test
- PR-AUC: **0.0595**  |  ROC-AUC: **0.5770**  |  thr: **0.9914**

```
              precision    recall  f1-score   support

           0      0.998     1.000     0.999    287506
           1      0.000     0.000     0.000       494

    accuracy                          0.998    288000
   macro avg      0.499     0.500     0.500    288000
weighted avg      0.997     0.998     0.997    288000

```
