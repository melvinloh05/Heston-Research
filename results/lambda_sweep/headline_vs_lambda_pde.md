# Headline contrast vs lambda_pde (CODE_AUDIT_2026-08-20 action 2)

rung3 - standard_pinn at the confirmatory cell. Negative CVaR diff = rung3 better;
`rel` is the pre-registered relative improvement. `pooled` is the registered
statistic (paired bootstrap over CRN paths); `seed` is its mandatory companion
(does the effect replicate across training runs). SENSITIVITY ANALYSIS — the
registered lambda_pde is 0.01 and no registered verdict reads any other rung.

## misspec cell

| lambda_pde | tc=0: rel (pooled CI) | tc=0.01: rel (pooled CI) | tc=0.02: rel (pooled CI) | seed-robust @tc=0 |
|---|---|---|---|---|
| 0 | +0.0295 [-0.067, -0.059] | -0.0307 [+0.138, +0.154] | -0.0576 [+0.440, +0.463] | yes |
| 0.0001 | +0.0741 [-0.173, -0.160] | -0.0560 [+0.249, +0.269] | -0.0936 [+0.696, +0.720] | yes |
| 0.0003 | +0.1003 [-0.239, -0.224] | -0.0535 [+0.236, +0.259] | -0.1013 [+0.745, +0.772] | yes |
| 0.001 | +0.1536 [-0.387, -0.368] | -0.0528 [+0.230, +0.259] | -0.1341 [+0.959, +0.991] | yes |
| 0.003 | +0.2208 [-0.605, -0.580] | -0.0374 [+0.159, +0.194] | -0.1623 [+1.132, +1.170] | yes |
| 0.01 **(registered)** | +0.3152 [-0.980, -0.944] | +0.0002 [-0.024, +0.024] *ns* | -0.1912 [+1.295, +1.345] | yes |

## in_model cell

| lambda_pde | tc=0: rel (pooled CI) | tc=0.01: rel (pooled CI) | tc=0.02: rel (pooled CI) | seed-robust @tc=0 |
|---|---|---|---|---|
| 0 | +0.0247 [-0.049, -0.040] | -0.0290 [+0.130, +0.144] | -0.0404 [+0.313, +0.332] | yes |
| 0.0001 | +0.0373 [-0.074, -0.061] | -0.0646 [+0.285, +0.302] | -0.0762 [+0.576, +0.598] | yes |
| 0.0003 | +0.0579 [-0.114, -0.099] | -0.0684 [+0.299, +0.318] | -0.0840 [+0.628, +0.652] | yes |
| 0.001 | +0.1066 [-0.216, -0.196] | -0.0838 [+0.361, +0.385] | -0.1154 [+0.839, +0.869] | yes |
| 0.003 | +0.1720 [-0.373, -0.348] | -0.0885 [+0.377, +0.407] | -0.1457 [+1.031, +1.068] | yes |
| 0.01 **(registered)** | +0.2722 [-0.665, -0.631] | -0.0690 [+0.292, +0.330] | -0.1724 [+1.187, +1.234] | yes |

