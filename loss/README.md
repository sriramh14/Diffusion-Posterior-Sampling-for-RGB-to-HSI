# Loss package

The package separates every training objective and validation metric into its
own file.

| File | Function | Use |
|---|---|---|
| `mrae.py` | `mrae`, `MRAELoss` | Relative spectral reconstruction loss |
| `l1.py` | `l1_loss` | Absolute reconstruction loss |
| `mse.py` | `mse_loss` | Squared reconstruction loss |
| `reconstruction.py` | `reconstruction_loss` | Selects MRAE, L1, or MSE from the main config |
| `prior.py` | `prior_l1_loss`, `prior_kd_loss` | Stage-2 compact-prior supervision |
| `rmse.py` | `rmse` | Validation RMSE |
| `psnr.py` | `psnr` | Validation PSNR |
| `sam.py` | `sam` | Spectral Angle Mapper in degrees by default |
| `ssim.py` | `ssim` | Local band-wise SSIM |
| `metrics.py` | `compute_metrics` | Runs all validation metrics |
| `utils.py` | `prepare_metric_tensors` | Range-aware metric preprocessing |

`main.py` imports only:

```python
from loss import (
    compute_metrics,
    prior_kd_loss,
    prior_l1_loss,
    reconstruction_loss,
)
```
