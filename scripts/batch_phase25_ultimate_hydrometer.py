#!/usr/bin/env python
"""Phase 25: The Ultimate Hydrometer — WDV Projection + Asymmetric Loss + Conditional Ensemble.

Current best: 13.67 (Phase 23 NM-optimized ensemble)
Final best: 13.61 (Conditional Ensemble t=150 w=20)

Key findings:
  - WDV Projection feature: inner product of spectrum with WDV as feature → -0.10 improvement
  - linear_tree: TERRIBLE (17-35 RMSE), don't use with preprocessed features
  - Asymmetric MSE (α=2.0) + WDV proj: 14.17 (best individual model innovation)
  - Conditional Ensemble: different weights for low/high moisture → -0.07 vs NM opt
  - Isotonic Regression: LEAKY in GroupKFold CV, do NOT use
  - Δx Generator (MLP): similar to WDV but less stable, adds diversity

Residual analysis (best model):
     0-30  : n= 675 RMSE=   7.68 bias=  +2.87
    30-60  : n= 263 RMSE=  13.26 bias=  +1.07
    60-100 : n= 171 RMSE=  15.72 bias=  -1.71
   100-150 : n= 140 RMSE=  16.04 bias=  -0.60
   150-200 : n=  55 RMSE=  27.65 bias=  -4.63
   200-300 : n=  18 RMSE=  48.12 bias= -31.21
   → 200+ region is the #1 bottleneck (bias=-31, 18 samples)
"""

# See batch_phase24_distribution_overdrive.py for the full implementation
# This file documents the Phase 25 experimental findings.
# Key experiments were run inline and results saved to runs/phase25_*/
