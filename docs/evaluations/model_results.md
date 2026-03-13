# Model Evaluation Results

Last updated: 2026-03-13

## Notes
- Evaluation joins `predictions` with `player_gw_stats` on (player_id, gw).
- If a GW has no actuals in `player_gw_stats`, it cannot be evaluated.

---

## baseline_rollavg_v0
Eval range: GW23–GW27  
Overall MAE: 1.0105

Per-GW MAE:
- GW23: 1.111139
- GW24: 1.072319
- GW25: 0.995594
- GW26: 1.045267
- GW27: 0.952774
- GW28: 0.950061
- GW29: 0.948960

---

## baseline_rollavg_v1
Eval range: GW28–GW29  
Overall MAE: 1.1589

Per-GW MAE:
- GW28: 1.171117
- GW29: 1.146769

---

## ridge_rollform_v1
Eval range: GW28–GW29  
Overall MAE: 0.9818

Per-GW MAE:
- GW28: 0.973880
- GW29: 0.989796