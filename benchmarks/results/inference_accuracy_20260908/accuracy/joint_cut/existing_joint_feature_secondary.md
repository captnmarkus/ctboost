# Existing joint-feature option: secondary comparison

The existing joint-feature/grouped-eight/scalar-cut setting is a plausible candidate for a future task-family preset, but this study does not justify changing the multiclass default. It won only 4/8 comparisons with current CTBoost, with material regressions, and used previously inspected development rows. Both arms were fixed before fitting; this is a secondary comparison, not the original experimental joint-cut gate or an independent confirmation. No new fit, promotion, or confirmation scoring was performed for this analysis.

The exact existing settings are `multiclass_feature_test="joint"`, `feature_test="grouped"`, `feature_test_bins=8`, `feature_test_adjustment="none"`, and scalar cut scoring. Other controls are identical to the current-default arm, including diagonal leaves, one_output_per_tree, Bernoulli subsampling, seed, and resource limits. No new joint-cut feature is required to use these controls.

Against current CTBoost, the option won 4/8 tasks with 2.1743% median relative log-loss improvement; worst regression was 12.6871% on splice. Against archived default XGBoost, it won 6/8 with 1.7334% median relative improvement and 2.0496% worst regression. The tiny hiva_agnostic advantage over XGBoost (0.002595% relative) counts as a mathematical win and should not be interpreted as a meaningful practical improvement.

| Dataset | Current CTBoost log loss | Existing joint-feature log loss | XGBoost log loss | Gain vs CTBoost | Gain vs XGBoost |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIC | 0.389078579 | 0.422052379 | 0.418782483 | -8.4748% | -0.780810% |
| SDSS17 | 0.10543776 | 0.100832427 | 0.102768031 | +4.3678% | +1.883468% |
| anneal | 0.0373887626 | 0.00594981087 | 0.0241101526 | +84.0866% | +75.322384% |
| hiva_agnostic | 0.170093358 | 0.170126134 | 0.170130549 | -0.0193% | +0.002595% |
| maternal_health_risk | 0.715633809 | 0.595195551 | 0.607422591 | +16.8296% | +2.012938% |
| splice | 0.124749987 | 0.140577176 | 0.137753742 | -12.6871% | -2.049624% |
| students_dropout_and_academic_success | 0.566300746 | 0.572022153 | 0.616077769 | -1.0103% | +7.150983% |
| website_phishing | 0.245984761 | 0.200075314 | 0.203294013 | +18.6635% | +1.583273% |

Training is more expensive overall: geometric fit-time ratio 1.6203, 90th-percentile ratio 2.9731. The geometric physical-tree ratio is 1.5752; total trees across the eight saved models increase from 6,057 to 9,665. These model sizes can increase inference work, so the quality result should not be presented as an inference improvement. Prediction-preserving shared traversal could reduce the cost of either arm, independently of this training choice.

| Dataset | Default iterations / trees | Joint-feature iterations / trees | Default fit seconds | Joint-feature fit seconds |
| --- | ---: | ---: | ---: | ---: |
| MIC | 100 / 800 | 135 / 1080 | 30.624 | 53.186 |
| SDSS17 | 354 / 1062 | 91 / 273 | 21.468 | 7.701 |
| anneal | 389 / 1945 | 1000 / 5000 | 16.429 | 48.686 |
| hiva_agnostic | 1 / 3 | 11 / 33 | 93.612 | 131.715 |
| maternal_health_risk | 134 / 402 | 161 / 483 | 0.301 | 0.636 |
| splice | 237 / 711 | 181 / 543 | 19.526 | 28.386 |
| students_dropout_and_academic_success | 132 / 396 | 310 / 930 | 6.639 | 13.196 |
| website_phishing | 246 / 738 | 441 / 1323 | 1.218 | 3.648 |

A prospective combination with grouped-eight for binary/regression and joint-feature/grouped-eight for multiclass would be a new family-level selection informed by these development results. The separately successful 37-task binary/regression gate cannot validate its multiclass branch. That combined recipe would need a frozen assessment and an explicit regression guard before promotion; no combined score or Elo is inferred here. The original fold 7 remains sealed.

[Exact per-task records and aggregate statistics](existing_joint_feature_secondary.json), [original study and failed joint-cut gate](README.md), [frozen arm parameters](plan.json).
