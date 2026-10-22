# Additional compression research, 16 October 2026

Research only. No new model, pruning operation, score, timing or experimental
selection is introduced by this note. The full-current-data CV protocol remains
frozen and active.

Recent work offers more relevant directions for tree inference than replacing
small machine-word operations with asymptotically faster large-integer
multiplication. Applicability still depends on the model and correctness
contract.

| Primary source | What it establishes | Implication for this study |
| --- | --- | --- |
| [Devos et al., ICML 2025, Level-wise Optimization and Pruning](https://proceedings.mlr.press/v267/devos25a.html) | LOP combines pruning/removal of trees with updated leaf predictions. The paper reports substantial compression on its experiments while mostly retaining predictive accuracy. | Potential future size/accuracy tradeoff, but it changes the fitted structure. It is outside the current fixed-topology correction experiment and requires a separate protocol. Published compression factors are not CTBoost latency measurements. |
| [Emine et al., AAAI 2025, Functionally-Identical Pruning](https://ojs.aaai.org/index.php/AAAI/article/view/33811) | The method seeks an ensemble with the same prediction function throughout the input space, using an optimization model and adversarially added points. | A possible future inference transformation. The abstract's functional guarantee does not demonstrate identical native float32 accumulation, categorical preprocessing, missing-value routes, probability clipping or serialization for CTBoost. Those are independent engineering obligations. |
| [Yajima and Matsui, ICML 2026, PINE](https://proceedings.mlr.press/v306/yajima26a.html) | PINE preserves prediction equivalence within a calibrated in-distribution region and controls that region with a conformal parameter. | Its regional guarantee differs from the current exact-prediction contract. It would also require a separately authorized calibration role; repeatedly used development data must not silently become that role. |
| [Liu and Mazumder, AISTATS 2023, ForestPrune](https://proceedings.mlr.press/v206/liu23h.html) | ForestPrune optimizes depth-layer pruning of existing ensembles. | Relevant for a future explicit compression tradeoff, but it alters tree topology and has not been tested here. |

These summaries use the authors' proceedings pages; implementation details and
full proofs have not been independently reproduced. No claim about practical
CTBoost speed, accuracy or formal equivalence follows from them.

Our engineering inference is that retaining the trained tree topology and
optimizing existing scalar leaf coefficients is currently the narrower next
accuracy hypothesis. The separately recorded free-intercept absolute-ridge
proposal changes the regularizer from a penalty on coefficient changes to a
penalty on final coefficients. It must pass its own numerical and native-fold
oracles before any real-data experiment. Compression can be considered after
the selected accuracy policy and its serving cost have been measured.
