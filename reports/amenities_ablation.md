# Step 10 amenities ablation

## Parsing and leakage control

The supplied `amenities` values are brace-wrapped CSV-style collections, not Python sets. The parser uses CSV quoting rules so an amenity such as `"Wide clearance to shower, toilet"` remains one token. Tokens are trimmed, Unicode-casefolded, and deduplicated within each listing.

E013 adds only the distinct amenity count. E014 also selects tokens present in at least 1% of the current training fold, capped at 100 indicators. Validation rows never participate in frequency selection. Each fold independently selected 93 indicators, and the intersection and union across folds were both 93, showing that the selected vocabulary was stable. Supplied tokens such as `translation missing: ...` remain unfiltered; no external semantic rules are introduced.

Both experiments otherwise use the complete E009 preprocessing and model configuration.

## Results

| Experiment | Amenity features | Mean RMSE | Std RMSE | Change vs E009 | Folds improved | Decision |
|---|---|---:|---:|---:|---:|---|
| E009 | None | 87.154584 | 3.478562 | 0.000000 | — | Previous best |
| E013 | Distinct count | 87.179464 | 3.434460 | +0.024880 | 2/5 | Reject |
| E014 | Count + 93 frequent indicators | **85.345138** | **3.478555** | **-1.809446** | **5/5** | **Retain** |

Per-fold RMSE:

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| E009 | 83.840567 | 90.057432 | 85.354560 | 92.467790 | 84.052570 |
| E013 | 83.734136 | 90.441945 | 85.368477 | 92.125821 | 84.226938 |
| E014 | **81.900978** | **89.657663** | **83.410757** | **89.465037** | **82.291253** |

## Decision

Amenity count alone is rejected because its mean RMSE is slightly worse than E009. Frequent amenity identities produce a substantial and consistent gain: E014 improves all five folds, lowers mean RMSE by `1.809446`, and leaves fold dispersion effectively unchanged.

E014 is the new best tabular model. A complete rerun reproduced every fold score, iteration count, selected-indicator count, negative-prediction count, and OOF SHA-256 hash. Its one negative OOF prediction is left unchanged for honest experiment scoring; submission constraints will be evaluated later.
