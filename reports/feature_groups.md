# Structured preprocessing and feature groups

## Scope

Step 5 assigns all 65 training columns to exactly one group. The structured preprocessor currently transforms 39 raw feature columns. Twenty high-cardinality or text columns are deferred to later steps, and six columns are excluded.

## Structured features used now

### Numeric columns (18)

`host_listings_count`, `accommodates`, `bathrooms`, `bedrooms`, `beds`, `guests_included`, `minimum_nights`, `maximum_nights`, `number_of_reviews`, `review_scores_rating`, `review_scores_accuracy`, `review_scores_cleanliness`, `review_scores_checkin`, `review_scores_communication`, `review_scores_location`, `review_scores_value`, `calculated_host_listings_count`, `reviews_per_month`

Numeric values are parsed with `pd.to_numeric`. Missing values are replaced with the median learned from the current fold's training rows.

### Currency column (1)

`extra_people`

Dollar signs and commas are removed before numeric conversion. Unlike the official baseline, `$25.00` becomes `25.0` rather than `-1`.

### Percentage column (1)

`host_response_rate`

Percent signs and commas are removed before numeric conversion. `95%` becomes `95.0`.

### Date columns (3)

`host_since`, `first_review`, `last_review`

Each date is converted to the number of days since the fixed origin `2000-01-01`. Invalid or missing dates become missing numeric values and are then imputed from fold-training rows.

### Low-cardinality categorical columns (16)

`host_response_time`, `host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `neighbourhood_group_cleansed`, `state`, `market`, `country_code`, `country`, `property_type`, `room_type`, `bed_type`, `instant_bookable`, `cancellation_policy`, `require_guest_profile_picture`, `require_guest_phone_verification`

Missing categories become `__MISSING__`. Categories are learned from the current fold's training rows and one-hot encoded. Validation or test categories not seen during fitting are represented by zeros across that field's learned one-hot columns.

## Deferred features

### High-cardinality categorical columns (7)

`host_id`, `host_name`, `host_location`, `host_neighbourhood`, `neighbourhood_cleansed`, `city`, `zipcode`

These are not one-hot encoded in Step 5 because they would create a large, sparse matrix and require separate evidence for frequency, target, hashing, or model-native categorical handling.

### Free-text columns (11)

`name`, `summary`, `space`, `description`, `neighborhood_overview`, `notes`, `transit`, `access`, `interaction`, `house_rules`, `host_about`

These are reserved for the text-model step.

### Structured-text columns (2)

`amenities`, `host_verifications`

These contain collections encoded as strings and require dedicated parsing.

## Excluded columns

| Column | Reason |
|---|---|
| `id` | Listing identifier; used for stable fold assignment, not as a model feature |
| `price` | Prediction target |
| `experiences_offered` | Constant in the training data |
| `host_acceptance_rate` | 100% missing |
| `is_business_travel_ready` | Constant in the training data |
| `square_feet` | Approximately 99% missing |

## Leakage boundary

Parsing currency, percentages, and dates is stateless. The numeric medians and one-hot category vocabulary are learned only by `StructuredPreprocessor.fit(training_fold)`. The fitted object is then reused for the matching validation fold and test data through `transform`.

## Real-data integration check

For Fold 0, the preprocessor was fitted on 26,830 rows and transformed:

- 26,830 fold-training rows to a `26,830 × 114` matrix;
- 6,708 validation rows to a `6,708 × 114` matrix;
- 17,337 test rows to a `17,337 × 114` matrix.

All three matrices had identical columns and contained zero missing or infinite values. The exact one-hot width can differ between separately fitted folds because category vocabularies are learned independently from each fold's training rows.
