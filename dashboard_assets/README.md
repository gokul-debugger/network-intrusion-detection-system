# Dashboard Release Assets

`reference_v1/` is the explicit deployment bundle for the NSL-KDD v3 preview.
It is not the original notebook model directory and does not replace research
studies. Model files are trusted repository-owned artifacts; never substitute
uploaded or third-party pickle/joblib files without an appropriate security review.

The manifest records input, study, explanation, model/data hashes, class order,
decision threshold, and runtime dependencies. A clean deployment verifies those
assets before loading models. Examples are category-balanced demonstration rows,
not a performance sample. See [the dashboard notes and dataset attribution](../docs/dashboard.md).
