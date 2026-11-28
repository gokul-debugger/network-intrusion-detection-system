# Reference Dashboard

The deployment entrypoint is `app/modern_app.py` on `feature/modern-upgrade`.
It is a richer, interactive preview using the frozen **NSL-KDD v3** reference
models. It is not a completed newer-dataset edition or a production IDS.
The original `app/app.py`, models, notebooks, results, `main`, and
`legacy/nsl-kdd` edition are not replaced by this deployment.

## Workspace

- **Connections:** labelled reference examples, a complete 41-feature manual
  form, and bounded CSV batches. Binary and multiclass outputs remain independent;
  disagreements are visible. Unseen categories are disclosed. A session threshold
  override never changes the frozen benchmark policy or published metrics.
- **Traffic logs:** immediate synthetic traffic using reserved documentation IPs,
  plus common/combined Apache/Nginx log uploads. Rule scores are separate from
  ML predictions and do not automatically block requests. IP display masking is
  enabled by default. Actual unique-IP counts are computed before masking.
- **Model evidence:** balanced precision/recall/F1 reporting, class support,
  confusion matrices, the harder dependent subset, validation threshold tradeoffs,
  original source hashes, and known limitations.
- **Explanations:** global SHAP reference importance and on-demand single-connection
  explanations from the selected fitted model. Units, background, output class,
  and reconstruction error are explicit. Visitor-specific explanations are cached
  only in that visitor's session, with a four-entry limit.

## Data Handling

Uploads are UTF-8 text, at most 2 MiB. Connection batches contain exactly the
41 features and at most 500 rows. Labels/difficulty, duplicate columns, missing
values, invalid indicators/rates, non-finite values, and oversized counts are
rejected rather than silently imputed. Numeric counts have a declared input
ceiling of 1e12. Log uploads accept at most 5,000 nonblank lines, with malformed
rows counted and excluded. No archive uploads, URL fetches, server-path selection,
model uploads, packet capture, arbitrary file writes, or background log persistence
are exposed.
Log dates outside 1970-2100 are rejected as unsupported rather than allowed to
overflow timestamp conversion.

The public model resources are shared and immutable after load; prediction/SHAP
access is serialized with a resource lock and uses bounded worker settings.
User uploads/results are not placed in global Streamlit data caches. Session
clearing removes application-private values and resets upload widgets. These
controls do not promise forensic erasure from a browser/provider or production
security guarantees. Only upload logs you are authorized to process, after
removing sensitive content. CSV exports neutralize spreadsheet-formula prefixes.

## Release Assets

`dashboard_assets/reference_v1/` contains an explicit, checksum-verified release
bundle: the selected v3 pipelines, category-balanced demonstration rows, training
SHAP background, frozen evidence, global SHAP importance, and a display subsample
of the validation curve. The exact selected threshold is retained. Demo rows
are the first ten available test records per category, not an accuracy sample.
Deployment model copies use lossless xz compression. The source study hashes are
retained, learned weights are not quantized, and predictions are checked against
the original models after compression.

The application loads only repository-owned model bytes whose checksums match
the release manifest, with recorded dependency/Python compatibility checks.
Joblib artifacts can execute code and are trusted because they are owned release
assets, not because hashes make arbitrary model authors safe. Uploaded data never
reaches a model-deserialization path.

The bundle is exported once from the original completed v3 and SHAP packages,
without fitting or overwriting either:

```bash
python -B scripts/export_dashboard_bundle.py --trust-local-models
```

Existing output destinations are refused. Another bundle needs a fresh path and
a deliberately reviewed application reference. The original `Models/` directory
and ignored research studies remain untouched. This small release bundle is
tracked separately so a clean hosted checkout has working model assets.

## Local Launch

Use Python 3.12 and the deployment dependencies, separately from research setup:

```bash
python -m pip install -r app/requirements.txt
streamlit run app/modern_app.py
```

The runtime does not fetch data, fit models, regenerate notebooks, or require a
second terminal/simulator. Synthetic logs and examples are immediately available.
No secrets are required by this application. SHAP is computed only on request.

The included Codespaces configuration uses the same Python 3.12 runtime and
`app/requirements.txt`, installs the Linux OpenMP runtime, and forwards port 8501.
It retains the repository's XSRF/CORS protections. A Codespace preview is a
development environment, not a permanent public deployment.

## Hosting

The Streamlit Cloud app must select:

```text
Repository: gokul-debugger/network-intrusion-detection-system
Branch: feature/modern-upgrade
Entrypoint: app/modern_app.py
Python: 3.12
Dependencies: app/requirements.txt
```

The root `packages.txt` requests the OpenMP runtime used by LightGBM on Linux.

`.streamlit/config.toml` enables XSRF/CORS protections, limits uploads, disables
usage telemetry, and suppresses detailed public exception displays. The checked
Cloud preview is a separate deployment, not a change to any existing hosted app.
Its live URL is recorded in the README only after the app is actually verified.
Deployment of the default `main` edition remains separate from the unfinished
newer-data upgrade. PyPI distribution is a later packaging step, not hosting.

## Attribution and Limits

NSL-KDD source: [University of New Brunswick dataset description](https://www.unb.ca/cic/datasets/nsl.html).
Dataset paper: M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, *A detailed analysis
of the KDD CUP 99 data set*, IEEE CISDA 2009, DOI
[10.1109/CISDA.2009.5356528](https://doi.org/10.1109/CISDA.2009.5356528).
The displayed examples/background and models derive from that benchmark.
The repository's MIT software license is not a claim of ownership of source data.

Historical test inspection, overlapping feature vectors, benchmark age, imperfect
rare-class performance, uncalibrated model scores, SHAP independence assumptions,
and web-log rule false positives remain limitations. Neither a successful demo
nor passing CI establishes operational effectiveness against modern attacks.
