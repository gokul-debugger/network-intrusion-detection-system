# Upgrade and Deployment Roadmap

This separates software releases, evaluation experiments, and hosted apps. A newer
evaluation number is not automatically a better model or a new dashboard release.

## Version Management

Maintain two usable editions in this repository, not merely a recovery copy:

| Branch | Responsibility |
|---|---|
| `main` | The default edition visitors see; becomes the complete modern-data edition after its acceptance checks pass |
| `legacy/nsl-kdd` | The original NSL-KDD edition with its own runnable notebook/demo workflow |
| `feature/modern-upgrade` | Incremental upgrade work, pushed and tested before integration into `main` |

Until the full upgrade is ready, `main` still contains the current stable project.
The legacy branch starts at commit `b006b99818c6c7c742228f772e67e3b4890a3ce9`.
It can receive deliberate compatibility/security fixes without automatically
receiving the modern feature schema or training workflow. Do not merge every
modern change into the legacy edition or claim it supports the new dataset.

The modern edition will include LightGBM, SHAP, validation-selected binary
thresholds, harder NSL-KDD reference evaluation, a richer dashboard, and a
separately trained newer-dataset workflow. NSL-KDD reference results remain
labelled; they do not become results for the newer dataset.

Push tested stage commits to the upgrade branch. Integrate the completed edition
through a reviewed pull request with passing CI, then keep `main` as its stable
home. Release tags can additionally identify tested releases, but are not a
substitute for the two-edition branch structure.

For two simultaneous hosted versions, create separate Streamlit apps targeting
`main` and `legacy/nsl-kdd`, with distinct URLs, labels, dependencies, and model
bundles. These deployments are planned, not already published. Do not redirect a
public stable app to unfinished work.
Evaluation run directories remain write-once and separately versioned regardless
of Git branch. Every study's archived source and manifest identify its actual code.

## Stages

| Stage | Work | Status |
|---|---|---|
| 1 | Optional LightGBM, validation-only threshold/model selection, harder-subset evaluation, regression tests | Implemented and checked locally; see the separate v3 smoke-study note |
| 2 | SHAP explanations for the selected supported model family | Planned |
| 3 | Richer, deployment-safe Streamlit dashboard with evidence views and validated inputs | Planned |
| 4 | A modern dataset with its own schema, provenance, split, and models | Planned |
| 5 | Installable dashboard/CLI distribution and hosted stable app | Planned; publish only after packaging and hosting checks |

### Model Evidence

Choose model and threshold from training-data validation only, before scoring
either historical test file. Retain per-class metrics, false-positive/negative
counts, and the default-0.5 comparison. A validation F1 optimum is not a calibrated
risk score or a justified operational security policy. Do not promise improved
recall or accuracy before examining reproducible evidence.

`KDDTest-21` excludes the easiest difficulty-21 records from the same benchmark.
It is not an external replication and cannot be pooled with the full test set
as though its rows were new independent observations.

### Explainability

Use SHAP's appropriate explainer for the selected model, with a declared
background drawn from training data. Match transformed feature names and class
order. Report whether values explain probability or raw/log-odds output and
check reconstruction/additivity in that scale. If the selected model is not a
supported tree model, use the correct explainer rather than silently explaining
a different tree model. Group one-hot categories back to original features for
readability without pretending they are causal effects. Keep explanations bounded
and cached; do not compute an entire dataset on every dashboard interaction.

### Dashboard

Add validated connection input/CSV uploads, per-class result views, threshold
tradeoffs, explanations, and downloadable results. Keep the NSL-KDD classifier
distinct from the rule-based web-access-log monitor. Apache/Nginx requests do not
provide the NSL-KDD model's 41 connection measurements.

The current log tab reads/writes a user-entered server path. Before public hosting,
replace that control in hosted mode with bounded in-memory uploads and
session-local demonstration logs. Do not expose server filesystem browsing,
arbitrary file writes, untrusted model uploads, cross-session log sharing, or raw
visitor logs in repository history. Do not describe uploaded logs as automatic
monitoring of a visitor's website.

### Modern Data

Prefer one dataset initially, after reviewing its official source and permitted
use. UNSW-NB15 and CIC-IDS2017 have different schemas and data-collection
assumptions. Audit flow/session, time, source/destination, and duplicate leakage
before defining splits. Fit preprocessing only on each training partition and
include simple baselines and imbalance-aware reports. The existing NSL-KDD model
cannot directly score a CSV from a different dataset just because both concern IDS.

## PyPI and Local Launching

PyPI distributes Python packages; it does not host a running website. A useful
distribution can provide a command such as the **proposed**, not yet installed,
`intrusion-dashboard` to launch Streamlit on the user's own machine.

Before publishing:

- Choose and check an available distribution name and a unique import namespace;
  avoid publishing the generic current `src` directory as a public API.
- Add `pyproject.toml`, a console entry point, and package-resource-based paths
  that work from an installed wheel, not only from a checkout's current directory.
- Keep large training data and research-only dependencies out of the base install.
- Give model bundles explicit source/license/provenance, hashes, schema, training
  dependencies, and a declared decision policy. Do not silently retrain on launch.
- Preserve legacy artifacts; export any new release bundle to a separate location.
- Prefer loading a tested model bundle over asking users to upload pickle/joblib
  files. Such files may execute Python code when loaded.
- Test the wheel in a clean Python environment from outside this repository,
  including local dashboard startup, packaged assets, and prediction smoke checks.
- Use CI build checks and PyPI Trusted Publishing. Confirm the release name/version
  before publishing; publishing is not the same as saving a local checkpoint.

A local PyPI-installed app and Streamlit Cloud can share tested application code,
but cloud uploads, filesystem access, secrets, and session isolation need separate
deployment tests.

## Streamlit Cloud

A GitHub repository or a local `streamlit run` session is not automatically a
Cloud app. Each deployment selects a repository, ref, entrypoint, dependencies,
and sharing configuration. The repository owner's workspace lists its deployments.

The current repo intentionally ignores `Models/*.pkl` and generated report CSVs.
Its current app therefore lacks those runtime assets in a clean GitHub checkout.
Resolve that with an explicitly versioned release bundle, not by committing all
training data, all local reports, or private progress notes.

Deployment acceptance checks:

- Clean-checkout and clean-wheel tests succeed on the supported Python version.
- The released models load with pinned compatible numerical/scikit-learn versions.
- Demonstration assets and provenance are present without rerunning notebooks.
- Public input validation, upload limits, and multi-session isolation pass.
- No arbitrary host paths or shared log writes are exposed.
- The hosted app actually renders and predicts after deployment, with safe failure states.
- Its verified live URL is added to the README, repository About field, and profile.

Deploy other portfolio apps individually after the same dependency/artifact audit.
Projects hosted on Render or GitHub Pages will not appear as Streamlit Cloud apps.
Do not change private repository visibility just to add a public dashboard link.

References: [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow),
[Python packaging](https://packaging.python.org/en/latest/tutorials/packaging-projects/),
[Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[Streamlit workspaces](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/explore-your-workspace),
[SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
