# Term Deposit Targeting Assistant

A Streamlit app that scores a bank client on how likely they are to subscribe to a
term deposit, so limited call-centre time goes to the people most likely to say yes.

Built on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro et al., 2014).

## What it does

**Score a client** — enter a client's profile, the details of the current contact,
their previous campaign history, and the economic climate at the time of the call.
The app returns:

- an estimated probability of subscribing, on a radial gauge with the 50% decision
  threshold marked;
- a verdict (*Worth calling* / *Deprioritise*);
- the lift over the 11.3% base subscription rate, i.e. how much better than calling
  at random this client is;
- a panel showing the economic indicators behind the chosen scenario and the
  historical subscription rate under those conditions.

**About the model** — a model card: headline metrics, why F1 rather than accuracy
was used to select the model, the confusion matrix in business terms (wasted calls
vs missed revenue), every model that was tried, and grouped feature importances.


Python 3.11+ is recommended. Pinned versions are in
[requirements.txt](requirements.txt); `scikit-learn` in particular must match the
version the model was pickled with.

## Files

| File | Purpose |
| --- | --- |
| [streamlit_app.py](streamlit_app.py) | The app: layout, styling, both pages |
| [app_utils.py](app_utils.py) | Input options and labels, value ranges, economic scenarios, and the feature engineering that turns form inputs into a model-ready row |
| [bank_final_model.pkl](bank_final_model.pkl) | The trained, tuned decision tree |
| [app_metadata.json](app_metadata.json) | Evaluation results exported from the training notebook — metrics, confusion matrix, model comparison, feature importances, scenario rates, and the column order the model expects after one-hot encoding |
| [.streamlit/config.toml](.streamlit/config.toml) | Dark theme matching the app's styling |

The `.pkl` file and `app_metadata.json` are produced by the training notebook
(`MLDP Codes Submission.ipynb`, kept with the coursework submission, not in this
repo). If either is missing the app fails with a readable message instead
of a traceback.

## The model

A `DecisionTreeClassifier` tuned with `RandomizedSearchCV` (5-fold CV, scored on F1):
`max_depth=10`, `min_samples_leaf=10`, `min_samples_split=2`, `criterion='gini'` —
230 leaves across 55 features. Trained on 28,831 clients, evaluated on a held-out
12,357-client test set.

| Metric | Value |
| --- | --- |
| Accuracy | 0.898 |
| Precision | 0.614 |
| Recall | 0.262 |
| F1 | 0.367 |

Only 11.3% of clients subscribe, so a model that predicts "no" for everyone already
scores 88.7% accuracy — accuracy cannot separate a good model from a lazy one here.
Recall matters because a missed subscriber is lost revenue; precision matters because
a false alarm wastes a call slot. F1, their harmonic mean, is what the model was
selected and tuned on.

Of the calls the model recommends, 61% convert, against 11.3% calling at random —
about a 5.5× improvement in call efficiency. It is a prioritisation tool, not a
replacement for broader outreach.

### Feature engineering

- `pdays` (days since last contact) uses `999` as a sentinel for "never contacted".
  It is replaced by an explicit `pdays_never_contacted` flag and a `pdays_bucket`
  category (`never` / `recent` / `medium` / `long_ago`), and the raw column is dropped.
- Categorical fields are one-hot encoded and reindexed to the column order recorded
  in `app_metadata.json`, so the feature row always matches the training layout.
- `duration` (call length) is deliberately excluded — it is only known *after* a call
  happens, so including it would leak the answer.

### What drives the prediction

The economic indicators dominate: `nr.employed` alone accounts for ~52% of importance,
followed by `pdays_bucket`, `euribor3m`, and `cons.conf.idx`. In other words, *when*
the bank calls matters more than *who* it calls. This is why the app asks for an
economic scenario rather than hiding those five indicators — the three presets
(Downturn / Neutral / Boom) are real combinations taken from the dataset, and the
historical subscription rate under each ranges from 5.3% to 56.8%.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository, with
`streamlit_app.py` as the entry point. Everything it needs is committed, so no build
step is required.
