import json
import os

import joblib
import pandas as pd
import streamlit as st

from app_utils import (CATEGORIES, ECON_FEATURES, ECON_LABELS, LABELS, RANGES,
                       SCENARIO_HELP, SCENARIOS, build_feature_row)

MODEL_PATH = 'bank_final_model.pkl'
COLUMNS_PATH = 'model_columns.pkl'
METADATA_PATH = 'app_metadata.json'

## Verdict colours. Kept in Python as well as CSS because the gauge stroke and
## the bar fill are set inline from the prediction.
ACCENT = '#22d3ee'
POSITIVE = '#4ade80'
NEGATIVE = '#f87171'

st.set_page_config(page_title='Term Deposit Targeting Assistant',
                   page_icon='◆', layout='wide')


st.markdown("""
<style>
  :root {
    --canvas:   #070b14;
    --panel:    #0d1524;
    --panel-2:  #111c30;
    --line:     #1c2942;
    --accent:   #22d3ee;
    --positive: #4ade80;
    --negative: #f87171;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono: ui-monospace, "JetBrains Mono", "Cascadia Mono", "SF Mono",
            Consolas, monospace;
  }

  /* Faint grid texture on the canvas -- reads as an instrument backdrop without
     competing with the content. */
  .stApp {
    background-color: var(--canvas);
    background-image:
      linear-gradient(rgba(34,211,238,.030) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34,211,238,.030) 1px, transparent 1px);
    background-size: 46px 46px;
  }

  .block-container { padding-top: 2.2rem; max-width: 1180px; }

  /* --- Page header ------------------------------------------------------- */
  .eyebrow { font-family: var(--mono); font-size: .72rem; letter-spacing: .19em;
             text-transform: uppercase; color: var(--accent);
             margin-bottom: .55rem; }
  .app-title { font-size: 2.05rem; font-weight: 700; letter-spacing: -.02em;
               line-height: 1.12; color: #f1f5f9; margin-bottom: .4rem; }
  .app-sub { color: var(--muted); font-size: .95rem; max-width: 62ch;
             margin-bottom: 1.5rem; }

  /* --- Panels ------------------------------------------------------------ */
  .panel { border: 1px solid var(--line); border-radius: 4px;
           background: var(--panel); padding: 1.4rem 1.5rem; }
  .panel-label { font-family: var(--mono); font-size: .68rem; letter-spacing: .16em;
                 text-transform: uppercase; color: var(--muted);
                 margin-bottom: .85rem; display: flex;
                 justify-content: space-between; align-items: center; }

  /* --- Radial gauge ------------------------------------------------------ */
  /* The readout is overlaid rather than drawn as SVG <text>; see gauge_svg().
     The arc's circle centre is viewBox (100,100), which maps to 107px down the
     199px-tall element, so the readout is centred on that point. */
  .gauge { position: relative; width: 215px; height: 199px; flex: 0 0 auto; }
  /* See gauge_svg()'s docstring for why this is a CSS transition and not an
     SMIL <animate> -- Streamlit reruns patch this element's attributes rather
     than replacing the node, which an SMIL animation does not react to. */
  .gauge-arc { transition: stroke-dashoffset .75s cubic-bezier(.2,.8,.2,1),
                           stroke .3s ease; }
  .gauge-mid { position: absolute; left: 50%; top: 107px;
               transform: translate(-50%, -50%); text-align: center;
               width: 150px; }
  .gauge-value { font-family: var(--mono); font-size: 2.15rem; font-weight: 700;
                 line-height: 1; }
  .gauge-cap { font-family: var(--mono); font-size: .6rem; letter-spacing: .15em;
               text-transform: uppercase; color: var(--muted); margin-top: .45rem; }
  .gauge-foot { position: absolute; left: 0; right: 0; bottom: 4px;
                text-align: center; font-family: var(--mono); font-size: .58rem;
                letter-spacing: .14em; text-transform: uppercase; color: #475569; }

  /* --- Verdict ----------------------------------------------------------- */
  .verdict-wrap { display: flex; gap: 1.7rem; align-items: center;
                  flex-wrap: wrap; }
  .verdict-side { flex: 1 1 210px; min-width: 200px; }
  .verdict { font-family: var(--mono); font-size: 1.12rem; font-weight: 700;
             letter-spacing: .07em; text-transform: uppercase;
             display: flex; align-items: center; gap: .55rem; }
  .verdict-yes { color: var(--positive); }
  .verdict-no  { color: var(--negative); }
  .diamond { font-size: .78rem; }
  .verdict-note { color: var(--muted); font-size: .9rem; line-height: 1.5;
                  margin-top: .6rem; }

  .lift { margin-top: 1.1rem; border-left: 2px solid var(--accent);
          padding: .6rem 0 .6rem .85rem; }
  .lift-value { font-family: var(--mono); font-size: 1.75rem; font-weight: 700;
                color: var(--accent); line-height: 1; }
  .lift-note { color: var(--muted); font-size: .82rem; margin-top: .3rem; }

  /* --- Ticker strip (economic indicators) -------------------------------- */
  .tick { display: grid; grid-template-columns: 1fr auto; gap: .1rem .8rem;
          padding: .62rem 0; border-bottom: 1px solid var(--line); }
  .tick:last-of-type { border-bottom: none; }
  .tick-name { font-family: var(--mono); font-size: .74rem; letter-spacing: .06em;
               text-transform: uppercase; color: var(--muted); }
  .tick-value { font-family: var(--mono); font-size: 1rem; font-weight: 700;
                color: var(--text); text-align: right; }
  .tick-track { grid-column: 1 / -1; height: 3px; background: var(--line);
                border-radius: 2px; margin-top: .4rem; position: relative; }
  .tick-dot { position: absolute; top: 50%; width: 7px; height: 7px;
              border-radius: 50%; background: var(--accent);
              transform: translate(-50%, -50%);
              box-shadow: 0 0 8px rgba(34,211,238,.85); }

  /* --- Stat readouts ----------------------------------------------------- */
  .stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; }
  .stat { border: 1px solid var(--line); border-top: 2px solid var(--accent);
          border-radius: 3px; background: var(--panel); padding: .95rem 1rem; }
  .stat-k { font-family: var(--mono); font-size: .66rem; letter-spacing: .14em;
            text-transform: uppercase; color: var(--muted); }
  .stat-v { font-family: var(--mono); font-size: 1.85rem; font-weight: 700;
            color: var(--text); line-height: 1.15; margin-top: .35rem; }
  .stat-n { color: var(--muted); font-size: .78rem; margin-top: .2rem; }

  .highlight { border: 1px solid rgba(34,211,238,.35); border-radius: 3px;
               background: rgba(34,211,238,.06); padding: .85rem 1rem;
               margin-top: 1rem; }
  .highlight-v { font-family: var(--mono); font-size: 1.4rem; font-weight: 700;
                 color: var(--accent); line-height: 1; }
  .highlight-n { color: var(--muted); font-size: .82rem; margin-top: .35rem;
                 line-height: 1.5; }

  /* --- Feature importance bars ------------------------------------------- */
  .fi { display: grid; grid-template-columns: 168px 1fr 58px;
        align-items: center; gap: .8rem; padding: .34rem 0; }
  .fi-name { font-family: var(--mono); font-size: .78rem; color: var(--text);
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .fi-track { height: 9px; background: var(--panel-2); border-radius: 2px;
              overflow: hidden; }
  .fi-fill { height: 100%; background: var(--accent); border-radius: 2px;
             animation: grow .7s cubic-bezier(.2,.8,.2,1) both; }
  .fi-pct { font-family: var(--mono); font-size: .78rem; color: var(--muted);
            text-align: right; }
  @keyframes grow { from { transform: scaleX(0); transform-origin: left; } }

  /* --- Confusion matrix -------------------------------------------------- */
  .cm-grid { display: grid; grid-template-columns: 108px 1fr 1fr; gap: 5px;
             max-width: 440px; }
  .cm-cell { padding: .9rem .6rem; border-radius: 3px; text-align: center;
             font-family: var(--mono); font-size: 1.2rem; font-weight: 700; }
  .cm-head { padding: .45rem .4rem; font-family: var(--mono); font-size: .66rem;
             letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
             text-align: center; align-self: end; line-height: 1.4; }
  .cm-side { padding: .85rem .5rem; font-family: var(--mono); font-size: .66rem;
             letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
             display: flex; align-items: center; line-height: 1.4; }
  .cm-good { background: rgba(74,222,128,.10); border: 1px solid rgba(74,222,128,.28);
             color: var(--positive); }
  .cm-bad  { background: rgba(248,113,113,.09); border: 1px solid rgba(248,113,113,.26);
             color: var(--negative); }
  .cm-sub { display: block; font-size: .62rem; font-weight: 500; letter-spacing: .07em;
            text-transform: uppercase; color: var(--muted); margin-top: .3rem; }

  /* --- Comparison table -------------------------------------------------- */
  .cmp { width: 100%; border-collapse: collapse; font-family: var(--mono);
         font-size: .82rem; }
  .cmp th { text-align: right; padding: .5rem .7rem; font-size: .66rem;
            letter-spacing: .12em; text-transform: uppercase; color: var(--muted);
            border-bottom: 1px solid var(--line); font-weight: 600; }
  .cmp th:first-child, .cmp td:first-child { text-align: left; }
  .cmp td { text-align: right; padding: .55rem .7rem; color: var(--text);
            border-bottom: 1px solid rgba(28,41,66,.55); }
  .cmp tr.win td { background: rgba(34,211,238,.07); color: var(--accent);
                   font-weight: 700; }
  .cmp tr.win td:first-child { box-shadow: inset 2px 0 0 var(--accent); }

  /* --- Key/value input summary ------------------------------------------- */
  .kv { display: grid; grid-template-columns: 1fr auto; gap: .35rem .9rem;
        font-family: var(--mono); font-size: .78rem; }
  .kv-k { color: var(--muted); }
  .kv-v { color: var(--text); text-align: right; }

  /* --- Empty state ------------------------------------------------------- */
  .placeholder { border: 1px dashed var(--line); border-radius: 4px;
                 padding: 3.4rem 1.6rem; text-align: center; }
  .placeholder-t { font-family: var(--mono); font-size: .8rem; letter-spacing: .16em;
                   text-transform: uppercase; color: var(--muted);
                   margin-top: .9rem; }
  .placeholder-s { color: var(--muted); font-size: .88rem; margin-top: .5rem;
                   opacity: .75; }

  /* Pulls the "Score this client" button away from the dashed placeholder
     border above it -- Streamlit's own inter-element gap is too tight here. */
  div[data-testid="stButton"] { margin-top: 1.1rem; }

  .footnote { color: var(--muted); font-size: .82rem; }
  .hr { height: 1px; background: var(--line); margin: 1.9rem 0 1.5rem; }

  /* --- Streamlit chrome overrides ---------------------------------------- */
  [data-testid="stSidebar"] { background: var(--panel);
                              border-right: 1px solid var(--line); }
  [data-testid="stSidebar"] h4 { font-family: var(--mono); font-size: .68rem !important;
                                 letter-spacing: .16em; text-transform: uppercase;
                                 color: var(--accent); margin: 1.1rem 0 .3rem; }
  .brand { font-family: var(--mono); font-size: .8rem; letter-spacing: .13em;
           text-transform: uppercase; color: var(--text); }
  .brand-dot { color: var(--positive); }
  .brand-sub { font-family: var(--mono); font-size: .62rem; letter-spacing: .12em;
               text-transform: uppercase; color: var(--muted); margin-top: .25rem; }
  .stDataFrame { border: 1px solid var(--line); border-radius: 3px; }

  /* --- Narrow screens ----------------------------------------------------- */
  /* The instrument tiles and the importance bars both assume a wide column;
     below ~640px they are re-stacked rather than squeezed. */
  @media (max-width: 640px) {
    .stat-row { grid-template-columns: repeat(2, 1fr); }
    .stat-v { font-size: 1.5rem; }
    .fi { grid-template-columns: 108px 1fr 48px; gap: .55rem; }
    .fi-name { font-size: .7rem; }
    .app-title { font-size: 1.6rem; }
    .verdict-wrap { justify-content: center; }
    .cmp { font-size: .74rem; }
    .cmp th, .cmp td { padding: .45rem .4rem; }
  }
  /* The comparison table has five columns of numbers and cannot usefully
     reflow, so it scrolls inside its own box instead of widening the page. */
  .cmp-scroll { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, encoding='utf-8') as f:
        return json.load(f)


def fail(message, detail):
    """Show a readable error instead of a traceback, then stop."""
    st.error(f'**{message}**\n\n{detail}')
    st.stop()


missing = [p for p in (MODEL_PATH, COLUMNS_PATH, METADATA_PATH)
           if not os.path.exists(p)]
if missing:
    fail('Required files are missing: ' + ', '.join(f'`{m}`' for m in missing),
         'Run every cell in `MLDP Codes Submission.ipynb` to produce the model '
         'files, then run `python make_app_metadata.py` to produce the metadata.')

try:
    model, model_columns = load_model()
    meta = load_metadata()
except Exception as exc:                                  ## noqa: BLE001
    fail('Could not load the model.', f'`{type(exc).__name__}: {exc}`')

FINAL = meta['final_model']
BUSINESS = meta['business']


GAUGE_TRACK = 'M 43.4 156.6 A 80 80 0 1 1 156.6 156.6'


def gauge_svg(prob, colour):
    offset = 100 - prob * 100
    return f"""
    <div class="gauge">
      <svg viewBox="0 0 200 185" width="215" height="199" role="img"
           aria-label="Estimated chance of subscribing: {prob:.1%}">
        <path d="{GAUGE_TRACK}" fill="none" stroke="#111c30" stroke-width="15"
              stroke-linecap="round"/>
        <path class="gauge-arc" d="{GAUGE_TRACK}" fill="none" stroke="{colour}"
              stroke-width="15" stroke-linecap="round" pathLength="100"
              stroke-dasharray="100"
              style="stroke-dashoffset:{offset:.2f};
                     filter: drop-shadow(0 0 7px {colour}66)"/>
        <!-- Decision threshold marker at 50%, i.e. top dead centre -->
        <path d="M 100 9 L 100 31" stroke="#64748b" stroke-width="2"/>
      </svg>
      <div class="gauge-mid">
        <div class="gauge-value" style="color:{colour}">{prob:.1%}</div>
        <div class="gauge-cap">Chance to subscribe</div>
      </div>
      <div class="gauge-foot">Threshold 50%</div>
    </div>"""


def ticker_strip(scenario):
    rows = []
    for feature in ECON_FEATURES:
        value = SCENARIOS[scenario][feature]
        across = [SCENARIOS[s][feature] for s in SCENARIOS]
        low, high = min(across), max(across)
        ## Guard against a flat indicator: if every preset shares a value there
        ## is no meaningful position, so the dot is centred.
        pos = 50.0 if high == low else (value - low) / (high - low) * 100
        rows.append(
            f'<div class="tick">'
            f'<div class="tick-name">{ECON_LABELS[feature]}</div>'
            f'<div class="tick-value">{value:,.3f}</div>'
            f'<div class="tick-track"><div class="tick-dot" '
            f'style="left:{pos:.1f}%"></div></div>'
            f'</div>')
    return ''.join(rows)


def importance_bars(items):
    """Feature importance as a terminal-style bar list.

    Bars are scaled against the largest value rather than 1.0, otherwise every
    bar after nr.employed would be too short to compare.
    """
    top = max(i['importance'] for i in items)
    rows = []
    for n, item in enumerate(items):
        width = item['importance'] / top * 100
        ## Fade down the list so the ranking reads even without the numbers.
        opacity = max(0.34, 1 - n * 0.075)
        rows.append(
            f'<div class="fi">'
            f'<div class="fi-name">{item["feature"]}</div>'
            f'<div class="fi-track"><div class="fi-fill" '
            f'style="width:{width:.1f}%;opacity:{opacity:.2f};'
            f'animation-delay:{n * 0.045:.2f}s"></div></div>'
            f'<div class="fi-pct">{item["importance"]:.1%}</div>'
            f'</div>')
    return ''.join(rows)


## sklearn class names -> the display names used in app_metadata.json, so the
## deployed model's row can be highlighted without hard-coding which one won.
MODEL_DISPLAY_NAMES = {
    'DecisionTreeClassifier': 'Decision Tree',
    'GradientBoostingClassifier': 'Gradient Boosted Tree',
    'RandomForestClassifier': 'Random Forest',
    'LogisticRegression': 'Logistic Regression',
}


def deployed_row_name(rows):
    display = MODEL_DISPLAY_NAMES.get(FINAL['name'])
    if display is None:
        return None
    return next((r['model'] for r in rows if r['model'].startswith(display)), None)


def comparison_table(rows, winner):
    """Model comparison, with the deployed model's row picked out."""
    body = []
    for row in rows:
        cls = ' class="win"' if row['model'] == winner else ''
        body.append(
            f'<tr{cls}><td>{row["model"]}</td>'
            f'<td>{row["accuracy"]:.3f}</td><td>{row["precision"]:.3f}</td>'
            f'<td>{row["recall"]:.3f}</td><td>{row["f1"]:.3f}</td></tr>')
    return (
        '<div class="cmp-scroll"><table class="cmp"><thead><tr><th>Model</th>'
        '<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
        f'</thead><tbody>{"".join(body)}</tbody></table></div>')


def stat_tiles(tiles):
    """A row of instrument-style readouts: (label, value, note) each."""
    cells = ''.join(
        f'<div class="stat"><div class="stat-k">{k}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-n">{n}</div></div>'
        for k, v, n in tiles)
    return f'<div class="stat-row">{cells}</div>'


def header(eyebrow, title, sub):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>'
                f'<div class="app-title">{title}</div>'
                f'<div class="app-sub">{sub}</div>', unsafe_allow_html=True)


st.sidebar.markdown(
    '<div class="brand"><span class="brand-dot">◆</span> Targeting Assistant</div>'
    '<div class="brand-sub">Term deposit propensity · v1.0</div>',
    unsafe_allow_html=True)
st.sidebar.markdown('<div class="hr" style="margin:1.1rem 0 .6rem"></div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio('Page', ['Score a client', 'About the model'],
                        label_visibility='collapsed')


def page_predict():
    header('Live scoring', 'Should we call this client?',
           'Enter a client\'s details on the left. The model estimates how likely '
           'they are to subscribe to a term deposit, so limited call-centre time '
           'goes to the people most likely to say yes.')


    def pick(field, label, help=None):
        return st.selectbox(
            label, CATEGORIES[field],
            format_func=lambda v: LABELS[field].get(v, v), help=help)

    sb = st.sidebar
    sb.markdown('#### Client')
    age = sb.slider('Age', RANGES['age']['min'], RANGES['age']['max'],
                    RANGES['age']['default'])
    with sb:
        job = pick('job', 'Occupation')
        marital = pick('marital', 'Marital status')
        education = pick('education', 'Education')
        default = pick('default', 'Has credit in default?')
        housing = pick('housing', 'Has a housing loan?')
        loan = pick('loan', 'Has a personal loan?')

    sb.markdown('#### This contact')
    with sb:
        contact = pick('contact', 'Contact method')
        month = pick('month', 'Month of contact')
        day_of_week = pick('day_of_week', 'Day of contact')
    campaign = sb.number_input(
        'Calls made in this campaign', RANGES['campaign']['min'],
        RANGES['campaign']['max'], RANGES['campaign']['default'],
        help='Including this one.')

    sb.markdown('#### Previous history')
    previous = sb.number_input(
        'Contacts in earlier campaigns', RANGES['previous']['min'],
        RANGES['previous']['max'], RANGES['previous']['default'])
    with sb:
        poutcome = pick('poutcome', 'Outcome of previous campaign')

    ## pdays uses 999 as a sentinel for "never contacted". The checkbox makes that
    ## explicit rather than expecting a user to type 999.
    never_contacted = sb.checkbox('Never contacted before this campaign',
                                  value=True)
    days_since = sb.number_input(
        'Days since last contact', RANGES['pdays']['min'], RANGES['pdays']['max'],
        RANGES['pdays']['default'], disabled=never_contacted,
        help='Only applies if the client was contacted in an earlier campaign.')
    pdays = 999 if never_contacted else days_since

    sb.markdown('#### Economic climate')
    scenario = sb.selectbox(
        'Conditions at time of call', list(SCENARIOS.keys()), index=1,
        help='Economy-wide conditions, not client details. This is the strongest '
             'driver in the model.')
    sb.caption(SCENARIO_HELP[scenario])

    raw = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous,
        'poutcome': poutcome,
        **SCENARIOS[scenario],
    }


    first_load = 'interacted' not in st.session_state
    if first_load:
        st.session_state['interacted'] = True

    left, right = st.columns([1.15, 1], gap='large')

    with left:
        if first_load:
            st.markdown(
                '<div class="placeholder">'
                '<div style="font-size:2.1rem;line-height:1;color:#22d3ee">◆</div>'
                '<div class="placeholder-t">Awaiting client</div>'
                '<div class="placeholder-s">Adjust any detail on the left, '
                'or press the button below.</div>'
                '</div>', unsafe_allow_html=True)
            st.button('Score this client', type='primary')
        else:
            try:
                row = build_feature_row(raw, model_columns)
                prob = float(model.predict_proba(row)[0, 1])
            except ValueError as exc:
                fail('Could not score this client.', str(exc))
            except Exception as exc:                      ## noqa: BLE001
                fail('Unexpected error while scoring.',
                     f'`{type(exc).__name__}: {exc}`')

            call = prob >= 0.5
            colour = POSITIVE if call else NEGATIVE
            lift = prob / BUSINESS['base_rate']

            st.markdown(f"""
            <div class="panel">
              <div class="panel-label"><span>Prediction</span>
                <span style="color:{colour}">● live</span></div>
              <div class="verdict-wrap">
                {gauge_svg(prob, colour)}
                <div class="verdict-side">
                  <div class="verdict {'verdict-yes' if call else 'verdict-no'}">
                    <span class="diamond">◆</span>
                    {'Worth calling' if call else 'Deprioritise'}
                  </div>
                  <div class="verdict-note">
                    {'This client is in the segment the model expects to convert.'
                     if call else
                     'Better to spend this call slot on a higher-scoring client.'}
                  </div>
                  <div class="lift">
                    <div class="lift-value">{lift:.1f}×</div>
                    <div class="lift-note">the {BUSINESS['base_rate']:.1%} average
                      subscription rate across all clients</div>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="footnote" style="margin-top:.9rem">Across the whole '
                f"test set this model converts <strong>{FINAL['precision']:.0%}"
                f'</strong> of the calls it recommends (vs '
                f"{BUSINESS['base_rate']:.1%} calling at random) and catches "
                f"<strong>{FINAL['recall']:.0%}</strong> of all subscribers. It is "
                'a prioritisation tool, not a replacement for broader outreach.'
                '</div>', unsafe_allow_html=True)

    with right:
        rate = meta['scenario_rates'][scenario]
        st.markdown(
            f'<div class="panel">'
            f'<div class="panel-label"><span>Economic climate</span>'
            f'<span style="color:#22d3ee">{scenario.upper()}</span></div>'
            f'{ticker_strip(scenario)}'
            f'<div class="highlight">'
            f'<div class="highlight-v">{rate["rate"]:.1%}</div>'
            f'<div class="highlight-n">of the {rate["n"]:,} clients historically '
            f'contacted in these conditions subscribed. Each preset is a real '
            f'combination taken from the dataset.</div></div>'
            f'</div>', unsafe_allow_html=True)

        with st.expander('Full input summary'):
            fields = [
                ('Age', str(age)), ('Occupation', LABELS['job'][job]),
                ('Marital status', LABELS['marital'][marital]),
                ('Education', LABELS['education'][education]),
                ('Credit in default', LABELS['default'][default]),
                ('Housing loan', LABELS['housing'][housing]),
                ('Personal loan', LABELS['loan'][loan]),
                ('Contact method', LABELS['contact'][contact]),
                ('Month', LABELS['month'][month]),
                ('Day', LABELS['day_of_week'][day_of_week]),
                ('Calls this campaign', str(campaign)),
                ('Earlier contacts', str(previous)),
                ('Previous outcome', LABELS['poutcome'][poutcome]),
                ('Days since last contact',
                 'Never contacted' if never_contacted else str(days_since)),
            ]
            st.markdown(
                '<div class="kv">' + ''.join(
                    f'<div class="kv-k">{k}</div><div class="kv-v">{v}</div>'
                    for k, v in fields) + '</div>', unsafe_allow_html=True)



def page_about():
    header('Model card', 'How this model works',
           'Evidence behind the predictions, measured on a '
           f"{FINAL['n_test']:,}-client test set the model never saw during "
           'training.')

    st.markdown(stat_tiles([
        ('F1 score', f"{FINAL['f1']:.3f}", 'Selection metric'),
        ('Precision', f"{FINAL['precision']:.1%}", 'Of flagged, subscribe'),
        ('Recall', f"{FINAL['recall']:.1%}", 'Of subscribers, found'),
        ('Efficiency', f"{BUSINESS['lift']:.1f}×", 'vs calling at random'),
    ]), unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap='large')

    with left:
        st.markdown('##### Why F1, and not accuracy')
        st.markdown(
            f"Only **{BUSINESS['base_rate']:.1%}** of clients subscribe, so a "
            'model that predicts "no" for everyone scores about 88.7% accuracy '
            'while being completely useless. Accuracy cannot tell a good model '
            'from a lazy one here.\n\n'
            '**Recall** matters because a missed subscriber is lost revenue — that '
            'client never gets called. **Precision** matters because a false alarm '
            'wastes a call slot on someone who was never going to convert. The '
            'bank cares about both, so **F1**, their harmonic mean, is used to '
            'select and tune the model.')

        st.markdown('##### Model')
        st.markdown(
            f"- **{FINAL['name']}**, tuned with `RandomizedSearchCV` "
            "(5-fold cross-validation, scored on F1)\n"
            f"- `max_depth={FINAL['max_depth']}`, "
            f"`min_samples_leaf={FINAL['min_samples_leaf']}`, "
            f"`min_samples_split={FINAL['min_samples_split']}`, "
            f"`criterion='{FINAL['criterion']}'`\n"
            f"- {FINAL['n_leaves']:,} leaves across {FINAL['n_features']} features\n"
            f"- Trained on {FINAL['n_train']:,} clients, tested on "
            f"{FINAL['n_test']:,}\n"
            '- `duration` (call length) is excluded — it is only known *after* a '
            'call happens, so using it would leak the answer')

    with right:
        st.markdown('##### Where it gets things right and wrong')
        tn, fp = meta['confusion_matrix'][0]
        fn, tp = meta['confusion_matrix'][1]
        st.markdown(f"""
        <div class="cm-grid">
          <div></div>
          <div class="cm-head">Model says<br>don't call</div>
          <div class="cm-head">Model says<br>call</div>
          <div class="cm-side">Did not<br>subscribe</div>
          <div class="cm-cell cm-good">{tn:,}<span class="cm-sub">correctly skipped</span></div>
          <div class="cm-cell cm-bad">{fp:,}<span class="cm-sub">wasted calls</span></div>
          <div class="cm-side">Subscribed</div>
          <div class="cm-cell cm-bad">{fn:,}<span class="cm-sub">missed revenue</span></div>
          <div class="cm-cell cm-good">{tp:,}<span class="cm-sub">won</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(
            f'<div class="footnote" style="margin-top:.9rem">Of the {tp + fp:,} '
            f'calls the model recommends, {tp:,} convert. It misses {fn:,} '
            'subscribers — the accepted cost of keeping the call list short.'
            '</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('##### Every model that was tried')
    st.caption('All trained on the same data and scored on the same test set. '
               'The baseline simply predicts the majority class.')
    st.markdown(
        comparison_table(meta['comparison'], deployed_row_name(meta['comparison'])),
        unsafe_allow_html=True)
    st.markdown(
        '<div class="footnote" style="margin-top:.9rem">Note how little Accuracy '
        'separates these models — every one sits near the 0.886 baseline. F1 is '
        'what actually distinguishes them.</div>', unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('##### What drives the prediction')
    st.caption('One-hot columns are grouped back under the original feature they '
               'came from.')
    st.markdown(importance_bars(meta['feature_importance']),
                unsafe_allow_html=True)
    st.markdown(
        '<div class="footnote" style="margin-top:1rem">The economic indicators '
        'dominate, which matches the exploratory analysis: <em>when</em> the bank '
        'calls matters more than <em>who</em> it calls.</div>',
        unsafe_allow_html=True)


if page == 'Score a client':
    page_predict()
else:
    page_about()

st.sidebar.markdown('<div class="hr" style="margin:1.4rem 0 .8rem"></div>',
                    unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="footnote" style="font-size:.75rem"><br>Data: UCI Bank Marketing '
    '(Moro et al., 2014).</div>', unsafe_allow_html=True)
