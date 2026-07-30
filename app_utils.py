import pandas as pd

CATEGORIES = {
    'job':         ['admin.', 'blue-collar', 'technician', 'services', 'management',
                    'retired', 'entrepreneur', 'self-employed', 'housemaid',
                    'unemployed', 'student', 'unknown'],
    'marital':     ['married', 'single', 'divorced', 'unknown'],
    'education':   ['university.degree', 'high.school', 'basic.9y',
                    'professional.course', 'basic.4y', 'basic.6y', 'unknown',
                    'illiterate'],
    'default':     ['no', 'unknown', 'yes'],
    'housing':     ['yes', 'no', 'unknown'],
    'loan':        ['no', 'yes', 'unknown'],
    'contact':     ['cellular', 'telephone'],
    'month':       ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep',
                    'mar', 'dec'],
    'day_of_week': ['thu', 'mon', 'wed', 'tue', 'fri'],
    'poutcome':    ['nonexistent', 'failure', 'success'],
}

## Human-readable labels. The raw dataset values are lowercase codes that mean
## little to a call-centre user, so every dropdown shows these instead.
LABELS = {
    'job': {
        'admin.': 'Administrative', 'blue-collar': 'Blue collar',
        'technician': 'Technician', 'services': 'Services',
        'management': 'Management', 'retired': 'Retired',
        'entrepreneur': 'Entrepreneur', 'self-employed': 'Self-employed',
        'housemaid': 'Housemaid', 'unemployed': 'Unemployed',
        'student': 'Student', 'unknown': 'Not recorded',
    },
    'marital': {
        'married': 'Married', 'single': 'Single', 'divorced': 'Divorced',
        'unknown': 'Not recorded',
    },
    'education': {
        'university.degree': 'University degree', 'high.school': 'High school',
        'basic.9y': 'Basic (9 years)', 'professional.course': 'Professional course',
        'basic.4y': 'Basic (4 years)', 'basic.6y': 'Basic (6 years)',
        'illiterate': 'Illiterate', 'unknown': 'Not recorded',
    },
    'default':     {'no': 'No', 'yes': 'Yes', 'unknown': 'Not recorded'},
    'housing':     {'yes': 'Yes', 'no': 'No', 'unknown': 'Not recorded'},
    'loan':        {'no': 'No', 'yes': 'Yes', 'unknown': 'Not recorded'},
    'contact':     {'cellular': 'Mobile phone', 'telephone': 'Landline'},
    'month': {
        'mar': 'March', 'apr': 'April', 'may': 'May', 'jun': 'June',
        'jul': 'July', 'aug': 'August', 'sep': 'September', 'oct': 'October',
        'nov': 'November', 'dec': 'December',
    },
    'day_of_week': {
        'mon': 'Monday', 'tue': 'Tuesday', 'wed': 'Wednesday',
        'thu': 'Thursday', 'fri': 'Friday',
    },
    'poutcome': {
        'nonexistent': 'No previous campaign', 'failure': 'Did not subscribe',
        'success': 'Subscribed',
    },
}

## Value ranges taken from the training data, used to bound the numeric inputs so
## a user cannot submit a client the model has never seen anything like.
RANGES = {
    'age':      {'min': 17, 'max': 98, 'default': 38},   ## median 38
    'campaign': {'min': 1,  'max': 56, 'default': 2},    ## median 2
    'previous': {'min': 0,  'max': 7,  'default': 0},    ## median 0
    'pdays':    {'min': 0,  'max': 27, 'default': 6},    ## median 6, excluding 999
}

ECON_FEATURES = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                 'euribor3m', 'nr.employed']

SCENARIOS = {
    'Downturn': {
        'emp.var.rate': -1.8, 'cons.price.idx': 93.876, 'cons.conf.idx': -40.0,
        'euribor3m': 0.682, 'nr.employed': 5008.7,
    },
    'Neutral': {
        'emp.var.rate': -1.8, 'cons.price.idx': 93.075, 'cons.conf.idx': -47.1,
        'euribor3m': 1.405, 'nr.employed': 5099.1,
    },
    'Boom': {
        'emp.var.rate': 1.4, 'cons.price.idx': 93.918, 'cons.conf.idx': -42.7,
        'euribor3m': 4.962, 'nr.employed': 5228.1,
    },
}

SCENARIO_HELP = {
    'Downturn': 'Low interest rates, shrinking employment. Term deposits look '
                'comparatively attractive.',
    'Neutral':  'Mid-cycle conditions, close to the dataset average.',
    'Boom':     'High interest rates, strong employment. Savers have better '
                'alternatives elsewhere.',
}

## Plain-English names for the economic indicators, used in the transparency panel.
ECON_LABELS = {
    'emp.var.rate':   'Employment variation rate',
    'cons.price.idx': 'Consumer price index',
    'cons.conf.idx':  'Consumer confidence index',
    'euribor3m':      '3-month Euribor rate (%)',
    'nr.employed':    'Number employed (thousands)',
}


def add_engineered_features(data):
    data = data.copy()

    data['pdays_never_contacted'] = (data['pdays'] == 999).astype(int)

    def bucket_pdays(v):
        if v == 999:
            return 'never'
        elif v <= 6:
            return 'recent'
        elif v <= 15:
            return 'medium'
        else:
            return 'long_ago'

    data['pdays_bucket'] = data['pdays'].apply(bucket_pdays)


    return data.drop(columns=['pdays'])


def build_feature_row(raw_inputs, model_columns):

    row = pd.DataFrame([raw_inputs])
    row = add_engineered_features(row)


    row = pd.get_dummies(row, drop_first=False)
    row = row.reindex(columns=model_columns, fill_value=0)

    if list(row.columns) != list(model_columns):
        raise ValueError(
            'Feature columns do not match the trained model. '
            'This usually means the notebook was re-run with different features '
            'but app_metadata.json was not refreshed.'
        )
    if row.isna().any().any():
        bad = row.columns[row.isna().any()].tolist()
        raise ValueError(f'Missing values in feature row: {bad}')

    return row
