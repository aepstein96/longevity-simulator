import json

import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.scenarios import LongevityScenario
import src.causes as causes

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    title="Longevity Simulator",
)
server = app.server

# --- Data Loading (for initial setup) ---
# Bucket labels and display order are defined in data/CDC/cause_categories.csv;
# the actual buckets present at runtime come from the columns of cause_fractions_total.csv.
CAUSE_LABELS = causes.bucket_labels()
CAUSE_ORDER = causes.bucket_order()

try:
    initial_causes = pd.read_csv('data/CDC/cause_fractions_total.csv', nrows=1).columns.tolist()
    available_causes = [c for c in initial_causes if c not in ['age_years', 'Unknown']]
except Exception:
    available_causes = [c for c in CAUSE_ORDER if c != 'COVID-19']

ordered_causes = [c for c in CAUSE_ORDER if c in available_causes] + \
                 [c for c in available_causes if c not in CAUSE_ORDER]

COLOR_BASELINE = "#6c757d"
COLOR_INTERVENTION = "#1f77b4"
COLOR_GAIN = "#2ca02c"
COLOR_LOSS = "#dc3545"  # red — shown when the scenario lives shorter than today

MEDIAN_CAP_AGE = 1000  # if the extrapolated median exceeds this, show ">1000"

# Gompertz formula colors: µ (mortality), A (waves), B (seawall)
COLOR_MU = "#6f42c1"   # purple
COLOR_A = "#d62728"    # red — waves / insults
COLOR_B = "#2a9d8f"    # teal — seawall / resistance

MILESTONE_AGES = [65, 80, 90, 100]
DEFAULT_SLOW_START = 25


# --- Helpers ---

def _survival_at(surv_series, age):
    """Survival probability at a given age (0 if beyond the series)."""
    if surv_series is None or len(surv_series) == 0:
        return 0.0
    if age in surv_series.index:
        return float(surv_series.loc[age])
    if age > surv_series.index.max():
        return float(surv_series.iloc[-1])
    return float(surv_series.reindex([age], method='nearest').iloc[0])


def _conditional_survival(surv, anchor_age):
    """Return S(t | alive at anchor_age) for t >= anchor_age."""
    if anchor_age is None or anchor_age <= 0:
        return surv
    anchor = _survival_at(surv, anchor_age)
    if anchor <= 0:
        return surv
    cond = (surv / anchor).clip(upper=1.0)
    return cond[cond.index >= anchor_age]


def _median_from_survival(surv, mort=None, cap=MEDIAN_CAP_AGE):
    """Find the age where survival drops below 0.5.

    If `surv` doesn't cross 50% within its range, extrapolate using the final
    mortality rate held constant: S(t) = S_last · (1−mx_last)^(t − age_last),
    solve for S(t) = 0.5. Returns float('inf') if the result exceeds `cap`.
    """
    below = surv[surv < 0.5]
    if len(below):
        return float(below.index.min())
    if mort is None or len(surv) == 0:
        return float('nan')
    last_age = float(surv.index[-1])
    last_S = float(surv.iloc[-1])
    last_mx = float(mort.iloc[-1])
    if last_S <= 0 or not (0 < last_mx < 1):
        return float('nan')
    extra = np.log(0.5 / last_S) / np.log(1 - last_mx)
    median = last_age + extra
    return float('inf') if median > cap else median


def _possessive(name):
    if not name:
        return ""
    nm = name.strip()
    if not nm:
        return ""
    return f"{nm}'" if nm.endswith('s') else f"{nm}'s"


def info_badge(tip_id, text):
    return html.Span([
        html.I(className="bi bi-info-circle ms-1", id=tip_id,
               style={"color": "#adb5bd", "cursor": "help"}),
        dbc.Tooltip(text, target=tip_id, placement="right"),
    ])


def _card_title(rest, poss):
    """Title with optional possessive name prefix (KPI cards and chart titles)."""
    return f"{poss} {rest}" if poss else rest.capitalize()


def _comparison_body(today_text, scenario_text, delta_text=None,
                     scenario_color=COLOR_INTERVENTION, show_scenario=True):
    rows = [
        html.Div([
            html.Span("Today", className="text-muted small"),
            html.Span(today_text, className="float-end"),
        ], className="mb-1"),
    ]
    if show_scenario:
        rows.append(html.Div([
            html.Span("With your changes", className="small",
                      style={"color": scenario_color}),
            html.Span(scenario_text, className="float-end fw-bold",
                      style={"color": scenario_color}),
        ]))
        if delta_text:
            rows.append(html.Div(delta_text, className="small text-muted mt-1"))
    return rows


def _chance_at_age(target_age, base_full, int_full, user_age):
    """Probability (in %) of reaching target_age, conditional on user_age if given.

    Returns (None, None) if user is already past the milestone.
    """
    if user_age is not None and user_age >= target_age:
        return None, None

    def _conditional_pct(surv):
        target = _survival_at(surv, target_age)
        if user_age is not None and user_age > 0:
            anchor = _survival_at(surv, user_age)
            if anchor > 0:
                return min(max(target / anchor * 100, 0), 100)
        return min(max(target * 100, 0), 100)

    return _conditional_pct(base_full), _conditional_pct(int_full)


def _reach_body(base_pct, int_pct, show_scenario=True,
                scenario_color=COLOR_INTERVENTION):
    if base_pct is None:
        return [html.Div("Already past this age", className="small text-muted")]
    delta = None
    if show_scenario and base_pct > 0.05:
        ratio = int_pct / base_pct
        if ratio >= 1.05:
            delta = f"{ratio:.1f}× more likely"
        elif ratio <= 0.95:
            delta = f"{ratio:.2f}× as likely"
    scen_txt = f"{int_pct:.1f}%" if int_pct is not None else "—"
    return _comparison_body(f"{base_pct:.1f}%", scen_txt, delta,
                            scenario_color=scenario_color,
                            show_scenario=show_scenario)


def _hover_surv(ages, surv_pct, mort_series):
    mx_at_age = mort_series.reindex(ages).to_numpy()
    out = []
    for age, pct, mx in zip(ages, surv_pct, mx_at_age):
        if np.isfinite(mx):
            mort_txt = f"{mx * 1000:.1f} in 1,000 die this year"
        else:
            mort_txt = ""
        out.append(
            f"<b>Age {int(age)}</b><br>"
            f"{pct:.1f}% still alive<br>"
            f"<span style='color:#6c757d'>{mort_txt}</span>"
        )
    return out


def _fmt_gompertz(x):
    if x is None or not np.isfinite(x):
        return html.Span("—")
    if 0.001 <= abs(x) < 1000:
        return html.Span(f"{x:.4f}")
    exp = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10 ** exp
    return html.Span([f"{mantissa:.2f} × 10", html.Sup(str(exp))])


def _gompertz_rows(base_val, int_val, show_intervention=True,
                   intervention_color=COLOR_INTERVENTION):
    rows = [
        html.Div([
            html.Span("Baseline: ", className="text-muted"),
            _fmt_gompertz(base_val),
        ]),
    ]
    if show_intervention:
        rows.append(html.Div([
            html.Span("Adjusted: ", style={"color": intervention_color}),
            html.Span(_fmt_gompertz(int_val),
                      style={"color": intervention_color, "fontWeight": 600}),
        ]))
    return rows


# --- Sidebar ---

sidebar = html.Div(
    [
        html.H2("Longevity Simulator", className="h4"),
        html.P(
            "See how curing diseases or slowing aging could change expected lifespan for a U.S. resident.",
            className="text-muted small",
        ),
        html.Hr(),

        html.H6("About you", className="mt-2"),
        html.Div("All fields optional.", className="small text-muted mb-2"),

        dbc.Input(id='user-name', type='text', placeholder='Name',
                  value='', maxLength=40, className="mb-2"),

        dbc.InputGroup([
            dbc.InputGroupText("Age"),
            dbc.Input(id='user-age', type='number',
                      min=0, max=100, step=1, value=40),
        ], className="mb-2"),

        dcc.Dropdown(
            id='sex-dropdown',
            options=[
                {'label': 'Other/Not specified', 'value': 'All'},
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'},
            ],
            value=None,
            clearable=True,
            placeholder='Sex (optional)',
            className="mb-1",
        ),

        html.Hr(),

        html.Div([
            html.H6("Cure these diseases", className="d-inline"),
            info_badge(
                "tip-causes",
                "Click a disease to remove all deaths from that cause — an upper-bound estimate of what a perfect cure would do. Click again to undo.",
            ),
        ], className="mt-2"),
        html.Div(
            [
                dbc.Button(
                    CAUSE_LABELS.get(c, c),
                    id={'type': 'cause-btn', 'cause': c},
                    color='primary',
                    outline=True,
                    size='sm',
                    n_clicks=0,
                    className='me-2 mb-2 rounded-pill',
                )
                for c in ordered_causes
            ],
            id='cause-buttons',
            className='d-flex flex-wrap mt-2',
        ),
        dcc.Store(id='selected-causes', data=[]),

        html.Hr(),

        html.Div([
            html.H6("Slow down your aging", className="d-inline"),
            info_badge(
                "tip-rate",
                "100% = normal aging. 50% = age half as fast. 0% = aging frozen. "
                "The intervention starts at your age.",
            ),
        ]),
        html.Div(id='aging-rate-label', className="small text-muted mt-1"),
        dcc.Slider(
            id='aging-rate-slider',
            min=0, max=150, step=5, value=100,
            marks={0: '0%', 50: '50%', 100: '100%', 150: '150%'},
            tooltip={"placement": "bottom", "always_visible": False},
        ),

        html.Hr(),
        dbc.Button(
            "Advanced settings",
            id="advanced-toggle", color="link", size="sm", className="p-0",
        ),
        dbc.Collapse(
            html.Div([
                html.Label("Gompertz fit options", className="mt-2 small"),
                dcc.Checklist(
                    id='gompertz-options',
                    options=[
                        {'label': ' Remove accidents before fit', 'value': 'remove_accidents'},
                        {'label': ' Use Makeham term', 'value': 'use_makeham'},
                    ],
                    value=['remove_accidents'],
                    className="small",
                ),
                html.Label("Show ages up to", className="mt-2 small"),
                dbc.Input(id='pad-to-input', type='number', value=120,
                          min=80, max=200, step=10, size="sm"),
            ]),
            id="advanced-collapse",
            is_open=False,
        ),

        html.Div(
            html.Img(
                src=app.get_asset_url('lockup-stacked.svg'),
                alt='Second Century Foundation',
                style={"width": "100%", "display": "block",
                       "opacity": 0.9},
            ),
            className="mt-auto pt-4",
        ),
    ],
    style={
        "position": "fixed",
        "top": 0, "left": 0, "bottom": 0,
        "width": "22rem",
        "padding": "1.25rem 1.25rem",
        "background-color": "#f8f9fa",
        "overflow-y": "auto",
        "border-right": "1px solid #e9ecef",
        "display": "flex",
        "flexDirection": "column",
    },
)


# --- KPI cards ---

def kpi_card(title_id, body_id):
    return dbc.Card(
        [
            dbc.CardHeader(html.Div(id=title_id, className="small text-uppercase text-muted")),
            dbc.CardBody(html.Div(id=body_id)),
        ],
        className="h-100 shadow-sm",
    )


content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(kpi_card("card-lifespan-title", "card-lifespan-body"), md=3),
                dbc.Col(kpi_card("card-gain-title", "card-gain-body"), md=3),
                dbc.Col(kpi_card("card-80-title", "card-80-body"), md=3),
                dbc.Col(kpi_card("card-100-title", "card-100-body"), md=3),
            ],
            className="mb-4 g-3 mt-1",
        ),

        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='survival-graph', config={"displayModeBar": False}), lg=7),
                dbc.Col(dcc.Graph(id='healthspan-graph', config={"displayModeBar": False}), lg=5),
            ],
            className="g-3",
        ),

        dbc.Card(
            dbc.CardBody([
                html.H6("How to read this", className="card-title"),
                html.P(
                    "The left chart shows the percentage still alive at each age — "
                    "your lifespan curve. The grey line is today; the blue line is the scenario. "
                    "The right chart shows the average number of chronic conditions a person "
                    "of that age has — your healthspan curve. Lower is healthier.",
                    className="small text-muted mb-0",
                ),
            ]),
            className="mt-4",
        ),

        dbc.Card([
            dbc.CardHeader(
                dbc.Button(
                    [
                        html.I(className="bi bi-chevron-right me-2", id="math-chevron"),
                        html.Span("Mathematical analysis", className="fw-semibold"),
                    ],
                    id="math-toggle",
                    color="link",
                    className="p-0 text-decoration-none text-dark",
                ),
            ),
            dbc.Collapse(
                dbc.CardBody([
                    html.P(
                        "Above age ~25, human mortality closely follows Gompertz' law — "
                        "the yearly risk of dying rises exponentially with age:",
                        className="small text-muted mb-2",
                    ),

                    # Formula — colored letters
                    html.Div(
                        [
                            html.Span("µ(t)",
                                      style={"color": COLOR_MU, "fontWeight": 700}),
                            html.Span("  =  ", style={"color": "#343a40"}),
                            html.Span("A",
                                      style={"color": COLOR_A, "fontWeight": 700}),
                            html.Span(" · e", style={"color": "#343a40"}),
                            html.Sup([
                                html.Span("B",
                                          style={"color": COLOR_B, "fontWeight": 700}),
                                html.Span(" · t", style={"color": "#343a40"}),
                            ]),
                        ],
                        style={
                            "fontSize": "1.75rem",
                            "fontFamily": "Georgia, 'Times New Roman', serif",
                            "textAlign": "center",
                            "padding": "0.75rem",
                            "margin": "0.25rem 0 1rem",
                            "backgroundColor": "#f8f9fa",
                            "borderRadius": "0.5rem",
                        },
                    ),

                    # Definitions
                    html.Div([
                        html.Div([
                            html.Span("µ(t)",
                                      style={"color": COLOR_MU, "fontWeight": 700,
                                             "fontSize": "1.05rem",
                                             "fontFamily": "Georgia, serif"}),
                            html.Span(" — mortality rate. ", className="fw-semibold"),
                            "The yearly probability of dying at age ",
                            html.Em("t"), ".",
                        ], className="mb-3"),

                        html.Div([
                            html.Div([
                                html.Span("A",
                                          style={"color": COLOR_A, "fontWeight": 700,
                                                 "fontSize": "1.05rem",
                                                 "fontFamily": "Georgia, serif"}),
                                html.Span(" — initial mortality rate.", className="fw-semibold"),
                                html.Span(
                                    " The yearly risk of dying at age 0 implied by the fit.",
                                    className="text-muted"),
                            ]),
                            html.Div(id="gompertz-a-values", className="ms-4 mt-1"),
                        ], className="mb-3"),

                        html.Div([
                            html.Div([
                                html.Span("B",
                                          style={"color": COLOR_B, "fontWeight": 700,
                                                 "fontSize": "1.05rem",
                                                 "fontFamily": "Georgia, serif"}),
                                html.Span(" — rate of exponential mortality growth.",
                                          className="fw-semibold"),
                                html.Span(
                                    " How fast yearly death risk multiplies as you age. "
                                    "B alone determines your maximum possible lifespan: "
                                    "smaller B ⇒ slower exponential climb ⇒ higher ceiling.",
                                    className="text-muted"),
                            ]),
                            html.Div(id="gompertz-b-values", className="ms-4 mt-1"),
                        ], className="mb-2"),
                    ], className="small"),

                    html.Div([
                        "Fitted from ages 25–100. ",
                        html.A(
                            "Read more here",
                            href="https://en.wikipedia.org/wiki/Gompertz%E2%80%93Makeham_law_of_mortality",
                            target="_blank", rel="noopener noreferrer",
                        ),
                        ".",
                    ], className="small text-muted fst-italic mt-2"),
                ]),
                id="math-collapse",
                is_open=False,
            ),
        ], className="mt-3"),

        dbc.Card(
            dbc.CardBody([
                html.H6("Sources & methods", className="card-title"),
                html.Div([
                    html.Strong("Data sources"),
                    html.Ul([
                        html.Li([
                            "U.S. mortality rates: ",
                            html.A(
                                "CDC/NCHS United States Life Tables (2010–2019)",
                                href="https://www.cdc.gov/nchs/products/life_tables.htm",
                                target="_blank", rel="noopener noreferrer",
                            ),
                            " — provides the annual probability of dying (mₓ) at each age.",
                        ]),
                        html.Li([
                            "Cause-of-death breakdown: ",
                            html.A(
                                "CDC Multiple Cause of Death database (2024)",
                                href="https://www.cdc.gov/nchs/nvss/mortality_public_use_data.htm",
                                target="_blank", rel="noopener noreferrer",
                            ),
                            " — ICD-10 codes on death certificates, aggregated into major categories (Cancer, Heart disease & stroke, Accidents, etc.).",
                        ]),
                        html.Li([
                            "Chronic-condition prevalence (for the healthspan curve): ",
                            html.A(
                                "IHME Global Burden of Disease (2023)",
                                href="https://vizhub.healthdata.org/gbd-results/",
                                target="_blank", rel="noopener noreferrer",
                            ),
                            " — age- and sex-specific prevalence rates for cardiovascular, cancer, neurological, chronic respiratory, and diabetes/kidney disease.",
                        ]),
                    ], className="small mb-2"),
                ], className="mb-2"),

                html.Div([
                    html.Strong("Methods"),
                    html.Ul([
                        html.Li([
                            html.Em("Survival curve: "),
                            "S(age) = ∏ (1 − mₓ) over all prior ages. The y-axis shows the fraction of a starting cohort still alive.",
                        ]),
                        html.Li([
                            html.Em("Curing a disease: "),
                            "We multiply each age's mortality rate by (1 − fraction of deaths from that cause at that age). This is an upper-bound estimate — a truly perfect cure, with no competing causes taking over.",
                        ]),
                        html.Li([
                            html.Em("Slowing aging: "),
                            "After the chosen start age, each year of chronological time advances biological age by the chosen fraction (e.g. 50% = age half as fast). 0% freezes mortality at the start-age rate.",
                        ]),
                        html.Li([
                            html.Em("Personal age: "),
                            "When you enter an age, curves are conditioned on being alive right now: S(age ∣ your age) = S(age) / S(your age).",
                        ]),
                        html.Li([
                            html.Em("Gompertz fit (advanced): "),
                            "mₓ ≈ a · exp(b · age), fit in log-space on ages 25–100. Useful for comparing the exponential rate of aging between scenarios.",
                        ]),
                    ], className="small mb-2"),
                ], className="mb-2"),

                html.Div([
                    html.Strong("Simplifying assumptions"),
                    html.Ul([
                        html.Li([
                            html.Em("Perfect cures: "),
                            "'curing' a disease removes it completely from the model — an upper bound. "
                            "Real-world treatments are never 100% effective.",
                        ]),
                        html.Li([
                            html.Em("No competing risks: "),
                            "removing one cause of death doesn't redistribute risk to other causes. "
                            "In reality, people who don't die of cancer eventually die of something else.",
                        ]),
                        html.Li([
                            html.Em("Constant mortality past age 100: "),
                            "we hold the age-100 mortality rate flat through age 120 because the data thins out — "
                            "this affects the long tail of the survival curve.",
                        ]),
                    ], className="small mb-2"),
                    html.Div("Treat results as thought experiments, not forecasts.",
                             className="small text-muted fst-italic"),
                ], className="mb-2"),
            ]),
            className="mt-3 mb-4",
        ),
    ],
    style={
        "margin-left": "23rem",
        "margin-right": "1.5rem",
        "padding": "1.5rem 1rem",
    },
)

app.layout = html.Div([sidebar, content])


# --- Callbacks ---

@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_advanced(n):
    return bool(n and n % 2 == 1)


@app.callback(
    Output("math-collapse", "is_open"),
    Output("math-chevron", "className"),
    Input("math-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_math(n):
    is_open = bool(n and n % 2 == 1)
    chev = "bi bi-chevron-down me-2" if is_open else "bi bi-chevron-right me-2"
    return is_open, chev


@app.callback(
    Output('aging-rate-label', 'children'),
    Input('aging-rate-slider', 'value'),
    Input('user-age', 'value'),
)
def label_aging_rate(value, user_age):
    start_age = user_age if user_age is not None else DEFAULT_SLOW_START
    start_suffix = f" starting at {int(start_age)}"
    if value is None or value == 100:
        return f"Normal aging (100%){start_suffix}"
    if value == 0:
        return f"Frozen aging (0%){start_suffix}"
    if value < 100:
        return f"Slowed aging ({value}%){start_suffix}"
    return f"Accelerated aging ({value}%){start_suffix}"


@app.callback(
    Output('selected-causes', 'data'),
    Input({'type': 'cause-btn', 'cause': ALL}, 'n_clicks'),
    State('selected-causes', 'data'),
    prevent_initial_call=True,
)
def toggle_cause(_n_clicks, selected):
    ctx = callback_context
    if not ctx.triggered or ctx.triggered[0]['value'] in (None, 0):
        return selected or []
    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        cause = json.loads(prop_id)['cause']
    except Exception:
        return selected or []
    selected = list(selected or [])
    if cause in selected:
        selected.remove(cause)
    else:
        selected.append(cause)
    return selected


@app.callback(
    Output({'type': 'cause-btn', 'cause': ALL}, 'outline'),
    Output({'type': 'cause-btn', 'cause': ALL}, 'color'),
    Input('selected-causes', 'data'),
    State({'type': 'cause-btn', 'cause': ALL}, 'id'),
)
def reflect_button_state(selected, ids):
    selected = set(selected or [])
    outlines = [i['cause'] not in selected for i in ids]
    colors = ['success' if i['cause'] in selected else 'primary' for i in ids]
    return outlines, colors


@app.callback(
    [
        Output('survival-graph', 'figure'),
        Output('healthspan-graph', 'figure'),
        Output('card-lifespan-title', 'children'),
        Output('card-lifespan-body', 'children'),
        Output('card-gain-title', 'children'),
        Output('card-gain-body', 'children'),
        Output('card-80-title', 'children'),
        Output('card-80-body', 'children'),
        Output('card-100-title', 'children'),
        Output('card-100-body', 'children'),
        Output('gompertz-a-values', 'children'),
        Output('gompertz-b-values', 'children'),
    ],
    [
        Input('user-name', 'value'),
        Input('user-age', 'value'),
        Input('sex-dropdown', 'value'),
        Input('selected-causes', 'data'),
        Input('aging-rate-slider', 'value'),
        Input('gompertz-options', 'value'),
        Input('pad-to-input', 'value'),
    ],
)
def update_dashboard(name, user_age, sex, removed_causes, aging_rate_percent, gompertz_opts, pad_to):
    if pad_to is None:
        pad_to = 120
    if removed_causes is None:
        removed_causes = []
    if aging_rate_percent is None:
        aging_rate_percent = 100
    if sex in (None, ''):
        sex = 'All'
    aging_rate = aging_rate_percent / 100.0
    slow_start = user_age if user_age is not None else DEFAULT_SLOW_START

    # --- Run simulation ---
    scenario = LongevityScenario(
        sex=sex,
        aging_rate=aging_rate,
        slow_aging_age=slow_start,
        removed_causes=removed_causes,
    )
    data = scenario.get_data(pad_to=pad_to)
    base_surv_full = data['baseline_survival']
    int_surv_full = data['intervention_survival']
    base_mort = data['baseline_mortality']
    int_mort = data['intervention_mortality']
    base_condition_count = data['baseline_condition_count']
    int_condition_count = data['intervention_condition_count']

    # Conditional survival (anchored at user's age) — what the user actually experiences
    base_surv = _conditional_survival(base_surv_full, user_age)
    int_surv = _conditional_survival(int_surv_full, user_age)

    # Whether the scenario differs from the baseline at all.
    intervention_active = bool(removed_causes) or aging_rate != 1.0

    # --- Lifespan numbers ---
    base_median = _median_from_survival(base_surv, base_mort)
    int_median = _median_from_survival(int_surv, int_mort)

    int_median_capped = int_median == float('inf')
    int_median_eff = float(MEDIAN_CAP_AGE) if int_median_capped else int_median
    if np.isfinite(base_median) and np.isfinite(int_median_eff):
        years_gained = int_median_eff - base_median
    else:
        years_gained = float('nan')

    # Color the intervention red if the scenario lives meaningfully shorter than today.
    intervention_color = (
        COLOR_LOSS if (np.isfinite(years_gained) and years_gained < -0.05)
        else COLOR_INTERVENTION
    )

    poss = _possessive(name)

    # --- KPI cards ---
    today_life = f"{base_median:.1f} yrs" if np.isfinite(base_median) else "—"
    if int_median_capped:
        scen_life = f">{MEDIAN_CAP_AGE} yrs"
    elif np.isfinite(int_median):
        scen_life = f"{int_median:.1f} yrs"
    else:
        scen_life = "—"
    life_delta = None
    if np.isfinite(years_gained) and abs(years_gained) > 0.05:
        prefix = ">+" if int_median_capped and years_gained > 0 else ("+" if years_gained > 0 else "")
        life_delta = f"{prefix}{years_gained:.1f} years vs. today"
    lifespan_title = _card_title("expected lifespan", poss)
    lifespan_body = _comparison_body(today_life, scen_life, life_delta,
                                     scenario_color=intervention_color,
                                     show_scenario=intervention_active)

    # Years gained card (single-number spotlight)
    gain_title = _card_title("years gained", poss)
    if not intervention_active:
        gain_body = [html.H3("—", className="mb-0"),
                     html.Div("No changes yet", className="small text-muted mt-1")]
    elif np.isfinite(years_gained):
        prefix = ">+" if int_median_capped and years_gained > 0 else (
            "+" if years_gained >= 0 else "")
        if years_gained > 0:
            big_color = COLOR_GAIN
        elif years_gained < 0:
            big_color = COLOR_LOSS
        else:
            big_color = COLOR_BASELINE
        gain_big = html.H3(f"{prefix}{years_gained:.1f} yrs",
                           className="mb-0 fw-bold",
                           style={"color": big_color})
        if np.isfinite(base_median) and base_median > 0 and years_gained > 0:
            pct_more = years_gained / base_median * 100
            longer_prefix = ">" if int_median_capped else ""
            sub = f"{longer_prefix}{pct_more:.0f}% longer than today"
        elif years_gained < 0:
            sub = "Shorter than today"
        else:
            sub = "Same as today"
        gain_body = [gain_big, html.Div(sub, className="small text-muted mt-1")]
    else:
        gain_body = [html.H3("—", className="mb-0"), html.Div("", className="small text-muted")]

    base_80, int_80 = _chance_at_age(80, base_surv_full, int_surv_full, user_age)
    base_100, int_100 = _chance_at_age(100, base_surv_full, int_surv_full, user_age)
    title_80 = _card_title("chance of reaching 80", poss)
    title_100 = _card_title("chance of reaching 100", poss)
    body_80 = _reach_body(base_80, int_80, show_scenario=intervention_active,
                          scenario_color=intervention_color)
    body_100 = _reach_body(base_100, int_100, show_scenario=intervention_active,
                           scenario_color=intervention_color)

    # --- Gompertz fits ---
    remove_accidents_fit = 'remove_accidents' in (gompertz_opts or [])
    use_makeham = 'use_makeham' in (gompertz_opts or [])
    base_fit_res = scenario.fit_curve(target='baseline', remove_accidents=remove_accidents_fit,
                                      use_makeham=use_makeham, fit_region=[25, 100])
    int_fit_res = scenario.fit_curve(target='intervention', remove_accidents=remove_accidents_fit,
                                     use_makeham=use_makeham, fit_region=[25, 100])

    a_base, b_base = float(base_fit_res['params'][0]), float(base_fit_res['params'][1])
    a_int, b_int = float(int_fit_res['params'][0]), float(int_fit_res['params'][1])
    gompertz_a_values = _gompertz_rows(a_base, a_int,
                                       show_intervention=intervention_active,
                                       intervention_color=intervention_color)
    gompertz_b_values = _gompertz_rows(b_base, b_int,
                                       show_intervention=intervention_active,
                                       intervention_color=intervention_color)

    # --- Survival figure ---
    ages_b = np.asarray(base_surv.index)
    vals_b = np.asarray(base_surv.values) * 100
    ages_i = np.asarray(int_surv.index)
    vals_i = np.asarray(int_surv.values) * 100

    # Shade between the two curves: green when scenario is above baseline
    # everywhere, red when it's below. (Handles edge cases like slow-aging
    # where the median is past pad_to and `years_gained > 0` would miss it.)
    diff = vals_i - vals_b
    fill_above = bool(intervention_active and len(diff) and diff.min() >= -1e-9 and diff.max() > 1e-9)
    fill_below = bool(intervention_active and len(diff) and diff.max() <= 1e-9 and diff.min() < -1e-9)

    if fill_above:
        fill_kwargs = {'fill': 'tonexty', 'fillcolor': 'rgba(44, 160, 44, 0.10)'}
    elif fill_below:
        fill_kwargs = {'fill': 'tonexty', 'fillcolor': 'rgba(220, 53, 69, 0.10)'}
    else:
        fill_kwargs = {}

    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=ages_b, y=vals_b, mode='lines', name='Today',
        line=dict(color=COLOR_BASELINE, width=2),
        text=_hover_surv(ages_b, vals_b, base_mort), hoverinfo='text',
    ))
    if intervention_active:
        fig_surv.add_trace(go.Scatter(
            x=ages_i, y=vals_i, mode='lines', name='With your changes',
            line=dict(color=intervention_color, width=3),
            text=_hover_surv(ages_i, vals_i, int_mort), hoverinfo='text',
            **fill_kwargs,
        ))

    shapes = [dict(type='line', xref='paper', x0=0, x1=1, yref='y', y0=50, y1=50,
                   line=dict(color='#adb5bd', width=1, dash='dot'))]
    annotations = [dict(xref='paper', x=0.01, y=50, yref='y', text='50% alive',
                        showarrow=False, font=dict(color='#6c757d', size=10), yshift=10)]
    if np.isfinite(base_median):
        shapes.append(dict(type='line', xref='x', x0=base_median, x1=base_median,
                           yref='y', y0=0, y1=50,
                           line=dict(color=COLOR_BASELINE, width=1, dash='dot')))
        annotations.append(dict(x=base_median, y=2, xref='x', yref='y',
                                text=f"Today: {base_median:.0f}",
                                showarrow=False, font=dict(color=COLOR_BASELINE, size=10),
                                bgcolor='rgba(248,249,250,0.85)'))
    # Only draw the scenario median marker if it falls within the displayed range.
    if intervention_active and np.isfinite(int_median) and int_median <= ages_b.max():
        shapes.append(dict(type='line', xref='x', x0=int_median, x1=int_median,
                           yref='y', y0=0, y1=50,
                           line=dict(color=intervention_color, width=1, dash='dot')))
        annotations.append(dict(x=int_median, y=6, xref='x', yref='y',
                                text=f"Scenario: {int_median:.0f}",
                                showarrow=False, font=dict(color=intervention_color, size=10),
                                bgcolor='rgba(248,249,250,0.85)'))

    # Milestone markers (skip ones already past for the user)
    if intervention_active:
        milestone_x, milestone_y, milestone_text = [], [], []
        for m_age in MILESTONE_AGES:
            if user_age is not None and m_age <= user_age:
                continue
            if m_age <= int_surv.index.max():
                int_pct = _survival_at(int_surv, m_age) * 100
                base_pct = _survival_at(base_surv, m_age) * 100
                milestone_x.append(m_age)
                milestone_y.append(int_pct)
                milestone_text.append(
                    f"<b>Age {m_age}</b><br>"
                    f"Scenario: {int_pct:.0f}% alive<br>"
                    f"Today: {base_pct:.0f}% alive"
                )
        if milestone_x:
            fig_surv.add_trace(go.Scatter(
                x=milestone_x, y=milestone_y, mode='markers', name='Milestones',
                marker=dict(size=9, color=intervention_color, line=dict(color='white', width=2)),
                text=milestone_text, hoverinfo='text', showlegend=False,
            ))

    surv_title = _card_title("chance of still being alive", poss)
    fig_surv.update_layout(
        title=dict(text=surv_title, font=dict(size=17)),
        xaxis=dict(title="Age (years)", showgrid=True, gridcolor='#eef0f2', zeroline=False),
        yaxis=dict(title="% still alive", range=[0, 100], ticksuffix='%',
                   showgrid=True, gridcolor='#eef0f2', zeroline=False),
        template="plotly_white", hovermode="x unified",
        shapes=shapes, annotations=annotations,
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        margin=dict(l=60, r=30, t=60, b=90), plot_bgcolor='white',
    )

    # --- Healthspan figure: expected number of chronic conditions ---
    ages_h_b = np.asarray(base_condition_count.index)
    vals_h_b = np.asarray(base_condition_count.values)
    ages_h_i = np.asarray(int_condition_count.index)
    vals_h_i = np.asarray(int_condition_count.values)
    if user_age is not None and user_age > 0:
        mask_b = ages_h_b >= user_age
        ages_h_b, vals_h_b = ages_h_b[mask_b], vals_h_b[mask_b]
        mask_i = ages_h_i >= user_age
        ages_h_i, vals_h_i = ages_h_i[mask_i], vals_h_i[mask_i]

    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(
        x=ages_h_b, y=vals_h_b, mode='lines', name='Today',
        line=dict(color=COLOR_BASELINE, width=2),
        hovertemplate="Age %{x}<br>%{y:.2f} chronic conditions (expected)<extra></extra>",
    ))
    if intervention_active:
        fig_health.add_trace(go.Scatter(
            x=ages_h_i, y=vals_h_i, mode='lines', name='With your changes',
            line=dict(color=intervention_color, width=3),
            hovertemplate="Age %{x}<br>%{y:.2f} chronic conditions (expected)<extra></extra>",
        ))

    health_title = _card_title("expected number of chronic conditions", poss)
    x_lo = float(user_age) if (user_age is not None and user_age > 0) else 0.0
    fig_health.update_layout(
        title=dict(text=health_title, font=dict(size=17)),
        xaxis=dict(title="Age (years)", showgrid=True, gridcolor='#eef0f2',
                   zeroline=False, range=[x_lo, 95]),
        yaxis=dict(title="Expected # of chronic conditions",
                   showgrid=True, gridcolor='#eef0f2', zeroline=False, rangemode='tozero'),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5),
        margin=dict(l=60, r=30, t=60, b=90), plot_bgcolor='white',
    )

    return (
        fig_surv, fig_health,
        lifespan_title, lifespan_body,
        gain_title, gain_body,
        title_80, body_80,
        title_100, body_100,
        gompertz_a_values, gompertz_b_values,
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
