import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.scenarios import LongevityScenario
import src.survival as surviva
import src.causes as causes  # Import causes to access logic if needed

# Initialize app with a bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server  # Expose server for deployment

# --- Data Loading (for initial setup) ---
try:
    initial_causes = pd.read_csv('data/cause_fractions_total.csv', nrows=1).columns.tolist()
    available_causes = [c for c in initial_causes if c not in ['age_years', 'Unknown']]
except Exception as e:
    available_causes = ['Cancer', 'Cardiovascular', 'Respiratory', 'External', 'Neurological'] # Fallback

# --- Layout ---

sidebar = html.Div(
    [
        html.H2("Longevity", className="display-4"),
        html.Hr(),
        html.P("Simulate interventions to extend human lifespan.", className="lead"),
        
        html.Label("Population", className="mt-3"),
        dcc.Dropdown(
            id='sex-dropdown',
            options=[
                {'label': 'All', 'value': 'All'},
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'}
            ],
            value='All',
            clearable=False
        ),
        
        html.Label("Remove Causes", className="mt-3"),
        dcc.Dropdown(
            id='cause-dropdown',
            options=[{'label': c, 'value': c} for c in available_causes],
            value=[],
            multi=True,
            placeholder="Select diseases to cure..."
        ),
        
        html.Hr(),
        html.H5("Aging Interventions", className="mt-3"),
        
        html.Label("Aging Rate (% of Normal)"),
        dbc.Input(
            id='aging-rate-input',
            type='number',
            value=100,
            min=0,
            step=5,
        ),
        
        html.Label("Slow aging starting at (Age)", className="mt-2"),
        dbc.Input(id='slow-aging-age', type='number', value=25, min=0, step=1),
        
        html.Hr(),
        html.H5("Gompertz Fit Settings", className="mt-3"),
        dcc.Checklist(
            id='gompertz-options',
            options=[
                {'label': ' Remove accidents for fit', 'value': 'remove_accidents'},
                {'label': ' Use Makeham term', 'value': 'use_makeham'}
            ],
            value=['remove_accidents'],
            className="mb-2"
        ),
        
        html.Label("Maximum X value (Age)", className="mt-2"),
        # Removed max=200 constraint
        dbc.Input(id='pad-to-input', type='number', value=120, min=80, step=10),
    ],
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "20rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
        "overflow-y": "auto"
    }
)

content = html.Div(
    [
        # KPIs Row
        dbc.Row(
            [
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Baseline Median Lifespan"),
                    dbc.CardBody(html.H4(id="kpi-baseline", className="card-title"))
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Intervention Median Lifespan"),
                    dbc.CardBody(html.H4(id="kpi-intervention", className="card-title"))
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Years Gained"),
                    dbc.CardBody(html.H4(id="kpi-gained", className="card-title", style={"color": "green"}))
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Gompertz Equations"),
                    dbc.CardBody([
                        html.Div(id="kpi-equation-base", className="small text-muted mb-1"),
                        html.Div(id="kpi-equation-int", className="small font-weight-bold")
                    ])
                ]), width=3),
            ],
            className="mb-4"
        ),
        
        # Graphs Row
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='survival-graph'), width=6),
                dbc.Col(dcc.Graph(id='mortality-graph'), width=6),
            ]
        )
    ],
    style={
        "margin-left": "22rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
)

app.layout = html.Div([sidebar, content])

# --- Callbacks ---

@app.callback(
    [
        Output('survival-graph', 'figure'),
        Output('mortality-graph', 'figure'),
        Output('kpi-baseline', 'children'),
        Output('kpi-intervention', 'children'),
        Output('kpi-gained', 'children'),
        Output('kpi-equation-base', 'children'),
        Output('kpi-equation-int', 'children')
    ],
    [
        Input('sex-dropdown', 'value'),
        Input('cause-dropdown', 'value'),
        Input('aging-rate-input', 'value'),
        Input('slow-aging-age', 'value'),
        Input('gompertz-options', 'value'),
        Input('pad-to-input', 'value')
    ]
)
def update_dashboard(sex, removed_causes, aging_rate_percent, slow_aging_age, gompertz_opts, pad_to):
    # Handle inputs
    if slow_aging_age is None: slow_aging_age = 25
    if pad_to is None: pad_to = 120
    if removed_causes is None: removed_causes = []
    if aging_rate_percent is None: aging_rate_percent = 100
    
    # Convert percentage to factor
    aging_rate = aging_rate_percent / 100.0
    
    # 1. Run Simulation
    scenario = LongevityScenario(
        sex=sex,
        aging_rate=aging_rate,
        slow_aging_age=slow_aging_age,
        removed_causes=removed_causes
    )
    
    data = scenario.get_data(pad_to=pad_to)
    
    base_surv = data['baseline_survival']
    int_surv = data['intervention_survival']
    base_mort = data['baseline_mortality']
    int_mort = data['intervention_mortality']
    
    # 2. Calculate KPIs
    base_median = survival.calculate_median_lifespan(base_surv)
    int_median = survival.calculate_median_lifespan(int_surv)
    years_gained = int_median - base_median
    
    kpi_base_text = f"{base_median:.1f} years"
    kpi_int_text = f"{int_median:.1f} years"
    kpi_gained_text = f"+{years_gained:.1f} years"
    
    # 3. Perform Gompertz Fit & Accident Removal logic
    remove_accidents_fit = 'remove_accidents' in (gompertz_opts or [])
    use_makeham = 'use_makeham' in (gompertz_opts or [])
    
    # Fit Baseline
    base_fit_res = scenario.fit_curve(
        target='baseline',
        remove_accidents=remove_accidents_fit,
        use_makeham=use_makeham,
        fit_region=[25, 100]
    )
    
    # Fit Intervention
    int_fit_res = scenario.fit_curve(
        target='intervention',
        remove_accidents=remove_accidents_fit,
        use_makeham=use_makeham,
        fit_region=[25, 100]
    )
    
    base_eq_text = f"Base: {base_fit_res['equation']}"
    int_eq_text = f"Int: {int_fit_res['equation']}"
    
    # 4. Prepare Plot Data (Accident Removal)
    # If "remove accidents" is checked, we also want to visually remove accidents 
    # from the scatter plots of mortality rates (as requested in point 1).
    # We can do this by creating a temp scenario or manually adjusting.
    # The 'fit_curve' logic internally removes accidents if requested, but 'get_data' doesn't automatically.
    
    plot_base_mort = base_mort
    plot_int_mort = int_mort
    mort_title = "Mortality Rate"
    
    if remove_accidents_fit:
        mort_title = "Mortality Rate (accidents removed)"
        
        # We need to compute mortality with accidents removed for plotting
        # Create a temp scenario or just use helper function
        # Since we are already inside, we can just use the internal logic or call helper
        # Simplest is to manually remove 'External' from the loaded baseline/intervention
        # But intervention might have already modified causes.
        
        # Let's use the Cause helper directly
        if 'External' in scenario.cause_fractions.columns:
            # Baseline (fresh copy)
            plot_base_mort = causes.remove_cause_from_lifetable(
                scenario.baseline_mortality.copy(), 
                scenario.cause_fractions, 
                'External'
            )
            plot_base_mort = scenario._pad_series(plot_base_mort, pad_to)
            
            # Intervention (needs careful handling: accidents removed on top of other interventions)
            # Re-calculating intervention mortality with accidents removed:
            # We can instantiate a temporary scenario that includes 'External' in removed_causes
            # But we must preserve other settings.
            
            temp_removed = removed_causes.copy()
            if 'External' not in temp_removed:
                temp_removed.append('External')
                
            temp_scenario = LongevityScenario(
                sex=sex,
                aging_rate=aging_rate,
                slow_aging_age=slow_aging_age,
                removed_causes=temp_removed
            )
            plot_int_mort = temp_scenario.get_data(pad_to=pad_to)['intervention_mortality']

    # 5. Generate Figures
    
    # Survival Curve
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Scatter(
        x=base_surv.index, y=base_surv.values,
        mode='lines', name='Baseline',
        line=dict(color='gray') # Solid gray
    ))
    fig_surv.add_trace(go.Scatter(
        x=int_surv.index, y=int_surv.values,
        mode='lines', name='Intervention',
        line=dict(color='lightblue') # Solid light blue
    ))
    fig_surv.update_layout(
        title="Survival Curve",
        xaxis_title="Age (years)",
        yaxis_title="Survival Probability",
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Mortality Rate (Log Scale)
    fig_mort = go.Figure()
    
    # Baseline Data
    fig_mort.add_trace(go.Scatter(
        x=plot_base_mort.index, y=plot_base_mort.values,
        mode='lines', name='Baseline',
        line=dict(color='gray') # Solid gray
    ))
    # Baseline Fit
    if len(base_fit_res['x']) > 0:
        fig_mort.add_trace(go.Scatter(
            x=base_fit_res['x'], y=base_fit_res['y_pred'],
            mode='lines', name='Baseline Fit',
            line=dict(color='black', dash='dash') # Dashed black
        ))
        
    # Intervention Data
    fig_mort.add_trace(go.Scatter(
        x=plot_int_mort.index, y=plot_int_mort.values,
        mode='lines', name='Intervention',
        line=dict(color='lightblue') # Solid light blue
    ))
    # Intervention Fit
    if len(int_fit_res['x']) > 0:
        fig_mort.add_trace(go.Scatter(
            x=int_fit_res['x'], y=int_fit_res['y_pred'],
            mode='lines', name='Intervention Fit',
            line=dict(color='darkblue', dash='dash') # Dashed dark blue
        ))
    
    fig_mort.update_layout(
        title=mort_title,
        xaxis_title="Age (years)",
        yaxis_title="Mortality Rate",
        yaxis_type="log",
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig_surv, fig_mort, kpi_base_text, kpi_int_text, kpi_gained_text, base_eq_text, int_eq_text

if __name__ == '__main__':
    app.run(debug=True)
