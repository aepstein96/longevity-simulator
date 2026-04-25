import pandas as pd
import numpy as np
import src.mortality as mortality
import src.causes as causes
import src.interventions as interventions
import src.survival as survival
import src.fitting as fitting
import src.healthspan as healthspan

class LongevityScenario:
    def __init__(self, sex='All', aging_rate=1.0, slow_aging_age=25, removed_causes=None):
        """
        Initialize a longevity scenario.
        
        Parameters
        ----------
        sex : str
            Population to use: 'All', 'Male', 'Female'
        aging_rate : float
            Rate of aging relative to normal (0.0 - 1.0).
            1.0 is normal aging, 0.0 is stopped aging.
        slow_aging_age : int
            Age at which to start slowing/stopping aging.
        removed_causes : list
            List of cause categories to remove (e.g., ['Cancer']).
        """
        self.sex = sex
        self.aging_rate = aging_rate
        self.slow_aging_age = slow_aging_age
        self.removed_causes = removed_causes if removed_causes else []
        
        # Map UI sex labels to data file suffixes
        self.sex_map = {
            'All': 'Total',
            'Male': 'Male',
            'Female': 'Female'
        }
        
        # Load data immediately or lazily? Lazily is probably fine, but we need it for calculations
        self._load_data()

    def _load_data(self):
        """Load mortality rates and cause fractions based on sex."""
        sex_key = self.sex_map.get(self.sex, 'Total')
        
        # Construct file paths
        mort_path = f'data/CDC/mortality_rates_{sex_key.lower()}.csv'
        cause_path = f'data/CDC/cause_fractions_{sex_key.lower()}.csv'
        
        self.baseline_mortality = mortality.load_mortality_rates(mort_path)
        self.cause_fractions = causes.load_cause_fractions(cause_path)

    def get_data(self, pad_to=120):
        """
        Calculate baseline and intervention mortality and survival curves.
        
        Parameters
        ----------
        pad_to : int
            Maximum age to extend the curves to.
            
        Returns
        -------
        dict
            Dictionary containing:
            - baseline_mortality (pd.Series)
            - baseline_survival (pd.Series)
            - intervention_mortality (pd.Series)
            - intervention_survival (pd.Series)
        """
        # --- Baseline ---
        # Just pad the baseline mortality
        base_mort = self.baseline_mortality.copy()
        base_mort = self._pad_series(base_mort, pad_to)
        base_survival = survival.calculate_survival_curve(base_mort)
        
        # --- Intervention ---
        # 1. Remove causes
        adj_mort = self.baseline_mortality.copy()
        if self.removed_causes:
            for cause in self.removed_causes:
                # We need to handle the case where cause might not be in the columns
                # The categorize_cause logic produces specific names
                if cause in self.cause_fractions.columns:
                    adj_mort = causes.remove_cause_from_lifetable(
                        adj_mort, self.cause_fractions, cause
                    )
        
        # 2. Apply aging intervention
        if self.aging_rate != 1.0:
            if self.aging_rate == 0.0:
                # Stop aging
                adj_mort = interventions.stop_aging(
                    adj_mort, final_age=self.slow_aging_age, pad_to_age=pad_to
                )
            else:
                # Slow or Fast aging
                # Note: interventions.slow_aging uses 'slow_factor' where 0.5 means half speed
                # My aging_rate=1.0 is normal, 0.5 is half speed, 2.0 is double speed. Matches slow_factor.
                adj_mort = interventions.slow_aging(
                    adj_mort, slow_factor=self.aging_rate, 
                    start_age=self.slow_aging_age, pad_to_age=pad_to
                )
        else:
            # Just pad if no aging intervention
            adj_mort = self._pad_series(adj_mort, pad_to)
            
        intervention_survival = survival.calculate_survival_curve(adj_mort)

        # Healthspan: expected number of chronic conditions by single-year age,
        # = sum of per-bucket prevalences. By linearity of expectation this is
        # exact regardless of how the conditions co-occur — no independence
        # assumption needed.
        default_buckets = list(healthspan.DEFAULT_HEALTHSPAN_BUCKETS)
        baseline_condition_count = healthspan.compute_expected_condition_count(
            sex=self.sex, buckets=default_buckets, pad_to=pad_to)

        # If the user "cured" any chronic-condition bucket, drop it from the
        # sum — that bucket's prevalence becomes 0 by definition.
        intervention_buckets = [b for b in default_buckets
                                if b not in self.removed_causes]
        if intervention_buckets == default_buckets:
            intervention_condition_count = baseline_condition_count.copy()
        elif intervention_buckets:
            intervention_condition_count = healthspan.compute_expected_condition_count(
                sex=self.sex, buckets=intervention_buckets, pad_to=pad_to)
        else:
            # All chronic buckets removed → E[#] = 0 at every age.
            intervention_condition_count = pd.Series(
                0.0, index=baseline_condition_count.index,
                name='E[# chronic conditions]')

        # Aging rescale (slow / freeze / accelerate) — same biological-age
        # remap that drives the mortality intervention, applied to the
        # condition-count curve so the scenario's x-axis is consistent.
        if self.aging_rate != 1.0:
            intervention_condition_count = healthspan.apply_aging_remap(
                intervention_condition_count,
                aging_rate=self.aging_rate,
                start_age=self.slow_aging_age,
            )

        return {
            'baseline_mortality': base_mort,
            'baseline_survival': base_survival,
            'intervention_mortality': adj_mort,
            'intervention_survival': intervention_survival,
            'baseline_condition_count': baseline_condition_count,
            'intervention_condition_count': intervention_condition_count,
        }

    def fit_curve(self, target='intervention', remove_accidents=True, use_makeham=False, fit_region=[25, 100]):
        """
        Fit Gompertz curve to mortality data.
        
        Parameters
        ----------
        target : str
            'baseline' or 'intervention'
        remove_accidents : bool
            Whether to remove external causes before fitting.
        use_makeham : bool
            Whether to use Makeham term.
        fit_region : list
            [start, end] age for fitting.
            
        Returns
        -------
        dict
            - params: tuple of parameters
            - equation: str representation
            - x: array of ages
            - y_pred: array of predicted rates
        """
        # Prepare data for fitting
        mort_to_fit = self.baseline_mortality.copy()
        
        # Determine causes to remove
        causes_to_remove = []
        if target == 'intervention':
            causes_to_remove.extend(self.removed_causes)
            
        if remove_accidents and 'External' not in causes_to_remove:
            causes_to_remove.append('External')
            
        # Remove causes
        for cause in causes_to_remove:
            if cause in self.cause_fractions.columns:
                mort_to_fit = causes.remove_cause_from_lifetable(
                    mort_to_fit, self.cause_fractions, cause
                )
        
        # Apply aging intervention ONLY if target is intervention
        if target == 'intervention' and self.aging_rate != 1.0:
            if self.aging_rate == 0.0:
                mort_to_fit = interventions.stop_aging(
                    mort_to_fit, final_age=self.slow_aging_age, pad_to_age=0
                )
            else:
                mort_to_fit = interventions.slow_aging(
                    mort_to_fit, slow_factor=self.aging_rate, 
                    start_age=self.slow_aging_age, pad_to_age=0
                )
        
        # Perform fit
        x_full, y_pred, params, cov = fitting.fit_gompertz(
            mort_to_fit, use_makeham=use_makeham, fit_region=fit_region
        )
        
        if use_makeham:
            eq_str = fitting.get_gompertz_equation(params[0], params[1], params[2])
        else:
            eq_str = fitting.get_gompertz_equation(params[0], params[1])
            
        return {
            'params': params,
            'equation': eq_str,
            'x': x_full,
            'y_pred': y_pred
        }

    def _pad_series(self, series, pad_to):
        """Helper to pad a series to a max age with constant last value."""
        series = series.sort_index()
        max_age = series.index.max()
        if pad_to > max_age:
            final_val = series.iloc[-1]
            pad_idx = np.arange(max_age + 1, pad_to + 1)
            pad_series = pd.Series(final_val, index=pad_idx)
            return pd.concat([series, pad_series])
        return series
