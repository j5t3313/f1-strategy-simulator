import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# FastF1 import
try:
    import fastf1
    FASTF1_AVAILABLE = True
    cache_dir = '.f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
except ImportError:
    FASTF1_AVAILABLE = False

# Bayesian modeling imports
try:
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# set page config
st.set_page_config(
    page_title="F1 Strategy Simulator 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class F1StrategySimulator:
    def __init__(self):
        # 2025 F1 Calendar
        self.circuit_data = [
            ('Australia', {'laps': 58, 'distance_km': 5.278, 'gp_name': 'Australian Grand Prix'}),
            ('China', {'laps': 56, 'distance_km': 5.451, 'gp_name': 'Chinese Grand Prix'}),
            ('Japan', {'laps': 53, 'distance_km': 5.807, 'gp_name': 'Japanese Grand Prix'}),
            ('Bahrain', {'laps': 57, 'distance_km': 5.412, 'gp_name': 'Bahrain Grand Prix'}),
            ('Saudi Arabia', {'laps': 50, 'distance_km': 6.174, 'gp_name': 'Saudi Arabian Grand Prix'}),
            ('Miami', {'laps': 57, 'distance_km': 5.41, 'gp_name': 'Miami Grand Prix'}),
            ('Imola', {'laps': 63, 'distance_km': 4.909, 'gp_name': 'Emilia Romagna Grand Prix'}),
            ('Monaco', {'laps': 78, 'distance_km': 3.337, 'gp_name': 'Monaco Grand Prix'}),
            ('Spain', {'laps': 66, 'distance_km': 4.655, 'gp_name': 'Spanish Grand Prix'}),
            ('Canada', {'laps': 70, 'distance_km': 4.361, 'gp_name': 'Canadian Grand Prix'}),
            ('Austria', {'laps': 71, 'distance_km': 4.318, 'gp_name': 'Austrian Grand Prix'}),
            ('Britain', {'laps': 52, 'distance_km': 5.891, 'gp_name': 'British Grand Prix'}),
            ('Hungary', {'laps': 70, 'distance_km': 4.381, 'gp_name': 'Hungarian Grand Prix'}),
            ('Belgium', {'laps': 44, 'distance_km': 7.004, 'gp_name': 'Belgian Grand Prix'}),
            ('Netherlands', {'laps': 72, 'distance_km': 4.259, 'gp_name': 'Dutch Grand Prix'}),
            ('Italy', {'laps': 53, 'distance_km': 5.793, 'gp_name': 'Italian Grand Prix'}),
            ('Azerbaijan', {'laps': 51, 'distance_km': 6.003, 'gp_name': 'Azerbaijan Grand Prix'}),
            ('Singapore', {'laps': 62, 'distance_km': 4.940, 'gp_name': 'Singapore Grand Prix'}),
            ('United States', {'laps': 56, 'distance_km': 5.513, 'gp_name': 'United States Grand Prix'}),
            ('Mexico', {'laps': 71, 'distance_km': 4.304, 'gp_name': 'Mexico City Grand Prix'}),
            ('Brazil', {'laps': 71, 'distance_km': 4.309, 'gp_name': 'São Paulo Grand Prix'}),
            ('Las Vegas', {'laps': 50, 'distance_km': 6.201, 'gp_name': 'Las Vegas Grand Prix'}),
            ('Qatar', {'laps': 57, 'distance_km': 5.380, 'gp_name': 'Qatar Grand Prix'}),
            ('Abu Dhabi', {'laps': 58, 'distance_km': 5.281, 'gp_name': 'Abu Dhabi Grand Prix'})
        ]
        
        # convert to dict for easy lookup
        self.circuits = {name: data for name, data in self.circuit_data}
        
        self.bayesian_models = {}
        self.data_cache = {}
        self.use_bayesian = BAYESIAN_AVAILABLE and FASTF1_AVAILABLE
    
    def calculate_fuel_consumption(self, circuit):
        """Calculate fuel consumption per lap: 105kg / laps (110kg/laps with 5kg in reserve)"""
        laps = self.circuits[circuit]['laps']
        return 105.0 / laps
        
    def calculate_fuel_corrected_laptime(self, raw_laptime, lap_number, total_laps, 
                                       fuel_consumption_per_lap, weight_effect=0.03):
        """Fuel corrected laptime calculation"""
        remaining_laps = total_laps - lap_number
        fuel_correction = remaining_laps * fuel_consumption_per_lap * weight_effect
        return raw_laptime - fuel_correction
    
    def load_real_f1_data(self, circuit_name):
        """Load race data for the selected circuit from 2024 season"""
        if not FASTF1_AVAILABLE:
            return None
            
        if circuit_name in self.data_cache:
            return self.data_cache[circuit_name]
            
        circuit_info = self.circuits.get(circuit_name)
        if not circuit_info:
            return None
            
        try:
            # load the specific GP's 2024 race data
            session = fastf1.get_session(2024, circuit_info['gp_name'], 'R')
            session.load()
            
            # process laps 
            laps = session.laps
            stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
            stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
            stints.dropna(subset=["LapTime_s"], inplace=True)
            stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1
            
            # filter valid data
            stints = stints[
                (stints["LapTime_s"] > 60) &
                (stints["LapTime_s"] < 300) &
                (stints["StintLap"] <= 50) &
                (stints["Compound"].isin(['SOFT', 'MEDIUM', 'HARD']))
            ]
            
            if len(stints) == 0:
                return None
            
            # apply fuel correction
            circuit_laps = circuit_info['laps']
            fuel_per_lap = 105.0 / circuit_laps
            weight_effect = 0.03
            
            stints['LapTime_FC'] = stints.apply(
                lambda row: self.calculate_fuel_corrected_laptime(
                    row['LapTime_s'], row['LapNumber'], circuit_laps, fuel_per_lap, weight_effect
                ), axis=1
            )
            
            # cache
            self.data_cache[circuit_name] = stints
            return stints
            
        except Exception as e:
            st.error(f"Could not load 2024 data for {circuit_info['gp_name']}: {str(e)}")
            return None
    
    def build_bayesian_tire_model(self, compound, circuit_name):
        """Build Bayesian tire model using 2024 race F1 data"""
        if not self.use_bayesian:
            return None
            
        model_key = f"{circuit_name}_{compound}"
        if model_key in self.bayesian_models:
            return self.bayesian_models[model_key]
        
        real_data = self.load_real_f1_data(circuit_name)
        
        if real_data is not None:
            compound_data = real_data[real_data['Compound'] == compound]
            
            if len(compound_data) > 15:  
                x_data = jnp.array(compound_data['StintLap'].values, dtype=jnp.float32)
                y_data = jnp.array(compound_data['LapTime_FC'].values, dtype=jnp.float32)
                
                def model(x, y=None):
                    alpha = numpyro.sample("alpha", dist.Normal(80, 5))
                    beta = numpyro.sample("beta", dist.Normal(0.03, 0.02))
                    sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
                    mu = alpha + beta * x
                    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

                kernel = NUTS(model)
                mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=False)
                mcmc.run(random.PRNGKey(0), x_data, y_data)
                
                self.bayesian_models[model_key] = {
                    'mcmc': mcmc,
                    'samples': mcmc.get_samples(),
                    'n_observations': len(compound_data),
                    'circuit': circuit_name,
                    'compound': compound
                }
                
                return self.bayesian_models[model_key]
        
        return None
    
    def calculate_tire_performance(self, compound, stint_lap, tire_age, base_laptime, circuit_name):
        """Calculate tire performance - use race data when available"""
        
        # try Bayesian model first
        if self.use_bayesian:
            model_key = f"{circuit_name}_{compound}"
            
            if model_key not in self.bayesian_models:
                self.build_bayesian_tire_model(compound, circuit_name)
            
            if model_key in self.bayesian_models:
                model_data = self.bayesian_models[model_key]
                samples = model_data['samples']
                
                effective_stint_lap = stint_lap + tire_age
                
                # sample from posterior - use single sample per call for proper stochastic behavior
                sample_idx = np.random.choice(len(samples['alpha']))
                
                alpha_sample = float(samples['alpha'][sample_idx])
                beta_sample = float(samples['beta'][sample_idx])
                sigma_sample = float(samples['sigma'][sample_idx])
                
                # calculate expected laptime from Bayesian model
                mu = alpha_sample + beta_sample * effective_stint_lap
                # add noise from the model's uncertainty
                prediction = mu + np.random.normal(0, sigma_sample)
                
                # Return the prediction directly (already accounts for degradation)
                return float(prediction)
        
        # fallback to physics-based model with compound-specific parameters
        compound_degradation = {
            'SOFT': 0.08,       
            'MEDIUM': 0.04,   
            'HARD': 0.02      
        }
        
        compound_offsets = {
            'SOFT': -0.8,     
            'MEDIUM': 0.0,    
            'HARD': +0.5      
        }
        
        # calculate deg
        deg_rate = compound_degradation.get(compound, 0.05)
        offset = compound_offsets.get(compound, 0.0)
        
        # age effect from previous usage
        age_effect = tire_age * 0.02
        
        # stint deg (linear + slight exponential for long stints)
        stint_degradation = deg_rate * stint_lap
        if stint_lap > 20:
            stint_degradation += 0.02 * (stint_lap - 20) ** 1.5
        
        return base_laptime + offset + age_effect + stint_degradation
    
    def validate_tire_allocation(self, strategy, tire_allocation):
        if not tire_allocation:
            return True, "Using fresh tires"
            
        required_sets = {}
        for stint in strategy:
            compound = stint['compound']
            required_sets[compound] = required_sets.get(compound, 0) + 1
            
        for compound, needed in required_sets.items():
            available = len([t for t in tire_allocation if t['compound'] == compound])
            if available < needed:
                return False, f"Need {needed} {compound} sets, only have {available}"
                
        return True, "Strategy is valid"
    
    def assign_tires_to_strategy(self, strategy, tire_allocation):
        if not tire_allocation:
            return [{'compound': stint['compound'], 'laps': stint['laps'], 'tire_age': 0} 
                   for stint in strategy]
            
        tire_sets = {compound: [] for compound in ['SOFT', 'MEDIUM', 'HARD']}
        for tire in tire_allocation:
            tire_sets[tire['compound']].append(tire)
        
        for compound in tire_sets:
            tire_sets[compound].sort(key=lambda x: x['age_laps'])
        
        enhanced_strategy = []
        for stint in strategy:
            compound = stint['compound']
            if tire_sets[compound]:
                tire_set = tire_sets[compound].pop(0)
                enhanced_strategy.append({
                    'compound': compound,
                    'laps': stint['laps'],
                    'tire_age': tire_set['age_laps']
                })
            else:
                raise ValueError(f"No more {compound} tire sets available!")
                
        return enhanced_strategy

    def simulate_race_strategy(self, circuit, strategy, tire_allocation=None, base_pace=80.0, 
                             pit_loss=22.0, num_simulations=1000, progress_bar=None):
        circuit_info = self.circuits[circuit]
        total_laps = circuit_info['laps']
        fuel_per_lap = self.calculate_fuel_consumption(circuit)
        
        is_valid, message = self.validate_tire_allocation(strategy, tire_allocation)
        if not is_valid:
            raise ValueError(f"Invalid strategy: {message}")
        
        enhanced_strategy = self.assign_tires_to_strategy(strategy, tire_allocation)
        
        results = []
        
        for sim in range(num_simulations):
            if progress_bar and sim % 100 == 0:
                progress_bar.progress(sim / num_simulations)
                
            race_time = 0
            current_lap = 1
            sim_base_pace = base_pace + np.random.normal(0, 0.5)
            
            for stint_idx, stint in enumerate(enhanced_strategy):
                compound = stint['compound']
                stint_length = stint['laps']
                tire_age = stint['tire_age']
                
                remaining_laps = total_laps - current_lap + 1
                stint_length = min(stint_length, remaining_laps)
                
                for stint_lap in range(1, stint_length + 1):
                    if current_lap > total_laps:
                        break
                    
                    raw_laptime = self.calculate_tire_performance(
                        compound, stint_lap, tire_age, sim_base_pace, circuit
                    )
                    
                    laptime = self.calculate_fuel_corrected_laptime(
                        raw_laptime, current_lap, total_laps, fuel_per_lap
                    )
                    
                    laptime += np.random.normal(0, 0.2)
                    race_time += laptime
                    current_lap += 1
                    
                if stint_idx < len(enhanced_strategy) - 1:
                    race_time += pit_loss
                        
            results.append(race_time)
            
        return np.array(results)

# predefined strategies
ALL_STRATEGIES = {
    "1-stop (M-H)": [{"compound": "MEDIUM", "laps": 20}, {"compound": "HARD", "laps": 24}],
    "1-stop (S-H)": [{"compound": "SOFT", "laps": 15}, {"compound": "HARD", "laps": 29}],
    "2-stop (M-M-H)": [{"compound": "MEDIUM", "laps": 15}, {"compound": "MEDIUM", "laps": 15}, {"compound": "HARD", "laps": 14}],
    "2-stop (M-M-S)": [{"compound": "MEDIUM", "laps": 16}, {"compound": "MEDIUM", "laps": 16}, {"compound": "SOFT", "laps": 12}],
    "2-stop (S-M-H)": [{"compound": "SOFT", "laps": 12}, {"compound": "MEDIUM", "laps": 16}, {"compound": "HARD", "laps": 16}],
    "2-stop (H-M-S)": [{"compound": "HARD", "laps": 16}, {"compound": "MEDIUM", "laps": 16}, {"compound": "SOFT", "laps": 12}],
    "1-stop (H-M)": [{"compound": "HARD", "laps": 25}, {"compound": "MEDIUM", "laps": 19}],
    "1-stop (H-S)": [{"compound": "HARD", "laps": 31}, {"compound": "SOFT", "laps": 13}],
    "2-stop (M-H-M)": [{"compound": "MEDIUM", "laps": 12}, {"compound": "HARD", "laps": 18}, {"compound": "MEDIUM", "laps": 14}]
}

def create_performance_plot(results, circuit):
    """Create performance visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Strategy Performance - {circuit}', fontsize=16, fontweight='bold')
    
    colors = ['steelblue', 'sandybrown', 'crimson', 'seagreen', 'indianred']
    
    # performance Distribution
    for i, (name, times) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax1.hist(times, bins=30, alpha=0.6, label=name, color=color, density=True)
        median = np.median(times)
        ax1.axvline(median, color=color, linestyle='--', linewidth=2)
                          
    ax1.set_xlabel("Race Time (seconds)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Performance Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # box plot
    data_for_box = [results[name] for name in results.keys()]
    labels_for_box = list(results.keys())
    
    box_plot = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for i, patch in enumerate(box_plot['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
        
    ax2.set_ylabel("Race Time (seconds)")
    ax2.set_title("Performance Spread")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # performance metrics
    strategies = list(results.keys())
    x_pos = np.arange(len(strategies))
    
    medians = [np.median(results[s]) for s in strategies]
    p95s = [np.percentile(results[s], 95) for s in strategies]
    
    ax3.bar(x_pos, medians, alpha=0.8, color='lightblue', label='Median')
    ax3.bar(x_pos, p95s, alpha=0.5, color='lightcoral', label='95th Percentile')
    
    ax3.set_xlabel("Strategy")
    ax3.set_ylabel("Race Time (seconds)")
    ax3.set_title("Performance Comparison")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strategies, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # cumulative distribution
    for i, (strategy, times) in enumerate(results.items()):
        sorted_times = np.sort(times)
        cumulative_prob = np.arange(1, len(times) + 1) / len(times)
        ax4.plot(sorted_times, cumulative_prob, label=strategy, 
                color=colors[i % len(colors)], linewidth=2)
    
    ax4.set_xlabel("Race Time (seconds)")
    ax4.set_ylabel("Cumulative Probability")
    ax4.set_title("Cumulative Distribution")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("🏎️ F1 Strategy Simulator 2025 🏎️")
    
    # Initialize simulator
    if 'simulator' not in st.session_state:
        st.session_state.simulator = F1StrategySimulator()
    
    sim = st.session_state.simulator
    
    # Sidebar configuration
    st.sidebar.header("🏁 Race Configuration")
    
    # Circuit selection - IN 2025 ORDER
    circuit_names = [name for name, _ in sim.circuit_data]
    circuit = st.sidebar.selectbox("Select Circuit", circuit_names)
    
    circuit_info = sim.circuits[circuit]
    st.sidebar.info(f"""
    **{circuit} Grand Prix**
    - Laps: {circuit_info['laps']}
    - Distance: {circuit_info['distance_km']:.3f} km/lap
    - Fuel/Lap: {sim.calculate_fuel_consumption(circuit):.2f} kg
    """)
    
    # Data availability status
    data_status_placeholder = st.sidebar.empty()
    
    # Strategy selection
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies",
        list(ALL_STRATEGIES.keys()),
        default=list(ALL_STRATEGIES.keys())[:3]
    )
    
    # Tire allocation
    use_custom_tires = st.sidebar.checkbox("Custom Tire Allocation")
    tire_allocation = None
    
    if use_custom_tires:
        tire_allocation = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            num_sets = st.sidebar.number_input(f"{compound} sets", 0, 8, 2, key=f"{compound}_sets")
            for i in range(num_sets):
                age = st.sidebar.number_input(f"{compound} set {i+1} age", 0, 50, 0, key=f"{compound}_{i}")
                tire_allocation.append({'compound': compound, 'age_laps': age})
    
    # circuit-specific base pace defaults
    circuit_base_paces = {
        'Australia': 82.0,
        'China': 95.0,
        'Japan': 92.0,
        'Bahrain': 93.0,
        'Saudi Arabia': 90.0,
        'Miami': 91.0,
        'Imola': 77.0,
        'Monaco': 75.0,
        'Spain': 78.0,
        'Canada': 75.0,
        'Austria': 67.0,
        'Britain': 88.0,
        'Hungary': 78.0,
        'Belgium': 107.0,
        'Netherlands': 72.0,
        'Italy': 83.0,
        'Azerbaijan': 102.0,
        'Singapore': 95.0,
        'United States': 96.0,
        'Mexico': 78.0,
        'Brazil': 72.0,
        'Las Vegas': 85.0,
        'Qatar': 84.0,
        'Abu Dhabi': 87.0
    }
    
    # circuit-specific pit loss defaults (based on pit lane length and speed limits)
    circuit_pit_losses = {
        'Australia': 21.5,
        'China': 20.8,
        'Japan': 20.2,
        'Bahrain': 19.8,
        'Saudi Arabia': 22.1,
        'Miami': 18.5,
        'Imola': 21.8,
        'Monaco': 16.2,  
        'Spain': 21.4,
        'Canada': 15.8,  
        'Austria': 18.9,
        'Britain': 20.5,
        'Hungary': 22.8,  
        'Belgium': 23.2,  
        'Netherlands': 20.1,
        'Italy': 15.9,   
        'Azerbaijan': 21.7,
        'Singapore': 22.5,
        'United States': 20.3,
        'Mexico': 21.1,
        'Brazil': 19.4,
        'Las Vegas': 19.6,
        'Qatar': 20.7,
        'Abu Dhabi': 21.3
    }
    
    # simulation parameters
    default_pace = circuit_base_paces.get(circuit, 80.0)
    base_pace = st.sidebar.slider(
        "Base Pace (s)", 
        60.0, 120.0, 
        default_pace, 
        0.1,
        help=f"Typical lap time for {circuit}. Default: {default_pace}s"
    )
    
    default_pit_loss = circuit_pit_losses.get(circuit, 22.0)
    pit_loss = st.sidebar.slider(
        "Pit Loss (s)", 
        15.0, 35.0, 
        default_pit_loss, 
        0.1,
        help=f"Time penalty for pit stop at {circuit}. Default: {default_pit_loss}s"
    )
    
    num_sims = st.sidebar.slider("Simulations", 100, 2000, 1000, 100)
    
    # check data availability for selected circuit
    if FASTF1_AVAILABLE and BAYESIAN_AVAILABLE:
        with data_status_placeholder:
            with st.spinner(f"Checking data for {circuit}..."):
                real_data = sim.load_real_f1_data(circuit)
                if real_data is not None:
                    st.success("✅ 2024 race data available")
                    
                    # show compound data availability
                    compound_counts = real_data['Compound'].value_counts()
                    st.write("**Tire data:**")
                    for compound in ['SOFT', 'MEDIUM', 'HARD']:
                        count = compound_counts.get(compound, 0)
                        if count > 15:
                            st.write(f"• {compound}: {count} laps ✅")
                        elif count > 0:
                            st.write(f"• {compound}: {count} laps ⚠️")
                        else:
                            st.write(f"• {compound}: No data ❌")
                else:
                    st.warning("⚠️ No 2024 data available")
                    st.info("Using basic physics model")
    elif FASTF1_AVAILABLE:
        data_status_placeholder.info("🏁 Real F1 data (basic modeling)")
    else:
        data_status_placeholder.warning("⚠️ Install FastF1 for real data")
    
    # main content
    if selected_strategies:
        # adjust strategies for circuit
        circuit_laps = circuit_info['laps']
        adjusted_strategies = {}
        
        for strategy_name in selected_strategies:
            strategy = ALL_STRATEGIES[strategy_name]
            total_original_laps = sum(stint['laps'] for stint in strategy)
            scale_factor = circuit_laps / total_original_laps
            
            adjusted_strategy = []
            remaining_laps = circuit_laps
            
            for i, stint in enumerate(strategy):
                if i == len(strategy) - 1:
                    # last stint gets all remaining laps
                    laps = remaining_laps
                else:
                    # scale intermediate stints proportionally
                    scaled_laps = stint['laps'] * scale_factor
                    # round to nearest integer but ensure minimum of 1 lap
                    laps = max(1, round(scaled_laps))
                    # don't exceed remaining laps
                    laps = min(laps, remaining_laps - (len(strategy) - i - 1))
                    remaining_laps -= laps
                    
                adjusted_strategy.append({'compound': stint['compound'], 'laps': laps})
            
            adjusted_strategies[strategy_name] = adjusted_strategy
        
        # display strategies
        st.subheader(f"🏁 Strategies for {circuit} ({circuit_laps} laps)")
        for name, strategy in adjusted_strategies.items():
            strategy_str = " → ".join([f"{stint['laps']}{stint['compound'][0]}" for stint in strategy])
            st.info(f"**{name}:** {strategy_str}")
        
        # run simulation
        if st.button("🏁 Run Analysis", type="primary"):
            with st.spinner("Running Monte Carlo simulations..."):
                # create progress tracking
                total_strategies = len(adjusted_strategies)
                strategy_progress = st.progress(0)
                current_strategy_text = st.empty()
                
                results = {}
                
                for i, (name, strategy) in enumerate(adjusted_strategies.items()):
                    # update strategy progress
                    current_strategy_text.text(f"Analyzing strategy {i+1}/{total_strategies}: {name}")
                    strategy_progress.progress(i / total_strategies)
                    
                    try:
                        times = sim.simulate_race_strategy(
                            circuit, strategy, tire_allocation,
                            base_pace, pit_loss, num_sims, None  # no individual progress bar
                        )
                        results[name] = times
                    except ValueError as e:
                        st.error(f"❌ {name}: {e}")
                        continue
                
                # complete the progress
                strategy_progress.progress(1.0)
                current_strategy_text.text("✅ All strategies evaluated")
                
                if results:
                    modeling_type = "Bayesian + Real Data" if sim.use_bayesian else "Basic Physics"
                    st.success(f"✅ Analysis complete using {modeling_type} modeling!")
                    
                    # store results
                    st.session_state.results = results
                    st.session_state.circuit = circuit
                
                # clean up progress indicators
                strategy_progress.empty()
                current_strategy_text.empty()
    
    # display results
    if 'results' in st.session_state:
        results = st.session_state.results
        current_circuit = st.session_state.circuit
        
        st.header("🏁 Analysis Results 🏁")
        
        # prepare summary data first
        summary_data = []
        for name, times in results.items():
            summary_data.append({
                "Strategy": name,
                "Median (s)": round(np.median(times), 1),
                "Mean (s)": round(np.mean(times), 1),
                "Std Dev (s)": round(np.std(times), 1),
                "5th %ile": round(np.percentile(times, 5), 1),
                "95th %ile": round(np.percentile(times, 95), 1),
                "Range": round(np.percentile(times, 95) - np.percentile(times, 5), 1)
            })
        
        df = pd.DataFrame(summary_data).sort_values("Median (s)")
        
        # export options
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        with col1:
            # prepare detailed results for export
            detailed_results = []
            for strategy_name, times in results.items():
                for sim_num, time in enumerate(times):
                    detailed_results.append({
                        'Strategy': strategy_name,
                        'Simulation': sim_num + 1,
                        'Race_Time_s': time,
                        'Circuit': current_circuit
                    })
            
            detailed_df = pd.DataFrame(detailed_results)
            csv_detailed = detailed_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Export Raw Data",
                data=csv_detailed,
                file_name=f"f1_strategy_raw_results_{current_circuit.lower()}.csv",
                mime="text/csv"
            )
        
        with col2:
            # export summary statistics
            summary_for_export = df.copy()
            summary_for_export['Circuit'] = current_circuit
            csv_summary = summary_for_export.to_csv(index=False)
            
            st.download_button(
                label="📈 Export Summary",
                data=csv_summary,
                file_name=f"f1_strategy_summary_{current_circuit.lower()}.csv",
                mime="text/csv"
            )
        
        with col3:
            # create PDF export functionality
            try:
                from matplotlib.backends.backend_pdf import PdfPages
                import io
                
                # create PDF in memory
                pdf_buffer = io.BytesIO()
                
                with PdfPages(pdf_buffer) as pdf:
                    # create the main plot
                    fig = create_performance_plot(results, current_circuit)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    
                    # create summary table as a figure
                    fig_table, ax = plt.subplots(figsize=(12, 8))
                    ax.axis('tight')
                    ax.axis('off')
                    
                    # create table data
                    table_data = df.values.tolist()
                    headers = df.columns.tolist()
                    
                    table = ax.table(cellText=table_data, colLabels=headers, 
                                   cellLoc='center', loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.2, 1.5)
                    
                    # style the table
                    for i in range(len(headers)):
                        table[(0, i)].set_facecolor('#4472C4')
                        table[(0, i)].set_text_props(weight='bold', color='white')
                    
                    plt.title(f'F1 Strategy Analysis Summary - {current_circuit}', 
                            fontsize=16, fontweight='bold', pad=20)
                    
                    pdf.savefig(fig_table, bbox_inches='tight')
                    plt.close(fig_table)
                
                pdf_buffer.seek(0)
                
                st.download_button(
                    label="📄 Export PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"f1_strategy_report_{current_circuit.lower()}.pdf",
                    mime="application/pdf"
                )
                
            except ImportError:
                st.info("📄 PDF export requires matplotlib - install it to enable this feature")
            except Exception as e:
                st.warning(f"📄 PDF export temporarily unavailable: {str(e)}")
        
        # plot 
        fig = create_performance_plot(results, current_circuit)
        st.pyplot(fig)
        
        # summary table
        st.dataframe(df, use_container_width=True)
        
        # risk analysis
        st.subheader("🏁 Risk Analysis")
        best_median = df["Median (s)"].min()
        
        risk_data = []
        for _, row in df.iterrows():
            time_penalty = row["Median (s)"] - best_median
            risk = row["Range"] / 2
            risk_data.append({
                "Strategy": row["Strategy"],
                "Time Penalty (s)": f"+{time_penalty:.1f}",
                "Risk (±s)": f"{risk:.1f}"
            })
        
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True)
        
        # head-to-head
        st.subheader("⚔️ Head-to-Head Win Rates")
        h2h_data = []
        strategy_names = list(results.keys())
        
        for strat1 in strategy_names:
            row = {'Strategy': strat1}
            for strat2 in strategy_names:
                if strat1 != strat2:
                    win_rate = np.mean(results[strat1] < results[strat2])
                    row[strat2] = f"{win_rate:.1%}"
                else:
                    row[strat2] = "—"
            h2h_data.append(row)
        
        h2h_df = pd.DataFrame(h2h_data).set_index('Strategy')
        st.dataframe(h2h_df, use_container_width=True)
        
        # recommendations
        best = df.iloc[0]
        most_consistent = df.loc[df['Std Dev (s)'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏁 Fastest Strategy", best['Strategy'], f"{best['Median (s)']}s")
        with col2:
            st.metric("🏁 Most Consistent", most_consistent['Strategy'], f"±{most_consistent['Std Dev (s)']}s")

if __name__ == "__main__":
    main()