# 🏎️ F1 Strategy Simulator 2025 🏎️

A comprehensive Formula 1 race strategy analysis tool that uses real F1 data and advanced statistical modeling to simulate and compare different pit stop strategies across all 2025 F1 circuits.

### 🏎️ Advanced Modeling
- **Real F1 Data Integration**: Uses FastF1 API to load actual 2024 race data for tire performance modeling
- **Bayesian Tire Modeling**: MCMC-based statistical models trained on real stint performance data
- **Stochastic Simulation**: Single-sample posterior draws per simulation step for realistic uncertainty
- **Fuel-Corrected Lap Times**: Accounts for changing fuel loads throughout the race
- **Monte Carlo Simulations**: Runs thousands of simulations for robust statistical analysis
- **Compound-Specific Degradation**: Different tire performance models for SOFT, MEDIUM, and HARD compounds

### 🏁 Circuit Coverage
- **All 24 F1 Circuits**: Complete 2025 calendar support
- **Circuit-Specific Parameters**: Lap counts, distances, and fuel consumption rates
- **Real Data Availability**: Shows which circuits have sufficient historical data

### 🏎️ Strategy Analysis
- **9 Pre-defined Strategies**: Common 1-stop and 2-stop strategies with realistic compound sequences
- **Custom Tire Allocation**: Define tire age and availability for advanced scenarios
- **Comprehensive Comparison**: Statistical analysis with confidence intervals and risk metrics
- **Performance Differentiation**: Clear strategy separation based on real tire performance data
- **Risk Assessment**: Understand the variability and reliability of each strategy option

### 🏁 Visualization & Export
- **Interactive Charts**: Distribution plots, box plots, and cumulative probability
- **Head-to-Head Analysis**: Win probability matrices between strategies
- **Multiple Export Options**:
  - Raw simulation data (CSV)
  - Summary statistics (CSV)
  - Complete PDF reports with plots and tables

## 🏎️ Quick Start 🏎️

### Online (Recommended)
Visit the live app: https://f1-strategy-sim.streamlit.app

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/j5t3313/f1-strategy-simulator.git
   cd f1-strategy-simulator
   ```

2. **Install dependencies**
 
```bash
   pip install -r requirements.txt
   ```
3. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🏎️ How to Use 🏎️

### 1. Select Circuit
Choose from any of the 24 F1 circuits in the 2025 calendar. The app will automatically adjust strategy parameters for circuit-specific lap counts and characteristics.

A spinner will appear telling you it is checking for race data availability. It may take a moment to load data via the FastF1 API.

### 2. Choose Strategies
Select from pre-defined strategies using the dropdown.

### 3. Configure Parameters
Parameters are automatically selected based on circuit selected, but you can manually adjust them if you'd like.
- **Base Pace**: Adjust the baseline lap time performance
- **Pit Loss**: Set the time penalty for pit stops
- **Tire Allocation**: Optionally define custom tire sets with age per set
- **Simulations**: Choose the number of Monte Carlo runs (100-2000). THESE TAKE TIME TO RUN - THE MORE SIMULATIONS YOU CHOOSE TO RUN, THE LONGER IT WILL TAKE. 

### 4. Analyze Results
The app provides comprehensive analysis including:
- Performance distributions and statistical summaries
- Risk analysis comparing strategies
- Head-to-head win probability matrices
- Export options for further analysis

### System Requirements
- Python 3.8+
- 4GB+ RAM (for Bayesian MCMC modeling)
- Internet connection (for F1 data download)
- ~500MB disk space (for F1 data cache)

### Key Dependencies
- **Streamlit**: Web application framework
- **FastF1**: Official F1 data API for real race data
- **JAX/NumPyro**: High-performance Bayesian statistical modeling with MCMC
- **Pandas/NumPy**: Data manipulation and numerical analysis
- **Matplotlib**: Advanced visualization and PDF export capabilities

## 🏎️ Technical Details 🏎️

### Data Sources
- **Real F1 Data**: 2024 season race data via FastF1 API for tire performance modeling
- **Circuit Information**: Official lap counts, distances, and fuel consumption calculations
- **Bayesian Tire Models**: MCMC models trained on fuel-corrected lap times vs. stint progression
- **Compound-Specific Parameters**: Different degradation rates and performance characteristics per tire type

### Modeling Approach
- **Fuel Correction**: `Laptime(FC) = Laptime - (Remaining_Laps × Fuel_Per_Lap × Weight_Effect)`
- **Tire Performance**: Bayesian posterior sampling from `laptime ~ Normal(α + β × stint_lap, σ)`
- **Stochastic Simulation**: Single posterior sample per tire performance calculation
- **Fallback Physics**: Compound-specific degradation models when insufficient data
- **Monte Carlo**: 1000+ simulations per strategy for statistical robustness

### Cache Management
The app automatically creates and manages a `.f1_cache` directory for F1 data caching, improving performance on subsequent runs.

## FAQs

### Why didn't you...
- **Include wet and intermediate tires?**: Because wet weather tire strategies are entirely dependent on, well, the weather and are, by their nature, reactive. This isn't the purpose of this project.
- **Factor in dirty air?**: This is a clean air simulation intended to determine which stategy, all other factors being equal, is the most likely to get the driver from the start of the race to the end the fastest. 

## 📄 License

This project is for educational and research purposes. F1 data is accessed through the official FastF1 API under their terms of service.

## ⚠️ Disclaimer

This simulator is for educational and research purposes only. Actual F1 race strategies depend on numerous factors not fully modeled in this application, including:
- **Real-time variables**: Weather conditions, track temperature, and surface evolution
- **Driver factors**: Individual performance variations, mistakes, and racecraft
- **Race dynamics**: Safety car deployments, VSC periods, and red flag situations  
- **Competitor strategy**: Real-time reactive decisions and strategic games
- **Technical issues**: Reliability problems, damage, and setup compromises
- **Regulatory factors**: Penalty risks, tire pressure monitoring, and technical regulations

**Model Limitations:**
- Based on 2024 data which may not reflect 2025 car/tire characteristics
- Assumes perfect execution without driver errors or suboptimal pit timing
- Linear tire degradation models may miss complex thermal and compound behaviors
- No modeling of track evolution, rubber buildup, or changing grip levels

Results should not be used for actual racing decisions, betting, or gambling purposes. You will absolutely regret it.

## 🏎️ Support

For questions, issues, or feature requests, email me:
jessica.5t3313@gmail.com

---

**Enjoy analyzing F1 strategies! 🏁**# f1-strategy-simulator
An F1 race strategy analysis tool using FastF1, Bayesian tire modeling, and Monte Carlo simulation to compare pit stop strategies across all F1 circuits. Built with Streamlit for interactive strategy optimization.
