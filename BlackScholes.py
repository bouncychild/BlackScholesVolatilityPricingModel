import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#####################
# Page configuration
st.set_page_config(
    page_title="Advanaced Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/*Main Styling*/
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
    width: auto;
    margin: 0 auto;
}
            
.metric-put {
    background-color: rgba(255, 204, 203, 0.3);
    color: black;
    border-radius: 10px;
    border-left: 5px solid #ffcccb;
}
            
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}
            
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
    font-weight: bold;
}    

/* Greeks styling */
.greeks-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
}  

.greek-box {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 10px;
    flex: 1;
    min-width: 120px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.greek-value {
    font-size: 1.2rem;
    font-weight: bold;
    text-align: center;
}

.greek-label {
    font-size: 0.9rem;
    text-align: center;
    color: #6c757d;
}

.stTabs [data-baseweb="tab] {
    padding: 8px 16px;
    border-radius: 4px 4px 0 0;
}
    
.stTabs [aria-selected="true] {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, 
                current_price: float, volatility: float, interest_rate: float):
        
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate
        self.call_price = None
        self.put_price = None
        self.call_delta = None
        self.put_delta = None
        self.gamma = None
        self.vega = None
        self.call_theta = None
        self.put_theta = None
        self.rho = None

    def calculate_prices(self):
        S = self.current_price
        K = self.strike
        T = self.time_to_maturity
        sigma = self.volatility
        r = self.interest_rate

        if T <= 0:
            # Handle expiration case
            self.call_price = max(S - K, 0)
            self.put_price = max(K - S, 0)
            return self.call_price, self.put_price

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        # Option prices
        self.call_price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        self.put_price = K * norm.cdf(-d2) - S * exp(-r * T) * norm.cdf(-d1)

        # Greeks
        # Delta 
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)
        
        # Gamma 
        self.gamma = norm.pdf(d1) / (S * sigma * sqrt(T))

        # Vega 
        self.vega = S * norm.pdf(d1) * sqrt(T) * 0.01 # per 1% change in vol

        # Theta
        self.call_theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)) / 365 # per day
        self.put_theta = (-S * norm.pdf(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)) / 365 # per day

        # Rho 
        self.call_rho = K * T * exp(-r * T) * norm.cdf(d2) * 0.01 # per 1% change
        self.put_rho = -K * T * exp(-r * T) * norm.cdf(-d2) * 0.01 # per 1% change

        return self.call_price, self.put_price

def plot_3d_surface(bs_model, spot_range, vol_range, option_type='call'):
    S_grid, V_grid = np.meshgrid(spot_range, vol_range)
    prices = np.zeros_like(S_grid)

    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            temp_bs = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot_range[j],
                volatility=vol_range[i],
                interest_rate=bs_model.interest_rate
            )
            temp_bs.calculate_prices()
            prices[i, j] = temp_bs.call_price if option_type == 'call' else temp_bs.put_price

    fig = go.Figure(data=[go.Surface(z=prices, x=spot_range, y=vol_range)])
    fig.update_layout(
        title=f'{option_type.upper()} Option Price Surface',
        scene=dict(
            xaxis_title='Spot Price',
            yaxis_title='Volatility',
            zaxis_title='Option Price'
        ),
        margin=dict(l=65, r=50, b=65, t=90),
        height=600
    )

    return fig
        
def plot_greeks(bs_model, spot_range, greek_type='delta'):
    greeks = {
        'delta': {'call': [], 'put': []},
        'gamma': [],
        'vega': [],
        'theta': {'call': [], 'put': []},
        'rho': {'call': [], 'put': []}
    }

    for S in spot_range:
        temp_bs = BlackScholes(
            time_to_maturity=bs_model.time_to_maturity,
            strike=bs_model.strike,
            current_price=S,
            volatility=bs_model.volatility,
            interest_rate=bs_model.interest_rate
        )
        temp_bs.calculate_prices()

        greeks['delta']['call'].append(temp_bs.call_delta)
        greeks['delta']['put'].append(temp_bs.put_delta)
        greeks['gamma'].append(temp_bs.gamma)
        greeks['vega'].append(temp_bs.vega)
        greeks['theta']['call'].append(temp_bs.call_theta)
        greeks['theta']['put'].append(temp_bs.put_theta)
        greeks['rho']['call'].append(temp_bs.call_rho)
        greeks['rho']['put'].append(temp_bs.put_rho)

    fig = go.Figure()

    if greek_type in ['delta', 'theta', 'rho']:
        fig.add_trace(go.Scatter(
            x=spot_range, y=greeks[greek_type]['call'],
            mode='lines', name=f'Call {greek_type.capitalize()}',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=spot_range, y=greeks[greek_type]['put'],
            mode='lines', name=f'Put {greek_type.capitalize()}',
            line=dict(color='red')
        ))
    else:
        fig.add_trace(go.Scatter(
            x=spot_range, y=greeks[greek_type],
            mode='lines', name=greek_type.capitalize(),
            line=dict(color='blue')
        ))


    fig.update_layout(
        title=f' {greek_type.capitalize()} vs. Spot Price',
        xaxis_title='Spot Price',
        yaxis_title=greek_type.capitalize(),
        height=500,
        margin=dict(l=50, r=50, b=50, t=80)
    )

    return fig

#Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("'Created by: Han-Wen'")

    current_price = st.number_input("Current Asset Price (S)", value=100.0, min_value=0.01, step=1.0)
    strike = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0)
    time_to_maturity = st.number_input("Time to Maturity (Years) (T)", value=1.0, min_value=0.0, max_value=10.0, step=0.1)
    volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=2.0, step=0.01)
    interest_rate = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=0.2, step=0.01)

    st.markdown("---")
    st.header("Visualization Parameters")

    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.5, step=1.0)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.5, step=1.0)
    vol_min = st.slider('Min Volatility', min_value=0.01, max_value=1.0, value=0.05, step=0.01)
    vol_max = st.slider('Max Volatility', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    num_points = st.slider('Number of Points in Grid', min_value=5, max_value=50, value=20)

# Main Page
st.title("Advanced Black-Scholes Option Pricing Model")

# Calculate option prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

# Display input parameters
with st.expander("Model Parameters", expanded=True):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Spot Price (S)", f"${current_price:.2f}")
    with col2:
        st.metric("Strike Price (K)", f"${strike:.2f}")
    with col3:
        st.metric("Time to Maturity (T)", f"{time_to_maturity:.2f} years")
    with col4:
        st.metric("Volatility (σ)", f"{volatility:.2%}")
    with col5: 
        st.metric("Risk-Free Rate (r)", f"{interest_rate:.2%}")


# Display option Prices
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Option Price</div>
                <div class="metric-label">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Option Price</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>                
        """, unsafe_allow_html=True)

# Display Greeks 
st.subheader("Option Greeks")
greeks_col1, greeks_col2, greeks_col3, greeks_col4, greeks_col5 = st.columns(5)
with greeks_col1:
    st.markdown("""
        <div class="greek-box">
            <div class="greek-label">Call Delta</div>
            <div class="greek-value">{:.4f}</div>
        </div>
        """.format(bs_model.call_delta), unsafe_allow_html=True)
with greeks_col2:
    st.markdown("""
    <div class="greek-box">
        <div class="greek-label">Put Delta</div>
        <div class="greek-value">{:.4f}</div>
    </div>
    """.format(bs_model.put_delta), unsafe_allow_html=True)
with greeks_col3:
    st.markdown("""
    <div class="greek-box">
        <div class="greek-label">Gamma</div>
        <div class="greek-value">{:.4f}</div>
    </div>
    """.format(bs_model.gamma), unsafe_allow_html=True)
with greeks_col4:
    st.markdown("""
    <div class="greek-box">
        <div class="greek-label">Vega</div>
        <div class="greek-value">{:.4f}</div>
    </div>
    """.format(bs_model.vega), unsafe_allow_html=True)
with greeks_col5:
    st.markdown("""
    <div class="greek-box">
        <div class="greek-label">Theta (per day)</div>
        <div class="greek-value">{:.4f}</div>
    </div>
    """.format(bs_model.call_theta), unsafe_allow_html=True)

# Visualization section
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["3D Price Surface", "Greeks Analysis", "Heatmaps"])

with tab1:
    st.header("3D Option Price Surface")
    spot_range = np.linspace(spot_min, spot_max, num_points)
    vol_range = np.linspace(vol_min, vol_max, num_points)

    col1, col2 = st.columns(2)
    with col1:
        fig_call = plot_3d_surface(bs_model, spot_range, vol_range, 'call')
        st.plotly_chart(fig_call, use_container_width=True)
    with col2:
        fig_put = plot_3d_surface(bs_model, spot_range, vol_range, 'put')
        st.plotly_chart(fig_put, use_container_width=True)

with tab2:
    st.header("Greeks Analysis")
    greek_type = st.selectbox("Select Greek to visualize", 
                             ['delta', 'gamma', 'vega', 'theta', 'rho'])
    
    spot_range = np.linspace(spot_min, spot_max, 100)
    fig = plot_greeks(bs_model, spot_range, greek_type)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Option Price Heatmaps")
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)
    
    # Calculate prices for heatmap
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_bs = BlackScholes(
                time_to_maturity=bs_model.time_to_maturity,
                strike=bs_model.strike,
                current_price=spot,
                volatility=vol,
                interest_rate=bs_model.interest_rate
            )
            temp_bs.calculate_prices()
            call_prices[i, j] = temp_bs.call_price
            put_prices[i, j] = temp_bs.put_price
    
    # Plot heatmaps
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(call_prices, 
                    xticklabels=np.round(spot_range, 1), 
                    yticklabels=np.round(vol_range, 2), 
                    annot=True, fmt=".1f", cmap="YlGn", ax=ax)
        ax.set_title('Call Option Prices')
        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Volatility')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(put_prices, 
                    xticklabels=np.round(spot_range, 1), 
                    yticklabels=np.round(vol_range, 2), 
                    annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title('Put Option Prices')
        ax.set_xlabel('Spot Price')
        ax.set_ylabel('Volatility')
        st.pyplot(fig)

# Model explanation
with st.expander("About the Black-Scholes Model"):
    st.markdown("""
    The Black-Scholes model is a mathematical model for pricing options contracts. The key assumptions are:
    
    - The option is European (can only be exercised at expiration)
    - No dividends are paid during the option's life
    - Markets are efficient (no arbitrage opportunities)
    - No transaction costs
    - Risk-free rate and volatility are known and constant
    - Returns are lognormally distributed
    
    The model calculates the theoretical price of options using the following formula for a call option:
    
    $$
    C = S_0 N(d_1) - K e^{-rT} N(d_2)
    $$
    
    Where:
    - $C$ = Call option price
    - $S_0$ = Current stock price
    - $K$ = Strike price
    - $T$ = Time to maturity
    - $r$ = Risk-free interest rate
    - $N$ = Cumulative standard normal distribution
    - $d_1 = \\frac{\\ln(S_0/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$
    - $d_2 = d_1 - \\sigma\\sqrt{T}$
    
    The put option price is calculated using put-call parity:
    
    $$
    P = K e^{-rT} N(-d_2) - S_0 N(-d_1)
    $$
    """)