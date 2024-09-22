import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mt

# Helper Functions
def convert_maturity_to_years(maturity):
    """Convert maturity strings like '1w', '1M', etc., to a fraction of years."""
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7 / 365
    elif 'M' in maturity:
        return int(maturity.replace('M', '')) * 30 / 365
    elif 'y' in maturity:
        return int(maturity.replace('y', ''))
    else:
        raise ValueError("Invalid maturity format. Use '1w', '1M', '3M', etc.")

def convert_maturity_to_days(maturity):
    """Convert maturity strings like '1w', '1M', etc., to days."""
    if 'w' in maturity:
        return int(maturity.replace('w', '')) * 7
    elif 'M' in maturity:
        return int(maturity.replace('M', '')) * 30
    elif 'y' in maturity:
        return int(maturity.replace('y', '')) * 365
    else:
        raise ValueError("Invalid maturity format. Use '1w', '1M', '3M', etc.")

def calculate_forward_price_fixed(df, asset, maturity_days, annual_rate=0.05):
    forward_col = f'Forward Price ({asset})'
    spot_col = asset
    df[forward_col] = np.nan
    for i in range(0, len(df), maturity_days):
        if i + maturity_days < len(df):
            forward_price = df.loc[df.index[i], spot_col] * mt.exp(annual_rate * (maturity_days / 365))
            df.loc[df.index[i:i+maturity_days], forward_col] = forward_price
        else:
            remaining_days = len(df) - i
            forward_price = df.loc[df.index[i], spot_col] * mt.exp(annual_rate * (remaining_days / 365))
            df.loc[df.index[i:], forward_col] = forward_price
    return df

def hedge_strategy_corrected(df, start_date, rewards_frequency, reward_amount, maturity, asset):
    maturity_period = convert_maturity_to_days(maturity)
    if asset not in df.columns:
        raise ValueError(f"Asset '{asset}' not found in data. Available assets: {', '.join(df.columns[1:])}")
    
    asset_data_start_date = df[df[asset].notna()]['Date'].min()
    start_date = pd.to_datetime(start_date) if pd.to_datetime(start_date) >= asset_data_start_date else asset_data_start_date
    
    df = df[df['Date'] >= start_date].copy()
    df = calculate_forward_price_fixed(df, asset, maturity_period)
    
    df[f'Notional Exchanged Forward ({asset})'] = 0.0
    df[f'Notional Exchanged Spot ({asset})'] = 0.0
    df[f'Cumulative Forward ({asset})'] = 0.0
    df[f'Cumulative Spot ({asset})'] = 0.0
    df[f'PnL ({asset})'] = 0.0  # Initialize PnL column
    
    reward_interval = {'Daily': 1, 'Weekly': 7, 'Monthly': 30}[rewards_frequency]
    forward_accumulation = 0
    last_maturity_date = 0
    
    for i in range(len(df)):
        if i % reward_interval == 0:
            spot_price = df.loc[df.index[i], asset]
            forward_price = df.loc[df.index[last_maturity_date], f'Forward Price ({asset})'] if i >= maturity_period else spot_price
            
            # Calculate notional exchanged at the spot price
            df.loc[df.index[i], f'Notional Exchanged Spot ({asset})'] = spot_price * reward_amount
            forward_accumulation += reward_amount
            
            # Calculate PnL
            if i >= maturity_period:
                df.loc[df.index[i], f'PnL ({asset})'] = (forward_price - spot_price) * reward_amount
            
            # Check if it's time to exchange at forward price
            if i - last_maturity_date >= maturity_period and i + maturity_period < len(df):
                df.loc[df.index[i + maturity_period], f'Notional Exchanged Forward ({asset})'] += forward_price * forward_accumulation
                forward_accumulation = 0
                last_maturity_date = i
        
        # Update cumulative values
        if i > 0:
            df.loc[df.index[i], f'Cumulative Spot ({asset})'] = (
                df.loc[df.index[i-1], f'Cumulative Spot ({asset})'] + df.loc[df.index[i], f'Notional Exchanged Spot ({asset})']
            )
            df.loc[df.index[i], f'Cumulative Forward ({asset})'] = (
                df.loc[df.index[i-1], f'Cumulative Forward ({asset})'] + df.loc[df.index[i], f'Notional Exchanged Forward ({asset})']
            )
        else:
            df.loc[df.index[i], f'Cumulative Spot ({asset})'] = df.loc[df.index[i], f'Notional Exchanged Spot ({asset})']
            df.loc[df.index[i], f'Cumulative Forward ({asset})'] = df.loc[df.index[i], f'Notional Exchanged Forward ({asset})']
    
    if forward_accumulation > 0:
        final_forward_price = df[f'Forward Price ({asset})'].iloc[-1]
        df.loc[df.index[-1], f'Notional Exchanged Forward ({asset})'] += forward_accumulation * final_forward_price
        df.loc[df.index[-1], f'Cumulative Forward ({asset})'] += forward_accumulation * final_forward_price
    
    final_spot_notional = f"{round(df[f'Cumulative Spot ({asset})'].iloc[-1]):,}"
    final_forward_notional = f"{round(df[f'Cumulative Forward ({asset})'].iloc[-1]):,}"
    difference = f"{int(final_forward_notional.replace(',', '')) - int(final_spot_notional.replace(',', '')):,}"
    returns = (int(final_forward_notional.replace(',', '')) / int(final_spot_notional.replace(',', '')) - 1) * 100
    
    st.subheader("Results")
    st.write(f"**Final accumulated notional with spot strategy:** {final_spot_notional} USD")
    st.write(f"**Final accumulated notional with forward strategy:** {final_forward_notional} USD")
    st.write(f"**Difference between forward and spot strategies:** {difference} USD")
    st.write(f"**Return of forward strategy relative to spot:** {returns:.2f}%")
    
    return df


def plot_results_adjusted(df, asset, rewards_frequency):
    st.subheader("Spot vs Forward Prices and PnL at Exchange Dates")
    
    # Determine the reward interval based on frequency
    reward_interval = {'Daily': 1, 'Weekly': 7, 'Monthly': 30}[rewards_frequency]
    
    # Extract the dates where exchanges occurred
    exchange_dates = df[df.index % reward_interval == 0]['Date']
    
    # Extract the corresponding spot and forward prices at those dates
    spot_prices_at_exchanges = df.loc[exchange_dates.index, asset]
    forward_prices_at_exchanges = df.loc[exchange_dates.index, f'Forward Price ({asset})']
    
    # Extract the PnL at each exchange date
    pnl_at_exchanges = df.loc[exchange_dates.index, f'PnL ({asset})']
    
    # Plotting the Spot vs Forward Prices
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(exchange_dates, spot_prices_at_exchanges, marker='', linestyle='-', color='blue', label='Spot Price at Exchange')
    ax.plot(exchange_dates, forward_prices_at_exchanges, marker='', linestyle='--', color='orange', label='Forward Price at Exchange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    


def calculate_option_payoff(option_type, is_bought, strike_price, spot_prices, premium):
    if option_type == 'Call':
        intrinsic_values = np.maximum(spot_prices - strike_price, 0)
    else:  # Put
        intrinsic_values = np.maximum(strike_price - spot_prices, 0)
    
    payoff = intrinsic_values - premium if is_bought else premium - intrinsic_values
    return payoff

def plot_payoffs(options):
    if options.empty:
        st.warning("No options added yet. Please add options to see the payoff diagram.")
        return
    
    spot_prices = np.linspace(50, 150, 500)  # Adjusted range centered on 100%
    total_payoff = np.zeros_like(spot_prices)
    
    for _, option in options.iterrows():
        payoff = calculate_option_payoff(
            option_type=option['Type'],
            is_bought=option['Position'] == 'Buy',
            strike_price=option['Strike Price'],
            spot_prices=spot_prices,
            premium=option['Premium']
        )
        total_payoff += payoff
    
    plt.figure(figsize=(8, 4))
    plt.step(spot_prices, total_payoff, label='Total Payoff', color='black', linewidth=2, linestyle='--', where='mid')
    plt.title('Options Payoff Diagram')
    plt.xlabel('Spot Price at Maturity (%)')
    plt.ylabel('Payoff (%)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def black_scholes_price(option_type, S, K, T, r, sigma):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Calculate premium as a percentage of the spot price
    premium_percentage = (price / S) * 100
    return premium_percentage

# Streamlit App Interface
st.set_page_config(page_title="Decimal Hedge - Strategies Simulator", layout="centered")

# Sidebar Navigation
st.sidebar.title("Decimal Hedge")
if st.sidebar.button("Home"):
    st.experimental_set_query_params(page="Home")
if st.sidebar.button("Forward Backtesting"):
    st.experimental_set_query_params(page="ForwardBacktesting")
if st.sidebar.button("Vanilla Options Payoff Simulator"):
    st.experimental_set_query_params(page="VanillaOptionsPayoffSimulator")

# Determine which page to show based on the query parameter
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["Home"])[0]

# Main Page Rendering
if page == "Home":
    st.title("Welcome to Decimal Hedge Strategies Simulator")
    st.markdown("""
    This application is designed to help our clients better understand the financial aspects of hedging their cryptocurrency assets.
    We offer various strategies that can be simulated to visualize potential outcomes and compare different hedging techniques.
    
    Explore the tools provided in this application to see how you can mitigate risks and optimize your returns in the volatile world of crypto.
    
    **Choose a strategy** from the sidebar to get started.
    """)

elif page == "ForwardBacktesting":
    st.title("Forward Backtesting")
    
    # Explanation of the Forward Backtesting Strategy
    st.markdown("""
    In this simulation, we aim to compare the effectiveness of using a spot strategy versus a forward strategy over a specified period. 
    A spot strategy involves exchanging the asset at its current market price (spot price) when rewards are claimed, while a forward strategy 
    locks-in a future price (forward price) today for delivery at a later date. By backtesting these strategies on historical data, 
    we can determine the Pnl you could reach by using a forward hedging strategy.
    
    You can customize the parameters of the simulation, such as the Hedge start date, the frequency of rewards, and the maturity of the forward 
    contracts, to see how different strategies would have performed.
    """)

    # Load data from GitHub
    github_url = 'https://raw.githubusercontent.com/decimalhedge/app/main/HP.xlsx'
    try:
        hp_df = pd.read_excel(github_url)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
    
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Hedge start date", value=pd.to_datetime("2018-03-01"))
    with col2:
        rewards_frequency = st.selectbox("Rewards Frequency", options=['Daily', 'Weekly', 'Monthly'])
    with col3:
        asset = st.selectbox("Asset", options=hp_df.columns[1:])
    
    col1, col2 = st.columns(2)
    with col1:
        reward_amount = st.number_input("Reward Amount in Kind", value=1.0, min_value=0.0)
    with col2:
        maturity = st.selectbox("Forward Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'], index=4)  # Default to '12M'
    
    if st.button("Run Hedging Strategy"):
        with st.spinner("Running hedging strategy simulation..."):
            hedged_df_corrected = hedge_strategy_corrected(hp_df, start_date, rewards_frequency, reward_amount, maturity, asset)
            plot_results_adjusted(hedged_df_corrected, asset, rewards_frequency)


elif page == "VanillaOptionsPayoffSimulator":
    st.title("Vanilla Options Payoff Simulator")
    st.markdown("""
    The Vanilla Options Payoff Simulator allows you to create and visualize different option strategies.
    You can add different option legs, specify their strike prices, maturities, and whether you are buying or selling them.
    
    This tool will calculate the premiums as a percentage of the current spot price and plot the combined payoff diagram for all the options you've added.
    This can help you better understand the potential outcomes of your strategies at different spot prices at maturity.
    """)

    if 'options_data' not in st.session_state:
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])
    
    st.subheader("Market Data")
    col1, col2 = st.columns(2)
    with col1:
        volatility = st.number_input("Volatility (σ)", value=0.2, min_value=0.0, format="%.2f")
    with col2:
        risk_free_rate = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, format="%.2f")
    
    st.subheader("Add New Option Leg")
    with st.form(key='option_form'):
        cols = st.columns(4)
        option_type = cols[0].selectbox("Option Type", options=['Call', 'Put'])
        position = cols[1].selectbox("Position", options=['Buy', 'Sell'])
        strike_price = cols[2].number_input("Strike Price (%)", value=100, min_value=0)
        maturity = cols[3].selectbox("Maturity", options=['1w', '1M', '3M', '6M', '12M', '24M', '36M'])

        # Convert maturity to years
        maturity_in_years = convert_maturity_to_years(maturity)

        premium_percentage = black_scholes_price(option_type, 100, strike_price, maturity_in_years, risk_free_rate, volatility)
        if position == "Sell":
            premium_percentage = -premium_percentage
        
        if st.form_submit_button(label="Add Option"):
            new_option = {
                'Type': option_type,
                'Position': position,
                'Strike Price': strike_price,
                'Premium': premium_percentage,  # Premium as a percentage
                'Volatility': volatility,
                'Maturity': maturity,  # Store the original maturity string
                'Risk-Free Rate': risk_free_rate
            }
            st.session_state.options_data = pd.concat([st.session_state.options_data, pd.DataFrame([new_option])], ignore_index=True)

    # Display current option legs and the sum of premiums
    st.subheader("Current Option Legs")
    if not st.session_state.options_data.empty:
        # Display column headers
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        cols[0].write("Position")
        cols[1].write("Type")
        cols[2].write("Strike Price")
        cols[3].write("Maturity")
        cols[4].write("Volatility")
        cols[5].write("Premium (%)")  # Updated to indicate percentage
        cols[6].write("Remove")

        for idx, option in st.session_state.options_data.iterrows():
            cols = st.columns([1, 1, 1, 1, 1, 1, 1])
            cols[0].write(option['Position'])
            cols[1].write(option['Type'])
            cols[2].write(option['Strike Price'])
            cols[3].write(option['Maturity'])
            cols[4].write(f"{option['Volatility'] * 100:.2f}%")
            cols[5].write(f"{option['Premium']:.2f} %")  # Display premium as a percentage
            remove_button = cols[6].button("❌", key=f"remove_{idx}")

            # Handle removal of option leg
            if remove_button:
                st.session_state.options_data = st.session_state.options_data.drop(idx).reset_index(drop=True)

    # Display sum of premiums
    total_premium = st.session_state.options_data['Premium'].sum()
    st.write(f"**Total Premium:** {total_premium:.2f} %")

    # Improved plotting section
    st.subheader("Options Payoff Diagram")
    if st.session_state.options_data.empty:
        st.warning("No options added yet. Please add options to see the payoff diagram.")
    else:
        plot_payoffs(st.session_state.options_data)

    # Handle reset action
    if st.button("Reset All Options"):
        st.session_state.options_data = pd.DataFrame(columns=['Type', 'Position', 'Strike Price', 'Premium', 'Volatility', 'Maturity', 'Risk-Free Rate'])

