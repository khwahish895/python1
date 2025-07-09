import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="₿ CryptoVision Pro",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for crypto-themed styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b69 100%);
        color: #ffffff;
    }
    
    .crypto-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #f7931a, #50af95, #627eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(247, 147, 26, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .price-card {
        background: linear-gradient(135deg, rgba(247, 147, 26, 0.1), rgba(80, 175, 149, 0.1));
        border: 2px solid rgba(247, 147, 26, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .price-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(247, 147, 26, 0.2);
        border-color: rgba(247, 147, 26, 0.5);
    }
    
    .bull-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.1));
        border: 2px solid rgba(34, 197, 94, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
    }
    
    .bear-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 2px solid rgba(239, 68, 68, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.1);
    }
    
    .neutral-card {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(139, 69, 19, 0.1));
        border: 2px solid rgba(168, 85, 247, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(168, 85, 247, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a3e, #2d1b69);
        border-right: 2px solid rgba(247, 147, 26, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #f7931a, #50af95);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(247, 147, 26, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(80, 175, 149, 0.4);
    }
    
    .metric-positive {
        color: #22c55e;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .metric-negative {
        color: #ef4444;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .metric-neutral {
        color: #a855f7;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        background: linear-gradient(45deg, #f7931a, #50af95);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .crypto-symbol {
        font-size: 2rem;
        margin-right: 10px;
    }
    
    .live-indicator {
        width: 12px;
        height: 12px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .alert-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
        border: 2px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
        backdrop-filter: blur(5px);
    }
    
    .portfolio-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(67, 56, 202, 0.1));
        border: 2px solid rgba(99, 102, 241, 0.4);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cryptocurrency data simulation
@st.cache_data
def generate_crypto_data():
    """Generate realistic cryptocurrency data"""
    cryptos = {
        'Bitcoin': {'symbol': '₿', 'price': 45000, 'volatility': 0.05},
        'Ethereum': {'symbol': 'Ξ', 'price': 3000, 'volatility': 0.07},
        'Binance Coin': {'symbol': 'BNB', 'price': 300, 'volatility': 0.08},
        'Cardano': {'symbol': 'ADA', 'price': 0.5, 'volatility': 0.10},
        'Solana': {'symbol': 'SOL', 'price': 100, 'volatility': 0.12},
        'Polkadot': {'symbol': 'DOT', 'price': 25, 'volatility': 0.09},
        'Dogecoin': {'symbol': 'DOGE', 'price': 0.08, 'volatility': 0.15},
        'Avalanche': {'symbol': 'AVAX', 'price': 35, 'volatility': 0.11}
    }
    
    data = []
    for name, info in cryptos.items():
        # Generate 30 days of historical data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = []
        volume = []
        
        current_price = info['price']
        for i in range(30):
            # Simulate price movement
            change = np.random.normal(0, info['volatility'])
            current_price *= (1 + change)
            prices.append(current_price)
            
            # Simulate volume
            base_volume = random.randint(1000000, 10000000)
            volume.append(base_volume * (1 + abs(change) * 5))
        
        for i, date in enumerate(dates):
            data.append({
                'Date': date,
                'Crypto': name,
                'Symbol': info['symbol'],
                'Price': prices[i],
                'Volume': volume[i],
                'Market_Cap': prices[i] * random.randint(10000000, 50000000),
                'Change_24h': np.random.normal(0, info['volatility']) * 100
            })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_portfolio_data():
    """Generate sample portfolio data"""
    portfolio = {
        'Bitcoin': {'amount': 0.5, 'avg_buy_price': 42000},
        'Ethereum': {'amount': 2.0, 'avg_buy_price': 2800},
        'Cardano': {'amount': 1000, 'avg_buy_price': 0.45},
        'Solana': {'amount': 10, 'avg_buy_price': 90}
    }
    return portfolio

def create_price_chart(df, crypto_name):
    """Create price chart for a specific cryptocurrency"""
    crypto_data = df[df['Crypto'] == crypto_name].sort_values('Date')
    
    fig = go.Figure()
    
    # Candlestick-style line chart
    fig.add_trace(go.Scatter(
        x=crypto_data['Date'],
        y=crypto_data['Price'],
        mode='lines+markers',
        name=f'{crypto_name} Price',
        line=dict(color='#f7931a', width=3),
        marker=dict(size=6, color='#50af95'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Price: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'📈 {crypto_name} Price Movement',
        template='plotly_dark',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#f7931a'),
        xaxis=dict(gridcolor='rgba(247, 147, 26, 0.2)'),
        yaxis=dict(gridcolor='rgba(247, 147, 26, 0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    
    return fig

def create_volume_chart(df, crypto_name):
    """Create volume chart for a specific cryptocurrency"""
    crypto_data = df[df['Crypto'] == crypto_name].sort_values('Date')
    
    fig = px.bar(
        crypto_data, x='Date', y='Volume',
        title=f'📊 {crypto_name} Trading Volume',
        color='Volume',
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#50af95'),
        xaxis=dict(gridcolor='rgba(80, 175, 149, 0.2)'),
        yaxis=dict(gridcolor='rgba(80, 175, 149, 0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_market_overview(df):
    """Create market overview chart"""
    latest_data = df.groupby('Crypto').last().reset_index()
    
    fig = px.treemap(
        latest_data,
        path=['Crypto'],
        values='Market_Cap',
        color='Change_24h',
        color_continuous_scale='RdYlGn',
        title='🌍 Cryptocurrency Market Overview'
    )
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#627eea')
    )
    
    return fig

def create_correlation_matrix(df):
    """Create correlation matrix for different cryptocurrencies"""
    pivot_df = df.pivot(index='Date', columns='Crypto', values='Price')
    correlation_matrix = pivot_df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect='auto',
        title='🔗 Cryptocurrency Price Correlations',
        color_continuous_scale='RdBu_r'
    )
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#a855f7')
    )
    
    return fig

def create_portfolio_pie_chart(portfolio_data, current_prices):
    """Create portfolio allocation pie chart"""
    portfolio_values = []
    labels = []
    
    for crypto, data in portfolio_data.items():
        current_price = current_prices.get(crypto, data['avg_buy_price'])
        value = data['amount'] * current_price
        portfolio_values.append(value)
        labels.append(f"{crypto}<br>${value:,.2f}")
    
    fig = px.pie(
        values=portfolio_values,
        names=labels,
        title='💼 Portfolio Allocation',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        template='plotly_dark',
        font=dict(color='white', family='Rajdhani'),
        title_font=dict(size=20, color='#06b6d4')
    )
    
    return fig

# Main title
st.markdown('<h1 class="crypto-header">₿ CRYPTOVISION PRO</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #f7931a; font-size: 1.3rem; margin-bottom: 2rem;">🚀 Advanced Cryptocurrency Analytics Dashboard 🚀</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f7931a, #50af95); border-radius: 15px; margin-bottom: 20px;">
    <h2 style="color: white; font-family: 'Rajdhani', sans-serif; margin: 0;">📊 CONTROL PANEL</h2>
</div>
""", unsafe_allow_html=True)

# Generate data
df = generate_crypto_data()
portfolio_data = generate_portfolio_data()

# Sidebar options
page = st.sidebar.selectbox(
    "🎯 SELECT VIEW",
    ["🏠 Dashboard", "📈 Price Analysis", "📊 Market Overview", "💼 Portfolio", "🔔 Alerts", "📰 News Simulator"]
)

# Live indicator
st.sidebar.markdown("""
<div style="padding: 15px; background: rgba(34, 197, 94, 0.1); border-radius: 10px; margin: 10px 0;">
    <span class="live-indicator"></span>
    <strong style="color: #22c55e;">LIVE DATA</strong>
</div>
""", unsafe_allow_html=True)

# Get latest prices for calculations
latest_prices = df.groupby('Crypto')['Price'].last().to_dict()

if page == "🏠 Dashboard":
    st.markdown('<h2 class="section-header">🏠 MARKET DASHBOARD</h2>', unsafe_allow_html=True)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_market_cap = df.groupby('Crypto')['Market_Cap'].last().sum()
    avg_change = df.groupby('Crypto')['Change_24h'].last().mean()
    active_cryptos = df['Crypto'].nunique()
    total_volume = df.groupby('Crypto')['Volume'].last().sum()
    
    with col1:
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #f7931a; margin: 0;">💎 TOTAL MARKET CAP</h3>
            <h2 style="color: white; margin: 10px 0;">${total_market_cap/1e12:.2f}T</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        color_class = "metric-positive" if avg_change > 0 else "metric-negative"
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #50af95; margin: 0;">📊 AVG 24H CHANGE</h3>
            <h2 class="{color_class}" style="margin: 10px 0;">{avg_change:+.2f}%</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #627eea; margin: 0;">🪙 ACTIVE CRYPTOS</h3>
            <h2 style="color: white; margin: 10px 0;">{active_cryptos}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #a855f7; margin: 0;">💹 TOTAL VOLUME</h3>
            <h2 style="color: white; margin: 10px 0;">${total_volume/1e9:.1f}B</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Price cards for top cryptocurrencies
    st.markdown('<h3 class="section-header">💰 TOP CRYPTOCURRENCIES</h3>', unsafe_allow_html=True)
    
    latest_data = df.groupby('Crypto').last().reset_index()
    latest_data = latest_data.nlargest(6, 'Market_Cap')
    
    cols = st.columns(3)
    for i, (_, crypto) in enumerate(latest_data.iterrows()):
        col = cols[i % 3]
        
        change = crypto['Change_24h']
        if change > 0:
            card_class = "bull-card"
            arrow = "📈"
            color = "#22c55e"
        else:
            card_class = "bear-card"
            arrow = "📉"
            color = "#ef4444"
        
        with col:
            st.markdown(f'''
            <div class="{card_class}">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span class="crypto-symbol">{crypto['Symbol']}</span>
                    <h3 style="margin: 0; color: white;">{crypto['Crypto']}</h3>
                </div>
                <h2 style="color: white; margin: 10px 0;">${crypto['Price']:,.2f}</h2>
                <div style="display: flex; align-items: center;">
                    <span style="margin-right: 8px;">{arrow}</span>
                    <span style="color: {color}; font-weight: 600;">{change:+.2f}%</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Market overview chart
    st.markdown('<h3 class="section-header">🌍 MARKET OVERVIEW</h3>', unsafe_allow_html=True)
    fig_market = create_market_overview(df)
    st.plotly_chart(fig_market, use_container_width=True)

elif page == "📈 Price Analysis":
    st.markdown('<h2 class="section-header">📈 PRICE ANALYSIS</h2>', unsafe_allow_html=True)
    
    # Crypto selection
    selected_crypto = st.selectbox(
        "🎯 SELECT CRYPTOCURRENCY",
        df['Crypto'].unique(),
        index=0
    )
    
    # Display current info
    latest_crypto_data = df[df['Crypto'] == selected_crypto].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #f7931a; margin: 0;">💰 CURRENT PRICE</h3>
            <h2 style="color: white; margin: 10px 0;">${latest_crypto_data['Price']:,.2f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        change = latest_crypto_data['Change_24h']
        color = "#22c55e" if change > 0 else "#ef4444"
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #50af95; margin: 0;">📊 24H CHANGE</h3>
            <h2 style="color: {color}; margin: 10px 0;">{change:+.2f}%</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="price-card">
            <h3 style="color: #627eea; margin: 0;">💹 VOLUME</h3>
            <h2 style="color: white; margin: 10px 0;">${latest_crypto_data['Volume']/1e6:.1f}M</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_price = create_price_chart(df, selected_crypto)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_volume = create_volume_chart(df, selected_crypto)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Technical indicators simulation
    st.markdown('<h3 class="section-header">⚙️ TECHNICAL INDICATORS</h3>', unsafe_allow_html=True)
    
    crypto_data = df[df['Crypto'] == selected_crypto].sort_values('Date')
    prices = crypto_data['Price'].values
    
    # Simple moving averages
    sma_7 = np.convolve(prices, np.ones(7)/7, mode='valid')
    sma_14 = np.convolve(prices, np.ones(14)/14, mode='valid')
    
    # RSI simulation
    rsi = 50 + np.random.normal(0, 15)
    rsi = max(0, min(100, rsi))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'''
        <div class="neutral-card">
            <h3 style="color: #a855f7; margin: 0;">📊 RSI (14)</h3>
            <h2 style="color: white; margin: 10px 0;">{rsi:.1f}</h2>
            <p style="color: #a855f7; margin: 0;">
                {'🔥 Overbought' if rsi > 70 else '❄️ Oversold' if rsi < 30 else '⚖️ Neutral'}
            </p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="neutral-card">
            <h3 style="color: #a855f7; margin: 0;">📈 SMA (7)</h3>
            <h2 style="color: white; margin: 10px 0;">${sma_7[-1]:,.2f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="neutral-card">
            <h3 style="color: #a855f7; margin: 0;">📈 SMA (14)</h3>
            <h2 style="color: white; margin: 10px 0;">${sma_14[-1]:,.2f}</h2>
        </div>
        ''', unsafe_allow_html=True)

elif page == "📊 Market Overview":
    st.markdown('<h2 class="section-header">📊 MARKET OVERVIEW</h2>', unsafe_allow_html=True)
    
    # Market summary
    col1, col2 = st.columns(2)
    
    with col1:
        # Top gainers
        latest_data = df.groupby('Crypto').last().reset_index()
        top_gainers = latest_data.nlargest(3, 'Change_24h')
        
        st.markdown('<h3 style="color: #22c55e;">📈 TOP GAINERS</h3>', unsafe_allow_html=True)
        for _, crypto in top_gainers.iterrows():
            st.markdown(f'''
            <div class="bull-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: white;">{crypto['Symbol']} {crypto['Crypto']}</h4>
                        <p style="margin: 5px 0; color: #22c55e;">${crypto['Price']:,.2f}</p>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="margin: 0; color: #22c55e;">+{crypto['Change_24h']:.2f}%</h3>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        # Top losers
        top_losers = latest_data.nsmallest(3, 'Change_24h')
        
        st.markdown('<h3 style="color: #ef4444;">📉 TOP LOSERS</h3>', unsafe_allow_html=True)
        for _, crypto in top_losers.iterrows():
            st.markdown(f'''
            <div class="bear-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: white;">{crypto['Symbol']} {crypto['Crypto']}</h4>
                        <p style="margin: 5px 0; color: #ef4444;">${crypto['Price']:,.2f}</p>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="margin: 0; color: #ef4444;">{crypto['Change_24h']:.2f}%</h3>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Correlation matrix
    st.markdown('<h3 class="section-header">🔗 PRICE CORRELATIONS</h3>', unsafe_allow_html=True)
    fig_corr = create_correlation_matrix(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Market data table
    st.markdown('<h3 class="section-header">📋 MARKET DATA</h3>', unsafe_allow_html=True)
    
    display_data = latest_data[['Crypto', 'Symbol', 'Price', 'Change_24h', 'Volume', 'Market_Cap']].copy()
    display_data['Price'] = display_data['Price'].apply(lambda x: f"${x:,.2f}")
    display_data['Change_24h'] = display_data['Change_24h'].apply(lambda x: f"{x:+.2f}%")
    display_data['Volume'] = display_data['Volume'].apply(lambda x: f"${x/1e6:.1f}M")
    display_data['Market_Cap'] = display_data['Market_Cap'].apply(lambda x: f"${x/1e9:.1f}B")
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)

elif page == "💼 Portfolio":
    st.markdown('<h2 class="section-header">💼 PORTFOLIO TRACKER</h2>', unsafe_allow_html=True)
    
    # Calculate portfolio value
    total_value = 0
    total_cost = 0
    
    for crypto, data in portfolio_data.items():
        current_price = latest_prices.get(crypto, data['avg_buy_price'])
        current_value = data['amount'] * current_price
        cost = data['amount'] * data['avg_buy_price']
        total_value += current_value
        total_cost += cost
    
    total_pnl = total_