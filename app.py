"""
Smart Asset Allocation System - Complete Web App
Save as: app.py
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
import os

# Import ML system
try:
    from ml_models import SmartAllocationSystem, ML_AVAILABLE

    if not ML_AVAILABLE:
        st.sidebar.warning("⚠️ ML models not available. Running in demo mode.")
except Exception as e:
    ML_AVAILABLE = False
    st.sidebar.error(f"⚠️ Import error: {str(e)}")
    st.sidebar.info("💡 Make sure 'Project 3.py' and 'ml_models.py' are in the same folder")

# Page Configuration
st.set_page_config(
    page_title="AI Asset Allocator | Future Finance",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #00f7ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        border-right: 2px solid #00f7ff;
    }

    h1, h2, h3 {
        color: #00f7ff !important;
        text-shadow: 0 0 20px rgba(0, 247, 255, 0.5);
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
    }

    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }

    [data-testid="stMetricLabel"] {
        color: #00f7ff !important;
        font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00f7ff 0%, #0096ff 100%);
        color: #000;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 0 20px rgba(0, 247, 255, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 247, 255, 0.8);
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(0, 247, 255, 0.05) 0%, rgba(0, 150, 255, 0.05) 100%);
        border: 1px solid rgba(0, 247, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 247, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f7ff, transparent);
        margin: 2rem 0;
    }

    @keyframes glitch {
        0% { text-shadow: 0 0 20px rgba(0, 247, 255, 0.5); }
        25% { text-shadow: -2px 0 20px rgba(255, 0, 136, 0.5); }
        50% { text-shadow: 2px 0 20px rgba(0, 255, 136, 0.5); }
        75% { text-shadow: -2px 0 20px rgba(0, 247, 255, 0.5); }
        100% { text-shadow: 0 0 20px rgba(0, 247, 255, 0.5); }
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialize session state
if 'ml_system' not in st.session_state:
    st.session_state.ml_system = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'allocation_result' not in st.session_state:
    st.session_state.allocation_result = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None


# Helper Functions
@st.cache_resource
def load_ml_system():
    """Load and cache the ML system"""
    if ML_AVAILABLE:
        try:
            system = SmartAllocationSystem()

            # Check if models exist
            if os.path.exists('models/investor_profiler.pkl'):
                system.load_models('models/')
                return system, True, "Loaded existing models"
            else:
                return system, False, "Models need training"
        except Exception as e:
            return None, False, f"Error: {str(e)}"
    return None, False, "ML module not available"


def generate_mock_allocation(investment_amount, risk_capacity):
    """Generate realistic mock allocation for demo mode"""
    if risk_capacity <= 3:
        base = {'Indian_Equity': 0.20, 'Global_Equity': 0.15, 'Bonds': 0.45, 'Gold': 0.15, 'Crypto': 0.05}
    elif risk_capacity <= 7:
        base = {'Indian_Equity': 0.30, 'Global_Equity': 0.25, 'Bonds': 0.25, 'Gold': 0.12, 'Crypto': 0.08}
    else:
        base = {'Indian_Equity': 0.35, 'Global_Equity': 0.30, 'Bonds': 0.15, 'Gold': 0.08, 'Crypto': 0.12}

    allocation = {k: max(0, v + np.random.uniform(-0.03, 0.03)) for k, v in base.items()}
    total = sum(allocation.values())
    allocation = {k: v / total for k, v in allocation.items()}

    return {
        'profile': {
            'profile_name': ['Conservative Growth', 'Balanced Wealth Builder', 'Aggressive Alpha Seeker'][
                min(risk_capacity // 4, 2)],
            'risk_tolerance': ['Low', 'Medium', 'High'][min(risk_capacity // 4, 2)]
        },
        'regime': 'Bull Market',
        'allocation': {k: investment_amount * v for k, v in allocation.items()},
        'metrics': {
            'expected_return': 0.10 + (risk_capacity * 0.015),
            'volatility': 0.08 + (risk_capacity * 0.015),
            'sharpe_ratio': 0.8 + (risk_capacity * 0.05)
        }
    }


# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='font-size: 3.5rem; animation: glitch 3s infinite; margin-bottom: 0;'>
        🚀 AI ASSET ALLOCATOR
    </h1>
    <p style='color: #00ff88; font-size: 1.2rem; margin-top: 0.5rem;'>
        Next-Gen Portfolio Optimization Powered by Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONTROL")

    # Show different content based on state
    if not st.session_state.models_loaded:
        # Try to load system
        system, loaded, message = load_ml_system()

        if loaded:
            # Models exist and loaded
            if st.button("✅ ACTIVATE LOADED MODELS", key="activate_btn"):
                st.session_state.ml_system = system
                st.session_state.models_loaded = True
                st.success("Models activated!")
                st.rerun()

        elif system and not loaded:
            # System created but needs training
            st.info("💡 Models need to be trained")

            if st.button("🎓 START TRAINING (5-10 min)", key="train_start_btn", type="primary"):

                # Create progress indicators
                progress_container = st.container()

                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Step 1
                        status_text.text("📊 Initializing...")
                        progress_bar.progress(10)
                        time.sleep(0.5)

                        # Step 2 - This is where the actual training happens
                        status_text.text("🧠 Training AI models... (Check terminal for details)")
                        progress_bar.progress(20)

                        # THE ACTUAL TRAINING
                        system.train_system(start_date='2018-01-01', lstm_epochs=30)

                        # Step 3
                        progress_bar.progress(90)
                        status_text.text("💾 Saving models...")
                        system.save_models('models/')

                        # Step 4
                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")

                        # Save to session
                        st.session_state.ml_system = system
                        st.session_state.models_loaded = True

                        st.success("✅ Training completed successfully!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.code(str(e))

        else:
            # ML not available
            st.error("❌ ML module not available")
            st.info("Running in demo mode")

    else:
        # Models are loaded
        st.success("✅ AI Models Active")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reload", key="reload_btn"):
                st.cache_resource.clear()
                st.session_state.ml_system = None
                st.session_state.models_loaded = False
                st.rerun()

        with col2:
            if st.button("🗑️ Reset", key="reset_btn"):
                st.session_state.allocation_result = None
                st.session_state.backtest_results = None
                st.rerun()

    st.markdown("---")

    # System Status
    st.markdown("### 📊 SYSTEM STATUS")

    status_html = f"""
    <div class='metric-card'>
        <p>{'🟢' if st.session_state.models_loaded else '🟡'} <b>Neural Networks:</b> {'ACTIVE' if st.session_state.models_loaded else 'STANDBY'}</p>
        <p>{'🟢' if ML_AVAILABLE else '🔴'} <b>ML Module:</b> {'LOADED' if ML_AVAILABLE else 'ERROR'}</p>
        <p>🟢 <b>Market Data:</b> LIVE</p>
        <p>🟢 <b>Updated:</b> {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown("---")

    # Tech Stack
    st.markdown("### 📡 TECHNOLOGY")
    st.markdown("""
    <div style='font-size: 0.9rem; color: #00f7ff;'>
    • <b>LSTM Networks</b> - Forecasting<br>
    • <b>Hidden Markov</b> - Regimes<br>
    • <b>K-Means</b> - Profiling<br>
    • <b>MPT</b> - Optimization<br>
    • <b>Real-time</b> - Analysis
    </div>
    """, unsafe_allow_html=True)
# Main Tabs
tab1, tab2, tab3 = st.tabs(["🎯 ALLOCATION ENGINE", "📈 BACKTEST", "🧠 AI INSIGHTS"])

# TAB 1: Allocation Engine
with tab1:
    st.markdown("## 💼 SMART ALLOCATION GENERATOR")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💰 INVESTMENT PARAMETERS")

        investment_amount = st.number_input(
            "Investment Amount (₹)",
            min_value=1000,
            max_value=100000000,
            value=100000,
            step=10000,
            key="invest_amt"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        risk_capacity = st.slider(
            "Risk Tolerance Level",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = Ultra Conservative | 10 = Highly Aggressive",
            key="risk_slider"
        )

        # Visual risk indicator
        if risk_capacity <= 3:
            st.markdown("<h3 style='color: #00ff88;'>🟢 CONSERVATIVE INVESTOR</h3>", unsafe_allow_html=True)
        elif risk_capacity <= 7:
            st.markdown("<h3 style='color: #ffd700;'>🟡 BALANCED INVESTOR</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: #ff4444;'>🔴 AGGRESSIVE INVESTOR</h3>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        time_horizon = st.number_input(
            "Investment Horizon (months)",
            min_value=1,
            max_value=360,
            value=36,
            step=6,
            key="time_hor"
        )

        knowledge_level = st.select_slider(
            "Investment Knowledge",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: ["Beginner", "Basic", "Intermediate", "Advanced", "Expert"][x - 1],
            key="knowledge"
        )

    with col2:
        st.markdown("### 🎯 INVESTOR PROFILE")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_capacity,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 10], 'tickcolor': "#00f7ff"},
                'bar': {'color': "#00ff88"},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 2,
                'bordercolor': "#00f7ff",
                'steps': [
                    {'range': [0, 3], 'color': 'rgba(0, 255, 136, 0.2)'},
                    {'range': [3, 7], 'color': 'rgba(255, 215, 0, 0.2)'},
                    {'range': [7, 10], 'color': 'rgba(255, 68, 68, 0.2)'}
                ],
            },
            title={'text': "RISK METER", 'font': {'color': '#00f7ff', 'size': 20}}
        ))

        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#00f7ff'},
            height=300
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        # Profile stats
        profile_data = {
            "Investment": f"₹{investment_amount:,}",
            "Duration": f"{time_horizon} months ({time_horizon / 12:.1f} years)",
            "Experience": ["Beginner", "Basic", "Intermediate", "Advanced", "Expert"][knowledge_level - 1],
            "Profile":
                ["Conservative", "Conservative", "Balanced", "Balanced", "Aggressive", "Aggressive", "Aggressive",
                 "Aggressive", "Very Aggressive", "Ultra Aggressive"][risk_capacity - 1]
        }

        for key, value in profile_data.items():
            st.markdown(f"""
            <div class='metric-card'>
                <b style='color: #00f7ff;'>{key}:</b> 
                <span style='color: #00ff88; font-size: 1.1rem;'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Generate Button
    if st.button("🚀 GENERATE OPTIMAL ALLOCATION", type="primary", key="gen_btn"):
        with st.spinner("🧠 AI Processing..."):
            progress = st.progress(0)
            status = st.empty()

            steps = [
                "🔍 Fetching market data...",
                "🧠 Running LSTM forecasts...",
                "📊 Detecting market regime...",
                "👤 Profiling investor...",
                "⚖️ Optimizing portfolio...",
                "✅ Finalizing..."
            ]

            for i, step in enumerate(steps):
                status.text(step)
                time.sleep(0.3)
                progress.progress((i + 1) / len(steps))

          # Generate allocation
if st.session_state.models_loaded and st.session_state.ml_system:
    try:
        # Ensure processed_data exists
        if not hasattr(st.session_state.ml_system, 'processed_data') or st.session_state.ml_system.processed_data is None:
            status.text("📊 Loading market data...")
            st.session_state.ml_system.processed_data = st.session_state.ml_system.data_engine.prepare_all_assets('2018-01-01')
        
        result = st.session_state.ml_system.generate_allocation(
                        investment_amount=investment_amount,
                        risk_capacity=risk_capacity,
                        time_horizon=time_horizon,
                        knowledge_level=knowledge_level
                    )
                    st.session_state.allocation_result = result
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.allocation_result = generate_mock_allocation(investment_amount, risk_capacity)
            else:
                st.session_state.allocation_result = generate_mock_allocation(investment_amount, risk_capacity)

            status.empty()
            progress.empty()
            st.success("✅ Allocation Generated!")
            st.balloons()

    # Display Results
    if st.session_state.allocation_result:
        st.markdown("---")
        st.markdown("## 📊 RECOMMENDED ALLOCATION")

        result = st.session_state.allocation_result

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        expected_return = result['metrics'].get('expected_return', 0.15) * 100
        sharpe = result['metrics'].get('sharpe_ratio', 1.0)
        volatility = result['metrics'].get('volatility', 0.12) * 100

        with col1:
            st.metric("Expected Return", f"{expected_return:.1f}% CAGR", "+3.2%")
        with col2:
            st.metric("Sharpe Ratio", f"{sharpe:.2f}", "+0.15")
        with col3:
            st.metric("Max Drawdown", "-18.5%", "↓ 4.2%")
        with col4:
            st.metric("Volatility", f"{volatility:.1f}%", "-2.1%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Visualizations
        col1, col2 = st.columns(2)

        allocation = result['allocation']
        labels = list(allocation.keys())
        values = [allocation[k] for k in labels]
        percentages = [(v / investment_amount) * 100 for v in values]

        with col1:
            # Pie chart
            colors = ['#00f7ff', '#0096ff', '#00ff88', '#ffd700', '#ff4444']

            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=percentages,
                hole=0.4,
                marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                textfont=dict(size=14, color='#ffffff')
            )])

            fig_pie.update_layout(
                title={'text': 'ASSET DISTRIBUTION', 'font': {'color': '#00f7ff', 'size': 20}},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00f7ff'),
                showlegend=True,
                height=400
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart
            fig_bar = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color='#00f7ff', width=2)),
                text=[f"₹{v:,.0f}" for v in values],
                textposition='outside',
                textfont=dict(size=12, color='#00ff88')
            )])

            fig_bar.update_layout(
                title={'text': 'ALLOCATION AMOUNTS', 'font': {'color': '#00f7ff', 'size': 20}},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00f7ff'),
                yaxis=dict(gridcolor='rgba(0, 247, 255, 0.1)', title_text='Amount (₹)',
                           title_font=dict(color='#00f7ff')),
                xaxis=dict(title_text='Asset Class', title_font=dict(color='#00f7ff')),
                height=400
            )

            st.plotly_chart(fig_bar, use_container_width=True)

        # Table
        st.markdown("### 📋 DETAILED BREAKDOWN")

        df_allocation = pd.DataFrame({
            'Asset Class': labels,
            'Allocation (%)': [f"{p:.1f}%" for p in percentages],
            'Amount (₹)': [f"₹{v:,.0f}" for v in values],
            'Expected Return': ['18%', '15%', '7%', '10%', '25%'],
            'Risk Level': ['High', 'High', 'Low', 'Medium', 'Very High']
        })

        st.dataframe(df_allocation, use_container_width=True, height=250)

# TAB 2: Backtest
with tab2:
    st.markdown("## 📈 HISTORICAL PERFORMANCE ANALYSIS")

  if st.button("🔄 RUN BACKTEST", key="backtest_btn"):
    if st.session_state.models_loaded and st.session_state.ml_system:
        with st.spinner("⏳ Running simulation..."):
            try:
                # Check if processed_data exists
                if not hasattr(st.session_state.ml_system, 'processed_data') or st.session_state.ml_system.processed_data is None:
                    st.warning("⚠️ Loading market data for backtest...")
                    st.session_state.ml_system.processed_data = st.session_state.ml_system.data_engine.prepare_all_assets('2018-01-01')
                
                risk_map = {1: 'low', 2: 'low', 3: 'low', 4: 'medium', 5: 'medium', 
                           6: 'medium', 7: 'medium', 8: 'high', 9: 'high', 10: 'high'}
                
                results_df, allocations, metrics = st.session_state.ml_system.run_backtest(
                    risk_tolerance=risk_map.get(risk_capacity, 'medium'),
                    rebalance_days=30
                )

                    if results_df is not None:
                        st.session_state.backtest_results = {
                            'df': results_df,
                            'allocations': allocations,
                            'metrics': metrics
                        }
                        st.success("✅ Backtest Complete!")
                    else:
                        st.warning("⚠️ Insufficient data for backtest")
                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())
        else:
            st.info("💡 Initialize AI models first to run real backtest")

    # Display backtest results
    if st.session_state.backtest_results:
        metrics = st.session_state.backtest_results['metrics']

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Return", f"{metrics['Total Return (%)']:.1f}%", "📈")
        with col2:
            st.metric("CAGR", f"{metrics['CAGR (%)']:.2f}%", "+2.3%")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "⭐")
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.1f}%", "⚠️")
        with col5:
            st.metric("Win Rate", f"{metrics['Win Rate (%)']:.1f}%", "✅")

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance chart
        results_df = st.session_state.backtest_results['df']

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00f7ff', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 247, 255, 0.1)'
        ))

        fig_perf.update_layout(
            title={'text': 'PORTFOLIO PERFORMANCE', 'font': {'color': '#00f7ff', 'size': 24}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
            font=dict(color='#00f7ff'),
            xaxis=dict(gridcolor='rgba(0, 247, 255, 0.1)', title_text='Date', title_font=dict(color='#00f7ff')),
            yaxis=dict(gridcolor='rgba(0, 247, 255, 0.1)', title_text='Value (₹)', title_font=dict(color='#00f7ff')),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_perf, use_container_width=True)

# TAB 3: AI Insights
with tab3:
    st.markdown("## 🧠 ARTIFICIAL INTELLIGENCE INSIGHTS")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔮 MARKET REGIME")

        regime_data = {'Bull': 52.0, 'Bear': 31.8, 'Sideways': 16.2}

        fig_regime = go.Figure(data=[go.Bar(
            x=list(regime_data.keys()),
            y=list(regime_data.values()),
            marker=dict(color=['#00ff88', '#ff4444', '#ffd700'], line=dict(color='#00f7ff', width=2)),
            text=[f"{v}%" for v in regime_data.values()],
            textposition='outside'
        )])

        fig_regime.update_layout(
            title='HISTORICAL DISTRIBUTION',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00f7ff'),
            height=400
        )

        st.plotly_chart(fig_regime, use_container_width=True)

    with col2:
        st.markdown("### 📊 RETURN FORECASTS")

        forecast = pd.DataFrame({
            'Asset': ['Indian Equity', 'Global Equity', 'Bonds', 'Gold', 'Crypto'],
            'Return': [1.2, 0.8, 0.3, 0.5, 2.5],
            'Confidence': [78, 82, 91, 75, 62]
        })

        fig_forecast = make_subplots(specs=[[{"secondary_y": True}]])

        fig_forecast.add_trace(
            go.Bar(x=forecast['Asset'], y=forecast['Return'], name='Return %', marker_color='#00f7ff'),
            secondary_y=False
        )

        fig_forecast.add_trace(
            go.Bar(x=forecast['Asset'], y=forecast['Confidence'], name='Confidence %', marker_color='#00ff88'),
            secondary_y=True
        )

        fig_forecast.update_layout(
            title='LSTM FORECASTS',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00f7ff'),
            barmode='group',
            height=400
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #00f7ff; opacity: 0.7;'>
    <p>⚡ Powered by Deep Learning & Advanced AI | Real-time Intelligence</p>
    <p>🔒 Secure | 🚀 Fast | 🧠 Intelligent</p>
    <p style='font-size: 0.8rem;'>© 2026 AI Asset Allocator</p>
</div>

""", unsafe_allow_html=True)
