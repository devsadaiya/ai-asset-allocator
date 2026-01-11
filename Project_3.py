"""
Smart Asset Allocation System - Advanced ML Implementation
Complete system with Feature Engineering, ML Models, and Backtesting
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn import hmm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.optimize import minimize
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("SMART ASSET ALLOCATION SYSTEM - AI/ML ENGINE")
print("=" * 80)


# ============================================================================
# MODULE 1: DATA COLLECTION & FEATURE ENGINEERING
# ============================================================================

class DataEngine:
    """Handles data collection and feature engineering for all asset classes"""

    def __init__(self):
        self.asset_tickers = {
            'Indian_Equity': '^NSEI',  # NIFTY 50
            'Global_Equity': '^GSPC',  # S&P 500
            'Gold': 'GC=F',  # Gold Futures
            'Bonds': '^TNX',  # 10-Year Treasury Yield
            'Crypto': 'BTC-USD',  # Bitcoin
            'VIX': '^VIX'  # Volatility Index
        }

    def fetch_data(self, start_date='2015-01-01', end_date=None):
        """Fetch historical data for all assets"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\n📊 Fetching data from {start_date} to {end_date}...")

        all_data = {}
        for name, ticker in self.asset_tickers.items():
            try:
                print(f"  → Downloading {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    all_data[name] = data
                else:
                    print(f"  ⚠️  No data for {name}")
            except Exception as e:
                print(f"  ❌ Error fetching {name}: {e}")

        return all_data

    def engineer_features(self, df, asset_name):
        """Create advanced features for ML models"""
        df = df.copy()

        # Ensure Close is a Series, not DataFrame
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

        # Price-based features
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))

        # Volatility features
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_60'] = df['returns'].rolling(window=60).std()

        # Momentum indicators
        df['momentum_20'] = close.pct_change(periods=20)
        df['momentum_60'] = close.pct_change(periods=60)
        df['roc'] = ((close - close.shift(10)) / close.shift(10)) * 100

        # Moving averages
        df['MA_20'] = close.rolling(window=20).mean()
        df['MA_50'] = close.rolling(window=50).mean()
        df['MA_200'] = close.rolling(window=200).mean()

        # Moving average crossovers
        df['MA_20_50_ratio'] = df['MA_20'] / df['MA_50']
        df['MA_50_200_ratio'] = df['MA_50'] / df['MA_200']

        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # MACD (Moving Average Convergence Divergence)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # Volume indicators (if available)
        if 'Volume' in df.columns:
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            df['volume_ma'] = volume.rolling(window=20).mean()
            df['volume_ratio'] = volume / df['volume_ma']

        # Trend strength
        df['trend_strength'] = np.where(close > df['MA_50'], 1, -1)

        # Higher highs and lower lows
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)

        # Drawdown
        df['cummax'] = close.cummax()
        df['drawdown'] = (close - df['cummax']) / df['cummax']

        print(f"  ✓ Engineered {len(df.columns)} features for {asset_name}")

        return df.dropna()

    def prepare_all_assets(self, start_date='2015-01-01'):
        """Fetch and engineer features for all assets"""
        raw_data = self.fetch_data(start_date)

        processed_data = {}
        for asset_name, df in raw_data.items():
            processed_data[asset_name] = self.engineer_features(df, asset_name)

        return processed_data


# ============================================================================
# MODULE 2: INVESTOR PROFILING (ML-Based Clustering)
# ============================================================================

class InvestorProfiler:
    """ML-based investor profiling using K-Means clustering"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.profile_definitions = {
            0: {
                'name': 'Conservative Growth',
                'description': 'Focus on capital preservation with modest growth',
                'equity_range': (20, 40),
                'risk_tolerance': 'Low'
            },
            1: {
                'name': 'Balanced Wealth Builder',
                'description': 'Balanced approach between growth and stability',
                'equity_range': (50, 60),
                'risk_tolerance': 'Medium'
            },
            2: {
                'name': 'Aggressive Alpha Seeker',
                'description': 'Maximum growth focus with higher risk acceptance',
                'equity_range': (65, 80),
                'risk_tolerance': 'High'
            }
        }

    def train_profiler(self, sample_data=None):
        """Train K-Means clustering model on investor data"""
        if sample_data is None:
            # Create synthetic training data representing different investor types
            np.random.seed(42)
            n_samples = 1000

            # Conservative investors
            conservative = np.column_stack([
                np.random.normal(50000, 20000, n_samples // 3),  # Investment amount
                np.random.normal(3, 1, n_samples // 3),  # Risk capacity (1-10)
                np.random.normal(60, 24, n_samples // 3),  # Time horizon (months)
                np.random.normal(2, 0.5, n_samples // 3)  # Knowledge (1-5)
            ])

            # Balanced investors
            balanced = np.column_stack([
                np.random.normal(200000, 80000, n_samples // 3),
                np.random.normal(6, 1, n_samples // 3),
                np.random.normal(36, 12, n_samples // 3),
                np.random.normal(3.5, 0.5, n_samples // 3)
            ])

            # Aggressive investors
            aggressive = np.column_stack([
                np.random.normal(500000, 200000, n_samples // 3),
                np.random.normal(9, 0.5, n_samples // 3),
                np.random.normal(24, 6, n_samples // 3),
                np.random.normal(4.5, 0.3, n_samples // 3)
            ])

            sample_data = np.vstack([conservative, balanced, aggressive])

        # Normalize features
        scaled_data = self.scaler.fit_transform(sample_data)

        # Train K-Means with 3 clusters
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.kmeans.fit(scaled_data)

        print("\n👤 Investor Profiler trained successfully!")
        print(f"   Cluster centers: {self.kmeans.cluster_centers_.shape}")

    def classify_investor(self, investment_amount, risk_capacity, time_horizon, knowledge):
        """Classify a new investor"""
        if self.kmeans is None:
            self.train_profiler()

        # Create feature vector
        features = np.array([[investment_amount, risk_capacity, time_horizon, knowledge]])
        features_scaled = self.scaler.transform(features)

        # Predict cluster
        cluster_id = self.kmeans.predict(features_scaled)[0]

        profile = self.profile_definitions[cluster_id]

        return {
            'cluster_id': cluster_id,
            'profile_name': profile['name'],
            'description': profile['description'],
            'equity_range': profile['equity_range'],
            'risk_tolerance': profile['risk_tolerance']
        }


# ============================================================================
# MODULE 3: MARKET REGIME DETECTION (Hidden Markov Model)
# ============================================================================

class RegimeDetector:
    """Detect market regimes using Hidden Markov Models"""

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: 'Bear Market',
            1: 'Sideways/Neutral',
            2: 'Bull Market'
        }

    def prepare_features(self, asset_data):
        """Prepare features for regime detection"""
        features = asset_data[['returns', 'volatility_20', 'momentum_20', 'RSI', 'MACD']].copy()
        features = features.dropna()
        return features

    def train(self, asset_data):
        """Train HMM on historical market data"""
        print("\n📈 Training Market Regime Detector (HMM)...")

        features = self.prepare_features(asset_data)
        features_scaled = self.scaler.fit_transform(features)

        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )

        self.model.fit(features_scaled)

        print(f"  ✓ HMM trained with {self.n_regimes} regimes")
        print(f"  ✓ Converged: {self.model.monitor_.converged}")

    def detect_regime(self, asset_data, return_probabilities=False):
        """Detect current market regime"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = self.prepare_features(asset_data)
        features_scaled = self.scaler.transform(features)

        # Predict hidden states
        hidden_states = self.model.predict(features_scaled)

        # Get current regime
        current_regime = hidden_states[-1]

        # Calculate regime probabilities
        if return_probabilities:
            probs = self.model.predict_proba(features_scaled)
            current_probs = probs[-1]
            return current_regime, self.regime_names[current_regime], current_probs

        return current_regime, self.regime_names[current_regime]

    def get_regime_statistics(self, asset_data):
        """Analyze regime characteristics"""
        features = self.prepare_features(asset_data)
        features_scaled = self.scaler.transform(features)
        hidden_states = self.model.predict(features_scaled)

        regime_stats = {}
        for regime_id in range(self.n_regimes):
            mask = hidden_states == regime_id
            regime_returns = features[mask]['returns']

            regime_stats[self.regime_names[regime_id]] = {
                'count': mask.sum(),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
            }

        return regime_stats


# ============================================================================
# MODULE 4: RETURN FORECASTING (LSTM Neural Network)
# ============================================================================

class ReturnForecaster:
    """LSTM-based return forecasting for asset classes"""

    def __init__(self, sequence_length=60, forecast_horizon=30):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'returns', 'volatility_20', 'momentum_20', 'RSI',
            'MACD', 'BB_width', 'MA_20_50_ratio'
        ]

    def prepare_sequences(self, data, target_col='returns'):
        """Create sequences for LSTM training"""
        # Check if all required features exist
        missing_features = [f for f in self.feature_columns if f not in data.columns]
        if missing_features:
            print(f"  ⚠️  Missing features: {missing_features}")
            return np.array([]), np.array([])

        features = data[self.feature_columns].values
        target = data[target_col].shift(-self.forecast_horizon).values

        # Check if we have enough data
        if len(features) < self.sequence_length + self.forecast_horizon + 50:
            print(f"  ⚠️  Insufficient data length: {len(features)}")
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon):
            sequence = features[i:i + self.sequence_length]
            # Check for NaN in sequence
            if not np.isnan(sequence).any():
                X.append(sequence)
                y.append(target[i + self.sequence_length])

        if len(X) == 0:
            return np.array([]), np.array([])

        return np.array(X), np.array(y)

    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),

            Dense(16, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )

        return model

    def train_asset_model(self, asset_name, asset_data, epochs=50, validation_split=0.2):
        """Train LSTM model for a specific asset"""
        print(f"\n🧠 Training LSTM for {asset_name}...")

        # Prepare data
        X, y = self.prepare_sequences(asset_data)

        # Remove NaN values
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        # Check if we have enough data
        if len(X) < 100:
            print(f"  ⚠️  Insufficient data for {asset_name} (only {len(X)} samples). Skipping...")
            return None

        # Check array dimensions
        if X.ndim != 3:
            print(f"  ⚠️  Invalid data shape for {asset_name}. Expected 3D array, got {X.ndim}D. Skipping...")
            return None

        n_samples, n_timesteps, n_features = X.shape

        if n_samples < 100 or n_timesteps != self.sequence_length:
            print(f"  ⚠️  Data shape issue for {asset_name}: {X.shape}. Skipping...")
            return None

        # Scale features
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)

        self.scalers[asset_name] = scaler

        # Build model
        model = self.build_model()

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Train
        history = model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        self.models[asset_name] = model

        final_loss = history.history['val_loss'][-1]
        final_mae = history.history['val_mae'][-1]
        print(f"  ✓ Training complete - Val Loss: {final_loss:.6f}, Val MAE: {final_mae:.6f}")

        return history

    def predict_return(self, asset_name, asset_data):
        """Predict future return for an asset"""
        if asset_name not in self.models:
            print(f"  ⚠️  No trained model for {asset_name}, using default estimate")
            return 0.0008  # Default ~0.08% daily return (~20% annual)

        try:
            model = self.models[asset_name]
            scaler = self.scalers[asset_name]

            # Prepare recent data
            recent_features = asset_data[self.feature_columns].values[-self.sequence_length:]

            # Check for NaN values
            if np.isnan(recent_features).any():
                print(f"  ⚠️  NaN values in recent data for {asset_name}, using default estimate")
                return 0.0008

            recent_scaled = scaler.transform(recent_features)
            recent_scaled = recent_scaled.reshape(1, self.sequence_length, len(self.feature_columns))

            # Predict
            prediction = model.predict(recent_scaled, verbose=0)[0][0]

            return prediction
        except Exception as e:
            print(f"  ⚠️  Error predicting for {asset_name}: {e}, using default estimate")
            return 0.0008

    def train_all_assets(self, processed_data, epochs=50):
        """Train models for all asset classes"""
        print("\n" + "=" * 80)
        print("TRAINING RETURN FORECASTING MODELS (LSTM)")
        print("=" * 80)

        for asset_name, asset_data in processed_data.items():
            if asset_name != 'VIX':  # Skip VIX for return prediction
                self.train_asset_model(asset_name, asset_data, epochs=epochs)


# ============================================================================
# MODULE 5: PORTFOLIO OPTIMIZATION
# ============================================================================

class PortfolioOptimizer:
    """Modern Portfolio Theory + Regime-aware optimization"""

    def __init__(self):
        self.asset_names = ['Indian_Equity', 'Global_Equity', 'Bonds', 'Gold', 'Crypto']

    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def optimize_portfolio(self, predicted_returns, risk_tolerance='medium',
                           market_regime='Bull Market', constraints=None):
        """
        Optimize portfolio allocation based on predicted returns and constraints

        Parameters:
        - predicted_returns: dict of asset returns
        - risk_tolerance: 'low', 'medium', 'high'
        - market_regime: 'Bear Market', 'Sideways/Neutral', 'Bull Market'
        - constraints: custom constraints dict
        """

        # Fill in missing assets with default returns
        default_returns = {
            'Indian_Equity': 0.0008,  # ~20% annual
            'Global_Equity': 0.0007,  # ~18% annual
            'Bonds': 0.0003,  # ~7% annual (conservative)
            'Gold': 0.0004,  # ~10% annual
            'Crypto': 0.0015  # ~40% annual (high risk)
        }

        # Merge predicted returns with defaults
        complete_returns = default_returns.copy()
        complete_returns.update(predicted_returns)

        returns_array = np.array([complete_returns[asset] for asset in self.asset_names])

        # Create covariance matrix (simplified - in production use historical data)
        volatilities = {
            'Indian_Equity': 0.20,
            'Global_Equity': 0.18,
            'Bonds': 0.05,
            'Gold': 0.15,
            'Crypto': 0.60
        }

        # Correlation matrix (typical values)
        corr_matrix = np.array([
            [1.00, 0.75, -0.10, 0.15, 0.30],  # Indian Equity
            [0.75, 1.00, -0.15, 0.20, 0.35],  # Global Equity
            [-0.10, -0.15, 1.00, 0.25, -0.05],  # Bonds
            [0.15, 0.20, 0.25, 1.00, 0.20],  # Gold
            [0.30, 0.35, -0.05, 0.20, 1.00]  # Crypto
        ])

        vol_array = np.array([volatilities[asset] for asset in self.asset_names])
        cov_matrix = np.outer(vol_array, vol_array) * corr_matrix

        # Define constraints based on risk tolerance and regime
        bounds = self.get_allocation_bounds(risk_tolerance, market_regime)

        # Objective: Maximize Sharpe Ratio
        def negative_sharpe(weights):
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, returns_array, cov_matrix)
            return -sharpe

        # Constraints: weights sum to 1
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Initial guess: equal weight
        initial_weights = np.array([1 / len(self.asset_names)] * len(self.asset_names))

        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )

        optimal_weights = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(optimal_weights, returns_array, cov_matrix)

        return {
            'weights': dict(zip(self.asset_names, optimal_weights)),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe
        }

    def get_allocation_bounds(self, risk_tolerance, market_regime):
        """Define allocation bounds based on risk profile and market regime"""

        # Base bounds
        if risk_tolerance == 'low':
            base_bounds = [
                (0.10, 0.30),  # Indian Equity
                (0.10, 0.25),  # Global Equity
                (0.25, 0.50),  # Bonds
                (0.05, 0.20),  # Gold
                (0.00, 0.05)  # Crypto
            ]
        elif risk_tolerance == 'medium':
            base_bounds = [
                (0.20, 0.40),
                (0.15, 0.35),
                (0.10, 0.30),
                (0.05, 0.15),
                (0.00, 0.15)
            ]
        else:  # high
            base_bounds = [
                (0.30, 0.50),
                (0.20, 0.40),
                (0.00, 0.20),
                (0.00, 0.15),
                (0.05, 0.25)
            ]

        # Adjust for market regime
        if market_regime == 'Bear Market':
            # Reduce equity exposure, increase defensive assets
            base_bounds[0] = (max(0, base_bounds[0][0] - 0.10), base_bounds[0][1] - 0.10)  # Lower equity
            base_bounds[2] = (base_bounds[2][0] + 0.10, min(0.60, base_bounds[2][1] + 0.15))  # More bonds
            base_bounds[3] = (base_bounds[3][0] + 0.05, min(0.30, base_bounds[3][1] + 0.10))  # More gold

        elif market_regime == 'Bull Market':
            # Increase equity exposure
            base_bounds[0] = (base_bounds[0][0], min(0.60, base_bounds[0][1] + 0.10))
            base_bounds[1] = (base_bounds[1][0], min(0.50, base_bounds[1][1] + 0.10))

        return base_bounds


# ============================================================================
# MODULE 6: BACKTESTING ENGINE
# ============================================================================

class Backtester:
    """Comprehensive backtesting framework"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.asset_names = ['Indian_Equity', 'Global_Equity', 'Bonds', 'Gold', 'Crypto']

    def run_backtest(self, processed_data, forecaster, regime_detector,
                     risk_tolerance='medium', rebalance_frequency=30):
        """
        Run historical backtest

        Parameters:
        - processed_data: dict of asset dataframes
        - forecaster: trained ReturnForecaster
        - regime_detector: trained RegimeDetector
        - risk_tolerance: investor risk profile
        - rebalance_frequency: days between rebalancing
        """

        print("\n" + "=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)

        # Get common date range
        date_ranges = [df.index for df in processed_data.values() if len(df) > 0]
        if not date_ranges:
            print("  ❌ No data available for backtesting")
            return None, None, None

        common_dates = date_ranges[0]
        for dates in date_ranges[1:]:
            common_dates = common_dates.intersection(dates)

        # Start from a point where we have enough history
        start_idx = forecaster.sequence_length + forecaster.forecast_horizon + 200

        if len(common_dates) < start_idx + 100:
            print(f"  ⚠️  Insufficient data for backtesting")
            print(f"  Available dates: {len(common_dates)}, Required: {start_idx + 100}")
            print(f"  Running simplified backtest with available data...")
            start_idx = max(100, len(common_dates) // 2)

        test_dates = common_dates[start_idx:]

        if len(test_dates) == 0:
            print("  ❌ No test dates available after filtering")
            print("  This usually happens when:")
            print("    - Not enough historical data was downloaded")
            print("    - Too many NaN values in the data")
            print("  Try running with an earlier start_date (e.g., 2015-01-01)")
            return None, None, None

        print(f"Backtest period: {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Number of trading days: {len(test_dates)}")
        print(f"Rebalance frequency: {rebalance_frequency} days")

        # Initialize tracking
        portfolio_values = []
        allocations_history = []
        dates_history = []

        current_weights = None
        days_since_rebalance = 0

        optimizer = PortfolioOptimizer()

        for i, current_date in enumerate(test_dates):
            # Rebalance if needed
            if current_weights is None or days_since_rebalance >= rebalance_frequency:

                # Get data up to current date
                historical_data = {}
                for asset_name in self.asset_names:
                    if asset_name in processed_data:
                        historical_data[asset_name] = processed_data[asset_name].loc[:current_date]

                # Detect market regime
                try:
                    regime_id, regime_name = regime_detector.detect_regime(
                        historical_data['Indian_Equity']
                    )
                except Exception as e:
                    regime_name = 'Bull Market'  # Default

                # Forecast returns
                predicted_returns = {}
                for asset_name in self.asset_names:
                    if asset_name in forecaster.models and asset_name in historical_data:
                        try:
                            pred = forecaster.predict_return(asset_name, historical_data[asset_name])
                            predicted_returns[asset_name] = pred
                        except:
                            pass

                # Optimize portfolio
                optimal_allocation = optimizer.optimize_portfolio(
                    predicted_returns,
                    risk_tolerance=risk_tolerance,
                    market_regime=regime_name
                )

                current_weights = optimal_allocation['weights']
                days_since_rebalance = 0

                allocations_history.append({
                    'date': current_date,
                    'regime': regime_name,
                    'weights': current_weights.copy()
                })

            # Calculate daily returns
            daily_returns = {}
            for asset_name in self.asset_names:
                if asset_name in processed_data:
                    asset_df = processed_data[asset_name]
                    if current_date in asset_df.index:
                        ret_value = asset_df.loc[current_date, 'returns']
                        # Ensure we get a scalar value, not a Series
                        if isinstance(ret_value, pd.Series):
                            daily_returns[asset_name] = float(ret_value.iloc[0]) if len(ret_value) > 0 else 0.0
                        else:
                            daily_returns[asset_name] = float(ret_value) if not pd.isna(ret_value) else 0.0
                    else:
                        daily_returns[asset_name] = 0.0
                else:
                    daily_returns[asset_name] = 0.0

            # Calculate portfolio return
            portfolio_return = sum(current_weights[asset] * daily_returns[asset]
                                   for asset in self.asset_names)

            # Update portfolio value
            if len(portfolio_values) == 0:
                portfolio_value = self.initial_capital * (1 + portfolio_return)
            else:
                portfolio_value = portfolio_values[-1] * (1 + portfolio_return)

            portfolio_values.append(portfolio_value)
            dates_history.append(current_date)
            days_since_rebalance += 1

        # Calculate performance metrics
        results_df = pd.DataFrame({
            'date': dates_history,
            'portfolio_value': portfolio_values
        })
        results_df.set_index('date', inplace=True)

        metrics = self.calculate_performance_metrics(results_df, allocations_history)

        return results_df, allocations_history, metrics

    def calculate_performance_metrics(self, results_df, allocations_history):
        """Calculate comprehensive performance metrics"""

        # Total return
        total_return = (results_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100

        # Calculate daily returns - ensure they are numeric
        portfolio_values = results_df['portfolio_value'].values
        daily_returns = pd.Series([
            (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
            if i > 0 else 0.0
            for i in range(len(portfolio_values))
        ])
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # CAGR
        n_years = len(results_df) / 252
        if n_years > 0:
            cagr = ((results_df['portfolio_value'].iloc[-1] / self.initial_capital) ** (1 / n_years) - 1) * 100
        else:
            cagr = 0.0

        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if volatility > 0:
            sharpe_ratio = (cagr / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (cagr / 100 - risk_free_rate) / downside_std if downside_std > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0.0

        # Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate (%)': win_rate,
            'Final Value': results_df['portfolio_value'].iloc[-1],
            'Number of Rebalances': len(allocations_history)
        }

        return metrics

        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (cagr / 100 - risk_free_rate) / (volatility / 100)

        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (cagr / 100 - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100

        # Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate (%)': win_rate,
            'Final Value': results_df['portfolio_value'].iloc[-1],
            'Number of Rebalances': len(allocations_history)
        }

        return metrics

    def plot_backtest_results(self, results_df, metrics, allocations_history):
        """Visualize backtest results"""

        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))

            # Portfolio value over time
            axes[0].plot(results_df.index, results_df['portfolio_value'],
                         linewidth=2, color='#2E86AB')
            axes[0].axhline(y=self.initial_capital, color='gray',
                            linestyle='--', alpha=0.5, label='Initial Capital')
            axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Portfolio Value ($)', fontsize=11)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Drawdown
            daily_returns = results_df['portfolio_value'].pct_change()
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100

            axes[1].fill_between(drawdown.index, drawdown.values, 0,
                                 color='#A23B72', alpha=0.6)
            axes[1].set_title('Drawdown (%)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Drawdown (%)', fontsize=11)
            axes[1].grid(True, alpha=0.3)

            # Allocation changes over time
            if len(allocations_history) > 0:
                dates = [a['date'] for a in allocations_history]
                asset_names = list(allocations_history[0]['weights'].keys())

                allocation_matrix = np.zeros((len(dates), len(asset_names)))
                for i, alloc in enumerate(allocations_history):
                    for j, asset in enumerate(asset_names):
                        allocation_matrix[i, j] = alloc['weights'][asset] * 100

                axes[2].stackplot(dates, allocation_matrix.T,
                                  labels=asset_names, alpha=0.8)
                axes[2].set_title('Asset Allocation Over Time', fontsize=14, fontweight='bold')
                axes[2].set_ylabel('Allocation (%)', fontsize=11)
                axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
            print("\n📊 Backtest visualization saved as 'backtest_results.png'")

            return fig
        except Exception as e:
            print(f"\n⚠️  Could not generate visualization: {e}")
            return None


# ============================================================================
# MODULE 7: MAIN EXECUTION ENGINE
# ============================================================================

class SmartAllocationSystem:
    """Main orchestrator for the entire AI system"""

    def __init__(self):
        self.data_engine = DataEngine()
        self.profiler = InvestorProfiler()
        self.regime_detector = RegimeDetector()
        self.forecaster = ReturnForecaster()
        self.optimizer = PortfolioOptimizer()
        self.backtester = Backtester()

        self.processed_data = None
        self.is_trained = False

    def train_system(self, start_date='2015-01-01', lstm_epochs=50):
        """Train all ML models"""
        print("\n" + "=" * 80)
        print("🚀 INITIALIZING SMART ASSET ALLOCATION SYSTEM")
        print("=" * 80)

        # Step 1: Fetch and process data
        print("\n[STEP 1/4] Data Collection & Feature Engineering")
        self.processed_data = self.data_engine.prepare_all_assets(start_date)

        # Step 2: Train investor profiler
        print("\n[STEP 2/4] Training Investor Profiler")
        self.profiler.train_profiler()

        # Step 3: Train regime detector
        print("\n[STEP 3/4] Training Market Regime Detector")
        self.regime_detector.train(self.processed_data['Indian_Equity'])

        # Analyze regimes
        regime_stats = self.regime_detector.get_regime_statistics(
            self.processed_data['Indian_Equity']
        )
        print("\n📊 Regime Statistics:")
        for regime, stats in regime_stats.items():
            print(f"  {regime}:")
            print(f"    - Days: {stats['count']}")
            print(f"    - Avg Return: {stats['avg_return']:.4f}")
            print(f"    - Volatility: {stats['volatility']:.4f}")
            print(f"    - Sharpe: {stats['sharpe']:.4f}")

        # Step 4: Train return forecasters
        print("\n[STEP 4/4] Training Return Forecasting Models")
        self.forecaster.train_all_assets(self.processed_data, epochs=lstm_epochs)

        self.is_trained = True
        print("\n" + "=" * 80)
        print("✅ SYSTEM TRAINING COMPLETE")
        print("=" * 80)

    def generate_allocation(self, investment_amount, risk_capacity,
                            time_horizon, knowledge_level):
        """Generate allocation for a new investor"""

        if not self.is_trained:
            raise ValueError("System not trained. Call train_system() first.")

        print("\n" + "=" * 80)
        print("💼 GENERATING SMART ALLOCATION")
        print("=" * 80)

        # Profile investor
        print("\n[1/4] Profiling Investor...")
        profile = self.profiler.classify_investor(
            investment_amount, risk_capacity, time_horizon, knowledge_level
        )
        print(f"  Profile: {profile['profile_name']}")
        print(f"  Risk Tolerance: {profile['risk_tolerance']}")

        # Detect current market regime
        print("\n[2/4] Detecting Market Regime...")
        regime_id, regime_name = self.regime_detector.detect_regime(
            self.processed_data['Indian_Equity']
        )
        print(f"  Current Regime: {regime_name}")

        # Forecast returns
        print("\n[3/4] Forecasting Asset Returns...")
        predicted_returns = {}
        for asset_name in self.optimizer.asset_names:
            if asset_name in self.processed_data:
                pred = self.forecaster.predict_return(
                    asset_name, self.processed_data[asset_name]
                )
                predicted_returns[asset_name] = pred
                print(f"  {asset_name}: {pred * 100:.2f}% (monthly)")
            else:
                print(f"  {asset_name}: Using default estimate (data unavailable)")

        # Optimize portfolio
        print("\n[4/4] Optimizing Portfolio...")
        risk_map = {'Low': 'low', 'Medium': 'medium', 'High': 'high'}
        optimal_allocation = self.optimizer.optimize_portfolio(
            predicted_returns,
            risk_tolerance=risk_map[profile['risk_tolerance']],
            market_regime=regime_name
        )

        # Convert to monetary amounts
        allocation_amounts = {
            asset: investment_amount * weight
            for asset, weight in optimal_allocation['weights'].items()
        }

        print("\n" + "=" * 80)
        print("📋 RECOMMENDED ALLOCATION")
        print("=" * 80)
        print(f"\nInvestment Amount: ₹{investment_amount:,.0f}")
        print(f"\nAsset Allocation:")
        for asset, amount in allocation_amounts.items():
            pct = (amount / investment_amount) * 100
            print(f"  {asset:20s}: ₹{amount:12,.0f} ({pct:5.1f}%)")

        print(f"\nExpected Annual Return: {optimal_allocation['expected_return'] * 12 * 100:.2f}%")
        print(f"Expected Volatility: {optimal_allocation['volatility'] * np.sqrt(12) * 100:.2f}%")
        print(f"Sharpe Ratio: {optimal_allocation['sharpe_ratio']:.2f}")

        return {
            'profile': profile,
            'regime': regime_name,
            'allocation': allocation_amounts,
            'metrics': optimal_allocation
        }

    def run_backtest(self, risk_tolerance='medium', rebalance_days=30):
        """Run comprehensive backtest"""

        if not self.is_trained:
            raise ValueError("System not trained. Call train_system() first.")

        results_df, allocations, metrics = self.backtester.run_backtest(
            self.processed_data,
            self.forecaster,
            self.regime_detector,
            risk_tolerance=risk_tolerance,
            rebalance_frequency=rebalance_days
        )

        print("\n" + "=" * 80)
        print("📊 BACKTEST PERFORMANCE METRICS")
        print("=" * 80)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:30s}: {value:10.2f}")
            else:
                print(f"{metric:30s}: {value}")

        # Plot results
        self.backtester.plot_backtest_results(results_df, metrics, allocations)

        return results_df, allocations, metrics

    def save_models(self, path='models/'):
        """Save all trained models"""
        import os
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.profiler, f'{path}investor_profiler.pkl')
        joblib.dump(self.regime_detector, f'{path}regime_detector.pkl')
        joblib.dump(self.forecaster, f'{path}return_forecaster.pkl')

        print(f"\n💾 Models saved to {path}")

    def load_models(self, path='models/'):
        """Load pre-trained models"""
        self.profiler = joblib.load(f'{path}investor_profiler.pkl')
        self.regime_detector = joblib.load(f'{path}regime_detector.pkl')
        self.forecaster = joblib.load(f'{path}return_forecaster.pkl')
        self.is_trained = True

        print(f"\n📂 Models loaded from {path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def get_user_inputs():
    """Interactive function to get user investment preferences"""
    print("\n" + "=" * 80)
    print("💰 WELCOME TO SMART ASSET ALLOCATION SYSTEM")
    print("=" * 80)
    print("\nPlease provide your investment details:\n")

    # Investment Amount
    while True:
        try:
            amount = input("1️⃣  Investment Amount (₹): ")
            investment_amount = float(amount.replace(',', ''))
            if investment_amount <= 0:
                print("   ⚠️  Please enter a positive amount")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    # Risk Capacity
    print("\n2️⃣  Risk Capacity (How much risk can you tolerate?)")
    print("   1 = Very Low Risk (Conservative)")
    print("   5 = Medium Risk (Balanced)")
    print("   10 = Very High Risk (Aggressive)")
    while True:
        try:
            risk = int(input("   Enter (1-10): "))
            if 1 <= risk <= 10:
                risk_capacity = risk
                break
            else:
                print("   ⚠️  Please enter a number between 1 and 10")
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    # Time Horizon
    print("\n3️⃣  Investment Time Horizon")
    print("   Enter the number of months you plan to invest for")
    print("   Examples: 12 months (1 year), 36 months (3 years), 60 months (5 years)")
    while True:
        try:
            time_horizon = int(input("   Enter months: "))
            if time_horizon <= 0:
                print("   ⚠️  Please enter a positive number of months")
                continue
            break
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    # Investment Knowledge
    print("\n4️⃣  Investment Knowledge Level")
    print("   1 = Beginner (New to investing)")
    print("   3 = Intermediate (Some experience)")
    print("   5 = Expert (Professional/Advanced)")
    while True:
        try:
            knowledge = int(input("   Enter (1-5): "))
            if 1 <= knowledge <= 5:
                knowledge_level = knowledge
                break
            else:
                print("   ⚠️  Please enter a number between 1 and 5")
        except ValueError:
            print("   ⚠️  Please enter a valid number")

    print("\n" + "=" * 80)
    print("✅ INPUT SUMMARY")
    print("=" * 80)
    print(f"💰 Investment Amount: ₹{investment_amount:,.0f}")
    print(f"📊 Risk Capacity: {risk_capacity}/10")
    print(f"⏱️  Time Horizon: {time_horizon} months ({time_horizon / 12:.1f} years)")
    print(f"📚 Knowledge Level: {knowledge_level}/5")
    print("=" * 80)

    confirm = input("\n✓ Proceed with these inputs? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("\n❌ Cancelled. Please restart the program.")
        return None

    return {
        'investment_amount': investment_amount,
        'risk_capacity': risk_capacity,
        'time_horizon': time_horizon,
        'knowledge_level': knowledge_level
    }


if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("🚀 SMART ASSET ALLOCATION SYSTEM - INITIALIZATION")
    print("=" * 80)
    print("\nWelcome! This system will:")
    print("  1. Train AI models on historical market data")
    print("  2. Analyze your investment profile")
    print("  3. Detect current market conditions")
    print("  4. Generate optimized asset allocation")
    print("  5. Run comprehensive backtests")

    proceed = input("\n⚡ Start training models? This may take 5-10 minutes (yes/no): ").strip().lower()

    if proceed not in ['yes', 'y']:
        print("\n❌ Exiting. Run again when ready!")
        exit()

    # Initialize system
    system = SmartAllocationSystem()

    # Train all models
    system.train_system(start_date='2018-01-01', lstm_epochs=30)

    print("\n" + "=" * 80)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("=" * 80)

    # Get user inputs interactively
    user_inputs = get_user_inputs()

    if user_inputs is None:
        exit()

    # Generate allocation with user's inputs
    print("\n" + "=" * 80)
    print("🔮 GENERATING YOUR PERSONALIZED ALLOCATION")
    print("=" * 80)

    allocation = system.generate_allocation(
        investment_amount=user_inputs['investment_amount'],
        risk_capacity=user_inputs['risk_capacity'],
        time_horizon=user_inputs['time_horizon'],
        knowledge_level=user_inputs['knowledge_level']
    )

    # Ask if user wants to see backtest
    print("\n" + "=" * 80)
    run_backtest = input("\n📊 Would you like to see historical backtest results? (yes/no): ").strip().lower()

    if run_backtest in ['yes', 'y']:
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE BACKTEST")
        print("=" * 80)

        # Map user's risk to backtest parameter
        if user_inputs['risk_capacity'] <= 3:
            bt_risk = 'low'
        elif user_inputs['risk_capacity'] <= 7:
            bt_risk = 'medium'
        else:
            bt_risk = 'high'

        results = system.run_backtest(
            risk_tolerance=bt_risk,
            rebalance_days=30
        )

        if results[0] is not None:
            results_df, allocations, metrics = results
            print("\n✅ Backtest completed successfully!")
        else:
            print("\n⚠️  Backtest could not be completed with current data.")
            print("    Consider training with more historical data (earlier start_date).")

    # Save models
    save_models = input("\n💾 Save trained models for future use? (yes/no): ").strip().lower()
    if save_models in ['yes', 'y']:
        system.save_models()

    print("\n" + "=" * 80)
    print("✅ SMART ALLOCATION SYSTEM - SESSION COMPLETE")
    print("=" * 80)
    print("\n📌 Your personalized allocation has been generated!")
    if run_backtest in ['yes', 'y'] and results[0] is not None:
        print("📊 Backtest visualization saved as 'backtest_results.png'")
    if save_models in ['yes', 'y']:
        print("💾 Models saved in 'models/' directory")
    print("\n🎯 Thank you for using Smart Asset Allocation System!")
    print("=" * 80)