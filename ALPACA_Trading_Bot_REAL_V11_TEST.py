import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
import warnings
import json
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import yfinance as yf
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import logging
import joblib
import sys
import re
from collections import Counter
import pennylane as qml
from filterpy.kalman import KalmanFilter 
from scipy.special import expit

# --- Platform-Specific Imports ---
try:
    import fcntl
    fcntl_available = True
except ImportError:
    fcntl_available = False
    logging.warning("fcntl module not found. File locking disabled (expected on Windows).")

# --- Check for TensorFlow ---
try:
    import tensorflow as tf
    tf_available = True
    logging.info(f"TensorFlow version {tf.__version__} found.")
    
    # --- MEMORY LEAK FIX: Set to run eagerly for easier debugging/resource management ---
    # This prevents accumulating static graphs, which can cause leaks.
    tf.config.run_functions_eagerly(False)
    
    # Optional: Configure GPU memory growth if GPU is used
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            logging.error(f"Error setting memory growth: {e}")
except ImportError:
    logging.error("TensorFlow not found. Trading will be disabled.")
    tf_available = False

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress specific TensorFlow warnings if needed
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1: Filter INFO, 2: Filter INFO+WARNING, 3: Filter INFO+WARNING+ERROR

import socket

# --- UPDATED ROBUST IPV4 OVERRIDE ---
orig_getaddrinfo = socket.getaddrinfo

def force_ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    # Force the family to AF_INET (IPv4) at the source
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = force_ipv4_getaddrinfo
# --- END OVERRIDE ---

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- File Paths & Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
CONFIG_PATH = '/app/config_Alpaca_REAL_V2.json'
MODELS_DIR = '/app/Models'
LOCK_FILE = os.path.join(script_dir, "trading_bot.lock")
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_FILENAME_FORMAT = "scalers_{ticker}_{interval}.joblib"

# --- Define Keras file formats ---
CONFIG_SUFFIX = "_config.json"
WEIGHTS_SUFFIX = ".weights.h5"

# --- Load Alpaca API Credentials ---
api = None
config = {}
try:
    logging.info(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as cf: config = json.load(cf)
    API_KEY = config["ALPACA_API_KEY"]
    API_SECRET = config["ALPACA_SECRET_KEY"]
    BASE_URL = config.get("ALPACA_BASE_URL", "https://api.alpaca.markets") # Default to paper trading
    logging.info(f"Initializing Alpaca API with Base URL: {BASE_URL}")
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    account = api.get_account()
    logging.info(f"Alpaca Connection OK. Account: {account.status}, Portfolio: ${float(account.portfolio_value):,.2f}")

except FileNotFoundError:
    logging.error(f"FATAL ERROR: Configuration file not found at {CONFIG_PATH}")
    api = None
except KeyError as e:
    logging.error(f"FATAL ERROR: Missing key in configuration file {CONFIG_PATH}: {e}")
    api = None
except Exception as e:
    logging.error(f"FATAL ERROR Initializing Alpaca API: {e}")
    api = None

# Exit early if API connection failed
if api is None:
    logging.critical("CRITICAL: Could not initialize Alpaca API. Exiting.")
    sys.exit(1)

# ---  Model Architecture & Feature Functions ---
class PPO(keras.Model):
    """ The CNN-based PPO model architecture using Keras (Must match training script exactly). """
    def __init__(self, num_features, window_size, output_size, hidden_size):
        super(PPO, self).__init__()

        # Keras initializers
        ortho_init_gain_sqrt2 = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        ortho_init_gain_1 = tf.keras.initializers.Orthogonal(gain=1.0)
        zero_init = tf.keras.initializers.Zeros()

        # --- UPGRADED: Causal Temporal Convolutional Network (WaveNet) ---
        self.feature_extractor = keras.Sequential([
            layers.InputLayer(input_shape=(window_size, num_features)),
            
            layers.Conv1D(filters=hidden_size, kernel_size=3, padding='causal', 
                          dilation_rate=1, activation='relu',
                          kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),
            
            layers.Conv1D(filters=hidden_size, kernel_size=3, padding='causal', 
                          dilation_rate=2, activation='relu',
                          kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),
                          
            layers.Conv1D(filters=hidden_size, kernel_size=3, padding='causal', 
                          dilation_rate=4, activation='relu',
                          kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),

            layers.Conv1D(filters=hidden_size, kernel_size=3, padding='causal', 
                          dilation_rate=8, activation='relu',
                          kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),
            
            # Slice the exact final timestep (t) which now contains 
            # the aggregated causal history of the entire window.
            layers.Lambda(lambda x: x[:, -1, :])
        ], name="causal_tcn_extractor")
        
        # --- Policy Head ---
        self.policy_net = keras.Sequential([
            layers.Dense(hidden_size, activation='relu',
                         kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),
            layers.Dense(output_size, name="policy_logits",
                         kernel_initializer=ortho_init_gain_1, bias_initializer=zero_init)
        ], name="policy_net")
        
        # --- Value Head ---
        self.value_net = keras.Sequential([
            layers.Dense(hidden_size, activation='relu',
                         kernel_initializer=ortho_init_gain_sqrt2, bias_initializer=zero_init),
            layers.Dense(1, name="value_output",
                         kernel_initializer=ortho_init_gain_1, bias_initializer=zero_init)
        ], name="value_net")

    def call(self, x, training=False):
        features = self.feature_extractor(x, training=training)
        policy_logits = self.policy_net(features, training=training)
        value = self.value_net(features, training=training)
        return value, policy_logits

# Feature calculation functions from the training script
def compute_rsi(series, period=14):
    delta = series.diff(1); gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=1).mean(); loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-9); rsi = 100.0 - (100.0 / (1.0 + rs)); return rsi.fillna(50.0)
def compute_moving_averages(df, feat_config):
    close_price = df['close']; fast_ma = close_price.rolling(window=feat_config['ma_fast_window'], min_periods=1).mean(); slow_ma = close_price.rolling(window=feat_config['ma_slow_window'], min_periods=1).mean()
    safe_close = close_price.replace(0, 1e-9); return (fast_ma - close_price) / safe_close, (slow_ma - close_price) / safe_close
def compute_bollinger_bands(df, feat_config):
    close_price = df['close']; window, std_mult = feat_config['bbands_window'], feat_config['bbands_std_mult']; mid_band = close_price.rolling(window=window, min_periods=1).mean()
    std_dev = close_price.rolling(window=window, min_periods=1).std().fillna(0.0); upper_band, lower_band = mid_band + (std_dev * std_mult), mid_band - (std_dev * std_mult)
    band_width = (upper_band - lower_band).replace(0, 1e-9); bb_percent = (close_price - lower_band) / band_width; return bb_percent.fillna(0.5).clip(-0.5, 1.5)
def compute_macd(df, feat_config):
    close_price = df['close']; ema_fast = close_price.ewm(span=feat_config['macd_fast_span'], adjust=False).mean(); ema_slow = close_price.ewm(span=feat_config['macd_slow_span'], adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=feat_config['macd_signal_span'], adjust=False).mean(); macd_hist = macd_line - signal_line
    price_std_norm = close_price.rolling(window=feat_config['macd_slow_span'], min_periods=1).std().replace(0, 1e-9).fillna(1e-9); return macd_hist / price_std_norm
def compute_atr(df, feat_config):
    high, low, close = df['high'], df['low'], df['close']; true_range = pd.concat([high - low, abs(high - close.shift(1)).fillna(0), abs(low - close.shift(1)).fillna(0)], axis=1).max(axis=1)
    atr = true_range.rolling(window=feat_config['atr_window'], min_periods=1).mean().ffill().fillna(0.0); return atr / close.replace(0, 1e-9)
def compute_obv_manual(df):
    close, volume = df['close'], df['volume']; signed_volume = volume * np.sign(close.diff()); signed_volume.iloc[0] = 0; obv = signed_volume.cumsum()
    obv_change_norm = obv.diff().fillna(0) / volume.replace(0, 1e-9); return obv_change_norm.replace([np.inf, -np.inf], 0).fillna(0)

# --- ADD Quantum Circuit Wrapper Function ---
def quantum_circuit_amplitude(value, q_config_runtime):
    """ Wrapper for the globally defined Amplitude Embedding QNode. """
    global _qml_circuit_for_amplitude # Use the globally defined QNode
    # Get num_qubits from the specific model's config, default to 3 if missing
    num_qubits_for_error_return = q_config_runtime.get("num_qubits", 3)

    if not _qml_circuit_for_amplitude:
        # Return NaNs matching the expected output dimension if QNode isn't ready
        return [np.nan] * num_qubits_for_error_return

    # Convert input numpy/float to a TensorFlow tensor
    value_tensor = tf.convert_to_tensor(float(value), dtype=tf.float32)
    try:
        # Call the global QNode
        result_tensor_list = _qml_circuit_for_amplitude(value_tensor)
        # Convert list of TensorFlow tensor outputs back to a list of numpy scalars
        return [res.numpy() for res in result_tensor_list]
    except Exception as e:
        # Log error and return NaNs matching expected dimension
        logging.error(f"Error in quantum_circuit_amplitude for value {value}: {e}", exc_info=True)
        return [np.nan] * num_qubits_for_error_return
# --- End Quantum Wrapper ---

# --- Global Variables & Bot Configuration ---
#tickers = config.get('TICKERS', ['INTC', 'IONQ', 'KO', 'KR', 'OXY', 'SIRI', 'T', 'PYPL']) #ORIGINAL TICKERS
tickers = config.get('TICKERS', ['IONQ','KO','OXY','BAC','GM','PFE','PYPL','FCX','SOFI','T','F','CCL'])
YF_INTERVAL = config.get('YF_INTERVAL', '1d')
INVESTMENT_AMOUNT = config.get('INVESTMENT_AMOUNT', 550)
POLLING_INTERVAL_SECONDS = config.get('POLLING_INTERVAL_SECONDS', 300)
HISTORICAL_DATA_PERIOD = config.get('HISTORICAL_DATA_PERIOD', "1y") # 1y is sufficient for live features
REQUIRED_FEATURE_BUFFER = config.get('REQUIRED_FEATURE_BUFFER', 250) # Increased for longer MAs
COOLDOWN_PERIOD_DAYS = config.get('COOLDOWN_PERIOD_DAYS', 1)
MAX_HOLD_DAYS = config.get('MAX_HOLD_DAYS', 5) # Close after 5 trading days
CONFIDENCE_THRESHOLD = config.get('CONFIDENCE_THRESHOLD', 0.40)

# --- Position Sizing Configuration ---
RISK_PER_TRADE_PCT = 0.015  # 1.5% of Net Liquidity (Adjustable)
#MAX_DOLLAR_PER_SLOT = 550   # Your 1/8th slot hard ceiling ($4,200 / 8)

# Separate Capital Weights for Directional Bias
# Shorts generally get less capital due to asymmetric risk and higher margin requirements.
LONG_CAPITAL_WEIGHT = config.get('LONG_CAPITAL_WEIGHT', 1.00)
SHORT_CAPITAL_WEIGHT = config.get('SHORT_CAPITAL_WEIGHT', 0.75)

# Longs: Widen SL to 3.5%, TP to 8%
STOP_LOSS_LONG_PCT = config.get('STOP_LOSS_LONG_PCT', 0.03) 
TAKE_PROFIT_LONG_PCT = 0.03
#TAKE_PROFIT_LONG_PCT = config.get('TAKE_PROFIT_LONG_PCT', 0.06) 

# Shorts: Tight SL at 2.0%, TP at 5.0%
STOP_LOSS_SHORT_PCT = config.get('STOP_LOSS_SHORT_PCT', 0.03) 
TAKE_PROFIT_SHORT_PCT = 0.03
#TAKE_PROFIT_SHORT_PCT = config.get('TAKE_PROFIT_SHORT_PCT', 0.06) 

POST_ORDER_DELAY_SECONDS = config.get('POST_ORDER_DELAY_SECONDS', 3)

ASSET_INDEX_MAP = config.get("asset_index_map", {
    # --- The Core Keepers ---
    'CSCO': 'XLK', 'HPE': 'XLK', 'IONQ': 'QTUM', 'PYPL': 'XLK', 'INTC': 'SOXX',
    
    # --- Financials & Fintech ---
    'BAC': 'XLF', 'SOFI': 'XLF', 'HOOD': 'XLF',
    
    # --- Defensive & Healthcare ---
    'KO': 'XLP', 'KR': 'XLP', 'PFE': 'XLV',
    
    # --- Energy, Materials & Commodities ---
    'OXY': 'XLE', 'FCX': 'XME', 'GOLD': 'XLB',
    
    # --- Consumer Discretionary & Transport ---
    'F': 'XLY', 'GM': 'XLY', 'CCL': 'XLY', 'UBER': 'XLI',
    
    # --- High Volatility & Telecom ---
    'PLTR': 'IGV', 'MARA': 'BLOK', 'T': 'XLC'
})
trade_states, models, scalers, last_proposed_signals = {}, {}, {}, {} 

# --- GHOST TRADING GLOBALS ---
ghost_positions = {}
ghost_history = {'long': [], 'short': []}
GHOST_MA_WINDOW = 5 # Number of recent virtual trades to average

# --- Load Models & Scalers (Per Ticker) ---
logging.info("Attempting to load PER-TICKER TensorFlow models and scalers...")
for ticker in tickers:
    # --- Construct base path (without .pth) ---
    base_model_filename = f"ppo_{ticker}_{YF_INTERVAL}_opt" # Assumes _opt suffix exists
    base_model_path = os.path.join(MODELS_DIR, base_model_filename)
    # --- Define specific file paths ---
    config_path = base_model_path + CONFIG_SUFFIX
    weights_path = base_model_path + WEIGHTS_SUFFIX
    scaler_path = os.path.join(MODELS_DIR, SCALER_FILENAME_FORMAT.format(ticker=ticker, interval=YF_INTERVAL))

    # --- Optimized Model Paths (_opt)
    base_model_filename_opt = f"ppo_{ticker}_{YF_INTERVAL}_opt"
    base_model_path_opt = os.path.join(MODELS_DIR, base_model_filename_opt)
    config_path_opt = base_model_path_opt + CONFIG_SUFFIX
    weights_path_opt = base_model_path_opt + WEIGHTS_SUFFIX

    # --- Default Model Paths (_def)
    base_model_filename_def = f"ppo_{ticker}_{YF_INTERVAL}_def"
    base_model_path_def = os.path.join(MODELS_DIR, base_model_filename_def)
    config_path_def = base_model_path_def + CONFIG_SUFFIX
    weights_path_def = base_model_path_def + WEIGHTS_SUFFIX

    # 2. Check for the Scaler first (required for any model)
    if not os.path.exists(scaler_path):
        logging.error(f"Required scaler for {ticker} is missing at {scaler_path}. Trading disabled for this ticker.")
        continue

    # 3. Determine which model files (opt or def) to load
    config_path = None
    weights_path = None
    base_model_path = None # For logging

    if os.path.exists(config_path_opt) and os.path.exists(weights_path_opt):
        logging.info(f"Found OPTIMIZED (_opt) model files for {ticker}.")
        config_path = config_path_opt
        weights_path = weights_path_opt
        base_model_path = base_model_path_opt
    elif os.path.exists(config_path_def) and os.path.exists(weights_path_def):
        logging.info(f"Found DEFAULT (_def) model files for {ticker}.")
        config_path = config_path_def
        weights_path = weights_path_def
        base_model_path = base_model_path_def
    else:
        logging.warning(f"No model files found for {ticker}. Looked for '_opt' and '_def' versions.")
        logging.warning(f" -> Checked for: {config_path_opt} AND {weights_path_opt}")
        logging.warning(f" -> Checked for: {config_path_def} AND {weights_path_def}")
        continue
    # -------------------------------------------

    try:
        # 1. Load configuration from JSON
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        # 2. Extract parameters and instantiate the Keras model
        ppo_cfg = loaded_config.get('ppo', {})
        env_cfg = loaded_config.get('env', {})

        num_features = loaded_config.get('input_size')
        # Correctly get window_size from env config
        window_size_loaded = env_cfg.get('window_size')
        output_size = env_cfg.get('action_size')
        hidden_size = ppo_cfg.get('hidden_size')

        # --- NEW PARANOIA CHECK ---
        expected_feature_list = loaded_config.get('feature_columns', [])
        if len(expected_feature_list) != num_features:
             raise ValueError(f"Config corruption for {ticker}: 'input_size' ({num_features}) does not match the number of items in 'feature_columns' ({len(expected_feature_list)}). Please retrain the model.")
        # --------------------------

        if not all([num_features, window_size_loaded, output_size, hidden_size]):
             raise KeyError("Loaded config missing required keys for model instantiation.")

        # Instantiate the Keras model
        ticker_model = PPO(
            num_features=num_features,
            window_size=window_size_loaded, # Use loaded value
            output_size=output_size,
            hidden_size=hidden_size
        )

        # 3. Build the model by calling it with dummy data (required before loading weights)
        dummy_input = tf.zeros((1, window_size_loaded, num_features))
        _ = ticker_model(dummy_input) # Build the model

        # 4. Load the saved weights
        ticker_model.load_weights(weights_path)
        ticker_model.trainable = False # Set to inference mode

        # 5. Load Scalers
        ticker_scalers = joblib.load(scaler_path)
        if not isinstance(ticker_scalers, dict):
            raise TypeError("Scaler file is not a valid dictionary.")

        # 6. Store in dictionaries
        models[ticker] = {
            "model": ticker_model,
            "config": loaded_config, # Store the full config
            "features": loaded_config.get('feature_columns', []), # Get feature list
            "window_size": window_size_loaded # Store loaded window size
        }
        scalers[ticker] = ticker_scalers
        logging.info(f"OK: Loaded TF model/scalers for {ticker} (Window: {window_size_loaded}, Features: {num_features})")

    except Exception as e:
        logging.error(f"FAIL: Could not load TF model/scaler for {ticker} from base path {base_model_path}: {e}", exc_info=True)

if not models:
    logging.critical("CRITICAL: No models were loaded. Exiting."); sys.exit(1)
# ---------------------------------------------

# --- Initialize Quantum Device and Define Global QNodes ---
qdevice = None
_qml_circuit_for_amplitude = None # For the primary (Kalman) quantum feature
q_relationship_circuit = None     # For the new asset-index relationship feature

# Check if *any* loaded model config requests quantum features
use_quantum_globally = False
for ticker in models:
    if models[ticker].get('config', {}).get('features', {}).get('use_quantum_feature', False):
        use_quantum_globally = True
        q_config_global = models[ticker]['config'].get('quantum', {}) # Use config from first model found
        break

if use_quantum_globally:
    try:
        logging.info("--- Initializing Quantum Resources ---")
        num_qubits_for_qnode = q_config_global.get("num_qubits", 3) # Default if missing

        # 1. Define the single quantum device
        qdevice = qml.device(
            q_config_global.get("backend", "default.qubit"),
            wires=num_qubits_for_qnode,
            shots=q_config_global.get("shots") # Will be None for exact expectation
        )
        logging.info(f"Quantum device '{q_config_global.get('backend', 'default.qubit')}' created with {num_qubits_for_qnode} wires.")

        # 2. Define QNode for the primary feature (MATCHING V2 ANGLE EMBEDDING)
        @qml.qnode(qdevice, interface='tf', diff_method='parameter-shift')
        def _actual_qnode_for_amplitude_embedding(v_tensor):
            # 1. Flatten the input and cast to float
            v_flat = tf.reshape(tf.cast(v_tensor, dtype=tf.float32), [-1])
            
            # 2. Scale the input [0, 1] to an angle [0, pi] for quantum rotation
            angle = v_flat[0] * np.pi 
            
            # 3. Apply Angle Embedding (Rotate all qubits by this angle)
            for i in range(num_qubits_for_qnode):
                qml.RY(angle, wires=i)
                qml.RX(angle, wires=i)
            
            # 4. Standard Entanglement
            for i in range(num_qubits_for_qnode - 1):
                qml.CNOT(wires=[i, i + 1])
            if num_qubits_for_qnode > 1:
                qml.CNOT(wires=[num_qubits_for_qnode - 1, 0])
            if num_qubits_for_qnode >= 2:
                qml.SWAP(wires=[0, 1])
            if num_qubits_for_qnode >= 3:
                qml.CZ(wires=[0, 2])
            
            # 5. Measure Expectation Value
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits_for_qnode)]

        _qml_circuit_for_amplitude = _actual_qnode_for_amplitude_embedding
        logging.info("Global Angle Embedding QNode defined successfully.")

        # 3. Define QNode for the relationship feature
        if len(qdevice.wires) >= 2:
            @qml.qnode(qdevice, interface='tf', diff_method='parameter-shift')
            def _actual_q_relationship_node(asset_input, index_input):
                angle1 = asset_input * np.pi
                angle2 = index_input * np.pi
                qml.RX(angle1, wires=0)
                qml.RX(angle2, wires=1)
                qml.CNOT(wires=[0, 1])
                qml.RY(angle1, wires=0)
                qml.RY(angle2, wires=1)
                return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
            q_relationship_circuit = _actual_q_relationship_node
            logging.info("Global Quantum Relationship QNode defined successfully.")
        else:
            logging.warning("Quantum device has fewer than 2 wires. Quantum relationship feature will be disabled.")
            q_relationship_circuit = None

    except Exception as e:
        logging.error(f"Failed to create quantum device or define global QNodes: {e}", exc_info=True)
        use_quantum_globally = False # Disable if setup fails
        qdevice = None
        _qml_circuit_for_amplitude = None
        q_relationship_circuit = None
else:
    logging.info("Quantum features not requested by any loaded model config. Skipping quantum setup.")
# --- End Quantum Setup ---

# --- ADD Quantum Circuit Wrapper Function ---
# (Place this near your other compute_* functions, e.g., around line 180)
def quantum_circuit_amplitude(value, q_config_runtime):
    """ Wrapper for the globally defined Amplitude Embedding QNode. """
    global _qml_circuit_for_amplitude
    num_qubits_for_error_return = q_config_runtime.get("num_qubits", 3)

    if not _qml_circuit_for_amplitude:
        return [np.nan] * num_qubits_for_error_return

    # --- ADDED: Handle NaN input gracefully ---
    float_value = float(value)
    if np.isnan(float_value):
        logging.warning(f"Quantum circuit received NaN input. Returning NaN array.")
        return [np.nan] * num_qubits_for_error_return
    # --- END ADDED ---

    value_tensor = tf.convert_to_tensor(float(value), dtype=tf.float32)
    try:
        result_tensor_list = _qml_circuit_for_amplitude(value_tensor)
        return [res.numpy() for res in result_tensor_list]
    except Exception as e:
        logging.error(f"Error in quantum_circuit_amplitude for value {value}: {e}", exc_info=True)
        return [np.nan] * num_qubits_for_error_return

# --- Data Fetching & Preprocessing Functions ---
def get_market_data(ticker, period=HISTORICAL_DATA_PERIOD, interval=YF_INTERVAL):
    """ 
    Fetches historical data directly from Alpaca (Source of Truth). 
    Replaces yfinance to ensure data matches execution venue.
    """
    # 1. Map '1d' string to Alpaca TimeFrame
    if interval == '1d':
        tf_interval = TimeFrame.Day
    elif interval == '1h':
        tf_interval = TimeFrame.Hour
    elif interval == '15m':
        tf_interval = TimeFrame.Minute * 15
    else:
        logging.warning(f"Interval {interval} not explicitly mapped, defaulting to Day.")
        tf_interval = TimeFrame.Day

    # 2. Calculate Start/End Dates
    # --- FIX: Buffer for Free Plan to avoid 'recent SIP data' error ---
    current_time = datetime.now(pytz.utc)
    # If market is open, we must look back 16+ minutes. We use 20 for safety.
    # If market is closed, this doesn't hurt (just misses post-market noise).
    end_dt = current_time - timedelta(minutes=20)
    
    if period == "1y":
        start_dt = end_dt - timedelta(days=365)
    elif period == "2y":
        start_dt = end_dt - timedelta(days=730)
    else:
        start_dt = end_dt - timedelta(days=365) # Default

    try:
        # 3. Fetch Asset Data
        logging.info(f"Fetching {ticker} data via Alpaca API (End: {end_dt.isoformat()})...")
        asset_bars = api.get_bars(ticker, tf_interval, start=start_dt.isoformat(), end=end_dt.isoformat(), adjustment='all').df
        
        if asset_bars.empty: 
            raise ValueError(f"No data returned from Alpaca for {ticker}")

        # Standardize columns to lowercase (Alpaca returns 'Close', 'Open' etc.)
        asset_bars.columns = [c.lower() for c in asset_bars.columns]
        
        # 4. Fetch Index Data (if mapped)
        index_ticker = ASSET_INDEX_MAP.get(ticker)
        final_df = asset_bars

        if index_ticker:
            try:
                logging.info(f"Fetching mapped index {index_ticker} for {ticker}...")
                index_bars = api.get_bars(index_ticker, tf_interval, start=start_dt.isoformat(), end=end_dt.isoformat(), adjustment='all').df
                
                if not index_bars.empty:
                    index_bars.columns = [c.lower() for c in index_bars.columns]
                    # Rename close to index_close for the merge
                    # Rename close to index_close for the merge
                    index_data = index_bars[['close']].rename(columns={'close': 'index_close'})
                    # Merge on Index (Timestamp)
                    final_df = pd.merge(asset_bars, index_data, left_index=True, right_index=True, how='left')
                    final_df['index_close'] = final_df['index_close'].ffill().bfill()
                else:
                    logging.warning(f"Index {index_ticker} returned empty data. Proceeding without index features.")
            except Exception as ix_e:
                logging.warning(f"Failed to fetch index {index_ticker}: {ix_e}. Proceeding without it.")

        # 5. Final Validation
        if len(final_df) < REQUIRED_FEATURE_BUFFER: 
            raise ValueError(f"Insufficient data ({len(final_df)} rows) from Alpaca.")
        
        # Alpaca get_bars.df is already TZ-aware (UTC), but we ensure it here
        if final_df.index.tz is None: 
            final_df = final_df.tz_localize('UTC')
        else: 
            final_df = final_df.tz_convert('UTC')
            
        return final_df
        
    except Exception as e:
        # Re-raise the exception with a clear, specific message
        raise ValueError(f"Failed to get/process market data for {ticker}: {e}") from e

def preprocess_data(data, ticker):
    """ Calculates features, scales, and returns a windowed TensorFlow tensor. """
    global use_quantum_globally, q_relationship_circuit, _qml_circuit_for_amplitude # Access global quantum flags/circuits

    if ticker not in models: raise ValueError(f"No model config/info for {ticker}")

    model_info, ticker_scalers = models[ticker], scalers[ticker]
    if 'config' not in model_info or 'features' not in model_info['config']:
        raise ValueError(f"Model config or features key missing for {ticker} in model_info.")

    feat_config = model_info['config']['features'] # Config used during *training*
    final_feature_columns = model_info['features'] # Actual feature list saved from training
    window_size_model = model_info['window_size'] # Window size used by the loaded model
    q_config = model_info['config'].get('quantum', {}) # Get quantum config specific to this model
    
    # Determine if quantum should run for *this specific ticker*
    use_quantum_for_ticker = feat_config.get('use_quantum_feature', False) and use_quantum_globally

    df = data.copy()

    # --- UPDATED FIX: Modern Pandas Filling ---
    core_cols = ['open', 'high', 'low', 'close', 'volume']
    cols_to_fill = [col for col in core_cols if col in df.columns]
    if 'index_close' in df.columns: 
        cols_to_fill.append('index_close')
        
    # Replace deprecated method='ffill' with modern .ffill().bfill()
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill()

    # Final safety for brand new tickers with zero history
    df[cols_to_fill] = df[cols_to_fill].fillna(0.0)
    
    # --- Feature Generation ---
    if feat_config.get("add_time_features"):
        df['day_of_week'] = df.index.dayofweek / 6.0
        df['month_of_year'] = (df.index.month - 1) / 11.0
        
    if 'ma_very_slow_window' in feat_config: 
        df['very_slow_ma_norm'] = (df['close'].rolling(window=feat_config['ma_very_slow_window'], min_periods=1).mean() - df['close']) / df['close'].replace(0,1e-9)
    
    df['fast_ma_norm'], df['slow_ma_norm'] = compute_moving_averages(df, feat_config)
    
    slow_ma_values = df['close'].rolling(window=feat_config['ma_slow_window'], min_periods=1).mean()
    df['slow_ma_slope'] = slow_ma_values.diff(periods=5).fillna(0)
    
    price_std_for_slope_norm = df['close'].rolling(window=feat_config['ma_slow_window'], min_periods=1).std().replace(0, 1e-9).fillna(1e-9)
    df['slow_ma_slope_norm'] = df['slow_ma_slope'] / price_std_for_slope_norm
    
    df['daily_return'] = df['close'].pct_change().fillna(0.0)
    df['volatility'] = df['daily_return'].rolling(window=20, min_periods=1).std().fillna(0.0)
    
    if feat_config.get("add_vol_of_vol"): 
        df['vol_of_vol'] = df['volatility'].diff().fillna(0.0)
        
    for period in feat_config.get('rsi_periods', []): 
        df[f'rsi_{period}'] = compute_rsi(df['close'], period)
        
    df['bollinger_percent'] = compute_bollinger_bands(df, feat_config)
    df['macd_norm'] = compute_macd(df, feat_config)
    df['atr_norm'] = compute_atr(df, feat_config)
    
    if feat_config.get('calculate_obv'): 
        df['obv_change_norm'] = compute_obv_manual(df)

    # --- Feature Interaction Loop ---
    if feat_config.get('feature_interaction_enabled'):
        rsi_periods_list = feat_config.get('rsi_periods', [])
        for period in rsi_periods_list:
            base_rsi_col = f'rsi_{period}'
            interaction_col_name = f'{base_rsi_col}_x_return'
            if base_rsi_col in df.columns:
                df[interaction_col_name] = df[base_rsi_col] * df['daily_return']

    rolling_low = df['close'].rolling(window=200, min_periods=1).min()
    rolling_high = df['close'].rolling(window=200, min_periods=1).max()
    channel_width = (rolling_high - rolling_low).replace(0, 1e-9)
    df['price_vs_200d_low'] = (df['close'] - rolling_low) / channel_width
    df['price_vs_200d_high'] = (rolling_high - df['close']) / channel_width

    # --- OPTIMIZATION: SLICE DATA ---
    required_depth = window_size_model + 50 
    if len(df) > required_depth:
        df = df.iloc[-required_depth:].copy()

    # --- Index-Related Features & Quantum Relationship ---
    if 'index_close' in df.columns:
        asset_return = df['close'].pct_change().fillna(0.0)
        index_return = df['index_close'].pct_change().fillna(0.0)

        df['asset_index_corr'] = asset_return.rolling(window=feat_config.get("correlation_window", 20)).corr(index_return).fillna(0.0)
        df['return_spread'] = (asset_return - index_return).fillna(0.0)
        
        # --- NEW MACRO FEATURES ---
        df['index_volatility'] = index_return.rolling(window=20, min_periods=1).std().fillna(0.0)
        df['ghost_regime_ma'] = df['return_spread'].rolling(window=5, min_periods=1).mean().fillna(0.0)
        # --------------------------
        
        if feat_config.get("add_adv_interactions"):
            df['vol_x_spread'] = df['volatility'] * df['return_spread']
            df['macd_x_corr'] = df['macd_norm'] * df['asset_index_corr']

        # Quantum Relationship Feature
        q_corr_col_name = 'quantum_asset_index_corr'
        if use_quantum_for_ticker and q_relationship_circuit is not None:
            scaled_asset_ret = asset_return.clip(-0.05, 0.05) / 0.05
            scaled_index_ret = index_return.clip(-0.05, 0.05) / 0.05
            try:
                quantum_corr_values = [
                    q_relationship_circuit(tf.convert_to_tensor(ar, dtype=tf.float32), tf.convert_to_tensor(ir, dtype=tf.float32)).numpy()
                    for ar, ir in zip(scaled_asset_ret, scaled_index_ret)
                ]
                df[q_corr_col_name] = quantum_corr_values
            except Exception as e_qcorr:
                logging.error(f"[{ticker}] Error calculating {q_corr_col_name}: {e_qcorr}")
                df[q_corr_col_name] = 0.0
    else:
        index_related_cols = ['asset_index_corr', 'return_spread', 'vol_x_spread', 'macd_x_corr', 'quantum_asset_index_corr']
        for col in index_related_cols:
            if col in final_feature_columns: df[col] = 0.0

    # --- 3. Primary Quantum Feature (Kalman Filter) ---
    q_kalman_col_name = q_config.get('feature_name', 'filtered_q_kalman')
    if use_quantum_for_ticker and _qml_circuit_for_amplitude is not None:
        try:
            # --- MATCHING V2 SCALING LOGIC (Tanh instead of expit) ---
            returns_col = df['daily_return'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            returns_np = returns_col.values.reshape(-1, 1)
            q_input_scaler_name = 'quantum_robust_scaler'
            
            # Use the pre-loaded scalers from the Orchestrator
            if q_input_scaler_name in ticker_scalers:
                standardized_returns = ticker_scalers[q_input_scaler_name].transform(returns_np).flatten()
            else:
                raise ValueError(f"No valid {q_input_scaler_name} found in scalers.")

            # Tanh preserves outlier variance without hard-clipping
            squeezed_data = np.tanh(standardized_returns) 
            normalized_data = (squeezed_data + 1.0) / 2.0  # Shift from [-1, 1] to [0, 1]
            quantum_input_raw = np.clip(normalized_data, 0.05, 0.95) # Safe buffer

            # KF1
            kf_data = KalmanFilter(dim_x=2, dim_z=1); kf_data.x = np.array([[quantum_input_raw[0]], [0.]])
            kf_data.F = np.array([[1., 1.], [0., 1.]]); kf_data.H = np.array([[1., 0.]])
            kf_data.P *= 1000.; kf_data.R = q_config.get('kf1_R', 5); kf_data.Q = q_config.get('kf1_Q', 0.01)
            
            filtered_data_list = [kf_data.predict() or kf_data.update(z) or kf_data.x[0,0] for z in quantum_input_raw]
            filtered_data = np.array(filtered_data_list)

            # Ensure absolute clipping. 
            quantum_input = np.clip(filtered_data, 0.0, 1.0)
            
            # --- QUANTUM EXECUTION (Sequential) ---
            logging.info(f"🚀 Executing Quantum Angle Embedding for {len(quantum_input)} states...")
            quantum_results_raw = np.array([quantum_circuit_amplitude(v, q_config) for v in quantum_input])
            
            # NaN Filtering
            nan_rows = np.isnan(quantum_results_raw).any(axis=1)
            valid_mask = ~nan_rows
            intermediate_valid_indices = df.index[valid_mask]
            
            quantum_feature = quantum_results_raw[valid_mask][:, 0]

            # KF2
            kf_quantum = KalmanFilter(dim_x=2, dim_z=1); kf_quantum.x = np.array([[quantum_feature[0]], [0.]])
            kf_quantum.F = np.array([[1.,1.],[0.,1.]]); kf_quantum.H = np.array([[1.,0.]])
            kf_quantum.P *= 1000.; kf_quantum.R = q_config.get('kf2_R', 5); kf_quantum.Q = q_config.get('kf2_Q', 0.01)
            
            filtered_q = []
            for z_val in quantum_feature:
                kf_quantum.predict(); kf_quantum.update(np.array([[z_val]])); filtered_q.append(kf_quantum.x[0,0])

            df[q_kalman_col_name] = pd.Series(filtered_q, index=intermediate_valid_indices).ffill().bfill().fillna(0.0)

        except Exception as e_qkalman:
            logging.error(f"[{ticker}] Quantum pipeline error: {e_qkalman}", exc_info=True)
            df[q_kalman_col_name] = 0.0

    # --- Final Column Check & Scaling ---
    for col in [c for c in final_feature_columns if c not in df.columns]:
        df[col] = 0.0

    df_processed = df[final_feature_columns].ffill().fillna(0.0).copy()
    df_scaled = df_processed.copy()

    # --- Scaling Modernized ---
    rsi_cols = [c for c in final_feature_columns if re.match(r'^rsi_\d+$', c)]
    other_cols = [c for c in final_feature_columns if c not in rsi_cols]

    if 'rsi' in ticker_scalers and rsi_cols:
        df_scaled[rsi_cols] = ticker_scalers['rsi'].transform(df_processed[rsi_cols])
    
    if 'standard' in ticker_scalers:
        cols_to_std = other_cols if 'rsi' in ticker_scalers else final_feature_columns
        df_scaled[cols_to_std] = ticker_scalers['standard'].transform(df_processed[cols_to_std])

    # --- Prepare final tensor ---
    latest_window = df_scaled.iloc[-window_size_model:]
    if len(latest_window) < window_size_model:
        raise ValueError(f"Rows ({len(latest_window)}) < window size ({window_size_model})")
        
    # --- 🔍 DIAGNOSTIC FEATURE CHECK ---
    # This will print the exact feature names and their live, scaled values 
    # right before they are fed into the neural network for a prediction.
    logging.info(f"\n--- [DIAGNOSTIC] Live Feature Array for {ticker} ---")
    latest_row = latest_window.iloc[-1]
    
    # Check if the number of features matches what the model expects
    if len(latest_row) != num_features:
        logging.critical(f"⚠️ FEATURE MISMATCH! Model expects {num_features}, but pipeline generated {len(latest_row)}.")
    else:
        logging.info(f"Feature count matches model expectations ({num_features}/{num_features}).")

    for idx, (col_name, val) in enumerate(latest_row.items()):
        # Highlight the custom quantum/macro features so they are easy to spot
        highlight = "✨ " if "quantum" in col_name or "kalman" in col_name or "ghost" in col_name else "  "
        logging.info(f"{idx+1:>2}. {highlight}{col_name:<25}: {val:>8.4f}")
    logging.info("--------------------------------------------------\n")
        
    return tf.convert_to_tensor(latest_window.values.astype(np.float32)[np.newaxis, ...]), df_scaled

def predict_step(model, state_tensor):
    return model(state_tensor, training=False)

def get_trading_signal(state_tensor, ticker):
    """ 
    Gets action signal with Entropy Filtering and 'Memory' Persistence Buffering.
    UPDATED: Strict 0.40 Confidence + Standard Deviation Margin Check.
    """
    global last_proposed_signals, tactical_regime
    default_hold = config.get('env', {}).get('action_hold', 0)
    
    # AUTOPILOT: Pulls from dynamic regime (Force a hard floor of 0.40 regardless of regime)
    threshold = max(tactical_regime["min_confidence"], 0.40)

    # Return 0.0 confidence if there is no state
    if ticker not in models or state_tensor is None:
        return default_hold, 0.0

    try:
        # 1. Get raw logits and apply Softmax
        _, policy_logits = predict_step(models[ticker]['model'], state_tensor)
        action_probs = tf.nn.softmax(policy_logits).numpy().flatten()
        
        # 2. Identify Best Action
        proposed_action = int(np.argmax(action_probs))
        confidence = action_probs[proposed_action]
        logging.info(f"[{ticker}] RAW PROPOSAL: {proposed_action} (Conf: {confidence:.2%})") 
        
        # --- STEP 1: ENTROPY FILTER (Standard Deviation Check) ---
        # Sort probabilities to find the gap between 1st and 2nd best choices
        sorted_probs = np.sort(action_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1]
        prob_std_dev = np.std(action_probs)
        
        # Require the margin to be strictly greater than the standard deviation, 
        # AND maintain a minimum absolute gap of 5%.
        if proposed_action in [1, 2] and (margin < prob_std_dev or margin < 0.05):
            logging.info(f"[{ticker}] Margin too thin (Margin: {margin:.2f} vs StdDev: {prob_std_dev:.2f}). Model is undecided. Forcing HOLD.")
            return default_hold, confidence

        # --- STEP 2: PERSISTENCE BUFFER (Memory-Based) ---
        prev_active_signal = last_proposed_signals.get(ticker)
        
        # Only update the buffer if the model is suggesting an ACTIVE action (1, 2, or 3)
        if proposed_action != default_hold:
            last_proposed_signals[ticker] = proposed_action 

        high_confidence_override = 0.60

        # Only require persistence for ENTERING a trade (1=Long, 2=Short)
        if proposed_action in [1, 2]:
            current_state = trade_states.get(ticker, {}).get('state')
            is_holding_long = (current_state == 'IN_POSITION' and proposed_action == 1)
            is_holding_short = (current_state == 'IN_POSITION' and proposed_action == 2)
            
            if is_holding_long or is_holding_short:
                logging.info(f"[{ticker}] Signal {proposed_action} aligns with open position. Holding.")
                return proposed_action, confidence

            if confidence > high_confidence_override:
                logging.info(f"[{ticker}] High Confidence ({confidence:.2f}) overrides persistence check.")
                last_proposed_signals[ticker] = proposed_action
                return proposed_action, confidence

            if proposed_action != prev_active_signal:
                logging.info(f"[{ticker}] New {proposed_action} signal. Waiting for 2nd cycle confirmation.")
                return default_hold, confidence
            else:
                logging.info(f"[{ticker}] {proposed_action} signal CONFIRMED (Memory match).")

        # --- FINAL CONFIDENCE GATE ---
        if confidence < threshold:
            logging.info(f"[{ticker}] Confidence {confidence:.2f} < {threshold:.2f} floor. Forcing HOLD.")
            return default_hold, confidence
        
        logging.info(f"[{ticker}] FINAL SIGNAL: {proposed_action} (Conf: {confidence:.2%}, Margin: {margin:.2f}, StdDev: {prob_std_dev:.2f})")
        return proposed_action, confidence

    except Exception as e:
        logging.error(f"[{ticker}] Signal generation error: {e}")
        return default_hold, 0.0

# --- Alpaca & Trading Logic ---
def get_current_price(ticker):
    try:
        snapshot = api.get_snapshot(ticker)
        if snapshot and snapshot.latest_trade: return float(snapshot.latest_trade.p)
        trade = api.get_latest_trade(ticker)
        return float(trade.p)
    except Exception as e:
        logging.warning(f"Could not get current price for {ticker}: {e}")
        return None

def check_if_market_open():
    try: return api.get_clock().is_open
    except Exception: return False

def cancel_existing_orders(ticker):
    try:
        orders = api.list_orders(status='open', symbols=[ticker])
        for order in orders: api.cancel_order(order.id)
        if orders: logging.info(f"Canceled {len(orders)} open order(s) for {ticker}.")
    except Exception as e:
        logging.error(f"Error canceling orders for {ticker}: {e}")

def escalate_open_orders(ticker, timeout_seconds=60):
    """
    Checks for open orders older than timeout_seconds and 
    escalates them to MARKET orders to ensure execution.
    Returns True if an escalation was triggered, False otherwise.
    """
    global api
    escalated = False
    
    try:
        open_orders = api.list_orders(status='open', symbols=[ticker])
        
        for order in open_orders:
            # Alpaca submitted_at is already UTC-aware
            submit_time = order.submitted_at
            now = datetime.now(pytz.utc)
            duration = (now - submit_time).total_seconds()

            if duration >= timeout_seconds:
                logging.warning(f"!!! {ticker} order {order.id} is stale ({duration:.0f}s). ESCALATING TO MARKET.")
                
                # 1. Cancel the stale limit order
                try:
                    api.cancel_order(order.id)
                    # Safety pause: allows Alpaca's matching engine to update order state
                    time.sleep(1) 
                    
                    # 2. Resubmit as a MARKET order
                    api.submit_order(
                        symbol=order.symbol,
                        qty=order.qty,
                        side=order.side,
                        type='market',
                        time_in_force='day'
                    )
                    logging.info(f" -> {ticker} successfully escalated to MARKET {order.side.upper()}.")
                    escalated = True
                    
                except Exception as e_inner:
                    logging.error(f"Failed to complete escalation steps for {ticker}: {e_inner}")

    except Exception as e:
        logging.error(f"Error during order escalation check for {ticker}: {e}")
    
    return escalated

# --- TACTICAL REGIME GLOBALS ---
tactical_regime = {
    "size_multiplier": 1.0,
    "min_confidence": config.get('CONFIDENCE_THRESHOLD', 0.28),
    "stop_loss_mult": 1.0, 
    "take_profit_target": 0.06,
    "breakeven_shield": False,
    "freeze_buys": False,
    # --- NEW MISSION CONTROL OVERRIDES ---
    "shorts_suspended": False,
    "elite_edge_active": False,
    "high_hit_rate_active": False,
    "panic_shock_active": False,
    "hard_stop_override": None
}

def update_tactical_regime():
    """Calculates 30-day rolling metrics and auto-adjusts bot parameters (Synced with Mission Control)."""
    global api, tactical_regime, config, ghost_history, GHOST_MA_WINDOW
    logging.info("Updating Tactical Regime parameters...")
    try:
        # Fetch minimal history for rolling math (60 days)
        history = api.get_portfolio_history(period='2M', timeframe='1D')
        df = pd.DataFrame({'timestamp': history.timestamp, 'equity': history.equity})
        
        # Convert timestamp to timezone-aware to match deposit logic
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.tz_localize('UTC')
        
        # --- SYNC FIX: APPEND LIVE EQUITY TICK TO MATCH MISSION CONTROL ---
        account = api.get_account()
        current_equity = float(account.equity)
        margin_util = (float(account.maintenance_margin) / current_equity * 100) if current_equity > 0 else 0
        
        live_row = pd.DataFrame([{'timestamp': pd.Timestamp.now(tz='UTC'), 'equity': current_equity}])
        # Use clean concat (avoids pandas warning)
        df = pd.concat([df, live_row], ignore_index=True)
        # ------------------------------------------------------------------

        if len(df) < 35: 
            logging.info("Not enough history for tactical adjustments. Using defaults.")
            return

        # --- SYNC: DEPOSIT ADJUSTMENT (Matches Mission Control) ---
        def apply_bot_deposit(df_to_adj, date_str, amount):
            ts = pd.Timestamp(date_str, tz='UTC')
            mask = df_to_adj['timestamp'] >= ts
            df_to_adj.loc[mask, 'equity'] -= amount
            return df_to_adj

        # Keep this list synced with your Mission Control script!
        df = apply_bot_deposit(df, "2026-01-24", 68.10)
        df = apply_bot_deposit(df, "2026-02-12", 69.81)
        df = apply_bot_deposit(df, "2026-02-16", 139.75)
        df = apply_bot_deposit(df, "2026-02-26", 69.71)
        df = apply_bot_deposit(df, "2026-03-04", 68.84)
        df = apply_bot_deposit(df, "2026-03-13", 69.61)
        df = apply_bot_deposit(df, "2026-03-21", 69.01)
        df = apply_bot_deposit(df, "2026-04-09", 69.30)
        df = apply_bot_deposit(df, "2026-04-15", 70.20)
        df = apply_bot_deposit(df, "2026-04-23", 70.40)
        df = apply_bot_deposit(df, "2026-04-29", 70.37)
        df = apply_bot_deposit(df, "2026-05-06", 71.11)
        df = apply_bot_deposit(df, "2026-05-14", 71.28)
        df = apply_bot_deposit(df, "2026-05-21", 70.12)
        # ----------------------------------------------------------
            
        df['daily_return'] = df['equity'].pct_change()
        
        # 30-Day Rolling Math
        window = 30
        roll_mean = df['daily_return'].rolling(window).mean()
        roll_std = df['daily_return'].rolling(window).std()
        
        latest_sharpe = ((roll_mean.iloc[-1] / roll_std.iloc[-1]) * (252 ** 0.5)) if roll_std.iloc[-1] > 0 else 0
        latest_win_rate = (df['daily_return'] > 0).astype(int).rolling(window).mean().iloc[-1] * 100
        
        active_days = (df['daily_return'] != 0).rolling(window).sum().iloc[-1]
        latest_sqn = (active_days ** 0.5) * (roll_mean.iloc[-1] / roll_std.iloc[-1]) if roll_std.iloc[-1] > 0 else 0
        
        rolling_peak = df['equity'].rolling(window, min_periods=1).max()
        dd_raw = (df['equity'] - rolling_peak) / rolling_peak
        latest_ulcer = ((dd_raw ** 2).rolling(window).mean().iloc[-1]) ** 0.5 * 100
        
        # --- 1. SHARPE (Position Sizing) ---
        if latest_sharpe < 0.5: tactical_regime["size_multiplier"] = 0.50
        elif latest_sharpe > 1.0: tactical_regime["size_multiplier"] = 1.0
              
        # --- 2. SQN (Selectivity) - FIXED BASELINE ---
        # Ghost Trading now evaluates market safety, so we lock to the base standard.
        tactical_regime["min_confidence"] = config.get('CONFIDENCE_THRESHOLD', 0.28) 

        # --- 3. ULCER INDEX (Defense) ---
        if latest_ulcer > 3.0: 
            tactical_regime["stop_loss_mult"] = 0.50 
            tactical_regime["breakeven_shield"] = True
        elif latest_ulcer < 1.5:
            tactical_regime["stop_loss_mult"] = 1.0
            tactical_regime["breakeven_shield"] = False
            
        # --- 4. WIN RATE (Take Profits) ---
        if latest_win_rate < 45.0: tactical_regime["take_profit_target"] = 0.03
        elif latest_win_rate > 50.0: tactical_regime["take_profit_target"] = 0.06
        
        # --- 5. MARGIN (Leverage) ---
        tactical_regime["freeze_buys"] = margin_util > 75.0

        # --- 6. GHOST TRADING GATES & SIZING TRACKERS ---
        tactical_regime["allow_longs"] = True
        tactical_regime["allow_shorts"] = True
        tactical_regime["long_ma"] = 0.0   # Track raw performance for sizing
        tactical_regime["short_ma"] = 0.0  # Track raw performance for sizing

        if len(ghost_history['long']) >= GHOST_MA_WINDOW:
            long_ma = sum(ghost_history['long'][-GHOST_MA_WINDOW:]) / GHOST_MA_WINDOW
            tactical_regime["allow_longs"] = long_ma >= -0.005 # Allow tiny drift
            tactical_regime["long_ma"] = long_ma # Expose to sizing logic

        if len(ghost_history['short']) >= GHOST_MA_WINDOW:
            short_ma = sum(ghost_history['short'][-GHOST_MA_WINDOW:]) / GHOST_MA_WINDOW
            tactical_regime["allow_shorts"] = short_ma >= -0.005
            tactical_regime["short_ma"] = short_ma # Expose to sizing logic

        logging.info(f"TACTICAL REGIME | Sharpe: {latest_sharpe:.2f} | SQN: {latest_sqn:.2f} | Ulcer: {latest_ulcer:.2f} | Win: {latest_win_rate:.1f}%")
        logging.info(f"GHOST REGIME | Long MA: {sum(ghost_history['long'][-GHOST_MA_WINDOW:])/GHOST_MA_WINDOW if ghost_history['long'] else 0:.2%} (Active: {tactical_regime['allow_longs']}) | Short MA: {sum(ghost_history['short'][-GHOST_MA_WINDOW:])/GHOST_MA_WINDOW if ghost_history['short'] else 0:.2%} (Active: {tactical_regime['allow_shorts']})")
        
    except Exception as e:
        logging.error(f"Error updating tactical regime: {e}")

def sync_mission_control_directives():
    """Reads Streamlit's dedicated override file and updates the bot's internal tactical regime."""
    global tactical_regime, STOP_LOSS_LONG_PCT
    override_path = '/app/system_override.json'
    
    if not os.path.exists(override_path): 
        return # Continue with bot defaults if Mission Control hasn't generated overrides yet
    
    try:
        with open(override_path, 'r') as f:
            state = json.load(f)
            
        directives = state.get("global_directives", {})
        if not directives: return
        
        # Override the bot's internal math with Streamlit's commands
        tactical_regime["size_multiplier"] = directives.get("sizing_multiplier", 1.0)
        tactical_regime["take_profit_target"] = directives.get("dynamic_take_profit_pct", 0.06)
        
        # Calculate the multiplier needed to force the exact Stop Loss Streamlit wants
        target_sl = directives.get("dynamic_stop_loss_pct", 0.03)
        tactical_regime["stop_loss_mult"] = target_sl / STOP_LOSS_LONG_PCT 
        
        # Override Ghost Gates
        tactical_regime["allow_longs"] = directives.get("ghost_gates", {}).get("long", True)
        tactical_regime["allow_shorts"] = directives.get("ghost_gates", {}).get("short", True)

        # --- NEW MISSION CONTROL OVERRIDES ---
        active_regime = directives.get('active_regime', 'STABLE')

        # 1. Shorts Suspended (Ghost Gate Closed)
        tactical_regime["shorts_suspended"] = not tactical_regime["allow_shorts"]

        # 2. Elite Edge (Base Sizing Restored)
        tactical_regime["elite_edge_active"] = (tactical_regime["size_multiplier"] >= 1.0)

        # 3. High Hit Rate (Trail/Hold Activated)
        tactical_regime["high_hit_rate_active"] = (tactical_regime["take_profit_target"] > 0.06)

        # 4. Regime Drift: PANIC / SHOCK (Tighten hard stops)
        tactical_regime["panic_shock_active"] = "PANIC" in active_regime or "SHOCK" in active_regime
        if tactical_regime["panic_shock_active"]:
            tactical_regime["hard_stop_override"] = 0.015  # -1.5%
        else:
            tactical_regime["hard_stop_override"] = None
        # ---------------------------------------
        
        logging.info(f"[MISSION CONTROL SYNC] Regime: {active_regime} | SL: {target_sl*100:.1f}% | TP: {tactical_regime['take_profit_target']*100:.1f}% | Size: {tactical_regime['size_multiplier']}x")
        
    except Exception as e:
        logging.error(f"Error reading system_override.json: {e}")

# --- Trading Logic Functions ---
def monitor_and_manage_positions():
    """
    Checks open positions for SL/TP triggers and Time Stops.
    Includes 'Order Chasing' logic via escalate_open_orders to prevent 
    stuck limit orders.
    """
    global api, trade_states, COOLDOWN_PERIOD_DAYS, tickers, models, MAX_HOLD_DAYS
    
    if api is None: 
        logging.error("API unavailable in monitor_and_manage.")
        return

    logging.info("Monitoring positions and checking for stale orders...")
    
    tickers_with_models = list(models.keys())
    current_time_utc = datetime.now(pytz.utc)

    # --- TASK 1: ESCALATE STALE ORDERS (Order Chaser) ---
    # We call the external function to handle the "Order Chasing" logic.
    # It should return a list of tickers that were escalated so we skip them this cycle.
    escalated_tickers = []
    for ticker in tickers_with_models:
        try:
            # We assume escalate_open_orders handles the 60s logic internally
            was_escalated = escalate_open_orders(ticker) 
            if was_escalated:
                escalated_tickers.append(ticker)
        except Exception as e_esc:
            logging.error(f"Error calling escalation for {ticker}: {e_esc}")

    # --- TASK 2: POSITION MONITORING (SL / TP / TIME) ---
    try:
        positions = api.list_positions()
    except Exception as e:
        logging.error(f"Error fetching positions: {e}. Cannot monitor.")
        return

    held_symbols_alpaca = {p.symbol: p for p in positions}
    position_closed_this_run = False

    for ticker in tickers_with_models:
        # If we just escalated an order to Market for this ticker, 
        # skip monitoring until the next cycle to allow the fill to settle.
        if ticker in escalated_tickers:
            continue

        current_state_info = trade_states.get(ticker, {'state': 'WAITING'})
        current_state = current_state_info.get('state')
        position = held_symbols_alpaca.get(ticker)

        try:
            # --- State Synchronization ---
            if position:
                # NEW: Check if this ticker is locked due to PDT
                pdt_lock = trade_states[ticker].get('pdt_lock_until')
                if pdt_lock:
                    pdt_lock_dt = datetime.fromisoformat(pdt_lock)
                    if pdt_lock_dt.tzinfo is None: pdt_lock_dt = pytz.utc.localize(pdt_lock_dt)
                    
                    if current_time_utc < pdt_lock_dt:
                        continue # Skip checking exits for this ticker until tomorrow
                    else:
                        del trade_states[ticker]['pdt_lock_until'] # Lock expired, resume normal operations
                        
                # FIX: Check if we are already waiting for an exit to fill
                if current_state == 'COOLDOWN':
                    logging.info(f"[{ticker}] Exit order is pending/filling. Waiting.")
                    continue # Skip SL checks to let the order fill or escalate
                elif current_state != 'IN_POSITION':
                    logging.warning(f"State sync: Alpaca position for {ticker} found. Syncing to IN_POSITION.")
                    trade_states[ticker] = {'state': 'IN_POSITION', 'entry_price': float(position.avg_entry_price)}
                    current_state = 'IN_POSITION'
                elif 'entry_price' not in current_state_info:
                    trade_states[ticker]['entry_price'] = float(position.avg_entry_price)

                # Sync Entry Time for Time Stop
                if 'entry_time' not in trade_states[ticker]:
                    try:
                        closed_orders = api.list_orders(status='closed', symbols=[ticker], limit=1)
                        if closed_orders and closed_orders[0].filled_at:
                            trade_states[ticker]['entry_time'] = closed_orders[0].filled_at.isoformat()
                        else:
                            trade_states[ticker]['entry_time'] = current_time_utc.isoformat()
                    except Exception:
                        trade_states[ticker]['entry_time'] = current_time_utc.isoformat()

            else: # No Alpaca position exists
                if current_state in ['IN_POSITION', 'PENDING_ENTRY']:
                    logging.warning(f"State mismatch: {ticker} disappeared from Alpaca. Manual liquidation assumed. Entering COOLDOWN.")
                    
                    # Force the cooldown even for manual closures
                    cd_time = current_time_utc + timedelta(days=COOLDOWN_PERIOD_DAYS)
                    trade_states[ticker] = {
                        'state': 'COOLDOWN', 
                        'cooldown_until': cd_time.isoformat()
                    }
                    current_state = 'COOLDOWN'

            # --- Exit Logic Checks ---
            if current_state == 'IN_POSITION' and position:
                qty = abs(float(position.qty))
                avg_entry = float(position.avg_entry_price)
                side = position.side
                current_price = get_current_price(ticker)
                if not current_price: continue

                # --- APPLY PANIC/SHOCK OVERRIDE ---
                if tactical_regime.get("panic_shock_active") and tactical_regime.get("hard_stop_override"):
                    dynamic_sl = tactical_regime["hard_stop_override"]
                else:
                    dynamic_sl = STOP_LOSS_LONG_PCT * tactical_regime["stop_loss_mult"]

                dynamic_tp = tactical_regime["take_profit_target"]

                if side == 'long':
                    stop_p = avg_entry * (1 - dynamic_sl)
                    prof_p = avg_entry * (1 + dynamic_tp)

                    # --- APPLY HIGH HIT RATE TRAIL ---
                    if tactical_regime.get("high_hit_rate_active"):
                        if current_price > avg_entry:
                            trail_stop = current_price * 0.98  # Trail 2% below current peak
                            stop_p = max(stop_p, trail_stop)

                    # DYNAMIC SHIELD: Trigger only when halfway to the profit target
                    shield_trigger = avg_entry * (1 + (dynamic_tp * 0.5))
                    if tactical_regime["breakeven_shield"] and current_price >= shield_trigger:
                        stop_p = max(stop_p, avg_entry * 1.002)
                        
                    close_side = 'sell'
                else: # short
                    if tactical_regime.get("panic_shock_active") and tactical_regime.get("hard_stop_override"):
                        dynamic_short_sl = tactical_regime["hard_stop_override"]
                    else:
                        dynamic_short_sl = STOP_LOSS_SHORT_PCT * tactical_regime["stop_loss_mult"]
                        
                    stop_p = avg_entry * (1 + dynamic_short_sl)
                    prof_p = avg_entry * (1 - dynamic_tp)

                    # --- APPLY HIGH HIT RATE TRAIL ---
                    if tactical_regime.get("high_hit_rate_active"):
                        if current_price < avg_entry:
                            trail_stop = current_price * 1.02  # Trail 2% above current trough
                            stop_p = min(stop_p, trail_stop)

                    # DYNAMIC SHIELD: Trigger only when halfway to the profit target (Shorts)
                    shield_trigger = avg_entry * (1 - (dynamic_tp * 0.5))
                    if tactical_regime["breakeven_shield"] and current_price <= shield_trigger:
                        stop_p = min(stop_p, avg_entry * 0.998) 
                        
                    close_side = 'buy'

                exit_reason = None
                
                # 1. Price-based Exit Check
                if (side == 'long' and current_price <= stop_p) or (side == 'short' and current_price >= stop_p):
                    exit_reason = f"Stop-Loss (Price: {current_price:.2f})"
                elif (side == 'long' and current_price >= prof_p) or (side == 'short' and current_price <= prof_p):
                    exit_reason = f"Take-Profit (Price: {current_price:.2f})"

                # 2. Time-based Exit Check (if no price trigger)
                if not exit_reason:
                    entry_time_str = trade_states[ticker].get('entry_time')
                    if entry_time_str:
                        entry_dt = datetime.fromisoformat(entry_time_str)
                        if entry_dt.tzinfo is None: entry_dt = pytz.utc.localize(entry_dt)
                        
                        calendar = api.get_calendar(start=entry_dt.date().isoformat(), end=current_time_utc.date().isoformat())
                        days_held = max(0, len(calendar) - 1)
                        if days_held >= MAX_HOLD_DAYS:
                            exit_reason = f"Time-Stop ({days_held} days)"

                # --- Execution: Protected Exit ---
                if exit_reason:
                    logging.info(f"!!! {ticker}: {exit_reason} condition met. Triggering exit.")
                    cancel_existing_orders(ticker)
                    
                    # FIX: Use market orders for Stop-Loss to prevent chasing. Limit for Take-Profit.
                    order_type = 'market' if 'Stop-Loss' in exit_reason else 'limit'
                    
                    limit_buffer = 0.005
                    l_price = current_price * (1 - limit_buffer) if close_side == 'sell' else current_price * (1 + limit_buffer)
                    
                    try:
                        if order_type == 'limit':
                            api.submit_order(
                                symbol=ticker, qty=qty, side=close_side,
                                type='limit', limit_price=round(l_price, 2),
                                time_in_force='day'
                            )
                        else:
                            api.submit_order(
                                symbol=ticker, qty=qty, side=close_side,
                                type='market', time_in_force='day'
                            )
                        
                        # FIX: Inject into Ghost Tracker ONLY upon successful order submission
                        try:
                            pnl_pct = (current_price - avg_entry) / avg_entry
                            if side == 'short':
                                pnl_pct = -pnl_pct
                            logging.info(f"[GHOST SYNC] Injecting live {side.upper()} trade result ({pnl_pct:.2%}) into Ghost Tracker.")
                            ghost_history[side].append(pnl_pct)
                        except Exception as e_ghost:
                            logging.error(f"Failed to calculate and sync live PnL to Ghost Engine for {ticker}: {e_ghost}")

                        cd_time = current_time_utc + timedelta(days=COOLDOWN_PERIOD_DAYS)
                        trade_states[ticker] = {'state': 'COOLDOWN', 'cooldown_until': cd_time.isoformat()}
                        position_closed_this_run = True
                        time.sleep(1) 
                    except Exception as e_exit:
                        error_str = str(e_exit).lower()
                        if 'pattern day trading' in error_str:
                            logging.error(f"!!! [{ticker}] SEC PDT RULE TRIGGERED! Alpaca blocked the exit to prevent a 90-day freeze. You are trapped in this position until tomorrow.")
                            # Lock further exit attempts for 24 hours to prevent API spam
                            lock_time = current_time_utc + timedelta(hours=24)
                            trade_states[ticker]['pdt_lock_until'] = lock_time.isoformat()
                        else:
                            logging.error(f"Failed to submit exit order for {ticker}: {e_exit}")

        except Exception as e_loop:
            logging.error(f"Error processing {ticker} in monitor loop: {e_loop}", exc_info=True)

    if not position_closed_this_run and held_symbols_alpaca:
        logging.info(" -> No new positions met SL/TP criteria.")

def monitor_ghost_positions():
    """
    Monitors virtual 'ghost' positions to evaluate current market regimes.
    Checks for virtual Stop Loss and Take Profit efficiently via bulk snapshot.
    """
    global ghost_positions, ghost_history, tactical_regime, api

    if not ghost_positions:
        return

    logging.info("Monitoring ghost positions...")
    tickers_to_remove = []
    
    # Extract list of active ghost tickers
    ghost_tickers = list(ghost_positions.keys())
    
    # O(1) Bulk fetch for all ghost prices
    current_prices = {}
    try:
        snapshots = api.get_snapshots(ghost_tickers)
        for ticker, snapshot in snapshots.items():
            if snapshot and snapshot.latest_trade:
                current_prices[ticker] = float(snapshot.latest_trade.p)
    except Exception as e:
        logging.error(f"Failed to fetch bulk snapshots for ghost positions: {e}")
        return # Skip this cycle if prices fail to load

    # Process logic locally
    for ticker, ghost_trade in ghost_positions.items():
        current_price = current_prices.get(ticker)
        if not current_price: 
            continue

        entry_price = ghost_trade.get('entry_price')
        side = ghost_trade.get('side')

        if not entry_price or not side:
            tickers_to_remove.append(ticker)
            continue

        dynamic_sl = STOP_LOSS_LONG_PCT * tactical_regime["stop_loss_mult"] if side == 'long' else STOP_LOSS_SHORT_PCT * tactical_regime["stop_loss_mult"]
        dynamic_tp = tactical_regime["take_profit_target"]

        if side == 'long':
            stop_p = entry_price * (1 - dynamic_sl)
            prof_p = entry_price * (1 + dynamic_tp)
        else: # short
            stop_p = entry_price * (1 + dynamic_sl)
            prof_p = entry_price * (1 - dynamic_tp)

        exit_reason = None
        pnl_pct = 0.0

        if (side == 'long' and current_price <= stop_p) or (side == 'short' and current_price >= stop_p):
            exit_reason = "Stop-Loss"
            pnl_pct = -dynamic_sl
        elif (side == 'long' and current_price >= prof_p) or (side == 'short' and current_price <= prof_p):
            exit_reason = "Take-Profit"
            pnl_pct = dynamic_tp

        if exit_reason:
            logging.info(f"[GHOST] {ticker} {side.upper()} closed via {exit_reason}. PnL: {pnl_pct:.2%}")
            ghost_history[side].append(pnl_pct)
            tickers_to_remove.append(ticker)

    # Cleanup
    for ticker in tickers_to_remove:
        del ghost_positions[ticker]

def generate_signals_and_enter_positions():
    """
    Analyzes all models, filters through Ghost Regime gates first, calculates R-Position sizes,
    and executes orders ensuring precise capital scaling.
    """
    global api, trade_states, models, tickers, POST_ORDER_DELAY_SECONDS, RISK_PER_TRADE_PCT, tactical_regime, ghost_positions

    if api is None:
        logging.error("API unavailable in generate_signals.")
        return
    
    logging.info(f"--- Portfolio Pass Start: R-Sizing (Risk: {RISK_PER_TRADE_PCT*100:.2f}%) ---")

    # Initialize Account Boundaries
    account = api.get_account()
    net_liquidity = float(account.equity)
    long_mv = float(account.long_market_value)
    short_mv = abs(float(account.short_market_value))
    
    # NEW: Pattern Day Trader (PDT) Protection Check
    daytrade_count = int(account.daytrade_count)
    is_pdt_restricted = (daytrade_count >= 3 and net_liquidity < 25000)
    if is_pdt_restricted:
        logging.warning(f"⚠️ PDT LIMIT REACHED ({daytrade_count}/3 Day Trades). All new entries will be routed to the Ghost Tracker to prevent being trapped.")
    
    true_available_cash = net_liquidity - long_mv - (short_mv * 1.5)
    available_buffer = true_available_cash * 0.98
    
    current_time_utc = datetime.now(pytz.utc)
    tickers_with_models = list(models.keys())
    
    desired_entries = []
    total_requested_cost = 0
    ghost_count = 0

    # --- PASS 1: Identify Signals, Gate via Regime, & Calculate IDEAL Sizes ---
    for ticker in tickers_with_models:
        current_state_info = trade_states.get(ticker, {'state': 'WAITING'})
        current_state = current_state_info.get('state')

        if current_state == 'COOLDOWN':
            cooldown_until_str = current_state_info.get('cooldown_until')
            cooldown_until = datetime.fromisoformat(cooldown_until_str) if isinstance(cooldown_until_str, str) else cooldown_until_str
            if cooldown_until and current_time_utc < cooldown_until:
                continue
            else:
                trade_states[ticker] = {'state': 'WAITING'}
                current_state = 'WAITING'

        try:
            market_data = get_market_data(ticker, period=HISTORICAL_DATA_PERIOD, interval=YF_INTERVAL)
            state_tensor, _ = preprocess_data(market_data, ticker)
            action, confidence = get_trading_signal(state_tensor, ticker)

            # A. EXIT SIGNAL LOGIC
            if current_state == 'IN_POSITION' and action == 3:
                logging.info(f"!!! MODEL SIGNAL: CLOSE {ticker} received.")
                try:
                    position = api.get_position(ticker)
                    current_price = get_current_price(ticker)
                    close_side = 'buy' if position.side == 'short' else 'sell'
                    cancel_existing_orders(ticker)
                    limit_buffer = 0.005
                    protected_limit = current_price * (1 - limit_buffer) if close_side == 'sell' else current_price * (1 + limit_buffer)
                    
                    api.submit_order(
                        symbol=ticker, qty=abs(float(position.qty)), 
                        side=close_side, type='limit', limit_price=round(protected_limit, 2),
                        time_in_force='day'
                    )
                    trade_states[ticker] = {'state': 'COOLDOWN', 'cooldown_until': (current_time_utc + timedelta(days=COOLDOWN_PERIOD_DAYS)).isoformat()}
                except Exception as e:
                    logging.error(f"Error during model-based exit for {ticker}: {e}")
                continue

            # B. ENTRY SIGNAL LOGIC
            if current_state == 'WAITING' and action in [1, 2]:
                current_price = get_current_price(ticker)
                if not current_price: continue
                
                action_name = "LONG" if action == 1 else "SHORT"

                # --- NEW PRE-PASS: GHOST TRADING ROUTING LOGIC ---
                is_blocked = False
                block_reason = ""

                if tactical_regime.get("freeze_buys", False):
                    is_blocked = True
                    block_reason = "Margin > 75% (Buys Frozen)"
                elif is_pdt_restricted:
                    is_blocked = True
                    block_reason = "PDT Limit Reached (No Day Trades Left)"
                elif action == 1 and not tactical_regime.get("allow_longs", True):
                    is_blocked = True
                    block_reason = "Longs Disabled by Market Regime"
                elif action == 2 and (not tactical_regime.get("allow_shorts", True) or tactical_regime.get("shorts_suspended", False)):
                    is_blocked = True
                    block_reason = "Shorts Suspended/Disabled by Mission Control"

                if is_blocked:
                    logging.info(f"[GHOST ROUTING] {action_name} {ticker} blocked by: {block_reason}. Sending to Ghost Tracker.")
                    ghost_positions[ticker] = {
                        'side': 'long' if action == 1 else 'short',
                        'entry_price': current_price,
                        'entry_time': current_time_utc.isoformat()
                    }
                    ghost_count += 1
                    continue # Skip size calculation and prevent adding to total_requested_cost
                # -------------------------------------------------

                # Calculate local volatility
                feat_cfg = models[ticker]['config']['features']
                atr_series = compute_atr(market_data, feat_cfg)
                atr_volatility = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else 0.03
                
                if atr_volatility > 0.20:
                    atr_volatility = 0.20
                
                # Dynamic Allocation
                active_ticker_count = len(tickers_with_models)
                
                # --- ELITE EDGE SIZING OVERRIDE ---
                if tactical_regime.get("elite_edge_active"):
                    current_multiplier = 1.0
                else:
                    current_multiplier = tactical_regime.get("size_multiplier", 1.0)
                    # If the Tactical Regime penalizes size, high confidence can only recover half of the penalty.
                    if confidence >= 0.65 and current_multiplier < 1.0:
                        penalty_recovery = (1.0 - current_multiplier) * 0.5 
                        current_multiplier = current_multiplier + penalty_recovery
                        logging.info(f"[{ticker}] High Conf ({confidence:.2f}) partially overrides regime. Multiplier adjusted to {current_multiplier:.2f}")
                # ----------------------------------

                base_slot_capital = (net_liquidity / active_ticker_count) * current_multiplier

                # --- ASYMMETRIC POSITION SIZING (GHOST ENGINE INTEGRATED) ---
                if action == 1: # LONG
                    ghost_ma = tactical_regime.get("long_ma", 0.0)
                    
                    # Ghost Engine Boost/Penalty for Longs
                    if ghost_ma > 0.015:   # Virtual longs are printing (>1.5% avg profit)
                        ghost_multiplier = 1.15
                    elif ghost_ma < 0.0:   # Virtual longs are struggling but not blocked yet
                        ghost_multiplier = 0.85
                    else:
                        ghost_multiplier = 1.0
                        
                    capital_weight = LONG_CAPITAL_WEIGHT * ghost_multiplier
                    logging.info(f"[{ticker}] LONG | Base: {LONG_CAPITAL_WEIGHT} | Ghost Mult: {ghost_multiplier:.2f} | Final Wt: {capital_weight:.2f}")

                elif action == 2: # SHORT
                    ghost_ma = tactical_regime.get("short_ma", 0.0)
                    
                    # Apply baseline volatility penalty to shorts
                    vol_penalty = 1.0 if atr_volatility < 0.10 else 0.80
                    
                    # Ghost Engine Boost/Penalty for Shorts
                    if ghost_ma > 0.015:   # Virtual shorts are crushing it (Definitive Bear Regime)
                        ghost_multiplier = 1.25 # Aggressive boost because shorts are working
                    elif ghost_ma < 0.0:   # Virtual shorts struggling (Chop/Grind up)
                        ghost_multiplier = 0.75 # Heavily penalize weak shorts
                    else:
                        ghost_multiplier = 1.0
                        
                    capital_weight = SHORT_CAPITAL_WEIGHT * vol_penalty * ghost_multiplier
                    logging.info(f"[{ticker}] SHORT | Vol Pen: {vol_penalty:.2f} | Ghost Mult: {ghost_multiplier:.2f} | Final Wt: {capital_weight:.2f}")
                else:
                    capital_weight = 1.00
                # ------------------------------------------------------------

                target_capital_deployment = base_slot_capital * capital_weight
                final_r_qty = int(target_capital_deployment / (current_price + 1e-9))

                if final_r_qty > 0:
                    cost = final_r_qty * current_price
                    desired_entries.append({
                        'ticker': ticker, 'qty': final_r_qty, 
                        'action': action, 'price': current_price, 'cost': cost
                    })
                    total_requested_cost += cost # Cost is now ONLY valid, unblocked trades
                    logging.info(f"[{ticker}] R-Size Proposal: {final_r_qty} shares (Est. Cost: ${cost:.2f})")

        except Exception as ticker_err:
            logging.error(f"Error analyzing {ticker}: {ticker_err}")

    # --- PASS 2: CAPITAL SCALING (Now accurate) ---
    scaling_factor = 1.0
    if total_requested_cost > available_buffer and total_requested_cost > 0:
        scaling_factor = available_buffer / total_requested_cost
        logging.warning(f"!!! TOTAL REQUEST (${total_requested_cost:.2f}) > AVAILABLE (${available_buffer:.2f}). Scaling by {scaling_factor:.2f}")

    # --- PASS 3: ORDER EXECUTION ---
    entered_count = 0
    for trade in desired_entries:
        final_qty = int(trade['qty'] * scaling_factor)
        if final_qty <= 0: continue
        
        ticker = trade['ticker']
        side = 'buy' if trade['action'] == 1 else 'sell'
        action_name = "LONG" if trade['action'] == 1 else "SHORT"
        
        asset = api.get_asset(ticker)
        if not asset.tradable or (trade['action'] == 2 and not asset.shortable):
            continue

        try:
            limit_buffer = 0.005
            l_price = trade['price'] * (1 + limit_buffer) if trade['action'] == 1 else trade['price'] * (1 - limit_buffer)
            
            logging.info(f">>> SUBMITTING {action_name}: {final_qty} {ticker} @ {l_price:.2f}")
            order = api.submit_order(
                symbol=ticker, qty=final_qty, side=side,
                type='limit', limit_price=round(l_price, 2),
                time_in_force='day'
            )
            
            trade_states[ticker] = {
                'state': 'PENDING_ENTRY', 
                'order_id': order.id,
                'entry_time': datetime.now(pytz.utc).isoformat()
            }
            entered_count += 1
            time.sleep(POST_ORDER_DELAY_SECONDS)
        except Exception as e:
            logging.error(f"API Entry Error for {ticker}: {e}")

    if entered_count > 0 or ghost_count > 0:
        logging.info(f"Portfolio Pass Complete. Live Entries: {entered_count} | Ghost Entries: {ghost_count}")
        
# --- Function to Sync Initial State ---
STATE_FILE_PATH = os.path.join(script_dir, "trading_state.json")

def initialize_trade_states():
    """
    Syncs the internal trade_states with current Alpaca positions and relevant open orders at startup.
    Loads persisted state first (including the Persistence Buffer) to recover Entry Times, 
    Cooldowns, and the last seen model signals.
    """
    global trade_states, last_proposed_signals, api, tickers, models 
    logging.info("Initializing and syncing trade states with Alpaca...")
    
    if api is None:
        logging.error("API unavailable, cannot sync states. Initializing all to WAITING.")
        trade_states = {ticker: {'state': 'WAITING'} for ticker in models.keys()} 
        return

    # 1. Default Initialization: Set all monitored tickers to WAITING
    tickers_with_models = list(models.keys()) 
    trade_states = {ticker: {'state': 'WAITING'} for ticker in tickers_with_models}
    last_proposed_signals = {} # Initialize empty persistence buffer

    # 2. Load Persisted State (The "Brain Transplant")
    if os.path.exists(STATE_FILE_PATH):
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                payload = json.load(f)
            
            # Handle the new payload structure vs old flat structure
            if isinstance(payload, dict) and "trade_states" in payload:
                loaded_states = payload.get("trade_states", {})
                last_proposed_signals = payload.get("last_proposed_signals", {})
                
                # --- NEW: Load Ghost Data ---
                global ghost_positions, ghost_history
                ghost_positions = payload.get("ghost_positions", {})
                ghost_history = payload.get("ghost_history", {'long': [], 'short': []})
                # -----------------------------
                
                logging.info(f"Restored Persistence Buffer and Ghost States.")
            else:
                # Fallback for the very first run after code change
                loaded_states = payload 
                logging.info("Old state file format detected. Persistence Buffer initialized as empty.")

            restored_count = 0
            for ticker, info in loaded_states.items():
                if ticker in tickers_with_models:
                    # Restore Datetime Objects from ISO Strings
                    if 'cooldown_until' in info and info['cooldown_until']:
                        try:
                            dt = datetime.fromisoformat(info['cooldown_until'])
                            if dt.tzinfo is None: dt = pytz.utc.localize(dt)
                            info['cooldown_until'] = dt
                        except ValueError:
                            logging.warning(f"Invalid cooldown datetime for {ticker}, resetting.")
                            info['cooldown_until'] = None

                    trade_states[ticker] = info
                    restored_count += 1
            
            logging.info(f"Successfully restored persisted states for {restored_count} tickers.")
        except Exception as e:
            logging.error(f"Failed to load state file: {e}. Starting with fresh states.")

    # 3. Sync with Real World (Alpaca API)
    initial_state_count = Counter() 

    try:
        # Get current positions
        positions = api.list_positions()
        current_positions_alpaca = {p.symbol: p for p in positions if p.symbol in tickers_with_models}
        logging.info(f"Found {len(current_positions_alpaca)} existing positions on Alpaca.")

        # Get relevant open orders
        open_orders_map = {} 
        try:
            all_init_orders = api.list_orders(status='open', limit=500, symbols=tickers_with_models)
            for o in all_init_orders:
                if o.symbol not in open_orders_map: 
                    if o.type in ['market', 'limit']: 
                        open_orders_map[o.symbol] = o
            logging.info(f"Found {len(open_orders_map)} relevant open BUY orders.")
        except Exception as oe:
            logging.warning(f"Could not fetch open orders during init: {oe}")

        # 4. Reconciliation Loop
        for ticker in tickers_with_models:
            state_info = trade_states.get(ticker, {'state': 'WAITING'}) 
            current_internal_state = state_info.get('state') 

            alpaca_pos = current_positions_alpaca.get(ticker)
            alpaca_order = open_orders_map.get(ticker)

            # Scenario A: Real Position Exists
            if alpaca_pos:
                if current_internal_state == 'COOLDOWN':
                    logging.info(f"Sync ({ticker}): Found position, but state is COOLDOWN (likely pending exit). Maintaining COOLDOWN.")
                    # Keep existing entry price if possible
                    if 'entry_price' not in state_info:
                        try: trade_states[ticker]['entry_price'] = float(alpaca_pos.avg_entry_price)
                        except: pass
                elif current_internal_state != 'IN_POSITION':
                    logging.info(f"Sync ({ticker}): Found position. State '{current_internal_state}' -> IN_POSITION.")
                    trade_states[ticker]['state'] = 'IN_POSITION'
                    try: trade_states[ticker]['entry_price'] = float(alpaca_pos.avg_entry_price)
                    except: pass

            # Scenario B: Real Order Exists
            elif alpaca_order:
                if current_internal_state != 'PENDING_ENTRY':
                    logging.info(f"Sync ({ticker}): Found open order. State '{current_internal_state}' -> PENDING_ENTRY.")
                
                trade_states[ticker] = {'state': 'PENDING_ENTRY', 'order_id': alpaca_order.id}

            # Scenario C: State Mismatch - Internal thinks busy, Alpaca says free
            elif current_internal_state == 'IN_POSITION':
                logging.warning(f"Sync ({ticker}): Position missing on startup. Assumed closed. Moving to COOLDOWN.")
                cd_time = datetime.now(pytz.utc) + timedelta(days=COOLDOWN_PERIOD_DAYS)
                trade_states[ticker] = {'state': 'COOLDOWN', 'cooldown_until': cd_time.isoformat()}            
            elif current_internal_state == 'PENDING_ENTRY':
                logging.warning(f"Sync ({ticker}): Internal state PENDING_ENTRY but no open order. Resetting to WAITING.")
                trade_states[ticker] = {'state': 'WAITING'}

            # Scenario D: Cooldown Management
            elif current_internal_state == 'COOLDOWN':
                cooldown_until = state_info.get('cooldown_until')
                valid_cooldown = False
                
                if isinstance(cooldown_until, datetime):
                    if cooldown_until.tzinfo is None: cooldown_until = pytz.utc.localize(cooldown_until)
                    valid_cooldown = True
                
                if valid_cooldown and datetime.now(pytz.utc) < cooldown_until:
                    logging.info(f"Sync ({ticker}): Enforcing COOLDOWN until {cooldown_until.strftime('%Y-%m-%d %H:%M')}.")
                else:
                    logging.info(f"Sync ({ticker}): Cooldown expired. Resetting to WAITING.")
                    trade_states[ticker] = {'state': 'WAITING'}

            initial_state_count[trade_states[ticker]['state']] += 1

        logging.info(f"State sync complete. Summary: {dict(initial_state_count)}")

    except Exception as e:
        logging.error(f"Error syncing states with Alpaca: {e}", exc_info=True)

def save_trade_states():
    """Saves the current trade_states and last_proposed_signals to a JSON file."""
    global trade_states, last_proposed_signals, ghost_positions, ghost_history
    try:
        # 1. Prepare trade_states for serialization
        states_to_save = {}
        for ticker, info in trade_states.items():
            serializable_info = info.copy()
            
            # Convert Cooldown datetime to string
            if 'cooldown_until' in serializable_info:
                cd = serializable_info['cooldown_until']
                if isinstance(cd, datetime):
                    serializable_info['cooldown_until'] = cd.isoformat()
            
            # Convert Entry Time datetime to string
            if 'entry_time' in serializable_info:
                et = serializable_info['entry_time']
                if isinstance(et, datetime):
                    serializable_info['entry_time'] = et.isoformat()
            
            states_to_save[ticker] = serializable_info

        # 2. Package dictionaries into a single payload (Bot State Only)
        payload = {
            "trade_states": states_to_save,
            "last_proposed_signals": last_proposed_signals,
            "ghost_positions": ghost_positions,
            "ghost_history": ghost_history
        }

        # 3. Write to file
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(payload, f, indent=4)
            
        logging.info(f"Bot state safely saved to {STATE_FILE_PATH}.")

    except Exception as e:
        logging.error(f"Failed to save trade states: {e}")

# === FIX: Add the missing locking function ===
def acquire_lock(lock_file_path):
    """Attempts to acquire an exclusive lock on a file. Returns the file handle if successful, None otherwise."""
    if not fcntl_available:
        logging.info("fcntl not available, skipping file lock.")
        if os.path.exists(lock_file_path):
            logging.warning(f"Lock file {lock_file_path} exists. Manual check recommended.")
        return "dummy_handle" # Return a non-None value to allow script to run

    try:
        f = open(lock_file_path, 'w')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logging.info(f"Successfully acquired lock on {lock_file_path}")
        f.write(str(os.getpid()))
        f.flush()
        return f
    except BlockingIOError:
        # Code to read PID from lock file can be added here if needed
        logging.error(f"Another instance appears to be running (lock file {lock_file_path} is locked). Exiting.")
        f.close()
        return None
    except Exception as e:
        logging.error(f"Error acquiring lock on {lock_file_path}: {e}")
        if 'f' in locals(): f.close()
        return None

import gspread
import os

def push_logs_to_cloud(log_file_path):
    """
    Pushes the last 50 lines of the local log file to Google Sheets.
    Includes re-authentication and error handling to prevent bot crashes.
    """
    try:
        # 1. Connect to Google Sheets
        # We re-authenticate every time to prevent the "12-hour token expiry" issue
        creds_path = '/app/google_creds.json'
        
        # specific check to help debug if path is wrong
        if not os.path.exists(creds_path):
            print(f"⚠️ [Cloud Log] Creds file missing at: {creds_path}")
            return

        gc = gspread.service_account(filename=creds_path)
        sh = gc.open("Angel_Bot_Logs")
        worksheet = sh.worksheet("logs")
        
        # 2. Read last 50 lines of local log safely
        if not os.path.exists(log_file_path):
            print(f"⚠️ [Cloud Log] Log file missing: {log_file_path}")
            return

        with open(log_file_path, 'r') as f:
            # Read all lines and take the last 50
            lines = f.readlines()[-600:]
        
        # 3. Clear and Update Sheet
        worksheet.clear()
        
        # Format for single column (List of Lists)
        cell_list = [[line.strip()] for line in lines] 
        
        # FIXED: Use named arguments to avoid DeprecationWarning
        worksheet.update(range_name='A1', values=cell_list)
        
    except Exception as e:
        # Crucial: Catch ANY error so the trading bot does not crash
        print(f"❌ [Cloud Log] Sync Failed: {e}")

# ------------------- Main Trading Loop -------------------
if __name__ == "__main__":
    # --- Configure Logging (Includes PID) ---
    log_format = f'%(asctime)s [%(levelname)s] [PID:{os.getpid()}] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', force=True) # Force reconfigures root logger

    # --- Acquire Lock ---
    lock_handle = acquire_lock(LOCK_FILE)
    if lock_handle is None:
        sys.exit(1) # Exit immediately if lock couldn't be acquired

    # --- Main execution block wrapped in try/finally to ensure lock release ---
    try:
        logging.info("="*60)
        logging.info("      ALPACA PPO TRADING BOT - POLLING VERSION (Per-Ticker Models)") # Updated Title
        logging.info("="*60)
        logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Configured Tickers: {len(tickers)} ({', '.join(tickers)})")
        logging.info(f"Models Successfully Loaded For: {len(models)} tickers ({', '.join(models.keys())})") # Log loaded models
        logging.info(f"Alpaca Base URL: {BASE_URL}")
        logging.info(f"YFinance Interval: {YF_INTERVAL}")
        logging.info(f"Investment per Trade: ${INVESTMENT_AMOUNT:,.2f}")
        logging.info(f"Long  Strategy | SL: {STOP_LOSS_LONG_PCT*100:.1f}% | TP: {TAKE_PROFIT_LONG_PCT*100:.1f}%")
        logging.info(f"Short Strategy | SL: {STOP_LOSS_SHORT_PCT*100:.1f}% | TP: {TAKE_PROFIT_SHORT_PCT*100:.1f}%")
        logging.info(f"Polling Interval: {POLLING_INTERVAL_SECONDS}s")
        logging.info(f"Required Feature Buffer: {REQUIRED_FEATURE_BUFFER} bars")
        logging.info(f"Models/Scalers Dir: {MODELS_DIR}")
        logging.info("Press Ctrl+C to stop the bot.")
        logging.info("="*60)

        # --- Initial Critical Checks (after lock acquired) ---
        if api is None:
            logging.critical("CRITICAL: Exiting - Alpaca API not available.")
            sys.exit(1)
        if not tf_available:
            logging.critical("CRITICAL: Exiting - TensorFlow not available.")
            sys.exit(1)
        if not models: # Check if the models dictionary is empty
             logging.critical("CRITICAL: Exiting - No PPO models were loaded successfully.")
             sys.exit(1)
        try:
            import pandas_ta
            logging.info(f"pandas_ta version {getattr(pandas_ta, '__version__', 'unknown')} found.")
        except ImportError:
            logging.critical("CRITICAL: pandas_ta library not found. Please install using 'pip install pandas_ta'. Exiting.")
            sys.exit(1)
        if not tickers:
            logging.critical("CRITICAL: Exiting - No tickers defined in config or defaults.")
            sys.exit(1)

        # Verify all needed scaler files exist *for loaded models*
        all_scalers_found = True
        tickers_missing_scalers = []
        for ticker in models.keys(): # Only check scalers for tickers where model loaded
             scaler_filename = SCALER_FILENAME_FORMAT.format(ticker=ticker, interval=YF_INTERVAL)
             scaler_path = os.path.join(MODELS_DIR, scaler_filename)
             if not os.path.exists(scaler_path):
                 logging.error(f"CRITICAL: Scaler file missing for {ticker} at {scaler_path} (Model was loaded!). Bot cannot trade this ticker.")
                 all_scalers_found = False
                 tickers_missing_scalers.append(ticker)

        # If scalers are missing, remove the corresponding models so the bot doesn't try to trade them
        if tickers_missing_scalers:
             for ticker in tickers_missing_scalers:
                  del models[ticker]
             logging.warning(f"Removed models for tickers with missing scalers: {tickers_missing_scalers}")
             if not models: # Check if *no* models remain after removing those with missing scalers
                  logging.critical("CRITICAL: Exiting - No models remain after removing those with missing scalers.")
                  sys.exit(1)
             logging.warning(f"Continuing with remaining models: {list(models.keys())}")

        # --- Initialize States (Sync with Alpaca for tickers with models) ---
        initialize_trade_states() # Now correctly initializes only for remaining models
        # -------------------------------------------

        last_market_status = None # Track market status changes
        cycle_count = 0

        # --- THE MAIN WHILE LOOP ---
        while True:
            cycle_count += 1
            current_time_utc = datetime.now(pytz.utc)
            market_open = check_if_market_open()

            if market_open != last_market_status:
                logging.info(f"--- Market Status Change: Now {'OPEN' if market_open else 'CLOSED'} ---")
                last_market_status = market_open
                
                if not market_open:
                    logging.info("Market closed. Initiating daily state save and memory flush.")
                    save_trade_states()
                    
                    # Force TensorFlow and Python memory release
                    try:
                        clear_session()
                        import gc
                        gc.collect()
                        logging.info("System memory cleared and ready for next session.")
                    except Exception as e:
                        logging.error(f"Memory cleanup failed: {e}")

            if market_open:
                logging.info(f"\n=== Cycle #{cycle_count} Start: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} (Market OPEN) ===")
                
                # 1. Bot calculates its internal regime
                update_tactical_regime()
                
                # 2. Bot syncs with Mission Control overrides
                sync_mission_control_directives() 
                
                monitor_and_manage_positions()
                monitor_ghost_positions()
                generate_signals_and_enter_positions()
                logging.info(f"=== Cycle #{cycle_count} End. Current states count: {Counter(ts.get('state', 'UNKNOWN') for ts in trade_states.values())} ===")
                
                import gc
                gc.collect()

                logging.info(f"Sleeping for {POLLING_INTERVAL_SECONDS} seconds...")
                sleep_interval = POLLING_INTERVAL_SECONDS
            else:
                # Log less frequently when market is closed
                if cycle_count % 12 == 1: # Log roughly every hour when closed (12 * 300s = 3600s)
                    logging.info(f"--- {current_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}: Market still CLOSED. Sleeping... ---")
                sleep_interval = 300 # Sleep longer when closed (e.g., 5 minutes)

            # --- CRITICAL UPDATE: CLOUD PUSH OUTSIDE MARKET_OPEN BLOCK ---
            # This ensures logs update even if market is closed
            try:
                # Assuming standard output is redirected to this file via cron/nohup
                LOG_FILE_PATH = '/app/cron_ALPACATradingBot_REAL_V11.log'
                
                # Simple check to avoid errors if file doesn't exist yet
                if os.path.exists(LOG_FILE_PATH) and os.path.getsize(LOG_FILE_PATH) > 0:
                    #pass
                    push_logs_to_cloud(LOG_FILE_PATH)
                    # logging.info("Logs pushed to Cloud.") # Uncomment to verify in local log
            except Exception as e:
                logging.error(f"Failed to push logs to cloud: {e}")
            # -------------------------------------------------------------

            # --- Sleep ---
            try:
                time.sleep(sleep_interval)
            except KeyboardInterrupt:
                logging.info("\nCtrl+C detected during sleep. Initiating shutdown...")
                break

    except KeyboardInterrupt:
        logging.info("\nCtrl+C detected during main execution. Stopping the trading bot gracefully...")

    except Exception as e:
        logging.critical(f"FATAL ERROR encountered in main execution: {e}", exc_info=True)

    finally:
        logging.info("--- Initiating shutdown sequence ---")
        logging.info("Attempting to save final trade states...")
        save_trade_states()

        # --- Release the Lock ---
        if fcntl_available and lock_handle and lock_handle != "dummy_handle":
            logging.info(f"Releasing lock file: {LOCK_FILE}")
            try:
                # Check PID before removing lock file
                remove_lock = True
                try:
                    # Ensure file pointer is at the beginning to read PID
                    lock_handle.seek(0)
                    pid_in_file = lock_handle.read().strip()
                    if pid_in_file.isdigit() and int(pid_in_file) != os.getpid():
                        logging.warning(f"Lock file PID ({pid_in_file}) does not match current PID ({os.getpid()}). Not removing lock file.")
                        remove_lock = False
                    elif not pid_in_file.isdigit():
                         logging.warning(f"Content in lock file ('{pid_in_file}') is not a valid PID. Not removing lock file.")
                         remove_lock = False
                except Exception as read_err:
                    logging.warning(f"Could not read PID from lock file before removal: {read_err}. Not removing lock file.")
                    remove_lock = False

                # Unlock the file first
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
                lock_handle.close() # Close the file handle

                # Remove the file only if PID matched
                if remove_lock and os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                    logging.info("Lock file removed.")
                elif remove_lock and not os.path.exists(LOCK_FILE):
                     logging.info("Lock file already removed.")

            except Exception as lock_release_e:
                logging.error(f"Error releasing or removing lock file {LOCK_FILE}: {lock_release_e}")
        elif lock_handle == "dummy_handle":
             logging.info("File locking was skipped (fcntl not available).")
        else:
             logging.info("No active lock handle to release.")
        # --- End Lock Release ---

        logging.info("--- Trading bot script finished ---")