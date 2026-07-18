import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests

# Initialize the FastAPI Web Application
app = FastAPI(title="S&P 500 Dynamic Portfolio Rebalancer")

# =========================================================================
# 🔄 1. MODEL & CONFIGURATION LOADING
# =========================================================================

# Load pre-trained Support Vector Machine (SVM) classifier model
# Ensure 'sp500_model.joblib' exists in the root of your deployment repository
model = joblib.load("sp500_model.joblib")

# Secure Environment Variable Extraction
# The key must be set inside Render's dashboard to prevent leaking raw tokens on GitHub
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

if not TIINGO_API_KEY:
    raise RuntimeError(
        "CRITICAL STARTUP FAILURE: The 'TIINGO_API_KEY' environment variable is missing. "
        "Please inject this variable inside your Render dashboard settings panel."
    )

# =========================================================================
# 📋 2. PYDANTIC DATA STRUCTS (API Payload Rules)
# =========================================================================

class PortfolioState(BaseModel):
    """
    Defines the incoming JSON contract structural payload sent from n8n / sheets.
    All data attributes are parsed as floats for consistent downstream math calculations.
    """
    monthly_investment_allowance: float  # Budget available for allocation this round (e.g., RM 500)
    current_piggy_bank_cash: float       # Strategic cash reserve pool on hand
    current_sp500_portfolio_value: float # Present valuation of existing index equity

# =========================================================================
# 🌐 3. TIINGO FINANCIAL INDICES DATA INGESTION UTILITY
# =========================================================================

def fetch_tiingo_history(ticker: str, items_count: int = 250):
    """
    Helper function to securely fetch End-Of-Day (EOD) financial market data from Tiingo.
    Authenticates requests via account tokens to safely bypass cloud data center IP blocks.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "token": TIINGO_API_KEY,
        "resampleFreq": "daily"
    }
    
    try:
        # Fetch REST payload with a explicit 10-second request timeout boundary
        response = requests.get(url, params=params, timeout=10)
        
        # Explicit error boundary checking for common HTTP failure situations
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid Tiingo API Token authentication configuration.")
        elif response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Tiingo upstream data failure for {ticker}: {response.text}")
            
        data = response.json()
        
        # Tiingo logs historical events chronologically (Oldest -> Newest).
        # We slice off the tail to grab exactly the lookback duration requested.
        if len(data) > items_count:
            data = data[-items_count:]
        return data
        
    except Exception as e:
        # Let explicit HTTP exceptions pass through, capture low-level connection drops here
        if isinstance(e, HTTPException): 
            raise e
        raise HTTPException(status_code=503, detail=f"Failed to connect to Tiingo data network infrastructure: {str(e)}")

# =========================================================================
# 🚀 4. REBALANCING EXECUTION CORE ROUTE
# =========================================================================

@app.post("/predict")
def execute_portfolio_strategy(state: PortfolioState):
    """
    Ingests user ledger variables, live-queries market indices data from Tiingo, 
    recomputes mathematical features, runs the SVM model, and emits actionable allocation directions.
    """
    
    # -------------------------------------------------------------------------
    # Phase A: Upstream Market Data Gathering & Dataframe Layout
    # -------------------------------------------------------------------------
    # Requesting 250 rows to safely guarantee a 220+ day rolling execution matrix for indicators
    sp500_raw = fetch_tiingo_history("spy", items_count=250)
    vix_raw = fetch_tiingo_history("vix", items_count=250)

    # Transform raw structural JSON lists straight into working Pandas DataFrames
    df_sp500 = pd.DataFrame(sp500_raw)
    df_vix = pd.DataFrame(vix_raw)

    if df_sp500.empty or df_vix.empty:
        raise HTTPException(status_code=502, detail="Upstream stock/index data source returned empty structures.")

    # Harmonize the date arrays and enforce index row sorting (Chronological Ascending order)
    df_sp500['date'] = pd.to_datetime(df_sp500['date'])
    df_sp500 = df_sp500.sort_values('date').set_index('date')
    
    df_vix['date'] = pd.to_datetime(df_vix['date'])
    df_vix = df_vix.sort_values('date').set_index('date')

    # Construct primary baseline operational framework dataframe
    df = pd.DataFrame(index=df_sp500.index)
    df['Close'] = df_sp500['close']
    df['Volume'] = df_sp500['volume']
    
    # Match Volatility Index prices across identical row date indices
    df['VIX_Close'] = df_vix['close']
    
    # Prune rows where market close holidays or tracking day data mismatch discrepancies occur
    df = df.dropna(subset=['Close', 'Volume', 'VIX_Close'])

    # -------------------------------------------------------------------------
    # Phase B: Technical Indicator Calculations (Re-generating Model Inputs)
    # -------------------------------------------------------------------------
    # Extract Calendar Features
    df['Feature_Month'] = df.index.month
    df['Feature_DayOfWeek'] = df.index.dayofweek
    
    # Extract Volatility Metrics
    df['Feature_VIX'] = df['VIX_Close']
    df['Feature_VIX_Change'] = df['Feature_VIX'].pct_change()
    
    # Trend Analysis: Price relative to its 200-day Simple Moving Average (SMA)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Feature_Price_to_SMA'] = df['Close'] / df['SMA_200']
    
    # Momentum Analysis: Relative Strength Index (RSI - 14 Days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['Feature_RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Variance and Structural Anomalies
    df['Feature_Daily_Return'] = df['Close'].pct_change()
    df['Feature_Volatility'] = df['Feature_Daily_Return'].rolling(window=21).std()
    df['Feature_Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['Feature_RSI_Trend'] = df['Feature_RSI'].diff(periods=3)
    
    # Clear out the initial NaN startup lookup rows caused by window bounds (e.g., 200 SMA lookback padding)
    processed_df = df.dropna()
    
    if processed_df.empty:
        raise HTTPException(
            status_code=422,
            detail="Mathematical processing layout failure: Insufficient historical trading days remaining."
        )
        
    # Extract the absolute latest compiled trading day record for prediction delivery
    latest_row = processed_df.tail(1)
    execution_date = latest_row.index[0].strftime('%Y-%m-%d')
    
    # Enforce precise variable structural positions matching the model training weights loop exactly
    cols = [
        "Feature_Month", "Feature_DayOfWeek", "Feature_Price_to_SMA", 
        "Feature_RSI", "Feature_Daily_Return", "Feature_Volatility", 
        "Feature_Volume_Ratio", "Feature_RSI_Trend", "Feature_VIX", "Feature_VIX_Change"
    ]
    X = latest_row[cols]
    
    # Feed the multi-dimensional tracking feature matrix to the SVM and parse classification label integer
    pred_class = int(model.predict(X)[0])
    
    # -------------------------------------------------------------------------
    # Phase C: Rules Engine & Dynamic Allocation Calculations
    # -------------------------------------------------------------------------
    allowance = state.monthly_investment_allowance
    piggy_cash = state.current_piggy_bank_cash
    portfolio = state.current_sp500_portfolio_value

    # --- REGIME CODE 2: SIDEWAYS / CONSOLIDATION MARKET ---
    if pred_class == 2:  
        market_regime = "Sideways (Consolidation)"
        action_signal = "BUY"
        amount_to_execute = allowance * 0.50
        cash_to_piggy = allowance * 0.50
        new_piggy_cash = piggy_cash + cash_to_piggy
        new_sp500_value = portfolio + amount_to_execute  
        source = f"Deploying 50% of monthly allowance (RM{amount_to_execute:.2f})."
        note = f"Market is calm. Investing RM{amount_to_execute:.2f} and redirecting RM{cash_to_piggy:.2f} into the piggy bank for future discounts."

    # --- REGIME CODE 1: BEARISH / MARKET SALE CONDITIONS ---
    elif pred_class == 1:
        market_regime = "Bearish (Downside Contraction)"
        action_signal = "BUY"
        piggy_contribution = piggy_cash * 0.30
        amount_to_execute = allowance + piggy_contribution
        new_piggy_cash = piggy_cash - piggy_contribution
        new_sp500_value = portfolio + amount_to_execute  
        source = f"100% of monthly allowance + RM{piggy_contribution:.2f} from Piggy Bank."
        note = f"Market fear detected! S&P 500 is on sale. Deploying all allowance and drawing heavily from your cash reserves to buy the dip."

    # --- REGIME CODE 0: BULLISH / ANOMALOUS HIGH PEAKS ---
    else:
        market_regime = "Bullish (Trend Expansion)"
        action_signal = "SELL / HOLD"
        profit_taken = portfolio * 0.10
        amount_to_execute = profit_taken
        new_piggy_cash = piggy_cash + allowance + profit_taken
        new_sp500_value = portfolio - profit_taken      
        source = f"Withdrawing 10% profit from S&P 500 value."
        note = f"Market is at an expensive peak. Monthly allowance saved as cash. Shaved RM{profit_taken:.2f} in profits to lock in gains."

    # Return compiled operational metadata back downstream to your pipeline layer
    return {
        "execution_date": execution_date,
        "market_regime": market_regime,
        "regime_code": pred_class,
        "action_signal": action_signal,
        "execution_details": {
            "target_asset": "S&P 500 ETF",
            "amount_to_execute": round(amount_to_execute, 2),
            "source_of_funds": source
        },
        "portfolio_updates": {
            "new_piggy_bank_cash": round(new_piggy_cash, 2),
            "new_sp500_portfolio_value": round(new_sp500_value, 2),
            "strategy_note": note
        }
    }