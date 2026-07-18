from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Initialize FastAPI application instance
app = FastAPI(title="S&P 500 Dynamic Portfolio Rebalancer")

# Load pre-trained Support Vector Machine (SVM) classifier model
model = joblib.load("sp500_model.joblib")

class PortfolioState(BaseModel):
    monthly_investment_allowance: float  # Dynamic user allowance parsed from Google Sheets ledger
    current_piggy_bank_cash: float       # Cash liquid reserves stored in the piggy bank
    current_sp500_portfolio_value: float # Market value evaluation of current stock tracker portfolio

@app.post("/predict")
def execute_portfolio_strategy(state: PortfolioState):
    # =========================================================================
    # 1. Background Feature Engineering (yfinance)
    # =========================================================================
    today = datetime.today()
    start_dt = (today - timedelta(days=450)).strftime('%Y-%m-%d')
    end_dt = (today + timedelta(days=2)).strftime('%Y-%m-%d')

    # Fetch fresh market underlying data
    sp500 = yf.download("^GSPC", start=start_dt, end=end_dt, progress=False)
    vix = yf.download("^VIX", start=start_dt, end=end_dt, progress=False)
    
    # Strips away the hierarchical ('Close', '^GSPC') structure back into standard flat indexes
    sp500.columns = sp500.columns.get_level_values(0)
    vix.columns = vix.columns.get_level_values(0)
    
    # Construct base operational DataFrame
    df = pd.DataFrame(index=sp500.index)
    df['Close'] = sp500['Close']
    df['Volume'] = sp500['Volume']
    df['VIX_Close'] = vix['Close']
    
    # Compute operational feature configurations mapping cleanly to the ML matrix array
    df['Feature_Month'] = df.index.month
    df['Feature_DayOfWeek'] = df.index.dayofweek
    df['Feature_VIX'] = df['VIX_Close']
    df['Feature_VIX_Change'] = df['Feature_VIX'].pct_change()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Feature_Price_to_SMA'] = df['Close'] / df['SMA_200']
    
    # Relative Strength Index (RSI) calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['Feature_RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Volatility and momentum variance checks
    df['Feature_Daily_Return'] = df['Close'].pct_change()
    df['Feature_Volatility'] = df['Feature_Daily_Return'].rolling(window=21).std()
    df['Feature_Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['Feature_RSI_Trend'] = df['Feature_RSI'].diff(periods=3)
    
    # Drop rolling indicator lookback null rows and isolate the absolute latest operational matrix row
    latest_row = df.dropna().tail(1)

    # Pulls the exact execution date timestamp from the isolated DataFrame index
    execution_date = latest_row.index[0].strftime('%Y-%m-%d')
    
    # Enforce strict matrix order preservation matching the training loop environment
    cols = [
        "Feature_Month", "Feature_DayOfWeek", "Feature_Price_to_SMA", 
        "Feature_RSI", "Feature_Daily_Return", "Feature_Volatility", 
        "Feature_Volume_Ratio", "Feature_RSI_Trend", "Feature_VIX", "Feature_VIX_Change"
    ]
    X = latest_row[cols]
    
    # Execute prediction mapping using the trained SVM architecture
    pred_class = int(model.predict(X)[0])
    
    # =========================================================================
    # 2. Dynamic Algorithmic Rebalancing Rules & [FIX 2] Portfolio Projections
    # =========================================================================
    allowance = state.monthly_investment_allowance
    piggy_cash = state.current_piggy_bank_cash
    portfolio = state.current_sp500_portfolio_value

    if pred_class == 2:  
        # SIDEWAYS MARKET: Regular, steady accumulation (50/50 split)
        market_regime = "Sideways (Consolidation)"
        action_signal = "BUY"
        amount_to_execute = allowance * 0.50
        cash_to_piggy = allowance * 0.50
        
        new_piggy_cash = piggy_cash + cash_to_piggy
        new_sp500_value = portfolio + amount_to_execute
        
        source = f"Deploying 50% of monthly allowance (RM{amount_to_execute:.2f})."
        note = f"Market is calm. Investing RM{amount_to_execute:.2f} and redirecting RM{cash_to_piggy:.2f} into the piggy bank for future discounts."

    elif pred_class == 1:
        # BEARISH MARKET: Market is at a discount. Buy aggressively!
        market_regime = "Bearish (Downside Contraction)"
        action_signal = "BUY"
        piggy_contribution = piggy_cash * 0.30
        amount_to_execute = allowance + piggy_contribution
        
        new_piggy_cash = piggy_cash - piggy_contribution
        new_sp500_value = portfolio + amount_to_execute  # 📈 [FIX 2] Portfolio grows by allowance + cash reserve injection
        
        source = f"100% of monthly allowance + RM{piggy_contribution:.2f} from Piggy Bank."
        note = f"Market fear detected! S&P 500 is on sale. Deploying all allowance and drawing heavily from your cash reserves to buy the dip."

    else:
        # BULLISH MARKET: Overextended tops. Stop buying, shave profits.
        market_regime = "Bullish (Trend Expansion)"
        action_signal = "SELL / HOLD"
        profit_taken = portfolio * 0.10
        amount_to_execute = profit_taken
        
        new_piggy_cash = piggy_cash + allowance + profit_taken
        new_sp500_value = portfolio - profit_taken
        
        source = f"Withdrawing 10% profit from S&P 500 value."
        note = f"Market is at an expensive peak. Monthly allowance saved as cash. Shaved RM{profit_taken:.2f} in profits to lock in gains."

    # Return structured JSON schema output to pipeline context nodes cleanly
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