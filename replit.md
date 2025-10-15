# AlgoTrader Pro - Replit Setup

## Project Overview

AlgoTrader Pro is a cryptocurrency algorithmic trading platform built with Streamlit that provides automated trading capabilities on the Bybit exchange. The system operates in both virtual (paper trading) and real trading modes, featuring:

- **Signal Generation**: Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Volume)
- **Machine Learning**: XGBoost-based signal filtering and scoring
- **Portfolio Management**: Position sizing, risk management, stop-loss/take-profit automation
- **Real-time Monitoring**: WebSocket connections for live market data
- **Performance Analytics**: Comprehensive trade tracking and reporting

## Current State

The application is successfully running on Replit with the following configuration:

### Environment
- **Python Version**: 3.11
- **Database**: PostgreSQL (Replit-managed)
- **Frontend**: Streamlit on port 5000
- **Trading Mode**: Virtual mode (paper trading) is active by default

### Key Features Working
✅ Database initialized with SQLAlchemy ORM  
✅ Virtual trading mode fully functional  
✅ Capital management system migrated from JSON to database  
✅ Multi-page Streamlit interface (Dashboard, Signals, Trades, Performance, Settings)  
✅ WebSocket background event loop for real-time data  
✅ API credentials configured for Bybit integration  

### API Integration Status
- Bybit API credentials are configured via Replit Secrets
- Currently experiencing 403 errors from Bybit API (may require API key permission updates on Bybit.com)
- Virtual mode works independently without API connectivity
- Real mode requires valid Bybit API credentials with proper permissions

## Recent Changes (October 15, 2025)

### Configuration Updates (Project Import)
1. **Streamlit Configuration** (`.streamlit/config.toml`):
   - Updated port from 5005 to 5000 (Replit standard)
   - Added `enableCORS = false` for Replit proxy compatibility
   - Added `enableXsrfProtection = false` for iframe embedding
   - Configured `address = "0.0.0.0"` for proper network binding

2. **Gitignore Enhancement**:
   - Added comprehensive Python-specific entries
   - Excluded virtual environments, cache files, build artifacts
   - Protected sensitive files (.env, logs, databases)

3. **Deployment Configuration**:
   - Configured for Autoscale deployment (stateless web app)
   - Run command: `streamlit run app.py --server.port=5000 --server.address=0.0.0.0`

### Feature Enhancements (October 15, 2025)

1. **Mainnet-Only Trading Enforcement**:
   - Removed testnet/sandbox options from settings UI
   - Added "Mainnet Only" information message in settings
   - Hardcoded `BYBIT_MAINNET=true` in environment configuration
   - BybitClient now exclusively connects to mainnet endpoints

2. **Batch Real Trade Execution**:
   - Added multi-select widget on Signals page to select multiple signals
   - New "Execute N Real Trade(s)" button for batch execution
   - Batch execution uses the same validated pipeline as single trades:
     * Normalizes signal keys (Symbol→symbol, Side→side, Entry→entry)
     * Sets required fields (qty, leverage)
     * Calls execute_real_trade helper for each signal
     * Full database persistence and error handling
     * Rollback support on failures
   - Shows aggregate success count after batch execution
   - Automatically syncs trades after completion

3. **Notification Credential Configuration**:
   - Added notification settings section to Settings page
   - Password-masked input fields for:
     * Discord Webhook URL
     * Telegram Bot Token and Chat ID
     * WhatsApp Number (via Twilio)
   - Credentials saved to session state and environment variables
   - Input validation with clear error messages

### Environment Variables
Required secrets configured in Replit Secrets:
- `BYBIT_API_KEY` - Bybit API key for live trading
- `BYBIT_API_SECRET` - Bybit API secret for authentication
- `DATABASE_URL` - PostgreSQL connection string (auto-configured by Replit)
- `BYBIT_MAINNET` - Set to `true` (mainnet-only enforcement)

## Project Architecture

### Frontend (Streamlit Multi-page App)
- **Main Entry**: `app.py` - Initializes engine, manages session state, sidebar controls
- **Pages**:
  - `pages/dashboard.py` - Overview, market data, portfolio summary
  - `pages/signals.py` - Trading signal generation and management
  - `pages/trades.py` - Active and historical trade tracking
  - `pages/performance.py` - Analytics and performance metrics
  - `pages/settings.py` - Configuration management

### Backend Components
- **Trading Engine** (`engine.py`) - Orchestrates all trading operations
- **Bybit Client** (`bybit_client.py`) - REST API + WebSocket communication
- **Automated Trader** (`automated_trader.py`) - Manages trading loops and execution
- **Signal Generator** (`signal_generator.py`) - Technical analysis and ML filtering

### Data Layer
- **Database Manager** (`db.py`) - SQLAlchemy ORM with PostgreSQL
- **Models**: Signals, Trades, WalletBalance, Settings
- **Error Recovery**: Automatic retry logic for transient database errors
- **Migration Support**: Capital data migrated from JSON to database

## User Preferences

*No specific user preferences have been recorded yet. This section will be updated as preferences are established.*

## How to Use

### Development Mode
1. The app is already running - access it via the Replit webview
2. Default mode is "Virtual" (paper trading with $100 simulated capital)
3. Switch to "Real" mode in the sidebar to enable live trading (requires valid API keys)

### Trading Modes
- **Virtual Mode**: 
  - Safe paper trading with simulated funds
  - No real money at risk
  - Perfect for testing strategies
  
- **Real Mode**:
  - Live trading on Bybit
  - Requires funded Bybit account
  - API keys must have trading permissions enabled
  - Syncs wallet balance and positions from Bybit

### Important Notes
- Always test strategies in Virtual mode first
- Ensure API keys have proper permissions on Bybit.com
- Monitor the Emergency Stop button for immediate halt of all trading
- Database automatically tracks all trades and signals

## Troubleshooting

### Bybit API 403 Errors
If you see 403 Forbidden errors in logs:
1. Log into Bybit.com → API Management
2. Verify API key has these permissions:
   - Read permission
   - Trade permission (for live trading)
   - Wallet permission (for balance sync)
3. Check if IP restrictions are enabled (Replit uses dynamic IPs)
4. Consider regenerating API keys if issues persist

### Database Issues
- Database is automatically initialized on first run
- SQLAlchemy handles connection pooling and retries
- Check DATABASE_URL secret if connection fails

### Port Already in Use
- Workflow configured for port 5000 (Replit standard)
- If conflicts occur, check for duplicate workflows

## Next Steps

Potential enhancements:
- Strategy backtesting module
- Multi-exchange support
- Advanced ML models for signal prediction
- Portfolio rebalancing automation
- Enhanced notification system (Discord, Telegram, WhatsApp)

---

*Last Updated: October 15, 2025*  
*Replit Project Import Completed*
