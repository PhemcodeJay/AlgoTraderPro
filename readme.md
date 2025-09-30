# AlgoTrader Pro

AlgoTrader Pro is a cryptocurrency algorithmic trading platform built with Streamlit that provides automated trading capabilities across multiple trading pairs on the Bybit exchange. The system operates in both virtual (paper trading) and real trading modes, featuring signal generation using technical indicators, machine learning filtering, portfolio management, and comprehensive performance analytics.

The platform is designed as a full-stack trading solution that can scan multiple cryptocurrency markets, generate trading signals based on technical analysis, execute trades automatically, and provide real-time monitoring and reporting capabilities through an intuitive web interface.

---

# System Architecture

## Frontend Architecture
- **Streamlit** is used as the primary web framework with a multi-page architecture.
- Main entry point (`app.py`) initializes the trading engine and manages session state.
- Individual pages live in the `pages/` directory (Dashboard, Signals, Trades, Performance Analytics, Settings).

## Backend Architecture
- **TradingEngine** (`engine.py`) → orchestrates trading operations.
- **BybitClient** (`bybit_client.py`) → handles API communication (REST + WebSocket).
- **AutomatedTrader** (`automated_trader.py`) → manages trading loops and execution.
- **SignalGenerator** (`signal_generator.py`) → creates technical analysis signals.

## Data Storage
- **PostgreSQL** via SQLAlchemy ORM for trades, signals, and settings.
- **Alembic** for database migrations.
- JSON config files: `settings.json`, `capital.json`, `virtual_trades.json`.

## Signal Generation
- Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Volume.
- ML filtering with XGBoost (`ml.py`) for signal scoring.

## Risk Management
- Position sizing based on account balance.
- Stop-loss and take-profit automation.
- Drawdown limits, leverage controls.
- Virtual trading mode for safe testing.

---

# External Dependencies

- **Bybit API** – market data + trade execution
- **pandas**, **numpy** – data wrangling
- **plotly** – interactive charts
- **scikit-learn**, **xgboost** – ML filtering
- **sqlalchemy**, **alembic** – database + migrations
- **psycopg2-binary** – PostgreSQL driver
- **requests**, **tenacity** – API handling
- **streamlit** – web UI
- **discord.py**, **telegram-bot**, WhatsApp integration – notifications

---

# 🚀 Setup Guide

## 1. Clone Repository
```bash
git clone https://github.com/yourusername/algotrader-pro.git
cd algotrader-pro
````

## 2. Environment Variables

Create a `.env` file:

```env
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
DB_URL=postgresql+psycopg2://trader:securepass@db:5432/algotrader
```

---

# 🐳 Dockerized Setup (Streamlit + PostgreSQL + pgAdmin4)

## docker-compose.yml

```yaml
version: "3.9"

services:
  db:
    image: postgres:15
    container_name: algotrader-db
    restart: always
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: securepass
      POSTGRES_DB: algotrader
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    container_name: algotrader-pgadmin
    restart: always
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@algotrader.local
      PGADMIN_DEFAULT_PASSWORD: adminpass
    ports:
      - "5050:80"
    depends_on:
      - db

  app:
    build: .
    container_name: algotrader-app
    restart: always
    environment:
      DB_URL: postgresql+psycopg2://trader:securepass@db:5432/algotrader
      BYBIT_API_KEY: your_api_key
      BYBIT_API_SECRET: your_api_secret
    volumes:
      - .:/app
    ports:
      - "8501:8501"
    depends_on:
      - db
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0

volumes:
  db_data:
```

## Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## requirements.txt

```
streamlit
sqlalchemy
psycopg2-binary
pandas
numpy
plotly
scikit-learn
xgboost
requests
python-dotenv
alembic
tenacity
```

---

# 🛠 Database Migrations with Alembic

## Initialize Alembic

```bash
alembic init migrations
```

This creates a `migrations/` folder and `alembic.ini`.

## Update `alembic.ini`

Set SQLAlchemy URL:

```ini
sqlalchemy.url = postgresql+psycopg2://trader:securepass@db:5432/algotrader
```

## Generate Migration

```bash
alembic revision --autogenerate -m "create initial tables"
```

## Apply Migration

```bash
alembic upgrade head
```

---

# 🔗 Access

* Streamlit App → [http://localhost:8501](http://localhost:8501)
* pgAdmin4 → [http://localhost:5050](http://localhost:5050)

  * Login: `admin@algotrader.local / adminpass`
  * Add Server → Host: `db`, User: `trader`, Password: `securepass`

---

# ✅ Development Flow

1. Edit code / models in Python.
2. Run `alembic revision --autogenerate -m "update tables"` when models change.
3. Run `alembic upgrade head` to apply DB schema changes.
4. Rebuild containers:

   ```bash
   docker-compose up --build
   ```

---

# 📊 Features in Progress

* Strategy backtesting module
* Multi-exchange support
* Portfolio rebalancing
* Advanced ML models