# engine.py
from trading_engine import TradingEngine
from automated_trader import AutomatedTrader
from bybit_client import BybitClient

def create_engine():
    client = BybitClient()
    engine = TradingEngine(client=client)
    trader = AutomatedTrader(engine, client)
    engine.trader = trader
    return engine
