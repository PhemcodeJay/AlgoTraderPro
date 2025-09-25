import os
import logging
import pandas as pd
import requests
import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

tz_utc3 = timezone(timedelta(hours=3))

def normalize_signal(signal: Any) -> Dict:
    if isinstance(signal, dict):
        return signal
    return {
        "symbol": getattr(signal, "symbol", "N/A"),
        "interval": getattr(signal, "interval", "N/A"),
        "signal_type": getattr(signal, "signal_type", "N/A"),
        "score": getattr(signal, "score", 0.0),
        "indicators": getattr(signal, "indicators", {}),
        "strategy": getattr(signal, "strategy", "Auto"),
        "side": getattr(signal, "side", "Buy"),
        "sl": getattr(signal, "sl", None),
        "tp": getattr(signal, "tp", None),
        "trail": getattr(signal, "trail", None),
        "liquidation": getattr(signal, "liquidation", None),
        "leverage": getattr(signal, "leverage", 10),
        "margin_usdt": getattr(signal, "margin_usdt", None),
        "entry": getattr(signal, "entry", None),
        "market": getattr(signal, "market", None),
        "created_at": getattr(signal, "created_at", None)
    }

def format_price_safe(value: Optional[float]) -> str:
    try:
        return f"{float(value):.4f}" if value is not None and value > 0 else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def format_currency_safe(value: Optional[float]) -> str:
    try:
        if value is None:
            return "0.00"
        return f"{float(value):.2f}"
    except (ValueError, TypeError, AttributeError):
        return "0.00"

def display_trades_table(trades: List[Dict], container, client=None, max_trades: int = 10):
    try:
        if not trades:
            container.info("🌙 No trades to display")
            return
            
        trades_data = []
        for trade in trades[:max_trades]:
            symbol = trade.get("symbol", "N/A")
            current_price = 0.0
            
            if client and hasattr(client, "get_current_price"):
                try:
                    current_price = client.get_current_price(symbol)
                except Exception as e:
                    logger.error(f"Error getting current price for {symbol}: {e}")
            
            qty = float(trade.get("qty", 0))
            entry_price = float(trade.get("entry_price", trade.get("price", 0)))
            side = trade.get("side", "Buy")
            
            # Calculate unrealized PnL
            if current_price > 0:
                if side.upper() in ["BUY", "LONG"]:
                    unreal_pnl = (current_price - entry_price) * qty
                else:
                    unreal_pnl = (entry_price - current_price) * qty
            else:
                unreal_pnl = 0.0
            
            # Use realized PnL if trade is closed, otherwise unrealized
            display_pnl = trade.get('pnl', unreal_pnl) if trade.get('status', '').lower() == 'closed' else unreal_pnl
            
            trades_data.append({
                "Symbol": symbol,
                "Side": side.title(),
                "Entry": f"${format_price_safe(entry_price)}",
                "Current": f"${format_price_safe(current_price)}" if current_price > 0 else "N/A",
                "P&L": f"${format_currency_safe(display_pnl)}",
                "Status": trade.get("status", "N/A").title(),
                "Mode": "Virtual" if trade.get("virtual", True) else "Real",
                "Score": f"{trade.get('score', 0):.1f}%" if trade.get('score') else "N/A",
                "Timestamp": trade.get("timestamp", "N/A")[:19] if trade.get("timestamp") else "N/A"
            })
        
        if trades_data:
            df = pd.DataFrame(trades_data)
            container.dataframe(df, use_container_width=True, height=min(400, len(trades_data) * 35 + 100))
        else:
            container.info("🌙 No trade data to display")
            
    except Exception as e:
        logger.error(f"Error displaying trades table: {e}")
        container.error(f"🚨 Error displaying trades: {e}")

def display_log_stats(log_file: str, container, refresh_key: str):
    try:
        if os.path.exists(log_file) and os.access(log_file, os.R_OK):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if not lines:
                container.info("🌙 No logs found")
                return
            
            # Count log levels
            error_count = sum(1 for line in lines if "ERROR" in line.upper())
            warning_count = sum(1 for line in lines if "WARNING" in line.upper())
            info_count = sum(1 for line in lines if "INFO" in line.upper())
            
            # Display metrics
            col1, col2, col3 = container.columns(3)
            col1.metric("Errors", error_count, delta=None if error_count == 0 else "⚠️")
            col2.metric("Warnings", warning_count)
            col3.metric("Info", info_count)
            
            # Show recent log entries
            recent_lines = lines[-20:]  # Show more recent entries
            with container.expander("Recent Log Entries", expanded=False):
                for line in recent_lines:
                    # Color code log levels
                    clean_line = line.strip()
                    if "ERROR" in clean_line:
                        st.error(clean_line)
                    elif "WARNING" in clean_line:
                        st.warning(clean_line)
                    else:
                        st.text(clean_line)
        else:
            container.warning("⚠️ Log file not accessible")
    except Exception as e:
        logger.error(f"Error displaying log stats: {e}")
        container.error(f"🚨 Error reading logs: {e}")

def get_trades_safe(db_instance, limit: int = 50) -> List[Dict]:
    try:
        trades = db_instance.get_trades(limit=limit)
        return [trade.to_dict() for trade in trades] if trades else []
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []

def get_open_trades_safe(db_instance) -> List[Dict]:
    try:
        trades = db_instance.get_open_trades()
        return [trade.to_dict() for trade in trades] if trades else []
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return []

def get_signals_safe(db_instance, limit: int = 20) -> List[Dict]:
    try:
        signals = db_instance.get_signals(limit=limit)
        return [signal.to_dict() for signal in signals] if signals else []
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return []

def get_ticker_snapshot() -> List[Dict]:
    """Fetch all USDT tickers with last price and 24h change from Bybit"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("retCode") == 0:
            tickers = data["result"].get("list", [])
            # Filter for USDT pairs only
            usdt_tickers = [ticker for ticker in tickers if ticker.get("symbol", "").endswith("USDT")]
            return usdt_tickers
    except Exception as e:
        logger.error(f"Error fetching ticker snapshot: {e}")
    return []

def calculate_portfolio_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    try:
        if not trades:
            return {"total_pnl": 0, "win_rate": 0, "total_trades": 0, "avg_pnl": 0}
        
        pnls = [float(trade.get("pnl", 0)) for trade in trades if trade.get("pnl") is not None]
        
        if not pnls:
            return {"total_pnl": 0, "win_rate": 0, "total_trades": len(trades), "avg_pnl": 0}
        
        total_pnl = sum(pnls)
        profitable_trades = len([pnl for pnl in pnls if pnl > 0])
        win_rate = (profitable_trades / len(pnls)) * 100 if pnls else 0
        avg_pnl = total_pnl / len(pnls) if pnls else 0
        
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "profitable_trades": profitable_trades,
            "avg_pnl": avg_pnl,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {"total_pnl": 0, "win_rate": 0, "total_trades": 0, "avg_pnl": 0}

def sync_real_wallet_balance(client, update_file: bool = True) -> Dict[str, Any]:
    """Sync real wallet balance with Bybit and optionally update capital.json"""
    try:
        if not client or not client.is_connected():
            logger.warning("Bybit client not connected")
            return {}
        
        balances = client.get_account_balance()
        usdt_balance = balances.get("USDT", {})
        
        balance_data = {
            "capital": usdt_balance.get("total", 0),
            "available": usdt_balance.get("available", 0),
            "used": usdt_balance.get("used", 0),
            "currency": "USDT"
        }
        
        if update_file:
            try:
                import json
                with open("capital.json", "r") as f:
                    capital = json.load(f)
                
                # Preserve start_balance if it exists
                balance_data["start_balance"] = capital.get("real", {}).get("start_balance", balance_data["capital"])
                capital["real"] = balance_data
                
                with open("capital.json", "w") as f:
                    json.dump(capital, f, indent=2)
                
                logger.info("Real balance synced with Bybit")
            except Exception as e:
                logger.error(f"Error updating capital.json: {e}")
        
        return balance_data
        
    except Exception as e:
        logger.error(f"Error syncing real wallet balance: {e}")
        return {}

def format_percentage(value: float) -> str:
    """Format percentage with appropriate color coding"""
    try:
        if value > 0:
            return f"+{value:.1f}%"
        else:
            return f"{value:.1f}%"
    except:
        return "0.0%"

def get_market_overview_data(symbols: Optional[List[str]] = None) -> List[Dict]:
    """Get market overview data for specified symbols"""
    try:
        if not symbols:
            symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        
        tickers = get_ticker_snapshot()
        overview_data = []
        
        for ticker in tickers:
            symbol = ticker.get("symbol", "")
            if symbol in symbols:
                price_change = float(ticker.get("price24hPcnt", 0)) * 100
                overview_data.append({
                    "symbol": symbol,
                    "price": float(ticker.get("lastPrice", 0)),
                    "change_24h": price_change,
                    "volume": float(ticker.get("volume24h", 0)),
                    "high_24h": float(ticker.get("highPrice24h", 0)),
                    "low_24h": float(ticker.get("lowPrice24h", 0))
                })
        
        return overview_data
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return []
