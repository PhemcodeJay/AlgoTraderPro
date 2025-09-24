import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import pytz
import indicators  # Use standalone functions from indicators.py
from ml import MLFilter
from notifications import send_all_notifications
from bybit_client import BybitClient
from db import Signal, db_manager
from utils import normalize_signal

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# -------------------------------
# Core Signal Utilities
# -------------------------------

def get_usdt_symbols(limit: int = 50) -> List[str]:
    """Fetch tradable USDT perpetual futures symbols from Bybit API."""
    try:
        client = BybitClient()
        if not client.is_connected():
            logger.error("BybitClient not connected, using fallback symbols")
            return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"][:limit]
        
        # Use requests since get_available_symbols is not defined
        data = requests.get("https://api.bybit.com/v5/market/tickers?category=linear").json()
        tickers = [i for i in data['result']['list'] if i['symbol'].endswith("USDT")]
        tickers.sort(key=lambda x: float(x['turnover24h']), reverse=True)
        symbols = [t['symbol'] for t in tickers[:limit]]
        logger.info(f"Fetched {len(symbols)} tradable USDT perpetual futures symbols from Bybit: {symbols[:5]}...")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"][:limit]

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
    """Calculate a score for a signal based on technical indicators."""
    score = analysis.get("score", 0)
    indicators = analysis.get("indicators", {})

    rsi = indicators.get("rsi", 50)
    if 20 <= rsi <= 30 or 70 <= rsi <= 80:
        score += 10
    elif rsi < 20 or rsi > 80:
        score += 5

    macd_hist = indicators.get("macd_histogram", 0)
    if abs(macd_hist) > 0.01:
        score += 8

    vol_ratio = indicators.get("volume_ratio", 1)
    if vol_ratio > 2:
        score += 12
    elif vol_ratio > 1.5:
        score += 6

    vol = indicators.get("volatility", 0)
    if 0.5 <= vol <= 3:
        score += 5
    elif vol > 5:
        score -= 10

    trend_score = indicators.get("trend_score", 0)
    score += trend_score * 3

    return min(100, max(0, score))

def classify_trend(ema9: float, ema21: float, sma20: float) -> str:
    """Classify trend based on EMA9, EMA21, and SMA20 alignment."""
    if ema9 is None or ema21 is None or sma20 is None:
        return "Scalp"
    if ema9 > ema21 > sma20:
        return "Trend"
    if ema9 > ema21:
        return "Swing"
    return "Scalp"

def enhance_signal(analysis: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance signal with additional parameters like SL, TP, and market type."""
    indicators = analysis.get("indicators", {})
    price = indicators.get("price", 0)
    atr = indicators.get("atr", 0)
    side = analysis.get("side", "BUY").upper()
    leverage = settings.get("LEVERAGE", 10)
    risk_pct = settings.get("RISK_PCT", 0.01)
    virtual_balance = settings.get("VIRTUAL_BALANCE", 10000)

    # Use previous standalone logic for TP/SL and trailing stop
    entry = price
    if side == "LONG":
        tp = round(entry * 1.015, 6)
        sl = round(entry * 0.985, 6)
        trail = round(entry * (1 - settings.get("ENTRY_BUFFER_PCT", 0.002)), 6)
        liq = round(entry * (1 - 1/leverage), 6)
    else:
        tp = round(entry * 0.985, 6)
        sl = round(entry * 1.015, 6)
        trail = round(entry * (1 + settings.get("ENTRY_BUFFER_PCT", 0.002)), 6)
        liq = round(entry * (1 + 1/leverage), 6)

    try:
        sl_diff = abs(entry - sl)
        margin = round((virtual_balance * risk_pct / sl_diff) * entry / leverage, 6)
    except ZeroDivisionError:
        margin = 5.0

    bb_upper = indicators.get("bb_upper", 0)
    bb_lower = indicators.get("bb_lower", 0)
    bb_slope = "Expanding" if bb_upper - bb_lower > price * 0.02 else "Contracting"

    vol = indicators.get("volatility", 0)
    if vol < 1:
        market_type = "Low Vol"
    elif vol < 3:
        market_type = "Normal"
    else:
        market_type = "High Vol"

    enhanced = analysis.copy()
    enhanced.update({
        "Symbol": enhanced.get("symbol", "UNKNOWN"),  # For notifications.py
        "Type": classify_trend(
            indicators.get("ema9"),
            indicators.get("ema21"),
            indicators.get("sma20")
        ),  # For notifications.py
        "Side": side,  # For notifications.py
        "Score": round(float(enhanced.get("score", 0)), 1),  # For notifications.py
        "Entry": round(float(entry), 6),  # For notifications.py
        "TP": round(float(tp), 6),  # For notifications.py
        "SL": round(float(sl), 6),  # For notifications.py
        "Trail": round(float(trail), 6),  # For notifications.py
        "Margin": round(float(margin), 6),  # For notifications.py
        "Market": market_type,  # For notifications.py
        "Liquidation": round(float(liq), 6),  # For notifications.py
        "entry": round(float(entry), 6),
        "sl": round(float(sl), 6),
        "tp": round(float(tp), 6),
        "trail": round(float(trail), 6),
        "liquidation": round(float(liq), 6),
        "margin_usdt": round(float(margin), 6),
        "bb_slope": bb_slope,
        "market": market_type,
        "leverage": leverage,
        "created_at": datetime.now(timezone.utc)
    })
    return enhanced

# -------------------------------
# Signal Generation
# -------------------------------

def generate_signals(
    symbols: List[str],
    interval: str = "60",
    top_n: int = 10,
    trading_mode: str = "virtual",
    settings: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generate trading signals for given symbols with multi-timeframe analysis."""
    if settings is None:
        settings = {
            "INTERVALS": ["15", "60", "240"],
            "MIN_VOLUME": 1000000,
            "MIN_ATR_PCT": 0.005,
            "RSI_OVERSOLD": 30,
            "RSI_OVERBOUGHT": 70,
            "LEVERAGE": 10,
            "RISK_PCT": 0.01,
            "VIRTUAL_BALANCE": 10000,
            "ENTRY_BUFFER_PCT": 0.002,
            "TOP_N_SIGNALS": top_n,
            "ML_THRESHOLD": 0.4
        }
    
    logger.info(f"Generating signals for {len(symbols)} symbols in {trading_mode} mode: {symbols[:5]}...")
    
    # Perform multi-timeframe analysis
    client = BybitClient()
    raw_analyses = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_symbol = {
            executor.submit(indicators.scan_multiple_symbols, [symbol], interval): symbol
            for symbol in symbols
        }
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                analysis = future.result()
                if analysis:
                    # Add multi-timeframe indicators
                    data = {}
                    for tf in settings["INTERVALS"]:
                        try:
                            candles = client.get_klines(symbol=symbol, interval=tf, limit=200)
                        except Exception as e:
                            logger.warning(f"BybitClient.get_klines failed for {symbol} on {tf}m: {e}, using requests")
                            url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={tf}&limit=200"
                            data_response = requests.get(url).json()
                            candles = data_response.get('result', {}).get('list', [])
                        
                        if len(candles) < 30:
                            logger.warning(f"Insufficient candles for {symbol} on {tf}m")
                            continue
                        
                        closes = [float(c["close"]) for c in candles]
                        highs = [float(c["high"]) for c in candles]
                        lows = [float(c["low"]) for c in candles]
                        volumes = [float(c["volume"]) for c in candles]
                        
                        data[tf] = {
                            "close": closes[-1],
                            "ema9": indicators.ema(closes, 9)[-1] if indicators.ema(closes, 9) else None,
                            "ema21": indicators.ema(closes, 21)[-1] if indicators.ema(closes, 21) else None,
                            "sma20": indicators.sma(closes, 20)[-1] if indicators.sma(closes, 20) else None,
                            "volume": volumes[-1]
                        }
                    
                    if data.get("60"):
                        analysis[0]["indicators"].update({
                            "ema9": data["60"].get("ema9"),
                            "ema21": data["60"].get("ema21"),
                            "sma20": data["60"].get("sma20")
                        })
                        # Multi-timeframe filtering
                        tf60 = data["60"]
                        if (tf60["volume"] < settings["MIN_VOLUME"] or 
                            (analysis[0]["indicators"].get("volatility", 0) < settings["MIN_ATR_PCT"]) or
                            not (settings["RSI_OVERSOLD"] < analysis[0]["indicators"].get("rsi", 50) < settings["RSI_OVERBOUGHT"])):
                            continue
                        
                        # Check timeframe alignment
                        sides = []
                        for tf in settings["INTERVALS"]:
                            d = data.get(tf, {})
                            if not d:
                                continue
                            if d["close"] > (analysis[0]["indicators"].get("bb_upper", float('inf'))):
                                sides.append("LONG")
                            elif d["close"] < (analysis[0]["indicators"].get("bb_lower", 0)):
                                sides.append("SHORT")
                            elif d["close"] > (d["ema21"] or float('inf')):
                                sides.append("LONG")
                            elif d["close"] < (d["ema21"] or 0):
                                sides.append("SHORT")
                            else:
                                sides.append("NEUTRAL")
                        
                        if len(set(s for s in sides if s != "NEUTRAL")) == 1 and not all(s == "NEUTRAL" for s in sides):
                            analysis[0]["side"] = sides[0] if sides[0] != "NEUTRAL" else next(s for s in sides if s != "NEUTRAL")
                            raw_analyses.append(analysis[0])
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

    if not raw_analyses:
        logger.warning("No analysis results")
        return []

    # Score and filter signals
    scored_signals = []
    for analysis in raw_analyses:
        score = calculate_signal_score(analysis)
        analysis["score"] = score
        min_score = 50 if trading_mode == "real" else 40
        if score >= min_score:
            scored_signals.append(analysis)

    if not scored_signals:
        logger.info("No significant signals found after scoring")
        return []

    # ML filtering
    ml_filter = MLFilter()
    try:
        filtered_signals = ml_filter.filter_signals(scored_signals, threshold=settings.get("ML_THRESHOLD", 0.4))
    except Exception as e:
        logger.warning(f"ML filter failed: {e}")
        filtered_signals = scored_signals

    # Enhance and store signals
    final_signals = []
    for s in filtered_signals[:top_n]:
        if isinstance(s, Signal):
            s_dict = normalize_signal(s)
            enhanced = enhance_signal(s_dict, settings)
        else:
            enhanced = enhance_signal(s, settings)
        
        final_signals.append(enhanced)

        # Save to DB
        signal_obj = Signal(
            symbol=str(enhanced.get("symbol", "UNKNOWN")),
            interval=str(interval),
            signal_type=str(enhanced.get("signal_type", "BUY")),
            score=float(enhanced.get("score", 0)),
            indicators=dict(enhanced.get("indicators", {})),
            strategy="Auto",
            side=str(enhanced.get("side", "BUY")),
            sl=float(enhanced.get("sl", 0)),
            tp=float(enhanced.get("tp", 0)),
            trail=float(enhanced.get("trail", 0)),
            liquidation=float(enhanced.get("liquidation", 0)),
            leverage=int(enhanced.get("leverage", 10)),
            margin_usdt=float(enhanced.get("margin_usdt", 0)),
            entry=float(enhanced.get("entry", 0)),
            market=str(enhanced.get("market", "futures")),
            created_at=enhanced.get("created_at", datetime.now(timezone.utc))
        )

        try:
            db_manager.add_signal(signal_obj)
        except Exception as e:
            logger.error(f"Failed to save signal {enhanced.get('symbol')} to DB: {e}")

    if final_signals and settings.get("NOTIFICATION_ENABLED", True):
        try:
            send_all_notifications(final_signals)
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    return final_signals

# -------------------------------
# Signal Summary
# -------------------------------

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of trading signals."""
    if not signals:
        return {"total": 0, "avg_score": 0, "top_symbol": "None"}

    total_signals = len(signals)
    avg_score = sum(float(s.get("score", 0)) for s in signals) / total_signals
    top_signal = max(signals, key=lambda x: float(x.get("score", 0)))
    top_symbol = top_signal.get("symbol", "Unknown")

    buy_signals = sum(1 for s in signals if s.get("side", "").upper() in ["BUY", "LONG"])
    sell_signals = total_signals - buy_signals

    market_types = {}
    for s in signals:
        market_types[s.get("market", "Unknown")] = market_types.get(s.get("market", "Unknown"), 0) + 1

    return {
        "total": total_signals,
        "avg_score": round(avg_score, 1),
        "top_symbol": top_symbol,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "market_distribution": market_types
    }

def analyze_single_symbol(symbol: str, interval: str = "60", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze a single symbol and return the enhanced signal dictionary."""
    if settings is None:
        settings = {
            "INTERVALS": ["15", "60", "240"],
            "MIN_VOLUME": 1000000,
            "MIN_ATR_PCT": 0.005,
            "RSI_OVERSOLD": 30,
            "RSI_OVERBOUGHT": 70,
            "LEVERAGE": 10,
            "RISK_PCT": 0.01,
            "VIRTUAL_BALANCE": 10000,
            "ENTRY_BUFFER_PCT": 0.002
        }
    
    raw_analyses = indicators.scan_multiple_symbols([symbol], interval, max_workers=1)
    if not raw_analyses:
        logger.warning(f"No analysis found for {symbol}")
        return {}

    analysis = raw_analyses[0]
    analysis["score"] = calculate_signal_score(analysis)
    
    # Add multi-timeframe indicators
    client = BybitClient()
    data = {}
    for tf in settings["INTERVALS"]:
        try:
            candles = client.get_klines(symbol=symbol, interval=tf, limit=200)
        except Exception as e:
            logger.warning(f"BybitClient.get_klines failed for {symbol} on {tf}m: {e}, using requests")
            url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={tf}&limit=200"
            data_response = requests.get(url).json()
            candles = data_response.get('result', {}).get('list', [])
        
        if len(candles) < 30:
            logger.warning(f"Insufficient candles for {symbol} on {tf}m")
            continue
        
        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        volumes = [float(c["volume"]) for c in candles]
        
        data[tf] = {
            "close": closes[-1],
            "ema9": indicators.ema(closes, 9)[-1] if indicators.ema(closes, 9) else None,
            "ema21": indicators.ema(closes, 21)[-1] if indicators.ema(closes, 21) else None,
            "sma20": indicators.sma(closes, 20)[-1] if indicators.sma(closes, 20) else None,
            "volume": volumes[-1]
        }
    
    if data.get("60"):
        analysis["indicators"].update({
            "ema9": data["60"].get("ema9"),
            "ema21": data["60"].get("ema21"),
            "sma20": data["60"].get("sma20")
        })
        # Multi-timeframe filtering
        tf60 = data["60"]
        if (tf60["volume"] < settings["MIN_VOLUME"] or 
            (analysis["indicators"].get("volatility", 0) < settings["MIN_ATR_PCT"]) or
            not (settings["RSI_OVERSOLD"] < analysis["indicators"].get("rsi", 50) < settings["RSI_OVERBOUGHT"])):
            return {}
        
        # Check timeframe alignment
        sides = []
        for tf in settings["INTERVALS"]:
            d = data.get(tf, {})
            if not d:
                continue
            if d["close"] > (analysis["indicators"].get("bb_upper", float('inf'))):
                sides.append("LONG")
            elif d["close"] < (analysis["indicators"].get("bb_lower", 0)):
                sides.append("SHORT")
            elif d["close"] > (d["ema21"] or float('inf')):
                sides.append("LONG")
            elif d["close"] < (d["ema21"] or 0):
                sides.append("SHORT")
            else:
                sides.append("NEUTRAL")
        
        if len(set(s for s in sides if s != "NEUTRAL")) != 1 or all(s == "NEUTRAL" for s in sides):
            return {}
        
        analysis["side"] = sides[0] if sides[0] != "NEUTRAL" else next(s for s in sides if s != "NEUTRAL")

    enhanced = enhance_signal(analysis, settings)

    # Save to DB
    signal_obj = Signal(
        symbol=str(enhanced.get("symbol", "UNKNOWN")),
        interval=str(interval),
        signal_type=str(enhanced.get("signal_type", "BUY")),
        score=float(enhanced.get("score", 0)),
        indicators=dict(enhanced.get("indicators", {})),
        strategy="Auto",
        side=str(enhanced.get("side", "BUY")),
        sl=float(enhanced.get("sl", 0)),
        tp=float(enhanced.get("tp", 0)),
        trail=float(enhanced.get("trail", 0)),
        liquidation=float(enhanced.get("liquidation", 0)),
        leverage=int(enhanced.get("leverage", 10)),
        margin_usdt=float(enhanced.get("margin_usdt", 0)),
        entry=float(enhanced.get("entry", 0)),
        market=str(enhanced.get("market", "futures")),
        created_at=enhanced.get("created_at", datetime.now(timezone.utc))
    )

    try:
        db_manager.add_signal(signal_obj)
    except Exception as e:
        logger.error(f"Failed to save signal {enhanced.get('symbol')} to DB: {e}")

    return enhanced

# -------------------------------
# Run Standalone
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    symbols = get_usdt_symbols(limit=20)
    signals = generate_signals(symbols, interval="60", top_n=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")