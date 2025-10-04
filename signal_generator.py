import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from indicators import scan_multiple_symbols, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications

from db import Signal, db_manager

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# -------------------------------
# Core Signal Utilities
# -------------------------------

def get_usdt_symbols(limit: int = 50) -> List[str]:
    try:
        symbols = get_top_symbols(limit)
        if not symbols:
            logger.warning("No symbols from API, using fallback list")
            symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        return symbols[:limit]
    except Exception as e:
        logger.error(f"Error fetching USDT symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
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

def enhance_signal(analysis: Dict[str, Any]) -> Any:
    indicators = analysis.get("indicators", {})
    price = indicators.get("price", 0)
    atr = indicators.get("atr", 0)
    side = analysis.get("side", "BUY")
    leverage = 10
    atr_multiplier = 2
    risk_reward = 2

    # Trade parameters
    sl_percent = 10  # Stop Loss: 10% below entry for buy, above entry for sell
    tp_percent = 50   # Take Profit: 50% above entry for buy, below entry for sell

    if side.lower() == "buy":
        sl = price * (1 - sl_percent / 10)   # Stop Loss 5% below entry
        tp = price * (1 + tp_percent / 10)   # Take Profit 25% above entry
        liq = price * (1 - 0.9 / leverage)    # Liquidation formula remains the same
    else:
        sl = price * (1 + sl_percent / 10)   # Stop Loss 5% above entry
        tp = price * (1 - tp_percent / 10)   # Take Profit 25% below entry
        liq = price * (1 + 0.9 / leverage)    # Liquidation formula for short

    print(f"SL: {sl}, TP: {tp}, LIQ: {liq}")


    trail = atr
    margin_usdt = 2.0

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
        "entry": round(price, 6),
        "sl": round(sl, 6),
        "tp": round(tp, 6),
        "trail": round(trail, 6),
        "liquidation": round(liq, 6),
        "margin_usdt": round(margin_usdt, 6),
        "bb_slope": bb_slope,
        "market": market_type,
        "leverage": leverage,
        "risk_reward": risk_reward,
        "atr_multiplier": atr_multiplier,
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
    trading_mode: str = "virtual"
) -> List[Dict[str, Any]]:
    logger.info(f"Generating signals for {len(symbols)} symbols in {trading_mode} mode")
    raw_analyses = scan_multiple_symbols(symbols, interval, max_workers=5)
    if not raw_analyses:
        logger.warning("No analysis results")
        return []

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
        filtered_signals = ml_filter.filter_signals(scored_signals)
    except Exception as e:
        logger.warning(f"ML filter failed: {e}")
        filtered_signals = scored_signals

    # Enhance and store signals
    final_signals = []
    for s in filtered_signals[:top_n]:
        if isinstance(s, Signal):
            s_dict = {
                "symbol": s.symbol,
                "interval": s.interval,
                "signal_type": s.signal_type,
                "score": s.score,
                "indicators": s.indicators,
                "side": s.side,
                # … add any other fields you need
            }
            enhanced = enhance_signal(s_dict)
        else:
            enhanced = enhance_signal(s)

        final_signals.append(enhanced)

        # Save to DB
        signal_obj = Signal(
            symbol=str(enhanced.get("symbol") or "UNKNOWN"),   # ensure str, fallback
            interval=interval,
            signal_type=str(enhanced.get("signal_type", "BUY")),
            score=float(enhanced.get("score", 0)),
            indicators=enhanced.get("indicators", {}),
            side=str(enhanced.get("side", "BUY")),
            sl=float(enhanced.get("sl") or 0),
            tp=float(enhanced.get("tp") or 0),
            trail=float(enhanced.get("trail") or 0),
            liquidation=float(enhanced.get("liquidation") or 0),
            leverage=int(enhanced.get("leverage", 10)),
            margin_usdt=float(enhanced.get("margin_usdt") or 0),
            entry=float(enhanced.get("entry") or 0),
            market=str(enhanced.get("market", "Unknown")),
            created_at=enhanced.get("created_at") or datetime.now(timezone.utc)  # always datetime
        )

        try:
            db_manager.add_signal(signal_obj)
        except Exception as e:
            logger.error(f"Failed to save signal {enhanced.get('symbol')} to DB: {e}")

    return final_signals

# -------------------------------
# Signal Summary
# -------------------------------

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
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

def analyze_single_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    """
    Analyze a single symbol and return the enhanced signal dictionary.
    """
    raw_analyses = scan_multiple_symbols([symbol], interval, max_workers=1)
    if not raw_analyses:
        logger.warning(f"No analysis found for {symbol}")
        return {}

    analysis = raw_analyses[0]
    analysis["score"] = calculate_signal_score(analysis)
    enhanced = enhance_signal(analysis)

    # Save to DB
    signal_obj = Signal(
    symbol=str(enhanced.get("symbol") or "UNKNOWN"),   # ensure str, fallback
    interval=interval,
    signal_type=str(enhanced.get("signal_type", "BUY")),
    score=float(enhanced.get("score", 0)),
    indicators=enhanced.get("indicators", {}),
    side=str(enhanced.get("side", "BUY")),
    sl=float(enhanced.get("sl") or 0),
    tp=float(enhanced.get("tp") or 0),
    trail=float(enhanced.get("trail") or 0),
    liquidation=float(enhanced.get("liquidation") or 0),
    leverage=int(enhanced.get("leverage", 10)),
    margin_usdt=float(enhanced.get("margin_usdt") or 0),
    entry=float(enhanced.get("entry") or 0),
    market=str(enhanced.get("market", "Unknown")),
    created_at=enhanced.get("created_at") or datetime.now(timezone.utc)  # always datetime
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

    if signals:
        send_all_notifications(signals)
