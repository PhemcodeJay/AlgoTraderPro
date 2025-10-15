import logging
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime, timezone
from indicators import scan_multiple_symbols, get_top_symbols, analyze_symbol
from utils import normalize_signal
from ml import MLFilter
from notifications import send_all_notifications
from db import Signal, db_manager

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# -------------------------------
# Core Signal Utilities
# -------------------------------

@st.cache_data
def get_usdt_symbols(limit: int = 50, _trading_mode: str = "virtual") -> List[str]:
    """
    Fetch top USDT symbols, using Bybit API for real mode if connected, otherwise fallback to defaults.
    """
    try:
        if _trading_mode == "real" and "bybit_client" in st.session_state and st.session_state.bybit_client and st.session_state.bybit_client.is_connected():
            try:
                symbols = get_top_symbols(limit)
                if not symbols:
                    logger.warning("No symbols from Bybit API, using fallback list")
                    symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
                logger.info(f"Fetched {len(symbols)} USDT symbols from Bybit API for real mode")
                return symbols[:limit]
            except Exception as e:
                logger.error(f"Error fetching USDT symbols from Bybit API: {e}", exc_info=True)
                return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        else:
            symbols = get_top_symbols(limit)
            if not symbols:
                logger.warning("No symbols from API, using fallback list")
                symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
            logger.info(f"Fetched {len(symbols)} USDT symbols for virtual mode")
            return symbols[:limit]
    except Exception as e:
        logger.error(f"Error in get_usdt_symbols: {e}", exc_info=True)
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
    """Calculate a score for a signal based on indicators."""
    score = analysis.get("score", 0)
    indicators = analysis.get("indicators", {})

    # RSI scoring (broadened to avoid overly restrictive filtering)
    rsi = indicators.get("rsi", 50)
    if rsi <= 30:  # Oversold
        score += 8
    elif rsi >= 70:  # Overbought
        score += 8
    elif 40 <= rsi <= 60:  # Neutral zone
        score += 4

    # Volume ratio scoring
    vol_ratio = indicators.get("volume_ratio", 1)
    if vol_ratio > 2:
        score += 15  # Increased weight for high volume
    elif vol_ratio > 1.2:
        score += 8

    # Volatility scoring
    vol = indicators.get("volatility", 0)
    if vol > 3:
        logger.info(f"High volatility ({vol}) detected for {analysis['symbol']}, boosting score")
        score += 12  # Increased reward for high volatility
    elif vol > 1:
        score += 6
    else:
        score += 3

    # Trend scoring
    trend_score = indicators.get("trend_score", 0)
    score += trend_score * 4  # Increased weight for trend alignment

    return min(100, max(0, score))

def is_market_bullish(interval: str = "60") -> bool:
    """Check if overall market (BTC) is bullish based on trend score."""
    try:
        btc_analysis = analyze_symbol("BTCUSDT", interval)
        if not btc_analysis:
            logger.warning("No BTC analysis available, assuming neutral market")
            return True  # Allow signals in neutral market
        btc_trend_score = btc_analysis["indicators"].get("trend_score", 0)
        return btc_trend_score >= 2  # Lowered threshold to allow more signals
    except Exception as e:
        logger.error(f"Error checking market trend: {e}", exc_info=True)
        return True  # Fallback to allow signals if BTC analysis fails

def enhance_signal(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance signal with additional parameters like SL/TP, leverage, etc."""
    indicators = analysis.get("indicators", {})
    price = indicators.get("price", 0)
    atr = indicators.get("atr", 0)
    side = analysis.get("side", "Buy").title()
    leverage = 15
    atr_multiplier = 3
    risk_reward = 3

    # Bollinger Bands slope
    bb_upper = indicators.get("bb_upper", 0)
    bb_lower = indicators.get("bb_lower", 0)
    bb_slope = "Expanding" if bb_upper - bb_lower > price * 0.02 else "Contracting"

    # Market type based on volatility
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
        "trail": round(atr, 6),
        "margin_usdt": 1.0,
        "bb_slope": bb_slope,
        "market": market_type,
        "leverage": leverage,
        "risk_reward": risk_reward,
        "atr_multiplier": atr_multiplier,
        "created_at": datetime.now(timezone.utc)
    })

    enhanced = normalize_signal(enhanced)
    return enhanced

# -------------------------------
# Signal Generation
# -------------------------------

@st.cache_data
def generate_signals(
    symbols: List[str],
    interval: str = "60",
    top_n: int = 10,
    _trading_mode: str = None
) -> List[Dict[str, Any]]:
    """
    Generate trading signals for given symbols, respecting trading mode from session state.
    """
    trading_mode = _trading_mode or st.session_state.get("trading_mode", "virtual")
    logger.info(f"Generating signals for {len(symbols)} symbols in {trading_mode} mode")
    
    # Check overall market trend (BTC) to align signals
    market_bullish = is_market_bullish(interval)
    logger.info(f"Market trend: {'Bullish' if market_bullish else 'Bearish or Neutral'}")
    
    raw_analyses = scan_multiple_symbols(symbols, interval, max_workers=5)
    if not raw_analyses:
        logger.warning("No analysis results from scan_multiple_symbols")
        return []

    scored_signals = []
    min_score = 30 if trading_mode == "real" else 40  # Lowered min_score for real mode
    for analysis in raw_analyses:
        score = calculate_signal_score(analysis)
        analysis["score"] = score
        if score < min_score:
            logger.info(f"Skipping {analysis['symbol']} due to low score: {score} < {min_score}")
            continue
        # Soften market trend filter to allow neutral market signals
        if analysis["signal_type"] == "buy" and not market_bullish and analysis["score"] < 60:
            logger.info(f"Skipping buy signal for {analysis['symbol']} due to non-bullish market and score {score}")
            continue
        elif analysis["signal_type"] == "sell" and market_bullish and analysis["score"] < 60:
            logger.info(f"Skipping sell signal for {analysis['symbol']} due to bullish market and score {score}")
            continue
        scored_signals.append(analysis)
        logger.info(f"Valid signal for {analysis['symbol']}: score={score}, type={analysis['signal_type']}")

    if not scored_signals:
        logger.warning("No signals passed scoring threshold")
        return []

    # ML filtering with fallback
    ml_filter = MLFilter()
    try:
        filtered_signals = ml_filter.filter_signals(scored_signals)
        logger.info(f"ML filter retained {len(filtered_signals)}/{len(scored_signals)} signals")
    except Exception as e:
        logger.error(f"ML filter failed: {e}", exc_info=True)
        filtered_signals = scored_signals  # Fallback to unfiltered signals
        logger.info("Bypassing ML filter due to error")

    # Sort by score and take top N
    filtered_signals.sort(key=lambda x: x["score"], reverse=True)
    top_signals = filtered_signals[:top_n]
    logger.info(f"Selected top {len(top_signals)} signals")

    # Enhance and store signals
    final_signals = []
    for analysis in top_signals:
        enhanced = enhance_signal(analysis)

        # Save to DB
        signal_obj = Signal(
            symbol=str(enhanced.get("symbol") or "UNKNOWN"),
            interval=interval,
            signal_type=str(enhanced.get("signal_type", "BUY")),
            score=float(enhanced.get("score", 0)),
            indicators=enhanced.get("indicators", {}),
            side=str(enhanced.get("side", "BUY")),
            sl=float(enhanced.get("sl") or 0),
            tp=float(enhanced.get("tp") or 0),
            trail=float(enhanced.get("trail") or 0),
            liquidation=float(enhanced.get("liquidation") or 0),
            leverage=int(enhanced.get("leverage", 15)),
            margin_usdt=float(enhanced.get("margin_usdt") or 0),
            entry=float(enhanced.get("entry") or 0),
            market=str(enhanced.get("market", "Unknown")),
            created_at=enhanced.get("created_at") or datetime.now(timezone.utc)
        )

        try:
            db_manager.add_signal(signal_obj)
            logger.info(f"Saved signal for {signal_obj.symbol} to database (score={signal_obj.score})")
        except Exception as e:
            logger.error(f"Failed to save signal {enhanced.get('symbol')} to DB: {e}", exc_info=True)

        final_signals.append(enhanced)

    if final_signals and trading_mode == "real":
        try:
            send_all_notifications(final_signals)
            logger.info(f"Sent notifications for {len(final_signals)} real mode signals")
        except Exception as e:
            logger.error(f"Failed to send notifications for real mode signals: {e}", exc_info=True)

    logger.info(f"Generated {len(final_signals)} final signals")
    return final_signals

# -------------------------------
# Signal Summary
# -------------------------------

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize signal statistics."""
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
    trading_mode = st.session_state.get("trading_mode", "virtual")
    logger.info(f"Analyzing single symbol {symbol} in {trading_mode} mode")

    # Check market trend for alignment
    market_bullish = is_market_bullish(interval)

    raw_analyses = scan_multiple_symbols([symbol], interval, max_workers=1)
    if not raw_analyses:
        logger.warning(f"No analysis found for {symbol}")
        return {}

    analysis = raw_analyses[0]
    analysis["score"] = calculate_signal_score(analysis)
    min_score = 30 if trading_mode == "real" else 40
    if analysis["score"] < min_score:
        logger.info(f"Signal score for {symbol} too low: {analysis['score']} < {min_score}")
        return {}
    if analysis["signal_type"] == "buy" and not market_bullish and analysis["score"] < 60:
        logger.info(f"Skipping buy signal for {symbol} due to non-bullish market and score {analysis['score']}")
        return {}
    elif analysis["signal_type"] == "sell" and market_bullish and analysis["score"] < 60:
        logger.info(f"Skipping sell signal for {symbol} due to bullish market and score {analysis['score']}")
        return {}

    enhanced = enhance_signal(analysis)

    # Save to DB
    signal_obj = Signal(
        symbol=str(enhanced.get("symbol") or "UNKNOWN"),
        interval=interval,
        signal_type=str(enhanced.get("signal_type", "BUY")),
        score=float(enhanced.get("score", 0)),
        indicators=enhanced.get("indicators", {}),
        side=str(enhanced.get("side", "BUY")),
        sl=float(enhanced.get("sl") or 0),
        tp=float(enhanced.get("tp") or 0),
        trail=float(enhanced.get("trail") or 0),
        liquidation=float(enhanced.get("liquidation") or 0),
        leverage=int(enhanced.get("leverage", 15)),
        margin_usdt=float(enhanced.get("margin_usdt") or 0),
        entry=float(enhanced.get("entry") or 0),
        market=str(enhanced.get("market", "Unknown")),
        created_at=enhanced.get("created_at") or datetime.now(timezone.utc)
    )

    try:
        db_manager.add_signal(signal_obj)
        logger.info(f"Saved signal for {symbol} to database (score={signal_obj.score})")
    except Exception as e:
        logger.error(f"Failed to save signal {enhanced.get('symbol')} to DB: {e}", exc_info=True)

    if trading_mode == "real":
        try:
            send_all_notifications([enhanced])
            logger.info(f"Sent notification for real mode signal {symbol}")
        except Exception as e:
            logger.error(f"Failed to send notification for {symbol}: {e}", exc_info=True)

    return enhanced

# -------------------------------
# Run Standalone
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize session state for standalone testing
    if "trading_mode" not in st.session_state:
        st.session_state.trading_mode = "virtual"
    
    symbols = get_usdt_symbols(limit=20, _trading_mode=st.session_state.trading_mode)
    signals = generate_signals(symbols, interval="60", top_n=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")

    if signals and st.session_state.trading_mode == "real":
        send_all_notifications(signals)