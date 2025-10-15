from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import requests

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# Constants
INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W"]
ML_ENABLED = True  # Feature flag for ML filtering

def get_candles(symbol: str, interval: str, limit: int = 200) -> List[Dict]:
    """Fetch candlestick data from Bybit API"""
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }
        
        # Pylance: ignore reportCallIssue for timeout parameter as it is valid in requests library
        response = requests.get(url, params=params, timeout=10)  # type: ignore[call-arg]
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") == 0 and "result" in data:
            klines = []
            for k in data["result"]["list"]:
                klines.append({
                    "time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
            return sorted(klines, key=lambda x: x["time"])
        return []
    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}")
        return []

def sma(prices: List[float], period: int) -> List[float]:
    """Simple Moving Average"""
    if len(prices) < period:
        return [0] * len(prices)
    
    sma_values = []
    for i in range(len(prices)):
        if i < period - 1:
            sma_values.append(0)
        else:
            avg = sum(prices[i-period+1:i+1]) / period
            sma_values.append(avg)
    return sma_values

def ema(prices: List[float], period: int) -> List[float]:
    """Exponential Moving Average"""
    if len(prices) < period:
        return [0] * len(prices)
    
    multiplier = 2 / (period + 1)
    ema_values = [prices[0]]  # Start with first price
    
    for i in range(1, len(prices)):
        ema_val = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
        ema_values.append(ema_val)
    
    return ema_values

def rsi(prices: List[float], period: int = 14) -> List[float]:
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return [50.0] * len(prices)
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    
    rsi_values = [50.0]  # Start with neutral RSI
    
    for i in range(period, len(gains)):
        avg_gain = sum(gains[i-period:i]) / period
        avg_loss = sum(losses[i-period:i]) / period
        
        if avg_loss == 0:
            rsi_val = 100
        else:
            rs = avg_gain / avg_loss
            rsi_val = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi_val)
    
    while len(rsi_values) < len(prices):
        rsi_values.insert(0, 50.0)
    
    return rsi_values

def stochastic_rsi(prices: List[float], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, List[float]]:
    """Stochastic RSI"""
    if len(prices) < period + 1:
        zero_list = [50.0] * len(prices)
        return {"stoch_rsi": zero_list, "k": zero_list, "d": zero_list}
    
    rsi_values = rsi(prices, period)
    stoch_rsi = []
    
    for i in range(period - 1, len(rsi_values)):
        rsi_slice = rsi_values[i-period+1:i+1]
        if not rsi_slice:
            stoch_rsi.append(50.0)
            continue
        lowest_rsi = min(rsi_slice)
        highest_rsi = max(rsi_slice)
        if highest_rsi == lowest_rsi:
            stoch_rsi.append(50.0)
        else:
            stoch_val = 100 * (rsi_values[i] - lowest_rsi) / (highest_rsi - lowest_rsi)
            stoch_rsi.append(stoch_val)
    
    while len(stoch_rsi) < len(prices):
        stoch_rsi.insert(0, 50.0)
    
    k_values = sma(stoch_rsi, smooth_k)
    d_values = sma(k_values, smooth_d)
    
    return {
        "stoch_rsi": stoch_rsi,
        "k": k_values,
        "d": d_values
    }

def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
    """Bollinger Bands"""
    if len(prices) < period:
        zero_list = [0.0] * len(prices)
        return {"upper": zero_list, "middle": zero_list, "lower": zero_list}
    
    sma_values = sma(prices, period)
    
    upper_band = []
    lower_band = []
    
    for i in range(len(prices)):
        if i < period - 1:
            upper_band.append(0.0)
            lower_band.append(0.0)
        else:
            price_slice = prices[i-period+1:i+1]
            std = np.std(price_slice)
            upper_band.append(sma_values[i] + (std * std_dev))
            lower_band.append(sma_values[i] - (std * std_dev))
    
    return {
        "upper": upper_band,
        "middle": sma_values,
        "lower": lower_band
    }

def calculate_indicators(candles: List[Dict]) -> Dict[str, Any]:
    """Calculate technical indicators (SMA 20, SMA 200, EMA 9, EMA 21, Bollinger Bands, RSI, Stochastic RSI, Volume)"""
    try:
        if not candles or len(candles) < 200:
            return {}
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        # Moving averages
        sma_20 = sma(closes, 20)
        sma_200 = sma(closes, 200)
        ema_9 = ema(closes, 9)
        ema_21 = ema(closes, 21)
        
        # Momentum indicators
        rsi_14 = rsi(closes, 14)
        stoch_rsi_data = stochastic_rsi(closes, period=14, smooth_k=3, smooth_d=3)
        
        # Volatility indicators
        bb_data = bollinger_bands(closes, period=20, std_dev=2)
        
        # Current values (last in arrays)
        current_price = closes[-1]
        current_sma_20 = sma_20[-1] if sma_20 else current_price
        current_sma_200 = sma_200[-1] if sma_200 else current_price
        current_ema_9 = ema_9[-1] if ema_9 else current_price
        current_ema_21 = ema_21[-1] if ema_21 else current_price
        current_rsi = rsi_14[-1] if rsi_14 else 50
        current_stoch_k = stoch_rsi_data["k"][-1] if stoch_rsi_data["k"] else 50
        current_stoch_d = stoch_rsi_data["d"][-1] if stoch_rsi_data["d"] else 50
        current_bb_upper = bb_data["upper"][-1] if bb_data["upper"] else 0
        current_bb_lower = bb_data["lower"][-1] if bb_data["lower"] else 0
        current_volume = volumes[-1] if volumes else 0
        
        # Volume analysis
        avg_volume = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Trend analysis
        trend_score = 0
        if len(sma_20) > 1 and len(sma_200) > 1:
            if sma_20[-1] > sma_200[-1]:
                trend_score += 1
            if closes[-1] > sma_20[-1]:
                trend_score += 1
            if sma_20[-1] > sma_20[-2]:
                trend_score += 1
            if ema_9[-1] > ema_21[-1]:
                trend_score += 1
        
        return {
            "price": current_price,
            "sma_20": current_sma_20,
            "sma_200": current_sma_200,
            "ema_9": current_ema_9,
            "ema_21": current_ema_21,
            "rsi": current_rsi,
            "stoch_k": current_stoch_k,
            "stoch_d": current_stoch_d,
            "bb_upper": current_bb_upper,
            "bb_lower": current_bb_lower,
            "bb_middle": bb_data["middle"][-1] if bb_data["middle"] else current_price,
            "volume": current_volume,
            "volume_ratio": volume_ratio,
            "trend_score": trend_score,
            "volatility": ((current_bb_upper - current_bb_lower) / current_price * 100) if current_price > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def get_top_symbols(limit: int = 100) -> List[str]:
    """Get top USDT trading pairs by volume"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear"}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") == 0 and "result" in data:
            tickers = data["result"]["list"]
            
            usdt_pairs = []
            for ticker in tickers:
                symbol = ticker.get("symbol", "")
                if symbol.endswith("USDT"):
                    volume = float(ticker.get("volume24h", 0))
                    price = float(ticker.get("lastPrice", 0))
                    if volume > 0 and price > 0:
                        usdt_pairs.append({
                            "symbol": symbol,
                            "volume": volume,
                            "price": price
                        })
            
            usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
            return [pair["symbol"] for pair in usdt_pairs[:limit]]
        
        return []
    except Exception as e:
        logger.error(f"Error getting top symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def analyze_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    """Comprehensive analysis of a single symbol with improved profitability criteria"""
    try:
        candles = get_candles(symbol, interval, 200)
        if not candles:
            return {}
        
        indicators = calculate_indicators(candles)
        if not indicators:
            return {}
        
        # Generate signal score based on indicators
        score = 0
        signals = []
        
        # Price vs Moving Averages - stricter criteria
        price = indicators.get("price", 0)
        sma_20 = indicators.get("sma_20", price)
        sma_200 = indicators.get("sma_200", price)
        ema_9 = indicators.get("ema_9", price)
        ema_21 = indicators.get("ema_21", price)
        
        # Require price above both SMA20 and SMA200 for bullish
        if price > sma_200 and price > sma_20 and ema_9 > ema_21:
            score += 25
            signals.append("BULLISH_MA_CROSS")
        elif price < sma_200 and price < sma_20 and ema_9 < ema_21:
            score += 25
            signals.append("BEARISH_MA_CROSS")
        
        # RSI signals - more precise zones
        rsi = indicators.get("rsi", 50)
        if 25 <= rsi <= 35:  # Sweet spot for oversold
            score += 25
            signals.append("RSI_OVERSOLD")
        elif 65 <= rsi <= 75:  # Sweet spot for overbought
            score += 25
            signals.append("RSI_OVERBOUGHT")
        elif rsi < 20 or rsi > 80:  # Too extreme
            score -= 10
        
        # Stochastic RSI signals - require crossover
        stoch_k = indicators.get("stoch_k", 50)
        stoch_d = indicators.get("stoch_d", 50)
        if stoch_k < 20 and stoch_k > stoch_d and abs(stoch_k - stoch_d) > 5:
            score += 25
            signals.append("STOCH_RSI_OVERSOLD")
        elif stoch_k > 80 and stoch_k < stoch_d and abs(stoch_k - stoch_d) > 5:
            score += 25
            signals.append("STOCH_RSI_OVERBOUGHT")
        
        # Bollinger Bands signals - must touch or exceed
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        bb_middle = indicators.get("bb_middle", price)
        if price <= bb_lower * 1.01:  # At or below lower band
            score += 20
            signals.append("BB_OVERSOLD")
        elif price >= bb_upper * 0.99:  # At or above upper band
            score += 20
            signals.append("BB_OVERBOUGHT")
        
        # Volume confirmation - require strong volume
        volume_ratio = indicators.get("volume_ratio", 1)
        if volume_ratio > 2.0:  # Very high volume
            score += 20
            signals.append("VOLUME_HIGH")
        elif volume_ratio > 1.5:
            score += 10
            signals.append("VOLUME_MEDIUM")
        elif volume_ratio < 1.2:  # Low volume - penalize
            score -= 15
        
        # Trend confirmation - critical for profitability
        trend_score = indicators.get("trend_score", 0)
        if trend_score >= 3:  # Strong bullish trend
            score += 20
            signals.append("TREND_BULLISH")
        elif trend_score <= 1:  # Strong bearish trend
            score += 20
            signals.append("TREND_BEARISH")
        else:  # Weak trend - avoid
            score -= 10
        
        # Volatility check
        vol = indicators.get("volatility", 0)
        if vol > 5:  # Too volatile
            score -= 20
        elif vol < 0.3:  # Too quiet
            score -= 15
        
        # Determine signal type and side - require multiple confirmations
        signal_type = "neutral"
        side = "Buy"
        
        buy_signals = ["RSI_OVERSOLD", "STOCH_RSI_OVERSOLD", "BB_OVERSOLD", "BULLISH_MA_CROSS"]
        sell_signals = ["RSI_OVERBOUGHT", "STOCH_RSI_OVERBOUGHT", "BB_OVERBOUGHT", "BEARISH_MA_CROSS"]
        
        buy_count = sum(1 for s in buy_signals if s in signals)
        sell_count = sum(1 for s in sell_signals if s in signals)
        
        # Require at least 2 confirmations + trend + volume
        if buy_count >= 2 and "TREND_BULLISH" in signals and ("VOLUME_HIGH" in signals or "VOLUME_MEDIUM" in signals):
            signal_type = "buy"
            side = "Buy"
        elif sell_count >= 2 and "TREND_BEARISH" in signals and ("VOLUME_HIGH" in signals or "VOLUME_MEDIUM" in signals):
            signal_type = "sell"
            side = "Sell"
        else:
            score -= 20  # Penalize weak signals
        
        # Calculate SL and TP with better ratios
        sl_percent = 0.08  # 8% stop loss
        tp_percent = 0.40  # 40% take profit (5:1 R/R)
        
        sl = price * (1 - sl_percent if side == "Buy" else 1 + sl_percent)
        tp = price * (1 + tp_percent if side == "Buy" else 1 - tp_percent)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "score": min(score, 100),
            "signal_type": signal_type,
            "side": side,
            "entry": price,
            "sl": round(sl, 4),
            "tp": round(tp, 4),
            "indicators": indicators,
            "signals": signals,
            "analysis_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {}

def scan_multiple_symbols(symbols: List[str], interval: str = "60", max_workers: int = 10) -> List[Dict]:
    """Scan multiple symbols concurrently"""
    try:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(analyze_symbol, symbol, interval): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol, timeout=60):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result.get("score", 0) > 0:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
        if len(results) < len(symbols):
            logger.warning(f"Only {len(results)} of {len(symbols)} symbols analyzed successfully")
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results
    except Exception as e:
        logger.error(f"Error scanning symbols: {e}")
        return []