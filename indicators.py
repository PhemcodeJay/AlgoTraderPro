import numpy as np
import requests
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# Constants
INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W"]
ML_ENABLED = True  # Feature flag for ML filtering

def get_candles(symbol: str, interval: str, limit: int = 100) -> List[Dict]:
    """Fetch candlestick data from Bybit API"""
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }
        
        response = requests.get(url, params=params, timeout=10)
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
    
    # Pad beginning with neutral values
    while len(rsi_values) < len(prices):
        rsi_values.insert(0, 50.0)
    
    return rsi_values

def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow:
        zero_list = [0.0] * len(prices)
        return {"macd": zero_list, "signal": zero_list, "histogram": zero_list}
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    histogram = [macd_val - sig_val for macd_val, sig_val in zip(macd_line, signal_line)]
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
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

def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """Average True Range"""
    if len(highs) < period or len(highs) != len(lows) or len(highs) != len(closes):
        return [0.0] * len(highs)
    
    true_ranges = []
    
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_ranges.append(max(tr1, tr2, tr3))
    
    # Calculate ATR using SMA of true ranges
    atr_values = [0.0]  # First value is 0
    
    for i in range(period, len(true_ranges) + 1):
        atr_val = sum(true_ranges[i-period:i]) / period
        atr_values.append(atr_val)
    
    # Pad to match input length
    while len(atr_values) < len(highs):
        atr_values.append(atr_values[-1] if atr_values else 0.0)
    
    return atr_values

def calculate_indicators(candles: List[Dict]) -> Dict[str, Any]:
    """Calculate all technical indicators for given candles"""
    try:
        if not candles or len(candles) < 20:
            return {}
        
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c["volume"] for c in candles]
        
        # Moving averages
        sma_20 = sma(closes, 20)
        sma_50 = sma(closes, 50)
        ema_20 = ema(closes, 20)
        
        # Momentum indicators
        rsi_14 = rsi(closes, 14)
        macd_data = macd(closes)
        
        # Volatility indicators
        bb_data = bollinger_bands(closes)
        atr_14 = atr(highs, lows, closes, 14)
        
        # Current values (last in arrays)
        current_price = closes[-1]
        current_rsi = rsi_14[-1] if rsi_14 else 50
        current_macd = macd_data["macd"][-1] if macd_data["macd"] else 0
        current_signal = macd_data["signal"][-1] if macd_data["signal"] else 0
        current_bb_upper = bb_data["upper"][-1] if bb_data["upper"] else 0
        current_bb_lower = bb_data["lower"][-1] if bb_data["lower"] else 0
        current_atr = atr_14[-1] if atr_14 else 0
        
        # Volume analysis
        avg_volume = sum(volumes[-10:]) / min(10, len(volumes)) if volumes else 0
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # Trend analysis
        trend_score = 0
        if len(sma_20) > 1 and len(sma_50) > 1:
            if sma_20[-1] > sma_50[-1]:
                trend_score += 1
            if closes[-1] > sma_20[-1]:
                trend_score += 1
            if sma_20[-1] > sma_20[-2]:
                trend_score += 1
        
        return {
            "price": current_price,
            "sma_20": sma_20[-1] if sma_20 else current_price,
            "sma_50": sma_50[-1] if sma_50 else current_price,
            "ema_20": ema_20[-1] if ema_20 else current_price,
            "rsi": current_rsi,
            "macd": current_macd,
            "macd_signal": current_signal,
            "macd_histogram": macd_data["histogram"][-1] if macd_data["histogram"] else 0,
            "bb_upper": current_bb_upper,
            "bb_lower": current_bb_lower,
            "bb_middle": bb_data["middle"][-1] if bb_data["middle"] else current_price,
            "atr": current_atr,
            "volume": volumes[-1] if volumes else 0,
            "volume_ratio": volume_ratio,
            "trend_score": trend_score,
            "volatility": (current_atr / current_price * 100) if current_price > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}

def get_top_symbols(limit: int = 50) -> List[str]:
    """Get top USDT trading pairs by volume"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "linear"}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") == 0 and "result" in data:
            tickers = data["result"]["list"]
            
            # Filter USDT pairs and sort by volume
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
            
            # Sort by volume and return top symbols
            usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
            return [pair["symbol"] for pair in usdt_pairs[:limit]]
        
        return []
    except Exception as e:
        logger.error(f"Error getting top symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def analyze_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    """Comprehensive analysis of a single symbol"""
    try:
        candles = get_candles(symbol, interval, 100)
        if not candles:
            return {}
        
        indicators = calculate_indicators(candles)
        if not indicators:
            return {}
        
        # Generate signal score based on multiple factors
        score = 0
        signals = []
        
        # RSI signals
        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            score += 25
            signals.append("RSI_OVERSOLD")
        elif rsi > 70:
            score += 25
            signals.append("RSI_OVERBOUGHT")
        
        # MACD signals
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal and indicators.get("macd_histogram", 0) > 0:
            score += 20
            signals.append("MACD_BULLISH")
        elif macd < macd_signal and indicators.get("macd_histogram", 0) < 0:
            score += 20
            signals.append("MACD_BEARISH")
        
        # Bollinger Bands signals
        price = indicators.get("price", 0)
        bb_upper = indicators.get("bb_upper", 0)
        bb_lower = indicators.get("bb_lower", 0)
        if price <= bb_lower:
            score += 15
            signals.append("BB_OVERSOLD")
        elif price >= bb_upper:
            score += 15
            signals.append("BB_OVERBOUGHT")
        
        # Trend signals
        trend_score = indicators.get("trend_score", 0)
        if trend_score >= 2:
            score += 15
            signals.append("TREND_BULLISH")
        
        # Volume confirmation
        volume_ratio = indicators.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            score += 10
            signals.append("VOLUME_HIGH")
        
        # Determine signal type and side
        signal_type = "neutral"
        side = "Buy"
        
        if "RSI_OVERSOLD" in signals or "BB_OVERSOLD" in signals:
            signal_type = "buy"
            side = "Buy"
        elif "RSI_OVERBOUGHT" in signals or "BB_OVERBOUGHT" in signals:
            signal_type = "sell"
            side = "Sell"
        elif "MACD_BULLISH" in signals and "TREND_BULLISH" in signals:
            signal_type = "buy"
            side = "Buy"
        elif "MACD_BEARISH" in signals:
            signal_type = "sell"
            side = "Sell"
        
        return {
            "symbol": symbol,
            "interval": interval,
            "score": min(score, 100),  # Cap at 100
            "signal_type": signal_type,
            "side": side,
            "indicators": indicators,
            "signals": signals,
            "analysis_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {}

def scan_multiple_symbols(symbols: List[str], interval: str = "60", max_workers: int = 10) -> List[Dict]:
    try:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(analyze_symbol, symbol, interval): symbol
                for symbol in symbols
            }
            for future in as_completed(future_to_symbol, timeout=60):  # Increase to 60s
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