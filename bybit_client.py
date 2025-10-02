import os
import hmac
import hashlib
import time
import json
import asyncio
import websockets
from typing import Dict, Any, Optional, List
import requests.adapters
from datetime import datetime
import requests
import threading
from dataclasses import dataclass
from logging_config import get_trading_logger
from exceptions import (
    APIException, APIConnectionException, APIRateLimitException, 
    APIAuthenticationException, APITimeoutException, APIDataException,
    APIErrorRecoveryStrategy, create_error_context
)

logger = get_trading_logger('api_client')

@dataclass
class RateLimitInfo:
    requests_per_second: int = 5
    requests_per_minute: int = 120
    last_request_time: float = 0
    request_count: int = 0
    minute_start: float = 0
    minute_count: int = 0

class BybitClient:
    def __init__(self):
        self.api_key: str = os.getenv("BYBIT_API_KEY", "")
        self.api_secret: str = os.getenv("BYBIT_API_SECRET", "")
        self.account_type: str = os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED").upper()

        # Mainnet only
        self.base_url: str = "https://api.bybit.com"
        self.ws_url: str = "wss://stream.bybit.com"

        # Optional sanity check
        if os.getenv("BYBIT_MAINNET", "true").lower() != "true":
            raise ValueError("Mainnet only mode is enabled. Please set BYBIT_MAINNET=true")
        
        # Connection and session management
        self.session: Optional[requests.Session] = None
        self.ws_connection = None
        self.loop = None
        self._price_cache = {}
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # Rate limiting
        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        
        # Error handling
        self.recovery_strategy = APIErrorRecoveryStrategy(max_retries=3, delay=1.0)
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # Timeout configuration
        self.request_timeout = 30  # seconds
        self.connect_timeout = 10  # seconds
        
        # Health monitoring
        self.last_successful_request = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Initialize connection
        self._initialize_session()
        self._test_connection()  # Test connection during initialization
        
        # Start background event loop for WebSocket
        self._start_background_loop()
        
        logger.info(f"BybitClient initialized - Environment: mainnet - Account Type: {self.account_type}")
    
    def _initialize_session(self):
        """Initialize HTTP session with proper configuration"""
        try:
            self.session = requests.Session()
            
            # Configure adapters for connection pooling and retries  
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=0  # We handle retries manually
            )
            
            self.session.mount('https://', adapter)
            self.session.mount('http://', adapter)
            
            logger.info("HTTP session initialized with connection pooling")
            
        except Exception as e:
            error_context = create_error_context(
                module=__name__,
                function='_initialize_session'
            )
            raise APIConnectionException(
                f"Failed to initialize HTTP session: {str(e)}",
                endpoint='session_init',
                context=error_context,
                original_exception=e
            )

    def _start_background_loop(self):
        """Start background event loop for async operations"""
        try:
            import threading
            def run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            
            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            logger.info("Background event loop started")
        except Exception as e:
            logger.error(f"Failed to start background loop: {e}")

    def _generate_signature(self, params: str, timestamp: str) -> str:
        """ producing: Generate API signature"""
        param_str = timestamp + self.api_key + "5000" + params
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self, params: str = "") -> Dict[str, str]:
        """ producing: Get authenticated headers"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(params, timestamp)
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }

    def _check_rate_limit(self) -> bool:
        """ checking: Check and enforce rate limits"""
        with self.rate_limit_lock:
            now = time.time()
            
            # Reset minute counter if needed
            if now - self.rate_limit.minute_start >= 60:
                self.rate_limit.minute_start = now
                self.rate_limit.minute_count = 0
            
            # Check per-second rate limit
            time_since_last = now - self.rate_limit.last_request_time
            if time_since_last < 1.0 / self.rate_limit.requests_per_second:
                sleep_time = (1.0 / self.rate_limit.requests_per_second) - time_since_last
                time.sleep(sleep_time)
                now = time.time()
            
            # Check per-minute rate limit
            if self.rate_limit.minute_count >= self.rate_limit.requests_per_minute:
                sleep_time = 60 - (now - self.rate_limit.minute_start)
                if sleep_time > 0:
                    logger.warning(f"Rate limit hit, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    self.rate_limit.minute_start = time.time()
                    self.rate_limit.minute_count = 0
            
            # Update counters
            self.rate_limit.last_request_time = time.time()
            self.rate_limit.minute_count += 1
            
            return True
    
    def _validate_api_credentials(self) -> bool:
        """ validating: Validate API credentials are present"""
        if not self.api_key or not self.api_secret:
            raise APIAuthenticationException(
                "API credentials not configured. Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables.",
                context=create_error_context(module=__name__, function='_validate_api_credentials')
            )
        return True
    
    def _handle_api_error(self, response_data: Dict, endpoint: str) -> None:
        """ handling: Handle API error responses"""
        ret_code = response_data.get("retCode", -1)
        ret_msg = response_data.get("retMsg", "Unknown error")
        
        # Map specific error codes to exceptions
        if ret_code == 10001:
            raise APIDataException(
                f"Request parameter error: {ret_msg}",
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif ret_code == 10002:
            raise APIAuthenticationException(
                f"Request timestamp expired: {ret_msg}",
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif ret_code == 10003:
            raise APIRateLimitException(
                f"Rate limit exceeded: {ret_msg}",
                retry_after=60,
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif ret_code == 10004:
            raise APIAuthenticationException(
                f"Invalid signature: {ret_msg}",
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif ret_code == 100028:
            raise APIException(
                f"Operation forbidden for unified account: {ret_msg}. Use cross margin mode.",
                error_code=str(ret_code),
                context=create_error_context(
                    module=__name__,
                    function='_handle_api_error',
                    extra_data={'endpoint': endpoint, 'ret_code': ret_code}
                )
            )
        else:
            raise APIException(
                f"API Error (code {ret_code}): {ret_msg}",
                error_code=str(ret_code),
                context=create_error_context(
                    module=__name__,
                    function='_handle_api_error',
                    extra_data={'endpoint': endpoint, 'ret_code': ret_code}
                )
            )
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """ making: Make authenticated API request with comprehensive error handling"""
        # Validate credentials
        self._validate_api_credentials()
        
        # Check rate limits
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        start_time = time.time()
        
        # Track request
        self.total_requests += 1
        
        for attempt in range(self.recovery_strategy.max_retries + 1):
            try:
                with self._connection_lock:
                    if not self.session:
                        raise APIConnectionException("HTTP session not initialized", endpoint=endpoint)
                    
                    if method.upper() == "GET":
                        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                        headers = self._get_headers(query_string)
                        response = self.session.get(
                            url, 
                            params=params, 
                            headers=headers, 
                            timeout=(self.connect_timeout, self.request_timeout)
                        )
                    else:
                        params_str = json.dumps(params) if params else ""
                        headers = self._get_headers(params_str)
                        response = self.session.post(
                            url, 
                            json=params, 
                            headers=headers, 
                            timeout=(self.connect_timeout, self.request_timeout)
                        )
                
                # Calculate response time
                response_time = (time.time() - start_time) * 1000
                
                # Check HTTP status
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise APIRateLimitException(
                        f"Rate limit exceeded on {endpoint}",
                        retry_after=retry_after,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'attempt': attempt}
                        )
                    )
                
                response.raise_for_status()
                
                # Parse JSON response
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise APIDataException(
                        f"Invalid JSON response from {endpoint}: {str(e)}",
                        response_data=response.text[:1000],  # Limit response data
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': response.status_code}
                        )
                    )
                
                # Check API response code
                if data.get("retCode") != 0:
                    self._handle_api_error(data, endpoint)
                
                # Success - update statistics
                self.successful_requests += 1
                self.last_successful_request = datetime.now()
                self.consecutive_errors = 0
                
                # Log successful request
                logger.info(
                    f"API request successful: {method} {endpoint}",
                    extra={
                        'endpoint': endpoint,
                        'method': method,
                        'response_time_ms': round(response_time, 2),
                        'status_code': response.status_code,
                        'attempt': attempt + 1
                    }
                )
                
                return data.get("result", {})
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                if not self.recovery_strategy.should_retry(e, attempt):
                    raise APITimeoutException(
                        f"Request timeout for {endpoint} after {attempt + 1} attempts: {str(e)}",
                        timeout_duration=self.request_timeout,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'total_attempts': attempt + 1}
                        ),
                        original_exception=e
                    )
                
                # Wait before retry
                if attempt < self.recovery_strategy.max_retries:
                    retry_delay = self.recovery_strategy.get_delay(attempt)
                    logger.warning(
                        f"API request failed (attempt {attempt + 1}), retrying in {retry_delay}s: {str(e)}",
                        extra={'endpoint': endpoint, 'attempt': attempt + 1, 'retry_delay': retry_delay}
                    )
                    time.sleep(retry_delay)
                
            except requests.exceptions.HTTPError as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                status_code = e.response.status_code if e.response else None
                
                if status_code == 401:
                    raise APIAuthenticationException(
                        f"Authentication failed for {endpoint}: {str(e)}",
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
                elif status_code == 403:
                    raise APIAuthenticationException(
                        f"Access forbidden for {endpoint}: {str(e)}",
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
                else:
                    raise APIConnectionException(
                        f"HTTP error for {endpoint}: {str(e)}",
                        endpoint=endpoint,
                        status_code=status_code,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
            
            except Exception as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                logger.error(
                    f"Unexpected error in API request to {endpoint}: {str(e)}",
                    extra={'endpoint': endpoint, 'attempt': attempt + 1}
                )
                
                raise APIException(
                    f"Unexpected error for {endpoint}: {str(e)}",
                    context=create_error_context(
                        module=__name__,
                        function='_make_request',
                        extra_data={'endpoint': endpoint, 'attempt': attempt + 1}
                    ),
                    original_exception=e
                )
        
        # Should not reach here, but just in case
        raise APIException(
            f"Maximum retry attempts exceeded for {endpoint}",
            context=create_error_context(
                module=__name__,
                function='_make_request',
                extra_data={'endpoint': endpoint, 'max_retries': self.recovery_strategy.max_retries}
            )
        )
    
    async def _make_request_async(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """ async wrapper: Async wrapper around _make_request using asyncio.to_thread"""
        return await asyncio.to_thread(self._make_request, method, endpoint, params)

    def is_connected(self) -> bool:
        """ checking: Check if client is connected and authenticated"""
        if not self._connected:
            # Attempt to verify connection if not connected
            try:
                self._test_connection()
            except Exception as e:
                logger.warning(f"Connection verification failed: {e}")
                self._connected = False
        return self._connected and bool(self.api_key and self.api_secret)
    
    def _test_connection(self) -> bool:
        """ testing: Test API connection with comprehensive error handling"""
        try:
            if not self.api_key or not self.api_secret:
                logger.error("API credentials missing in .env")
                self._connected = False
                return False

            # Use a simple endpoint for connection testing
            result = self._make_request("GET", "/v5/market/time", {})
            self._connected = True
            
            logger.info(
                f"API connection test successful",
                extra={
                    'endpoint': '/v5/market/time',
                    'environment': 'mainnet',
                    'account_type': self.account_type
                }
            )
            return True
            
        except APIAuthenticationException as e:
            self._connected = False
            logger.error(
                f"API authentication failed during connection test: {str(e)}",
                extra={
                    'error_type': 'authentication',
                    'environment': 'mainnet',
                    'account_type': self.account_type
                }
            )
            return False
            
        except APIRateLimitException as e:
            self._connected = False
            logger.warning(
                f"Rate limit hit during connection test: {str(e)}",
                extra={'error_type': 'rate_limit', 'retry_after': e.retry_after}
            )
            return False
            
        except APIException as e:
            self._connected = False
            logger.error(
                f"API error during connection test: {str(e)}",
                extra={'error_type': 'api_error', 'error_code': e.error_code}
            )
            return False
            
        except Exception as e:
            self._connected = False
            logger.error(
                f"Unexpected error during connection test: {str(e)}",
                extra={'error_type': 'unexpected'}
            )
            return False
    
    def get_connection_health(self) -> Dict[str, Any]:
        """ retrieving: Get comprehensive connection health information"""
        health_info = {
            'connected': self._connected,
            'environment': 'mainnet',
            'account_type': self.account_type,
            'api_configured': bool(self.api_key and self.api_secret),
            'last_successful_request': self.last_successful_request.isoformat() if self.last_successful_request else None,
            'consecutive_errors': self.consecutive_errors,
            'statistics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': round((self.successful_requests / max(self.total_requests, 1)) * 100, 2)
            },
            'rate_limiting': {
                'requests_per_second': self.rate_limit.requests_per_second,
                'requests_per_minute': self.rate_limit.requests_per_minute,
                'current_minute_count': self.rate_limit.minute_count
            }
        }

        # Determine overall health status
        if not health_info['api_configured']:
            health_info['status'] = 'unconfigured'
        elif not health_info['connected']:
            health_info['status'] = 'disconnected'
        elif self.consecutive_errors > 5:
            health_info['status'] = 'degraded'
        else:
            health_info['status'] = 'healthy'
            
        return health_info

    from typing import TYPE_CHECKING, Dict
    if TYPE_CHECKING:
        from db import WalletBalance
    def get_account_balance(self) -> "Dict[str, 'WalletBalance']":  # note the quotes around WalletBalance
        from db import WalletBalance
        """ retrieving: Get account wallet balance as WalletBalance objects keyed by coin symbol"""
        try:
            result = self._make_request(
                "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
            )

            balances: Dict[str, "WalletBalance"] = {}

            if result and "list" in result and result["list"]:
                wallet = result["list"][0]
                coins = wallet.get("coin", [])

                for coin in coins:
                    symbol = coin.get("coin", "")
                    if not symbol:
                        continue

                    available = float(coin.get("availableToWithdraw", 0) or 0.0)
                    total = float(coin.get("walletBalance", 0) or 0.0)
                    used = total - available

                    balances[symbol] = WalletBalance(
                        trading_mode="real",
                        capital=total,          # total balance → capital
                        available=available,
                        used=used,
                        start_balance=total,    # initial snapshot
                        currency=symbol,
                        updated_at=datetime.utcnow(),
                    )

            return balances

        except Exception as e:
            logger.error(f"Error getting account balance from Bybit: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        """ retrieving: Get current market price for symbol"""
        try:
            # Try cache first
            if symbol in self._price_cache:
                cache_time, price = self._price_cache[symbol]
                if time.time() - cache_time < 10:  # 10 second cache
                    return price

            # Fetch from API
            result = self._make_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": symbol
            })
            
            if result and "list" in result and result["list"]:
                price = float(result["list"][0].get("lastPrice", 0))
                self._price_cache[symbol] = (time.time(), price)
                return price
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_current_over_price(self, symbol: str) -> float:
        """ retrieving: Alias for get_current_price for backward compatibility"""
        return self.get_current_price(symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """ retrieving: Get historical kline/candlestick data"""
        try:
            result = self._make_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit)
            })
            
            if result and "list" in result:
                klines = []
                for k in result["list"]:
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
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        leverage: Optional[int] = 10,
        mode: str = "CROSS"  # Default to CROSS for unified accounts
    ) -> Dict:
        """
        Place a market trading order with leverage, cross margin mode for unified accounts,
        and TP/SL based on ROI percentages (25% TP ROI, -5% SL ROI).

        Args:
            symbol (str): Trading pair (e.g., 'BTCUSDT').
            side (str): 'buy' or 'sell'.
            qty (float): Order quantity in base currency.
            leverage (Optional[int]): Leverage multiplier (default: 10).
            mode (str): Margin mode ('CROSS' or 'ISOLATED', default: 'CROSS').

        Returns:
            Dict: Order details or error information.
        """
        try:
            leverage = leverage or 10
            if leverage <= 0:
                raise ValueError("Leverage must be positive")

            # Validate side
            side_lower = side.lower()
            if side_lower not in ["buy", "sell"]:
                raise ValueError("side must be 'buy' or 'sell'")

            # Validate quantity
            if qty <= 0:
                raise ValueError("Quantity must be positive")

            # Determine margin mode
            if self.account_type == "UNIFIED":
                mode = "CROSS"
                logger.info(f"Unified account detected, using CROSS margin mode for {symbol}")
            else:
                trade_mode = 1 if mode.upper() == "ISOLATED" else 0
                lev_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "tradeMode": trade_mode,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage)
                }
                await asyncio.get_running_loop().run_in_executor(
                    None, self._make_request, "POST", "/v5/position/switch-isolated", lev_params
                )

            # Get current price (try cache first)
            entry_price = None
            if symbol in self._price_cache:
                cache_time, price = self._price_cache[symbol]
                if time.time() - cache_time < 10:  # 10-second cache
                    entry_price = price
            if not entry_price:
                entry_price = self.get_current_price(symbol)
            if entry_price <= 0:
                raise APIDataException(f"Invalid entry price for {symbol}: {entry_price}")

            # Calculate stop loss and take profit (ROI-based: 25% TP ROI, -5% SL ROI)
            tp_roi_pct = 25.0  # 25% TP ROI
            sl_roi_pct = -5.0  # -5% SL ROI
            tp_movement_pct = tp_roi_pct / 100 / leverage  # e.g., 25% / 10x = 2.5% price move
            sl_movement_pct = 5 / 100 / leverage  # e.g., 5% / 10x = 0.5% price move

            if side_lower == "buy":
                take_profit = entry_price * (1 + tp_movement_pct)  # e.g., +2.5%
                stop_loss = entry_price * (1 - sl_movement_pct)    # e.g., -0.5%
            else:  # sell
                take_profit = entry_price * (1 - tp_movement_pct)  # e.g., -2.5%
                stop_loss = entry_price * (1 + sl_movement_pct)    # e.g., +0.5%

            # Validate TP/SL
            if take_profit <= 0 or stop_loss <= 0:
                raise APIDataException(f"Invalid TP ({-take_profit}) or SL ({stop_loss}) for {symbol}")

            # Build order params
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.title(),
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",  # Immediate or Cancel
                "stopLoss": str(round(stop_loss, 2)),
                "takeProfit": str(round(take_profit, 2)),
                "tpTriggerBy": "LastPrice",
                "slTriggerBy": "LastPrice"
            }

            # Place order
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._make_request, "POST", "/v5/order/create", params
            )

            if result:
                logger.info(
                    f"Order placed: {symbol}, {side}, qty={qty}, leverage={leverage}, "
                    f"TP={take_profit:.2f} ({tp_roi_pct}% ROI), SL={stop_loss:.2f} ({sl_roi_pct}% ROI)"
                )
                return {
                    "order_id": result.get("orderId"),
                    "symbol": symbol,
                    "accountType": self.account_type,
                    "side": side,
                    "qty": qty,
                    "price": entry_price,
                    "status": "pending",
                    "timestamp": datetime.now(),
                    "virtual": False,
                    "leverage": leverage,
                    "stopLoss": str(round(stop_loss, 2)),
                    "takeProfit": str(round(take_profit, 2)),
                    "margin_mode": mode.upper()
                }
            return {}

        except APIException as e:
            if e.error_code == "100028":
                logger.warning(f"Unified account error for {symbol}: {e}. Retrying with CROSS margin mode.")
                return await self.place_order(symbol, side, qty, leverage, mode="CROSS")
            logger.error(f"Error placing order for {symbol}: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}")
            return {"error": str(e)}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """ canceling: Cancel an order"""
        try:
            result = self._make_request("POST", "/v5/order/cancel", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            return bool(result)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """ retrieving: Get open orders"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
                
            result = self._make_request("GET", "/v5/order/realtime", params)
            
            if result and "list" in result:
                orders = []
                for order in result["list"]:
                    orders.append({
                        "order_id": order.get("orderId"),
                        "symbol": order.get("symbol"),
                        "side": order.get("side"),
                        "qty": float(order.get("qty", 0)),
                        "price": float(order.get("price", 0)),
                        "status": order.get("orderStatus"),
                        "timestamp": datetime.fromtimestamp(int(order.get("createdTime", 0)) / 1000)
                    })
                return orders
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """ retrieving: Get current positions asynchronously"""
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol

            result = await self._make_request_async("GET", "/v5/position/list", params)

            if result and "list" in result:
                positions: List[Dict] = []
                for pos in result["list"]:
                    size = float(pos.get("size", 0))
                    if size > 0:  # Only active positions
                        positions.append({
                            "symbol": pos.get("symbol"),
                            "side": pos.get("side"),
                            "size": size,
                            "entry_price": float(pos.get("avgPrice", 0)),
                            "mark_price": float(pos.get("markPrice", 0)),
                            "unrealized_pnl": float(pos.get("unrealisedPnl", 0)),
                            "leverage": float(pos.get("leverage", 10)),
                        })
                return positions
            return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def start_websocket(self, symbols: List[str]):
        """ starting: Start WebSocket connection for real-time data"""
        try:
            if not self.loop:
                logger.error("Event loop not available")
                return

            async def websocket_handler():
                uri = f"{self.ws_url}/v5/public/linear"
                try:
                    async with websockets.connect(uri) as websocket:
                        self.ws_connection = websocket
                        
                        # Subscribe to tickers
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}" for symbol in symbols]
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                        
                        async for message in websocket:
                            data = json.loads(message)
                            if data.get("topic", "").startswith("tickers."):
                                ticker_data = data.get("data", {})
                                symbol = ticker_data.get("symbol")
                                price = float(ticker_data.get("lastPrice", 0))
                                if symbol and price > 0:
                                    self._price_cache[symbol] = (time.time(), price)
                                    
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    self.ws_connection = None

            # Schedule WebSocket in background loop
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(websocket_handler(), self.loop)
                logger.info("WebSocket connection started")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")

    def close(self):
        """ closing: Close connections and cleanup"""
        try:
            if self.ws_connection and self.loop:
                asyncio.run_coroutine_threadsafe(self.ws_connection.close(), self.loop)
            if self.session:
                self.session.close()
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            logger.info("Bybit client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")