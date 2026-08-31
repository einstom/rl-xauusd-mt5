"""MT5 order execution: bracket market orders (entry + SL + TP) via order_send.

Sizing is fixed-fractional: the caller passes the account risk budget
(equity × risk_fraction) and the SL distance in price units; lots follow from
the symbol's tick value.  The RL agent never chooses lot size — only
direction and bracket shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config import CFG

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - non-Windows dev machines
    mt5 = None


def _require_mt5():
    if mt5 is None:
        raise ImportError(
            "The MetaTrader5 package is not installed (Windows-only). "
            "Run this module on the machine that hosts the MT5 terminal."
        )
    return mt5


@dataclass
class BracketPlan:
    """Everything needed to place (and later reconstruct) one bracket trade."""
    direction: int          # +1 buy, -1 sell
    lots: float
    entry_price: float      # market price used for the request
    sl: float
    tp: float
    sl_distance: float      # price units
    tp_r: float             # TP as R-multiple of the SL risk
    risk_cash: float        # money at risk if SL is hit (before slippage)


def ensure_symbol(symbol: str):
    """Make the symbol visible/subscribed and return its symbol_info."""
    _require_mt5()
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select({symbol}) failed: {mt5.last_error()}")
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"symbol_info({symbol}) returned None")
    return si


def money_per_price_unit_per_lot(symbol_info) -> float:
    """Account-currency value of a 1.0 price-unit move for 1.0 lot.

    For XAUUSD with 100 oz contracts and a USD account this is 100.0
    (tick_value / tick_size generalises it across brokers/symbols)."""
    tick_size = symbol_info.trade_tick_size or symbol_info.point
    tick_value = symbol_info.trade_tick_value
    if not tick_size or not tick_value:
        raise RuntimeError(f"Symbol {symbol_info.name}: missing tick size/value "
                           f"({tick_size}, {tick_value})")
    return tick_value / tick_size


def fixed_fractional_lots(risk_cash: float, sl_distance: float, symbol_info) -> float:
    """Lots such that hitting the SL loses ≈ risk_cash; floored to volume_step.

    Returns 0.0 when even the broker's minimum lot would risk more than
    ~1.5× the budget — the caller must then skip the trade instead of
    silently over-risking.
    """
    if sl_distance <= 0 or risk_cash <= 0:
        return 0.0
    mppl = money_per_price_unit_per_lot(symbol_info)
    raw = risk_cash / (sl_distance * mppl)

    step = symbol_info.volume_step or 0.01
    lots = math.floor(raw / step) * step
    lots = round(lots, 8)

    vmin, vmax = symbol_info.volume_min, symbol_info.volume_max
    if lots < vmin:
        min_risk = vmin * sl_distance * mppl
        if min_risk <= risk_cash * 1.5:
            return vmin
        return 0.0
    return min(lots, vmax)


def _min_stop_distance(symbol_info) -> float:
    """Broker minimum SL/TP distance from the current price, in price units."""
    return (symbol_info.trade_stops_level or 0) * symbol_info.point


def build_bracket_plan(
    symbol: str,
    direction: int,
    atr: float,
    sl_atr_mult: float,
    tp_r: float,
    risk_cash: float,
) -> Optional[BracketPlan]:
    """Turn the agent's bracket choice into concrete prices/lots at the
    current market.  Returns None when the trade cannot be sized safely."""
    _require_mt5()
    si = ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick({symbol}) failed: {mt5.last_error()}")

    price = tick.ask if direction > 0 else tick.bid
    sl_distance = max(sl_atr_mult * atr, _min_stop_distance(si) + si.spread * si.point)
    tp_distance = tp_r * sl_distance

    digits = si.digits
    sl = round(price - direction * sl_distance, digits)
    tp = round(price + direction * tp_distance, digits)

    lots = fixed_fractional_lots(risk_cash, sl_distance, si)
    if lots <= 0:
        print(f"[exec] SKIP: risk budget {risk_cash:.2f} cannot cover minimum lot "
              f"at SL distance {sl_distance:.2f}")
        return None

    return BracketPlan(direction=direction, lots=lots, entry_price=price,
                       sl=sl, tp=tp, sl_distance=sl_distance, tp_r=tp_r,
                       risk_cash=risk_cash)


def _allowed_filling_modes(symbol_info) -> list[int]:
    """Filling modes to try, most-preferred first, from the symbol's bitmask
    (SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2); RETURN as last resort."""
    _require_mt5()
    flags = symbol_info.filling_mode
    modes = []
    if flags & 2:
        modes.append(mt5.ORDER_FILLING_IOC)
    if flags & 1:
        modes.append(mt5.ORDER_FILLING_FOK)
    modes.append(mt5.ORDER_FILLING_RETURN)
    return modes


def place_bracket_order(plan: BracketPlan, symbol: str = None,
                        magic: int = None, comment: str = None,
                        max_retries: int = 3):
    """Send a market order with attached SL/TP.  Retries across filling modes
    and on requotes/price drift.  Returns the order_send result on success,
    raises RuntimeError with the retcode on definitive failure."""
    m = _require_mt5()
    symbol = symbol or CFG.mt5_symbol
    magic = magic if magic is not None else CFG.mt5_magic
    comment = comment or CFG.mt5_order_comment
    si = ensure_symbol(symbol)

    order_type = m.ORDER_TYPE_BUY if plan.direction > 0 else m.ORDER_TYPE_SELL
    retryable = {m.TRADE_RETCODE_REQUOTE, m.TRADE_RETCODE_PRICE_CHANGED,
                 m.TRADE_RETCODE_PRICE_OFF}

    last_result = None
    for attempt in range(max_retries):
        tick = m.symbol_info_tick(symbol)
        price = tick.ask if plan.direction > 0 else tick.bid
        for filling in _allowed_filling_modes(si):
            request = {
                "action": m.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": plan.lots,
                "type": order_type,
                "price": price,
                "sl": plan.sl,
                "tp": plan.tp,
                "deviation": CFG.mt5_deviation_points,
                "magic": magic,
                "comment": comment,
                "type_time": m.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = m.order_send(request)
            last_result = result
            if result is None:
                raise RuntimeError(f"order_send returned None: {m.last_error()}")
            if result.retcode == m.TRADE_RETCODE_DONE:
                side = "BUY" if plan.direction > 0 else "SELL"
                print(f"[exec] {side} {plan.lots} {symbol} @ {result.price} "
                      f"SL={plan.sl} TP={plan.tp} (deal {result.deal}, order {result.order})")
                return result
            if result.retcode == m.TRADE_RETCODE_INVALID_FILL:
                continue                      # next filling mode
            if result.retcode in retryable:
                break                         # refresh price, next attempt
            raise RuntimeError(
                f"order_send failed: retcode={result.retcode} ({result.comment})")
    raise RuntimeError(
        f"order_send gave up after {max_retries} attempts: "
        f"retcode={getattr(last_result, 'retcode', '?')} "
        f"({getattr(last_result, 'comment', '?')})")


def get_open_position(symbol: str = None, magic: int = None):
    """This bot's open position on the symbol (or None). One position max by
    construction — the strategy is single-bracket."""
    m = _require_mt5()
    symbol = symbol or CFG.mt5_symbol
    magic = magic if magic is not None else CFG.mt5_magic
    positions = m.positions_get(symbol=symbol)
    if not positions:
        return None
    ours = [p for p in positions if p.magic == magic]
    return ours[0] if ours else None


def close_position(position, max_retries: int = 3):
    """Close an open position at market (used for the agent's flat/flip)."""
    m = _require_mt5()
    symbol = position.symbol
    si = ensure_symbol(symbol)
    is_long = position.type == m.POSITION_TYPE_BUY
    order_type = m.ORDER_TYPE_SELL if is_long else m.ORDER_TYPE_BUY

    last_result = None
    for attempt in range(max_retries):
        tick = m.symbol_info_tick(symbol)
        price = tick.bid if is_long else tick.ask
        for filling in _allowed_filling_modes(si):
            request = {
                "action": m.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": CFG.mt5_deviation_points,
                "magic": position.magic,
                "comment": f"{CFG.mt5_order_comment}-close",
                "type_time": m.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            result = m.order_send(request)
            last_result = result
            if result is None:
                raise RuntimeError(f"order_send returned None: {m.last_error()}")
            if result.retcode == m.TRADE_RETCODE_DONE:
                print(f"[exec] closed position {position.ticket} @ {result.price}")
                return result
            if result.retcode == m.TRADE_RETCODE_INVALID_FILL:
                continue
            if result.retcode in {m.TRADE_RETCODE_REQUOTE, m.TRADE_RETCODE_PRICE_CHANGED,
                                  m.TRADE_RETCODE_PRICE_OFF}:
                break
            raise RuntimeError(
                f"close failed: retcode={result.retcode} ({result.comment})")
    raise RuntimeError(
        f"close gave up after {max_retries} attempts: "
        f"retcode={getattr(last_result, 'retcode', '?')}")


def account_equity() -> float:
    m = _require_mt5()
    acct = m.account_info()
    if acct is None:
        raise RuntimeError(f"account_info() failed: {m.last_error()}")
    return float(acct.equity)
