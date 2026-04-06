"""Paper Trading Engine — virtual P&L for PhaseBreak predictions.

Tracks: if we shorted on POSSIBLE/BUBBLE signal and covered 30 days later,
what would our P&L be? Compares to buy-and-hold benchmark.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
from rich.console import Console
from rich.table import Table

log = structlog.get_logger()
console = Console()

PORTFOLIO_PATH = Path(__file__).parent.parent.parent / "data" / "PAPER_PORTFOLIO.json"
INITIAL_CAPITAL = 100_000.0


@dataclass
class Trade:
    """Single paper trade."""

    ticker: str
    signal_date: str
    signal_verdict: str
    signal_strength: str
    entry_price: float
    exit_price: float | None = None
    exit_date: str | None = None
    pnl_pct: float | None = None
    status: str = "OPEN"  # OPEN | CLOSED | EXPIRED


@dataclass
class Portfolio:
    """Paper trading portfolio."""

    capital: float = INITIAL_CAPITAL
    trades: list[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    n_wins: int = 0
    n_losses: int = 0
    max_drawdown: float = 0.0
    peak_capital: float = INITIAL_CAPITAL

    def add_signal(self, ticker: str, verdict: str, strength: str, price: float) -> None:
        """Record a new signal as virtual short position."""
        if verdict not in ("BUBBLE", "POSSIBLE"):
            return
        # Don't duplicate
        if any(t.ticker == ticker and t.status == "OPEN" for t in self.trades):
            return

        self.trades.append(
            Trade(
                ticker=ticker,
                signal_date=time.strftime("%Y-%m-%d"),
                signal_verdict=verdict,
                signal_strength=strength,
                entry_price=price,
            )
        )
        log.info("paper_trade_opened", ticker=ticker, price=price, verdict=verdict)

    def check_exits(self, current_prices: dict[str, float], days_limit: int = 30) -> None:
        """Close trades that hit profit target, stop loss, or time limit."""
        from datetime import datetime, timedelta

        today = datetime.now()

        for trade in self.trades:
            if trade.status != "OPEN":
                continue

            signal_dt = datetime.strptime(trade.signal_date, "%Y-%m-%d")
            days_open = (today - signal_dt).days

            if trade.ticker in current_prices:
                current = current_prices[trade.ticker]
                pnl_pct = ((trade.entry_price - current) / trade.entry_price) * 100  # short P&L

                # Exit conditions
                if pnl_pct > 20:  # profit target: 20% drop
                    trade.exit_price = current
                    trade.exit_date = time.strftime("%Y-%m-%d")
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "CLOSED"
                    self._update_capital(pnl_pct)
                    log.info("paper_trade_profit", ticker=trade.ticker, pnl=f"{pnl_pct:.1f}%")

                elif pnl_pct < -15:  # stop loss: 15% against us
                    trade.exit_price = current
                    trade.exit_date = time.strftime("%Y-%m-%d")
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "CLOSED"
                    self._update_capital(pnl_pct)
                    log.info("paper_trade_stopped", ticker=trade.ticker, pnl=f"{pnl_pct:.1f}%")

                elif days_open >= days_limit:  # time exit
                    trade.exit_price = current
                    trade.exit_date = time.strftime("%Y-%m-%d")
                    trade.pnl_pct = round(pnl_pct, 2)
                    trade.status = "EXPIRED"
                    self._update_capital(pnl_pct)
                    log.info(
                        "paper_trade_expired",
                        ticker=trade.ticker,
                        pnl=f"{pnl_pct:.1f}%",
                        days=days_open,
                    )

    def _update_capital(self, pnl_pct: float) -> None:
        """Update capital and track drawdown."""
        # 10% of capital per trade
        position_size = self.capital * 0.10
        pnl_dollar = position_size * (pnl_pct / 100)
        self.capital += pnl_dollar
        self.total_pnl += pnl_dollar

        if pnl_pct > 0:
            self.n_wins += 1
        else:
            self.n_losses += 1

        self.peak_capital = max(self.peak_capital, self.capital)
        current_dd = (self.capital - self.peak_capital) / self.peak_capital * 100
        self.max_drawdown = min(self.max_drawdown, current_dd)

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from closed trades."""
        pnls = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        if len(pnls) < 2:
            return 0.0
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        if std_pnl == 0:
            return 0.0
        # Assume ~12 trades per year
        return round(float(mean_pnl / std_pnl * np.sqrt(12)), 2)

    def print_summary(self) -> None:
        """Print portfolio summary."""
        table = Table(title="Paper Trading Portfolio", show_header=True, header_style="bold")
        table.add_column("Ticker")
        table.add_column("Signal")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Status")
        table.add_column("Date")

        for t in self.trades:
            pnl_str = f"{t.pnl_pct:+.1f}%" if t.pnl_pct is not None else "—"
            color = (
                "green" if (t.pnl_pct or 0) > 0 else ("red" if (t.pnl_pct or 0) < 0 else "white")
            )
            table.add_row(
                t.ticker,
                f"{t.signal_verdict} ({t.signal_strength})",
                f"${t.entry_price:,.2f}",
                f"${t.exit_price:,.2f}" if t.exit_price else "—",
                f"[{color}]{pnl_str}[/{color}]",
                t.status,
                t.signal_date,
            )
        console.print(table)

        win_rate = (
            self.n_wins / (self.n_wins + self.n_losses) * 100
            if (self.n_wins + self.n_losses) > 0
            else 0
        )
        console.print(f"\n  Capital: ${self.capital:,.2f} (started ${INITIAL_CAPITAL:,.2f})")
        console.print(
            f"  Total P&L: ${self.total_pnl:+,.2f} ({self.total_pnl / INITIAL_CAPITAL * 100:+.1f}%)"
        )
        console.print(f"  Win rate: {win_rate:.0f}% ({self.n_wins}W / {self.n_losses}L)")
        console.print(f"  Max drawdown: {self.max_drawdown:.1f}%")
        console.print(f"  Sharpe ratio: {self.sharpe_ratio()}")

    def save(self) -> None:
        """Save portfolio to JSON."""
        data = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M"),
            "capital": round(self.capital, 2),
            "initial_capital": INITIAL_CAPITAL,
            "total_pnl": round(self.total_pnl, 2),
            "total_return_pct": round(self.total_pnl / INITIAL_CAPITAL * 100, 2),
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "max_drawdown_pct": round(self.max_drawdown, 2),
            "sharpe_ratio": self.sharpe_ratio(),
            "trades": [
                {
                    "ticker": t.ticker,
                    "signal_date": t.signal_date,
                    "signal_verdict": t.signal_verdict,
                    "signal_strength": t.signal_strength,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "exit_date": t.exit_date,
                    "pnl_pct": t.pnl_pct,
                    "status": t.status,
                }
                for t in self.trades
            ],
        }
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> Portfolio:
        """Load portfolio from JSON."""
        if not PORTFOLIO_PATH.exists():
            return cls()
        with open(PORTFOLIO_PATH, encoding="utf-8") as f:
            data = json.load(f)
        p = cls(
            capital=data.get("capital", INITIAL_CAPITAL),
            total_pnl=data.get("total_pnl", 0),
            n_wins=data.get("n_wins", 0),
            n_losses=data.get("n_losses", 0),
            max_drawdown=data.get("max_drawdown_pct", 0),
            peak_capital=max(data.get("capital", INITIAL_CAPITAL), INITIAL_CAPITAL),
        )
        for t in data.get("trades", []):
            p.trades.append(Trade(**t))
        return p
