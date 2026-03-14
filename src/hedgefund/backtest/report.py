"""Backtest report generation with charts and metrics.

Provides two report generators:

- :class:`BacktestReport` -- the original file-based report that saves
  individual chart images alongside an HTML file.
- :class:`SelfContainedReport` -- a Jinja2-based generator that embeds all
  charts as base64 data URIs for a single portable HTML file.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import structlog

from hedgefund.backtest.metrics import MetricsCalculator
from hedgefund.types import BacktestMetrics

log = structlog.get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Original file-based report (preserved for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report - {title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #7fdbff; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #16213e; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #0f3460; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
        .metric-card .label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        .positive {{ color: #00e676; }}
        .negative {{ color: #ff1744; }}
        .neutral {{ color: #ffab40; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px 15px; text-align: right; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; color: #7fdbff; text-align: left; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .chart-container {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 15px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; }}
        .footer {{ text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #333; }}
        .heatmap {{ margin: 15px 0; }}
        .heatmap td {{ text-align: center; width: 60px; padding: 8px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Backtest Report</h1>
    <p>Generated: {generated_at} | Period: {start_date} to {end_date}</p>

    <h2>Performance Summary</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Total Return</div>
            <div class="value {return_class}">{total_return}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe Ratio</div>
            <div class="value {sharpe_class}">{sharpe_ratio}</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{max_drawdown}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value {winrate_class}">{win_rate}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value neutral">{total_trades}</div>
        </div>
        <div class="metric-card">
            <div class="label">Profit Factor</div>
            <div class="value {pf_class}">{profit_factor}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sortino Ratio</div>
            <div class="value {sortino_class}">{sortino_ratio}</div>
        </div>
        <div class="metric-card">
            <div class="label">Expectancy</div>
            <div class="value {exp_class}">{expectancy}</div>
        </div>
    </div>

    <h2>Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Annualized Return</td><td>{annualized_return}</td></tr>
        <tr><td>Calmar Ratio</td><td>{calmar_ratio}</td></tr>
        <tr><td>Winning Trades</td><td>{winning_trades}</td></tr>
        <tr><td>Losing Trades</td><td>{losing_trades}</td></tr>
        <tr><td>Avg Win</td><td>{avg_win}</td></tr>
        <tr><td>Avg Loss</td><td>{avg_loss}</td></tr>
        <tr><td>Avg Win/Loss Ratio</td><td>{win_loss_ratio}</td></tr>
    </table>

    {equity_chart_section}
    {drawdown_chart_section}
    {monthly_returns_section}
    {trade_distribution_section}

    <div class="footer">
        HedgeFund AI Options Trading Agent | Backtest Engine v1.0
    </div>
</div>
</body>
</html>
"""


class BacktestReport:
    """Generate HTML backtest reports with embedded charts.

    This is the original file-based reporter that writes chart PNGs
    alongside the HTML file.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_calc = MetricsCalculator()

    def generate(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        metrics: BacktestMetrics,
        title: str = "Strategy Backtest",
        save_charts: bool = True,
    ) -> str:
        """Generate full HTML report.

        Args:
            equity_curve: Time-indexed portfolio value series.
            trades: Trade log DataFrame.
            metrics: Pre-calculated metrics.
            title: Report title.
            save_charts: Whether to save chart images.

        Returns:
            Path to generated HTML file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"backtest_{timestamp}"
        report_dir = self.output_dir / report_name
        report_dir.mkdir(parents=True, exist_ok=True)

        # Generate charts
        equity_section = ""
        drawdown_section = ""
        monthly_section = ""
        trade_dist_section = ""

        if save_charts:
            try:
                equity_path = self._plot_equity_curve(equity_curve, report_dir)
                equity_section = f'<div class="chart-container"><h2>Equity Curve</h2><img src="{equity_path.name}" /></div>'

                dd_path = self._plot_drawdown(equity_curve, report_dir)
                drawdown_section = f'<div class="chart-container"><h2>Drawdown</h2><img src="{dd_path.name}" /></div>'

                if not trades.empty:
                    dist_path = self._plot_trade_distribution(trades, report_dir)
                    trade_dist_section = f'<div class="chart-container"><h2>Trade P&L Distribution</h2><img src="{dist_path.name}" /></div>'

                monthly = self.metrics_calc.monthly_returns(equity_curve)
                if not monthly.empty:
                    monthly_section = self._monthly_returns_html(monthly)

                plt.close("all")
            except Exception:
                log.warning("chart_generation_failed", exc_info=True)

        # Value formatting helpers
        def pct(v: float) -> str:
            return f"{v * 100:.2f}%"

        def flt(v: float) -> str:
            return f"{v:.2f}"

        def dollar(v: float) -> str:
            return f"${v:,.2f}"

        def css_class(v: float) -> str:
            return "positive" if v > 0 else "negative" if v < 0 else "neutral"

        wl_ratio = (
            f"{metrics.avg_win / metrics.avg_loss:.2f}"
            if metrics.avg_loss > 0
            else "N/A"
        )

        html = REPORT_TEMPLATE.format(
            title=title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            start_date=equity_curve.index[0].strftime("%Y-%m-%d") if len(equity_curve) > 0 else "N/A",
            end_date=equity_curve.index[-1].strftime("%Y-%m-%d") if len(equity_curve) > 0 else "N/A",
            total_return=pct(metrics.total_return),
            return_class=css_class(metrics.total_return),
            sharpe_ratio=flt(metrics.sharpe_ratio),
            sharpe_class=css_class(metrics.sharpe_ratio),
            max_drawdown=pct(metrics.max_drawdown),
            win_rate=pct(metrics.win_rate),
            winrate_class="positive" if metrics.win_rate > 0.5 else "negative",
            total_trades=str(metrics.total_trades),
            profit_factor=flt(metrics.profit_factor),
            pf_class=css_class(metrics.profit_factor - 1),
            sortino_ratio=flt(metrics.sortino_ratio),
            sortino_class=css_class(metrics.sortino_ratio),
            expectancy=dollar(metrics.expectancy),
            exp_class=css_class(metrics.expectancy),
            annualized_return=pct(metrics.annualized_return),
            calmar_ratio=flt(metrics.calmar_ratio),
            winning_trades=str(metrics.winning_trades),
            losing_trades=str(metrics.losing_trades),
            avg_win=dollar(metrics.avg_win),
            avg_loss=dollar(metrics.avg_loss),
            win_loss_ratio=wl_ratio,
            equity_chart_section=equity_section,
            drawdown_chart_section=drawdown_section,
            monthly_returns_section=monthly_section,
            trade_distribution_section=trade_dist_section,
        )

        report_path = report_dir / "report.html"
        report_path.write_text(html)

        log.info("backtest_report_generated", path=str(report_path))
        return str(report_path)

    def _plot_equity_curve(self, equity: pd.Series, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(equity.index, equity.values, color="#00d4ff", linewidth=1.5)
        ax.fill_between(equity.index, equity.values, alpha=0.1, color="#00d4ff")
        ax.set_title("Equity Curve", fontsize=14, color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(alpha=0.2)

        path = output_dir / "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return path

    def _plot_drawdown(self, equity: pd.Series, output_dir: Path) -> Path:
        dd = self.metrics_calc.drawdown_series(equity)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dd.index, dd.values, color="#ff1744", alpha=0.6)
        ax.plot(dd.index, dd.values, color="#ff1744", linewidth=0.5)
        ax.set_title("Drawdown", fontsize=14, color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.grid(alpha=0.2)

        path = output_dir / "drawdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return path

    def _plot_trade_distribution(self, trades: pd.DataFrame, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        pnl = trades["pnl"]
        ax.hist(pnl, bins=50, color="#00d4ff", alpha=0.7, edgecolor="#333")
        ax.axvline(x=0, color="#ffab40", linestyle="--", linewidth=1)
        ax.axvline(x=pnl.mean(), color="#00e676", linestyle="--", linewidth=1, label=f"Mean: ${pnl.mean():,.0f}")
        ax.set_title("Trade P&L Distribution", fontsize=14, color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")
        ax.grid(alpha=0.2)

        path = output_dir / "trade_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        return path

    def _monthly_returns_html(self, monthly: pd.DataFrame) -> str:
        """Generate monthly returns heatmap as HTML table."""
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        html = '<h2>Monthly Returns</h2><table class="heatmap">'
        html += "<tr><th>Year</th>"
        for m in month_names:
            html += f"<th>{m}</th>"
        html += "<th>Annual</th></tr>"

        for year in monthly.index:
            html += f"<tr><td><strong>{year}</strong></td>"
            annual = 0.0
            for month in range(1, 13):
                if month in monthly.columns:
                    val = monthly.loc[year, month]
                    if pd.notna(val):
                        color = "#00e676" if val > 0 else "#ff1744"
                        bg = f"rgba(0,230,118,{min(abs(val)*5, 0.3):.2f})" if val > 0 else f"rgba(255,23,68,{min(abs(val)*5, 0.3):.2f})"
                        html += f'<td style="background:{bg};color:{color}">{val*100:.1f}%</td>'
                        annual += val
                    else:
                        html += "<td>-</td>"
                else:
                    html += "<td>-</td>"
            color = "#00e676" if annual > 0 else "#ff1744"
            html += f'<td style="color:{color}"><strong>{annual*100:.1f}%</strong></td></tr>'

        html += "</table>"
        return html


# ══════════════════════════════════════════════════════════════════════════════
# Self-contained Jinja2-based report (new)
# ══════════════════════════════════════════════════════════════════════════════

_JINJA_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report &mdash; {{ title }}</title>
<style>
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d;
          --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Oxygen, Ubuntu, sans-serif; background: var(--bg); color: var(--text);
         padding: 2rem; line-height: 1.6; }
  h1 { color: var(--accent); margin-bottom: .25rem; }
  h2 { color: var(--accent); margin: 1.5rem 0 .75rem; font-size: 1.2rem; }
  .subtitle { color: var(--muted); margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 1rem; margin-bottom: 2rem; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 1rem; }
  .card .label { font-size: .75rem; color: var(--muted); text-transform: uppercase; }
  .card .value { font-size: 1.4rem; font-weight: 600; margin-top: .25rem; }
  .positive { color: #3fb950; }
  .negative { color: #f85149; }
  img.chart { width: 100%; max-width: 900px; border-radius: 8px;
              border: 1px solid var(--border); margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }
  th, td { padding: .5rem .75rem; text-align: right; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-size: .75rem; text-transform: uppercase; }
  td:first-child, th:first-child { text-align: left; }
  footer { margin-top: 3rem; color: var(--muted); font-size: .8rem; text-align: center; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="subtitle">Generated {{ generated_at }} | {{ n_bars }} bars | {{ date_range }}</p>

<h2>Key Metrics</h2>
<div class="grid">
{% for m in metric_cards %}
  <div class="card">
    <div class="label">{{ m.label }}</div>
    <div class="value {{ m.css }}">{{ m.value }}</div>
  </div>
{% endfor %}
</div>

<h2>Equity Curve</h2>
<img class="chart" src="data:image/png;base64,{{ equity_chart }}" alt="equity curve">

<h2>Drawdown</h2>
<img class="chart" src="data:image/png;base64,{{ drawdown_chart }}" alt="drawdown">

{% if monthly_heatmap %}
<h2>Monthly Returns</h2>
<img class="chart" src="data:image/png;base64,{{ monthly_heatmap }}" alt="monthly returns">
{% endif %}

{% if trade_dist_chart %}
<h2>Trade Return Distribution</h2>
<img class="chart" src="data:image/png;base64,{{ trade_dist_chart }}" alt="trade distribution">
{% endif %}

<h2>Metrics Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{% for row in metrics_table %}
<tr><td>{{ row.metric }}</td><td class="{{ row.css }}">{{ row.value }}</td></tr>
{% endfor %}
</table>

<footer>hedgefund backtest report</footer>
</body>
</html>
"""


# ── Chart helpers (base64-encoded) ────────────────────────────────────────────


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _equity_chart_b64(
    equity: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    x = dates if dates is not None else np.arange(len(equity))
    ax.plot(x, equity, color="#58a6ff", linewidth=1.2)
    ax.fill_between(x, equity, equity.min(), alpha=0.08, color="#58a6ff")
    ax.set_title("Equity Curve", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Portfolio Value", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _drawdown_chart_b64(
    dd: np.ndarray,
    dates: pd.DatetimeIndex | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    x = dates if dates is not None else np.arange(len(dd))
    ax.fill_between(x, 0, -dd, color="#f85149", alpha=0.5)
    ax.plot(x, -dd, color="#f85149", linewidth=0.8)
    ax.set_title("Drawdown", color="#c9d1d9", fontsize=12)
    ax.set_ylabel("Drawdown %", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15)
    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _monthly_heatmap_b64(monthly_df: pd.DataFrame) -> str | None:
    if monthly_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3, len(monthly_df) * 0.5)), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    data = monthly_df.values * 100
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        color="#8b949e",
    )
    ax.set_yticks(range(len(monthly_df)))
    ax.set_yticklabels(monthly_df.index.astype(str), color="#8b949e")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 5 else "#c9d1d9")
    ax.set_title("Monthly Returns (%)", color="#c9d1d9", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.6)
    return _fig_to_base64(fig)


def _trade_distribution_b64(trade_returns: np.ndarray) -> str | None:
    if len(trade_returns) < 2:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.hist(trade_returns * 100, bins=50, color="#58a6ff", alpha=0.7, edgecolor="#30363d")
    ax.axvline(0, color="#f85149", linestyle="--", linewidth=1)
    ax.set_title("Trade Return Distribution", color="#c9d1d9", fontsize=12)
    ax.set_xlabel("Return (%)", color="#8b949e")
    ax.set_ylabel("Frequency", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, alpha=0.15)
    return _fig_to_base64(fig)


# ── Self-contained Jinja2 report ──────────────────────────────────────────────


class SelfContainedReport:
    """Generate a single portable HTML backtest report.

    All charts are embedded as base64 data URIs so the report is a
    single file with no external dependencies.  Uses Jinja2 for
    templating and matplotlib for chart rendering.
    """

    def __init__(self, title: str = "Backtest Report") -> None:
        self.title = title
        self._log = log.bind(component="self_contained_report")

    def generate(
        self,
        metrics: BacktestMetrics,
        equity_curve: np.ndarray | pd.Series,
        *,
        dates: pd.DatetimeIndex | None = None,
        trade_returns: np.ndarray | None = None,
        monthly_returns_df: pd.DataFrame | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """Build the HTML report and optionally write it to disk.

        Args:
            metrics: Computed backtest metrics.
            equity_curve: Portfolio value over time.
            dates: Timestamps corresponding to the equity curve.
            trade_returns: Per-trade returns for the distribution chart.
            monthly_returns_df: Pivoted DataFrame (year x month) of returns.
            output_path: If given, write the HTML file here.

        Returns:
            The rendered HTML string.
        """
        from jinja2 import Template

        eq = np.asarray(equity_curve, dtype=np.float64).ravel()

        # Drawdown series.
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1.0)

        metric_cards = self._build_metric_cards(metrics)
        metrics_table = self._build_metrics_table(metrics)

        equity_b64 = _equity_chart_b64(eq, dates)
        drawdown_b64 = _drawdown_chart_b64(dd, dates)
        heatmap_b64 = _monthly_heatmap_b64(monthly_returns_df) if monthly_returns_df is not None else None
        trade_dist_b64 = _trade_distribution_b64(trade_returns) if trade_returns is not None else None

        date_range = ""
        if dates is not None and len(dates) >= 2:
            date_range = f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}"

        template = Template(_JINJA_TEMPLATE)
        html = template.render(
            title=self.title,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            n_bars=len(eq),
            date_range=date_range,
            metric_cards=metric_cards,
            equity_chart=equity_b64,
            drawdown_chart=drawdown_b64,
            monthly_heatmap=heatmap_b64,
            trade_dist_chart=trade_dist_b64,
            metrics_table=metrics_table,
        )

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html)
            self._log.info("report_written", path=str(path))

        return html

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_metric_cards(m: BacktestMetrics) -> list[dict[str, str]]:
        def _fmt_pct(v: float) -> tuple[str, str]:
            css = "positive" if v >= 0 else "negative"
            return f"{v:+.2%}", css

        def _fmt_ratio(v: float) -> tuple[str, str]:
            css = "positive" if v >= 0 else "negative"
            return f"{v:.2f}", css

        cards: list[dict[str, str]] = []
        val, css = _fmt_pct(m.total_return)
        cards.append({"label": "Total Return", "value": val, "css": css})
        val, css = _fmt_pct(m.annualized_return)
        cards.append({"label": "Ann. Return", "value": val, "css": css})
        val, css = _fmt_ratio(m.sharpe_ratio)
        cards.append({"label": "Sharpe Ratio", "value": val, "css": css})
        val, css = _fmt_ratio(m.sortino_ratio)
        cards.append({"label": "Sortino Ratio", "value": val, "css": css})
        val, css = _fmt_pct(-abs(m.max_drawdown))
        cards.append({"label": "Max Drawdown", "value": val, "css": css})
        cards.append({"label": "Win Rate", "value": f"{m.win_rate:.1%}", "css": ""})
        val, css = _fmt_ratio(m.profit_factor)
        cards.append({"label": "Profit Factor", "value": val, "css": css})
        cards.append({"label": "Total Trades", "value": str(m.total_trades), "css": ""})
        return cards

    @staticmethod
    def _build_metrics_table(m: BacktestMetrics) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []

        def _add(metric: str, value: str, css: str = "") -> None:
            rows.append({"metric": metric, "value": value, "css": css})

        _add("Total Return", f"{m.total_return:+.2%}", "positive" if m.total_return >= 0 else "negative")
        _add("Annualized Return", f"{m.annualized_return:+.2%}", "positive" if m.annualized_return >= 0 else "negative")
        _add("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
        _add("Sortino Ratio", f"{m.sortino_ratio:.3f}")
        _add("Calmar Ratio", f"{m.calmar_ratio:.3f}")
        _add("Max Drawdown", f"{m.max_drawdown:.2%}", "negative")
        _add("Win Rate", f"{m.win_rate:.1%}")
        _add("Profit Factor", f"{m.profit_factor:.2f}")
        _add("Expectancy", f"{m.expectancy:.4f}")
        _add("Total Trades", str(m.total_trades))
        _add("Winning Trades", str(m.winning_trades))
        _add("Losing Trades", str(m.losing_trades))
        _add("Average Win", f"{m.avg_win:.4f}", "positive")
        _add("Average Loss", f"{m.avg_loss:.4f}", "negative")
        return rows
