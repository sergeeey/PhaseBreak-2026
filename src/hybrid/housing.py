"""PhaseBreak Hybrid 2.0 — Housing Market Module.

High-frequency housing data loader + deseasonalization + LPPLS analysis.

Data Sources:
- Zillow ZHVI (monthly): files.zillowstatic.com/research/public_csvs/zhvi/
- FRED Case-Shiller (monthly): via fredapi
- Redfin (weekly): manual download from redfin.com/news/data-center/

Usage:
    from src.housing.loader import HousingDataLoader
    loader = HousingDataLoader()
    data = loader.load_zhvi_metro("Austin, TX")
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass
class HousingEpisode:
    """A housing market episode with LPPLS analysis."""

    region: str
    region_type: str  # metro, city, state, zip
    start_date: str
    end_date: str
    n_points: int
    deseasonalized: bool

    # LPPLS results
    tc_estimate: float | None = None
    tc_date_str: str | None = None
    quality_score: float = 0.0
    r_squared: float = 0.0
    m_param: float | None = None
    omega_param: float | None = None
    verdict: str = "UNKNOWN"  # BUBBLE, POSSIBLE, NO_BUBBLE

    # Raw data
    dates: list[str] = field(default_factory=list)
    prices: list[float] = field(default_factory=list)
    prices_deseasonalized: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "region_type": self.region_type,
            "period": f"{self.start_date} to {self.end_date}",
            "n_points": self.n_points,
            "deseasonalized": self.deseasonalized,
            "tc_estimate": self.tc_estimate,
            "tc_date_str": self.tc_date_str,
            "quality_score": round(self.quality_score, 4),
            "r_squared": round(self.r_squared, 4),
            "m_param": self.m_param,
            "omega_param": self.omega_param,
            "verdict": self.verdict,
        }


# ─── Zillow ZHVI Direct URLs ──────────────────────────────────────────
# Source: https://www.zillow.com/research/data/
# Updated monthly on the 16th
# These are direct CSV URLs — no API key needed

ZHVI_URLS = {
    "metro": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "city": "https://files.zillowstatic.com/research/public_csvs/zhvi/City_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "state": "https://files.zillowstatic.com/research/public_csvs/zhvi/State_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "county": "https://files.zillowstatic.com/research/public_csvs/zhvi/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "zip": "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
}

# MSA Name Mapping — Zillow uses short names like "Austin, TX"
MSA_MAPPING = {
    "Austin": "Austin, TX",
    "Miami": "Miami, FL",
    "Phoenix": "Phoenix, AZ",
    "Las Vegas": "Las Vegas, NV",
    "Boise": "Boise City, ID",
    "Denver": "Denver, CO",
    "Seattle": "Seattle, WA",
    "San Francisco": "San Francisco, CA",
    "Los Angeles": "Los Angeles, CA",
    "New York": "New York, NY",
    "Chicago": "Chicago, IL",
    "Dallas": "Dallas, TX",
    "Atlanta": "Atlanta, GA",
    "Tampa": "Tampa, FL",
    "Orlando": "Orlando, FL",
    "Charlotte": "Charlotte, NC",
    "Nashville": "Nashville, TN",
    "Salt Lake City": "Salt Lake City, UT",
    "San Diego": "San Diego, CA",
    "Portland": "Portland, OR",
    "Houston": "Houston, TX",
    "Boston": "Boston, MA",
    "Washington": "Washington, DC",
    "Philadelphia": "Philadelphia, PA",
    "Detroit": "Detroit, MI",
    "Minneapolis": "Minneapolis, MN",
    "Riverside": "Riverside, CA",
}

# Hot housing markets to monitor (PhaseBreak watchlist)
HOT_MARKETS = [
    "Austin", "Miami", "Phoenix", "Las Vegas", "Boise",
    "Denver", "Tampa", "Orlando", "Nashville", "Salt Lake City",
]


class HousingDataLoader:
    """Load and prepare housing data for LPPLS analysis.

    Features:
    - Direct Zillow ZHVI CSV download (no API key)
    - Automatic deseasonalization (critical for housing!)
    - MSA name mapping
    - Data validation and interpolation
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Args:
            cache_dir: Directory to cache downloaded CSVs (default: data/housing_cache/)
        """
        self.cache_dir = Path(cache_dir or "data/housing_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_dfs: dict[str, pd.DataFrame] = {}

    def load_zhvi_metro(
        self,
        region: str,
        years_back: int = 5,
        deseasonalize: bool = True,
    ) -> HousingEpisode:
        """Load ZHVI data for a metro area and prepare for LPPLS.

        Args:
            region: Metro area name (short or full MSA name)
            years_back: How many years of data to load
            deseasonalize: Apply seasonal decomposition

        Returns:
            HousingEpisode with prepared data
        """
        # Resolve MSA name
        msa_name = self._resolve_msa_name(region)

        # Load data
        df = self._load_zhvi_csv("metro")

        # Filter to region
        region_df = self._filter_region(df, msa_name, "metro")

        if region_df.empty:
            raise ValueError(
                f"No ZHVI data found for '{region}' (tried '{msa_name}'). "
                f"Available regions: {list(df['RegionName'].unique()[:10])}..."
            )

        # Prepare time series
        return self._prepare_episode(
            region=msa_name,
            region_type="metro",
            df=region_df,
            years_back=years_back,
            deseasonalize=deseasonalize,
        )

    def load_zhvi_city(
        self,
        city: str,
        years_back: int = 5,
        deseasonalize: bool = True,
    ) -> HousingEpisode:
        """Load ZHVI data for a city."""
        df = self._load_zhvi_csv("city")
        region_df = self._filter_region(df, city, "city")

        if region_df.empty:
            raise ValueError(f"No ZHVI data found for city '{city}'")

        return self._prepare_episode(
            region=city,
            region_type="city",
            df=region_df,
            years_back=years_back,
            deseasonalize=deseasonalize,
        )

    def scan_hot_markets(
        self,
        years_back: int = 5,
        deseasonalize: bool = True,
    ) -> list[HousingEpisode]:
        """Scan all hot housing markets for LPPLS signals.

        Returns:
            List of HousingEpisode for each market
        """
        results = []
        df = self._load_zhvi_csv("metro")

        for market in HOT_MARKETS:
            try:
                msa_name = self._resolve_msa_name(market)
                region_df = self._filter_region(df, msa_name, "metro")

                if region_df.empty:
                    log.warning("no_data_for_market", market=market, msa=msa_name)
                    continue

                episode = self._prepare_episode(
                    region=msa_name,
                    region_type="metro",
                    df=region_df,
                    years_back=years_back,
                    deseasonalize=deseasonalize,
                )
                results.append(episode)
                log.info(
                    "market_scanned",
                    market=market,
                    n_points=episode.n_points,
                    latest_price=episode.prices[-1] if episode.prices else None,
                )

            except Exception as e:
                log.error("market_scan_failed", market=market, error=str(e))

        return results

    # ─── Internal Methods ──────────────────────────────────────────────

    def _resolve_msa_name(self, region: str) -> str:
        """Resolve short region name to full MSA name."""
        # Check exact match first
        if region in MSA_MAPPING:
            return MSA_MAPPING[region]

        # Check if it's already a full MSA name
        if any(region in v for v in MSA_MAPPING.values()):
            return region

        # Try partial match
        for short, full in MSA_MAPPING.items():
            if short.lower() in region.lower():
                return full

        # Return as-is (might work for city-level data)
        return region

    def _load_zhvi_csv(self, level: str) -> pd.DataFrame:
        """Load ZHVI CSV from Zillow (with caching)."""
        cache_file = self.cache_dir / f"zhvi_{level}.csv"

        # Check cache
        if level in self._cached_dfs:
            return self._cached_dfs[level]

        if cache_file.exists():
            log.info("loading_cached_zhvi", level=level, path=str(cache_file))
            df = pd.read_csv(cache_file)
            self._cached_dfs[level] = df
            return df

        # Download from Zillow
        url = ZHVI_URLS.get(level)
        if not url:
            raise ValueError(f"Unknown ZHVI level: {level}")

        log.info("downloading_zhvi", level=level, url=url)

        try:
            import httpx

            resp = httpx.get(url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()

            df = pd.read_csv(io.BytesIO(resp.content))
            df.to_csv(cache_file, index=False)
            self._cached_dfs[level] = df

            log.info(
                "zhvi_downloaded",
                level=level,
                rows=len(df),
                cols=len(df.columns),
                cached=str(cache_file),
            )
            return df

        except Exception as e:
            log.error("zhvi_download_failed", level=level, url=url, error=str(e))

            # Fallback: try to use cached file if exists
            if cache_file.exists():
                log.warning("using_stale_cache", path=str(cache_file))
                df = pd.read_csv(cache_file)
                self._cached_dfs[level] = df
                return df

            raise

    def _filter_region(
        self, df: pd.DataFrame, region_name: str, level: str
    ) -> pd.DataFrame:
        """Filter ZHVI DataFrame to specific region."""
        # ZHVI format: RegionName, RegionID, SizeRank, then date columns
        if level == "metro":
            # Try exact match first
            exact_match = df[df["RegionName"] == region_name]
            if not exact_match.empty:
                return exact_match

            # Try contains match
            mask = df["RegionName"].str.contains(region_name, case=False, na=False)
            filtered = df[mask]

            # If multiple matches, take the first (most likely the main MSA)
            if len(filtered) > 1:
                log.warning(
                    "multiple_msa_matches",
                    region=region_name,
                    matches=filtered["RegionName"].tolist()[:5],
                )
                return filtered.iloc[:1]

            return filtered

        elif level == "city":
            mask = df["RegionName"].str.contains(region_name, case=False, na=False)
            return df[mask]

        else:
            mask = df["RegionName"].str.contains(region_name, case=False, na=False)
            return df[mask]

    def _prepare_episode(
        self,
        region: str,
        region_type: str,
        df: pd.DataFrame,
        years_back: int,
        deseasonalize: bool,
    ) -> HousingEpisode:
        """Prepare housing data for LPPLS analysis."""
        # Extract date columns (format: "2024-01-31", "2024-02-29", etc.)
        date_cols = [c for c in df.columns if c.startswith("20")]

        if not date_cols:
            raise ValueError(f"No date columns found in ZHVI data for {region}")

        # Get price series
        prices_raw = df[date_cols].iloc[0].values.astype(float)

        # Parse dates
        dates = pd.to_datetime(date_cols)

        # Create DataFrame for processing
        ts_df = pd.DataFrame({"date": dates, "price": prices_raw})
        ts_df = ts_df.dropna()
        ts_df = ts_df.sort_values("date")

        log.info(
            "before_filter",
            region=region,
            n_points_before=len(ts_df),
            first_date=str(ts_df["date"].iloc[0]),
            last_date=str(ts_df["date"].iloc[-1]),
        )

        # Filter to recent years
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years_back)
        ts_df = ts_df[ts_df["date"] >= cutoff]

        log.info(
            "after_filter",
            region=region,
            n_points_after=len(ts_df),
            cutoff=str(cutoff),
        )

        if len(ts_df) < 24:
            raise ValueError(
                f"Too few data points for {region}: {len(ts_df)} (need >= 24)"
            )

        # Interpolate any missing months (ZHVI is monthly but may have gaps)
        ts_df = ts_df.set_index("date")
        ts_df = ts_df.asfreq("ME")  # Month End frequency (ZHVI uses end-of-month)
        ts_df["price"] = ts_df["price"].interpolate(method="linear")
        ts_df = ts_df.dropna()
        ts_df = ts_df.reset_index()

        # Deseasonalize (CRITICAL for housing!)
        prices_final = ts_df["price"].values.copy()
        prices_deseasonalized = None

        if deseasonalize and len(ts_df) >= 24:
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Need at least 2 full seasonal cycles (24 months for monthly data)
                result = seasonal_decompose(
                    ts_df["price"],
                    model="multiplicative",
                    period=12,  # Annual seasonality
                    extrapolate_trend="freq",
                )

                # Deseasonalized = trend + resid (remove seasonal component)
                prices_deseasonalized = (result.trend + result.resid).values
                prices_deseasonalized = np.nan_to_num(
                    prices_deseasonalized, nan=np.nanmean(prices_deseasonalized)
                )

                log.info(
                    "deseasonalized",
                    region=region,
                    seasonal_strength=float(
                        result.seasonal.std() / result.trend.std()
                        if result.trend.std() > 0
                        else 0
                    ),
                )

            except Exception as e:
                log.warning("deseasonalization_failed", region=region, error=str(e))
                prices_deseasonalized = prices_final.copy()
        else:
            prices_deseasonalized = prices_final.copy()

        # Build episode
        episode = HousingEpisode(
            region=region,
            region_type=region_type,
            start_date=str(ts_df["date"].iloc[0].date()),
            end_date=str(ts_df["date"].iloc[-1].date()),
            n_points=len(ts_df),
            deseasonalized=deseasonalize,
            dates=[str(d.date()) for d in ts_df["date"]],
            prices=prices_final.tolist(),
            prices_deseasonalized=prices_deseasonalized.tolist(),
        )

        log.info(
            "episode_prepared",
            region=region,
            n_points=episode.n_points,
            period=f"{episode.start_date} to {episode.end_date}",
            latest_price=episode.prices[-1],
        )

        return episode

    def clear_cache(self) -> None:
        """Clear cached CSV files."""
        for f in self.cache_dir.glob("*.csv"):
            f.unlink()
        self._cached_dfs.clear()
        log.info("cache_cleared", path=str(self.cache_dir))
