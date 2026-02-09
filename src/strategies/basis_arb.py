"""Basis Arbitrage Strategy - Signal Generation

This strategy generates signals for crypto basis arbitrage based on funding rate forecasts.

**Model**:
- Forecast = 8 × (mean funding over last 4 days)
- Hurdle = 15% annual → 0.00329 for 8 days
- Select top 10 tickers exceeding the hurdle

**Output**:
- Daily signal files indicating which positions to hold
- Tracks entry/exit signals for execution layer
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.data_helper import save_df_to_csv, load_df_from_csv


class BasisArbSignalGenerator:
    """Generate signals for basis arbitrage strategy based on funding rate forecasts."""
    
    def __init__(
        self,
        lookback_window: int = 4,
        forward_sum_window: int = 8,
        hurdle_annual: float = 0.2,
        top_n: int = 10,
        min_hold_days: int = 8,
        funding_4d_folder: str = "data/funding_4d",
        signals_folder: str = "data/signals",
        project_root: Optional[str] = None
    ):
        """Initialize the signal generator.
        
        Args:
            lookback_window: Days to average for forecast (default: 4)
            forward_sum_window: Days to forecast ahead (default: 8)
            hurdle_annual: Annual hurdle rate (default: 0.2 = 20%)
            top_n: Number of top positions to hold (default: 10)
            min_hold_days: Minimum intended hold days for reporting/logic (default: 8)
            funding_4d_folder: Folder containing 4-day funding rate files
            signals_folder: Folder to save signal files
            project_root: Root directory of the project
        """
        self.lookback_window = lookback_window
        self.forward_sum_window = forward_sum_window
        self.hurdle_annual = hurdle_annual
        self.top_n = top_n
        self.min_hold_days = min_hold_days
        
        # Calculate 8-day hurdle from annual
        self.hurdle_8day = hurdle_annual * (forward_sum_window / 365)
        
        # Set up paths
        if project_root is None:
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = project_root
        self.funding_4d_folder = os.path.join(project_root, funding_4d_folder)
        self.signals_folder = os.path.join(project_root, signals_folder)
        
        # Create signals folder if it doesn't exist
        os.makedirs(self.signals_folder, exist_ok=True)
        
    def get_latest_funding_4d_file(self, as_of_date: Optional[datetime] = None) -> Optional[str]:
        """Get the most recent 4-day funding rate file.
        
        Args:
            as_of_date: Reference date (default: today)
            
        Returns:
            Path to the latest funding_rate file or None if not found
        """
        if as_of_date is None:
            as_of_date = datetime.now()
            
        # List all funding_rate files
        if not os.path.exists(self.funding_4d_folder):
            print(f"Folder not found: {self.funding_4d_folder}")
            return None
            
        files = [f for f in os.listdir(self.funding_4d_folder) 
                if f.startswith('funding_rate_') and f.endswith('.csv')]
        
        if not files:
            print(f"No funding rate files found in {self.funding_4d_folder}")
            return None
        
        # Sort by date in filename and get latest
        files.sort(reverse=True)
        latest_file = files[0]
        
        return os.path.join(self.funding_4d_folder, latest_file)
    
    def load_funding_4d_data(self, file_path: str) -> pd.DataFrame:
        """Load 4-day averaged funding rate data.
        
        Args:
            file_path: Path to the funding rate CSV file
            
        Returns:
            DataFrame with columns: TICKER, EXCHANGE, FUNDING_RATE
        """
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ['TICKER', 'EXCHANGE', 'FUNDING_RATE']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"File missing required columns: {required_cols}")
        
        return df
    
    def calculate_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 8-day forecast based on 4-day average funding rate.
        
        Args:
            df: DataFrame with TICKER, EXCHANGE, FUNDING_RATE columns
            
        Returns:
            DataFrame with additional FORECAST column
        """
        df = df.copy()
        
        # Forecast = 8 × (mean funding over last 4 days)
        # The input is already the 4-day mean, so just multiply by 8
        df['FORECAST'] = df['FUNDING_RATE'] * self.forward_sum_window
        
        return df
    
    def apply_hurdle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter tickers that exceed the hurdle rate.
        
        Args:
            df: DataFrame with FORECAST column
            
        Returns:
            Filtered DataFrame with tickers exceeding hurdle
        """
        df = df.copy()
        
        # Keep only tickers where forecast exceeds hurdle
        df_filtered = df[df['FORECAST'] >= self.hurdle_8day].copy()
        
        print(f"Tickers exceeding hurdle ({self.hurdle_8day:.6f}): {len(df_filtered)}")
        
        return df_filtered
    
    def select_top_n(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top N tickers by forecast.
        
        Args:
            df: DataFrame with FORECAST column
            
        Returns:
            DataFrame with top N tickers
        """
        df = df.copy()
        
        # Sort by forecast descending and take top N
        df_sorted = df.sort_values('FORECAST', ascending=False)
        df_top = df_sorted.head(self.top_n).copy()
        
        print(f"Selected top {self.top_n} tickers")
        
        return df_top
    
    def _prepare_previous_signals(
        self, 
        previous_signals: Optional[pd.DataFrame], 
        as_of_date: datetime
    ) -> pd.DataFrame:
        """Prepare previous signals with ENTRY_DATE and HOLD_DAYS."""
        if previous_signals is None or len(previous_signals) == 0:
            return pd.DataFrame(columns=['TICKER', 'EXCHANGE', 'ENTRY_DATE', 'HOLD_DAYS'])
        
        prev = previous_signals.copy()
        
        # Ensure ENTRY_DATE exists
        if 'ENTRY_DATE' not in prev.columns:
            if 'SIGNAL_DATE' in prev.columns:
                prev['ENTRY_DATE'] = prev['SIGNAL_DATE']
            else:
                prev['ENTRY_DATE'] = as_of_date.strftime('%Y-%m-%d')
        
        prev['ENTRY_DATE'] = pd.to_datetime(prev['ENTRY_DATE'])

        as_of_ts = pd.Timestamp(as_of_date).normalize()
        prev['HOLD_DAYS'] = (as_of_ts - prev['ENTRY_DATE']).dt.days
        
        return prev[['TICKER', 'EXCHANGE', 'ENTRY_DATE', 'HOLD_DAYS']]
    
    def generate_signals(
        self, 
        as_of_date: Optional[datetime] = None,
        previous_signals: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate trading signals for the given date.
        
        Args:
            as_of_date: Date to generate signals for (default: today)
            previous_signals: Previous signal DataFrame (optional)
            
        Returns:
            DataFrame with signals (TICKER, EXCHANGE, FORECAST, SIGNAL)
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        print(f"\n{'='*60}")
        print(f"Generating signals for: {as_of_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Load latest 4-day funding data
        funding_file = self.get_latest_funding_4d_file(as_of_date)
        if funding_file is None:
            raise FileNotFoundError("No funding rate file found")
        
        print(f"Loading data from: {os.path.basename(funding_file)}")
        df = self.load_funding_4d_data(funding_file)
        print(f"Total tickers loaded: {len(df)}")
        
        # Calculate forecast
        df = self.calculate_forecast(df)
        
        # Apply hurdle filter (current universe above hurdle)
        df_above = self.apply_hurdle(df)
        
        # Prepare previous positions
        prev_prepared = self._prepare_previous_signals(previous_signals, as_of_date)
        
        if len(prev_prepared) == 0:
            # No previous positions → just take top N above hurdle
            df_selected = self.select_top_n(df_above)
            df_selected['ENTRY_DATE'] = as_of_date.strftime('%Y-%m-%d')
            df_selected['HOLD_DAYS'] = 0
            df_selected['KEEP_REASON'] = 'NEW_TOP'
        else:
            # Keep previous positions if still above hurdle
            prev_keep = prev_prepared.merge(
                df_above[['TICKER', 'EXCHANGE', 'FUNDING_RATE', 'FORECAST']],
                on=['TICKER', 'EXCHANGE'],
                how='inner'
            )
            prev_keep['KEEP_REASON'] = 'ABOVE_HURDLE'
            
            # Remaining slots for new entries
            remaining_slots = self.top_n - len(prev_keep)
            
            # Add new candidates only if there is capacity
            if remaining_slots > 0:
                prev_set = set(zip(prev_keep['TICKER'], prev_keep['EXCHANGE']))
                new_candidates = df_above[
                    ~df_above.set_index(['TICKER', 'EXCHANGE']).index.isin(prev_set)
                ].sort_values('FORECAST', ascending=False).head(remaining_slots).copy()
                
                new_candidates['ENTRY_DATE'] = as_of_date.strftime('%Y-%m-%d')
                new_candidates['HOLD_DAYS'] = 0
                new_candidates['KEEP_REASON'] = 'NEW_TOP'
                
                df_selected = pd.concat([prev_keep, new_candidates], ignore_index=True)
            else:
                df_selected = prev_keep.copy()
                if remaining_slots < 0:
                    print(
                        f"Warning: {len(prev_keep)} existing positions above hurdle exceed top_n={self.top_n}. "
                        f"Keeping all existing positions."
                    )
        
        # Add signal column (1 = LONG position)
        df_selected['SIGNAL'] = 1
        
        # Add metadata
        df_selected['SIGNAL_DATE'] = as_of_date.strftime('%Y-%m-%d')
        df_selected['HURDLE_8DAY'] = self.hurdle_8day
        df_selected['MIN_HOLD_DAYS'] = self.min_hold_days
        
        # Reorder columns
        df_selected = df_selected[([
            'TICKER', 'EXCHANGE', 'FUNDING_RATE', 'FORECAST',
            'SIGNAL', 'SIGNAL_DATE', 'ENTRY_DATE', 'HOLD_DAYS',
            'HURDLE_8DAY', 'MIN_HOLD_DAYS', 'KEEP_REASON'
        ])]
        
        return df_selected
    
    def save_signals(self, df_signals: pd.DataFrame, as_of_date: Optional[datetime] = None) -> str:
        """Save signals to CSV file.
        
        Args:
            df_signals: DataFrame with signals
            as_of_date: Date for the signals (default: today)
            
        Returns:
            Path to the saved file
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Generate filename
        date_str = as_of_date.strftime('%Y%m%d')
        time_str = as_of_date.strftime('%H%M%S')
        filename = f"basis_arb_signals_{date_str}_{time_str}.csv"
        file_path = os.path.join(self.signals_folder, filename)
        
        # Save to CSV
        df_signals.to_csv(file_path, index=False)
        print(f"\nSignals saved to: {file_path}")
        
        return file_path
    
    def load_previous_signals(self) -> Optional[pd.DataFrame]:
        """Load the most recent previous signals file.
        
        Returns:
            DataFrame with previous signals or None if not found
        """
        if not os.path.exists(self.signals_folder):
            return None
        
        files = [f for f in os.listdir(self.signals_folder) 
                if f.startswith('basis_arb_signals_') and f.endswith('.csv')]
        
        if not files:
            return None
        
        # Sort by filename (contains date) and get second latest
        files.sort(reverse=True)
        if len(files) < 2:
            return None
        
        prev_file = files[1]
        prev_path = os.path.join(self.signals_folder, prev_file)
        
        return pd.read_csv(prev_path)
    
    def compare_signals(
        self, 
        current_signals: pd.DataFrame, 
        previous_signals: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Compare current signals with previous to identify entries and exits.
        
        Args:
            current_signals: Current signal DataFrame
            previous_signals: Previous signal DataFrame (optional)
            
        Returns:
            Dictionary with 'entries', 'exits', and 'holds' lists
            Each list contains tuples of (TICKER, EXCHANGE)
        """
        current_positions = set(
            zip(current_signals['TICKER'], current_signals['EXCHANGE'])
        )
        
        if previous_signals is None or len(previous_signals) == 0:
            return {
                'entries': list(current_positions),
                'exits': [],
                'holds': []
            }
        
        previous_positions = set(
            zip(previous_signals['TICKER'], previous_signals['EXCHANGE'])
        )
        
        entries = list(current_positions - previous_positions)
        exits = list(previous_positions - current_positions)
        holds = list(current_positions & previous_positions)
        
        return {
            'entries': entries,
            'exits': exits,
            'holds': holds
        }
    
    def print_signal_summary(
        self, 
        df_signals: pd.DataFrame,
        comparison: Optional[Dict[str, List[Tuple[str, str]]]] = None
    ):
        """Print a summary of the generated signals.
        
        Args:
            df_signals: DataFrame with signals
            comparison: Comparison with previous signals (optional)
        """
        print(f"\n{'='*60}")
        print("SIGNAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total positions: {len(df_signals)}")
        print(f"Hurdle (8-day): {self.hurdle_8day:.6f} ({self.hurdle_annual:.1%} annual)")
        
        if len(df_signals) > 0:

            print(f"\nPositions:")
            for idx, row in df_signals.head(10).iterrows():
                print(f"  {row['TICKER']:20s} ({row['EXCHANGE']:12s}): {row['FORECAST']:.6f}")
        
        if comparison:
            print(f"\n{'='*60}")
            print("POSITION CHANGES")
            print(f"{'='*60}")
            print(f"New entries: {len(comparison['entries'])}")
            if comparison['entries']:
                for ticker, exchange in comparison['entries'][:5]:
                    print(f"  + {ticker} ({exchange})")
                if len(comparison['entries']) > 5:
                    print(f"  ... and {len(comparison['entries']) - 5} more")
            
            print(f"\nExits: {len(comparison['exits'])}")
            if comparison['exits']:
                for ticker, exchange in comparison['exits'][:5]:
                    print(f"  - {ticker} ({exchange})")
                if len(comparison['exits']) > 5:
                    print(f"  ... and {len(comparison['exits']) - 5} more")
            
            print(f"Holds: {len(comparison['holds'])}")
        
        print(f"{'='*60}\n")
    
    def run(self, as_of_date: Optional[datetime] = None) -> str:
        """Run the complete signal generation pipeline.
        
        Args:
            as_of_date: Date to generate signals for (default: today)
            
        Returns:
            Path to the saved signals file
        """
        # Load previous signals for comparison/holding logic
        df_previous = self.load_previous_signals()
        
        # Generate signals (keep above-hurdle positions)
        df_signals = self.generate_signals(as_of_date, df_previous)
        
        # Compare signals
        comparison = self.compare_signals(df_signals, df_previous)
        
        # Print summary
        self.print_signal_summary(df_signals, comparison)
        
        # Save signals
        file_path = self.save_signals(df_signals, as_of_date)
        
        return file_path


def main():
    """Main entry point for signal generation."""

    # Load configuration from JSON file
    config_path = os.path.join(
        Path(__file__).parent.parent.parent,
        "config",
        "basis_arb_config.json"
    )

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
    else:
        print(f"Config file not found: {config_path}")
        print("Using default parameters")
        config = {}

    # Initialize generator
    generator = BasisArbSignalGenerator(
        lookback_window=config.get('parameters', {}).get('lookback_window', 4),
        forward_sum_window=config.get('parameters', {}).get('forward_sum_window', 8),
        hurdle_annual=config.get('parameters', {}).get('hurdle_annual', 0.20),
        top_n=config.get('parameters', {}).get('top_n', 10),
        min_hold_days=config.get('parameters', {}).get('min_hold_days', 8),
        funding_4d_folder=config.get('data_paths', {}).get('funding_4d_folder', 'data/funding_4d'),
        signals_folder=config.get('data_paths', {}).get('signals_folder', 'data/signals')
    )
    
    # Run signal generation
    signals_file = generator.run()
    
    print(f"\nSignal generation complete!")


if __name__ == "__main__":
    main()
