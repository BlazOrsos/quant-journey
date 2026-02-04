"""Crypto Basis Arbitrage Pipeline

This pipeline orchestrates the complete workflow for generating basis arbitrage signals:
1. Fetch current symbols from exchanges (Binance, Bybit, Hyperliquid)
2. Download funding rate data for all symbols
3. Generate trading signals based on funding rate forecasts

CRITICAL: Steps execute sequentially, one after another, in this exact order.

Usage:
    python -m src.pipelines.crypto_basis_arbitrage_pipeline
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pipeline():
    """Execute the complete basis arbitrage pipeline.
    
    This function runs three jobs in strict sequential order:
    1. symbols_data_job - Fetch current trading symbols
    2. funding_data_job - Download and process funding rate data
    3. basis_arb - Generate trading signals
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "CRYPTO BASIS ARBITRAGE PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    # Track overall success
    pipeline_success = True
    
    # ========================================================================
    # STEP 1: Fetch Current Symbols
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1/3: Fetching Current Trading Symbols")
    print("=" * 80 + "\n")
    
    try:
        from jobs.symbols_data_job import main as symbols_job_main
        symbols_job_main()
        print("\n✓ Step 1 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Step 1 FAILED: {str(e)}\n")
        print("Pipeline aborted due to symbols fetch failure")
        pipeline_success = False
        return pipeline_success
    
    # ========================================================================
    # STEP 2: Download and Process Funding Rate Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2/3: Downloading and Processing Funding Rate Data")
    print("=" * 80 + "\n")
    
    try:
        from jobs.funding_data_job import run_full_funding_pipeline
        run_full_funding_pipeline()
        print("\n✓ Step 2 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Step 2 FAILED: {str(e)}\n")
        print("Pipeline aborted due to funding data processing failure")
        pipeline_success = False
        return pipeline_success
    
    # ========================================================================
    # STEP 3: Generate Trading Signals
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3/3: Generating Basis Arbitrage Trading Signals")
    print("=" * 80 + "\n")
    
    try:
        from strategies.basis_arb import main as basis_arb_main
        basis_arb_main()
        print("\n✓ Step 3 completed successfully\n")
    except Exception as e:
        print(f"\n✗ Step 3 FAILED: {str(e)}\n")
        pipeline_success = False
        return pipeline_success
    
    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Status: SUCCESS - All steps completed")
    print("=" * 80 + "\n")
    
    return pipeline_success


def main():
    """Main entry point for the pipeline."""
    success = run_pipeline()
    
    # Exit with appropriate code
    if not success:
        print("\nPipeline failed. Please check errors above.")
        sys.exit(1)
    else:
        print("\nPipeline executed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
