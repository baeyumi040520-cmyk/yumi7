import polars as pl
import os
import sys

# 7-Eleven NPD Framework - Phase 1 Step 1.1 Analysis
def run_npd_baseline():
    b2_path = '7eleven_npd_framework/data/processed/B2_POS_SALE.parquet'
    
    # 1. Load Data with Schema Handling
    # Getting original column names to rename them safely
    b2_schema = pl.scan_parquet(b2_path).collect_schema().names()
    mapping = {
        b2_schema[0]: 'SALE_DATE',
        b2_schema[1]: 'SALE_TIME',
        b2_schema[2]: 'STORE_CODE',
        b2_schema[3]: 'POS_NO',
        b2_schema[4]: 'TRADE_NO',
        b2_schema[5]: 'ITEM_CODE',
        b2_schema[6]: 'SALE_QTY',
        b2_schema[7]: 'SALE_AMT'
    }
    
    b2_lazy = pl.scan_parquet(b2_path).rename(mapping)
    
    # 2. Date Conversion & Launch Date Definition
    b2_with_date = b2_lazy.with_columns(
        pl.col('SALE_DATE').str.to_date('%Y%m%d').alias('sale_dt')
    )
    
    # Launch Date: First appearance of ITEM_CODE
    launch_dates = b2_with_date.group_by('ITEM_CODE').agg(
        pl.col('sale_dt').min().alias('launch_dt')
    )
    
    # 3. Join and Calculate Days Since Launch
    sales_enriched = b2_with_date.join(launch_dates, on='ITEM_CODE')
    sales_enriched = sales_enriched.with_columns(
        (pl.col('sale_dt') - pl.col('launch_dt')).dt.total_days().alias('days_after')
    )
    
    # 4. Aggregate Cumulative Sales for 1w, 2w, 4w
    # Filter only positive sales for normal items
    cum_sales = sales_enriched.group_by('ITEM_CODE').agg([
        pl.col('SALE_AMT').filter(pl.col('days_after') <= 7).sum().alias('cum_1w'),
        pl.col('SALE_AMT').filter(pl.col('days_after') <= 14).sum().alias('cum_2w'),
        pl.col('SALE_AMT').filter(pl.col('days_after') <= 28).sum().alias('cum_4w')
    ]).collect()
    
    # 5. Output Statistics
    print("\n--- [Phase 1.1] NPD Cumulative Sales Distribution Summary ---")
    print(f"Total Unique Items Identified: {len(cum_sales):,}")
    
    for period in ['cum_1w', 'cum_2w', 'cum_4w']:
        # Filter items that actually sold during the period
        active_sales = cum_sales.filter(pl.col(period) > 0)[period]
        if active_sales.len() == 0:
            print(f"{period}: No sales data found.")
            continue
            
        q50 = active_sales.median()
        q80 = active_sales.quantile(0.8)
        q90 = active_sales.quantile(0.9)
        q95 = active_sales.quantile(0.95)
        mean_val = active_sales.mean()
        
        print(f"\n[{period} Distribution]")
        print(f"  Count of active items: {active_sales.len():,}")
        print(f"  Mean Sales: {int(mean_val):,}")
        print(f"  Median (Q50): {int(q50):,}")
        print(f"  Top 20% (Q80): {int(q80):,}  <-- Hit Candidate Threshold")
        print(f"  Top 10% (Q90): {int(q90):,}  <-- Big Hit Threshold")
        print(f"  Top 5%  (Q95): {int(q95):,}  <-- Mega Hit Threshold")

if __name__ == "__main__":
    try:
        run_npd_baseline()
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)
