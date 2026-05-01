import pandas as pd
import numpy as np
import os

USD_TO_VND = 25450


def _read_csv_if_exists(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)


def _to_datetime(df, columns):
    if df is None:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _clip_outliers_iqr(df, columns, factor=1.5):
    if df is None:
        return df
    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _add_calendar_features(df, date_col):
    if date_col not in df.columns:
        return df
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['week_of_year'] = df[date_col].dt.isocalendar().week.astype('Int64')
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def load_and_merge_order_data(data_path="data/raw"):
    """
    Merge orders.csv with order_items.csv, products.csv, and promotions.csv.
    Handles NaN in promo_id and discount_amount.
    Converts currency to VND if necessary.
    """
    # Load datasets
    orders = pd.read_csv(os.path.join(data_path, 'orders.csv'))
    order_items = pd.read_csv(os.path.join(data_path, 'order_items.csv'))
    products = pd.read_csv(os.path.join(data_path, 'products.csv'))
    promotions = pd.read_csv(os.path.join(data_path, 'promotions.csv'))
    
    # Merge order_items with products to get unit prices/costs
    merged_data = order_items.merge(products, on='product_id', how='left')
    
    # Merge with orders
    merged_data = merged_data.merge(orders, on='order_id', how='left')
    
    # Merge with promotions
    merged_data = merged_data.merge(promotions, on='promo_id', how='left')
    
    # Handle NaN in promo_id and discount_amount
    merged_data['promo_id'] = merged_data['promo_id'].fillna('NO_PROMO')
    merged_data['discount_amount'] = merged_data['discount_amount'].fillna(0)
    
    # Currency Conversion Logic (Assuming original data might be in USD based on context)
    # The user requested all currency values (Revenue, COGS, Refund) be calculated in VND.
    # We should ensure calculations account for the 25,450 VND reference rate.
    
    # Example calculation (adjust depending on exact column names in actual CSVs)
    if 'item_price_usd' in merged_data.columns:
        merged_data['item_price_vnd'] = merged_data['item_price_usd'] * USD_TO_VND
        
    return merged_data

def preprocess_all(data_path="data/raw"):
    """
    End-to-end preprocessing pipeline:
    1) Load raw tables
    2) Join order-level and product-level tables
    3) Clean missing values and cap outliers
    4) Build time-based features
    5) Export modeling datasets for train and forecast

    Returns
    -------
    dict
        {
            'train_df': daily training dataset,
            'forecast_df': future dates for inference,
            'meta': metadata about data coverage,
            'transactions_df': enriched transaction-level dataset
        }
    """
    # 1) Load raw tables
    orders = _read_csv_if_exists(data_path, 'orders.csv')
    order_items = _read_csv_if_exists(data_path, 'order_items.csv')
    products = _read_csv_if_exists(data_path, 'products.csv')
    promotions = _read_csv_if_exists(data_path, 'promotions.csv')
    customers = _read_csv_if_exists(data_path, 'customers.csv')
    shipments = _read_csv_if_exists(data_path, 'shipments.csv')
    returns = _read_csv_if_exists(data_path, 'returns.csv')
    payments = _read_csv_if_exists(data_path, 'payments.csv')
    web_traffic = _read_csv_if_exists(data_path, 'web_traffic.csv')

    required_tables = {
        'orders.csv': orders,
        'order_items.csv': order_items,
        'products.csv': products,
    }
    missing_required = [name for name, table in required_tables.items() if table is None]
    if missing_required:
        raise FileNotFoundError(f"Missing required input file(s): {missing_required}")

    # Parse date columns early for consistent feature engineering
    orders = _to_datetime(orders, ['order_date'])
    products = _to_datetime(products, [])
    promotions = _to_datetime(promotions, ['start_date', 'end_date'])
    customers = _to_datetime(customers, ['signup_date'])
    shipments = _to_datetime(shipments, ['ship_date', 'delivery_date'])
    returns = _to_datetime(returns, ['return_date'])
    web_traffic = _to_datetime(web_traffic, ['date'])

    # 2) Join tables into transaction-level dataset
    tx = order_items.merge(products, on='product_id', how='left', suffixes=('', '_product'))
    tx = tx.merge(orders, on='order_id', how='left', suffixes=('', '_order'))

    if customers is not None and 'customer_id' in tx.columns:
        customer_cols = [col for col in ['customer_id', 'city', 'signup_date', 'gender', 'age_group', 'acquisition_channel'] if col in customers.columns]
        tx = tx.merge(customers[customer_cols], on='customer_id', how='left')

    if promotions is not None and 'promo_id' in tx.columns:
        tx = tx.merge(promotions, on='promo_id', how='left', suffixes=('', '_promo'))

    if shipments is not None:
        tx = tx.merge(shipments, on='order_id', how='left', suffixes=('', '_ship'))

    if payments is not None:
        payment_cols = [col for col in ['order_id', 'payment_value', 'installments'] if col in payments.columns]
        if payment_cols:
            tx = tx.merge(payments[payment_cols], on='order_id', how='left')

    if returns is not None:
        return_agg = returns.groupby(['order_id', 'product_id'], as_index=False).agg(
            return_quantity=('return_quantity', 'sum'),
            refund_amount=('refund_amount', 'sum')
        )
        tx = tx.merge(return_agg, on=['order_id', 'product_id'], how='left')

    # 3) Clean nulls and standardize numerics
    for col in ['quantity', 'unit_price', 'discount_amount', 'price', 'cogs', 'shipping_fee', 'payment_value', 'installments', 'return_quantity', 'refund_amount']:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors='coerce')

    fill_zero_cols = ['discount_amount', 'shipping_fee', 'payment_value', 'installments', 'return_quantity', 'refund_amount']
    for col in fill_zero_cols:
        if col in tx.columns:
            tx[col] = tx[col].fillna(0)

    if 'promo_id' in tx.columns:
        tx['promo_id'] = tx['promo_id'].fillna('NO_PROMO')

    if 'quantity' in tx.columns:
        tx['quantity'] = tx['quantity'].fillna(0).clip(lower=0)
    if 'unit_price' in tx.columns:
        tx['unit_price'] = tx['unit_price'].fillna(0).clip(lower=0)
    if 'cogs' in tx.columns:
        tx['cogs'] = tx['cogs'].fillna(0).clip(lower=0)
    if 'discount_amount' in tx.columns:
        tx['discount_amount'] = tx['discount_amount'].clip(lower=0)

    outlier_cols = ['quantity', 'unit_price', 'discount_amount', 'shipping_fee', 'payment_value', 'refund_amount']
    tx = _clip_outliers_iqr(tx, outlier_cols)

    # Currency conversion helper if USD column exists
    if 'item_price_usd' in tx.columns:
        tx['item_price_vnd'] = tx['item_price_usd'] * USD_TO_VND

    # 4) Feature engineering
    tx = _to_datetime(tx, ['order_date', 'ship_date', 'delivery_date', 'signup_date'])

    tx['gross_revenue'] = tx['quantity'] * tx['unit_price']
    tx['net_revenue'] = (tx['gross_revenue'] - tx['discount_amount']).clip(lower=0)
    tx['line_cogs'] = tx['quantity'] * tx['cogs']
    tx['is_returned'] = (tx.get('return_quantity', 0) > 0).astype(int)
    tx['refund_amount'] = tx.get('refund_amount', 0).fillna(0)
    tx['net_revenue_after_refund'] = (tx['net_revenue'] - tx['refund_amount']).clip(lower=0)

    if 'ship_date' in tx.columns and 'delivery_date' in tx.columns:
        tx['delivery_lead_days'] = (tx['delivery_date'] - tx['ship_date']).dt.days

    if 'signup_date' in tx.columns and 'order_date' in tx.columns:
        tx['customer_tenure_days'] = (tx['order_date'] - tx['signup_date']).dt.days

    if {'start_date', 'end_date', 'order_date'}.issubset(tx.columns):
        tx['is_promo_active'] = (
            (tx['order_date'] >= tx['start_date']) &
            (tx['order_date'] <= tx['end_date'])
        ).fillna(False).astype(int)
    else:
        tx['is_promo_active'] = 0

    tx = _add_calendar_features(tx, 'order_date')

    # Keep only records with valid order_date for downstream daily modeling
    tx = tx[tx['order_date'].notna()].copy()

    # 5) Build train_df (daily) and forecast_df (future horizon)
    daily = tx.groupby('order_date', as_index=False).agg(
        Revenue=('net_revenue_after_refund', 'sum'),
        COGS=('line_cogs', 'sum'),
        orders=('order_id', 'nunique'),
        items=('quantity', 'sum'),
        avg_discount=('discount_amount', 'mean'),
        returned_items=('return_quantity', 'sum')
    )

    if web_traffic is not None and {'date', 'sessions', 'unique_visitors', 'page_views'}.issubset(web_traffic.columns):
        traffic_daily = web_traffic.groupby('date', as_index=False).agg(
            sessions=('sessions', 'sum'),
            unique_visitors=('unique_visitors', 'sum'),
            page_views=('page_views', 'sum'),
            bounce_rate=('bounce_rate', 'mean'),
            avg_session_duration_sec=('avg_session_duration_sec', 'mean')
        )
        daily = daily.merge(traffic_daily, left_on='order_date', right_on='date', how='left')
        daily = daily.drop(columns=['date'])

    daily = daily.sort_values('order_date').reset_index(drop=True)
    daily = _add_calendar_features(daily, 'order_date')

    # Fill missing traffic features with rolling fallback values
    traffic_cols = ['sessions', 'unique_visitors', 'page_views', 'bounce_rate', 'avg_session_duration_sec']
    for col in traffic_cols:
        if col in daily.columns:
            fallback = daily[col].dropna().tail(28).median()
            if pd.isna(fallback):
                fallback = 0
            daily[col] = daily[col].fillna(fallback)

    # Build forecast_df from sample_submission.csv (competition-provided file).
    # This ensures we predict exactly the required dates in the correct order.
    sample_sub_path = os.path.join(data_path, 'sample_submission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        forecast_df = pd.DataFrame({
            'order_date': pd.to_datetime(sample_sub['Date'], errors='coerce')
        })
        forecast_df = forecast_df[forecast_df['order_date'].notna()].copy()
        # Preserve the original row order — do NOT sort or shuffle.
        forecast_df = forecast_df.reset_index(drop=True)
    elif daily.empty:
        forecast_df = pd.DataFrame(columns=['order_date'])
    else:
        # Fallback: generate 30 days after training end date
        last_date = daily['order_date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), periods=30, freq='D',
        )
        forecast_df = pd.DataFrame({'order_date': future_dates})

    forecast_df = _add_calendar_features(forecast_df, 'order_date')

    for col in traffic_cols:
        if col in daily.columns:
            value = daily[col].tail(28).median()
            if pd.isna(value):
                value = 0
            forecast_df[col] = value

    meta = {
        'transaction_rows': int(len(tx)),
        'train_rows': int(len(daily)),
        'forecast_rows': int(len(forecast_df)),
        'date_min': None if daily.empty else str(daily['order_date'].min().date()),
        'date_max': None if daily.empty else str(daily['order_date'].max().date()),
        'features_for_model': [
            col for col in daily.columns
            if col not in {'order_date', 'Revenue', 'COGS'}
        ]
    }

    return {
        'train_df': daily,
        'forecast_df': forecast_df,
        'meta': meta,
        'transactions_df': tx
    }
