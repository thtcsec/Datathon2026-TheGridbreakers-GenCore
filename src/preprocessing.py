import pandas as pd
import numpy as np
import os

USD_TO_VND = 25450

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
    Function to process Master, Transaction, and Operational tables.
    """
    # Implementation for joining other tables
    # (Master, Operational, Transaction layers as specified in the datathon)
    pass
