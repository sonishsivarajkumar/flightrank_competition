"""
Feature Engineering Module - Optimized for FlightRank 2025 Competition
"""

import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    """High-performance feature engineering for flight ranking"""
    
    def __init__(self):
        self.feature_cache = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive and optimized feature engineering
        """
        df = df.copy()
        
        # Time-based features (vectorized)
        df = self._add_time_features(df)
        
        # Flight structure features
        df = self._add_flight_structure_features(df)
        
        # Pricing and policy features
        df = self._add_pricing_features(df)
        
        # Carrier and user preference features
        df = self._add_carrier_features(df)
        
        # Service quality features
        df = self._add_service_features(df)
        
        # Route complexity features
        df = self._add_route_features(df)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features efficiently"""
        
        # Request time features
        if 'requestDate' in df.columns:
            df['requestDate'] = pd.to_datetime(df['requestDate'], errors='coerce')
            df['request_hour'] = df['requestDate'].dt.hour
            df['request_day_of_week'] = df['requestDate'].dt.dayofweek
            df['request_month'] = df['requestDate'].dt.month
            df['request_is_weekend'] = (df['request_day_of_week'] >= 5).astype(np.int8)
            df['request_is_business_hours'] = ((df['request_hour'] >= 9) & (df['request_hour'] <= 17)).astype(np.int8)
        
        # Flight timing features for both legs
        for leg in [0, 1]:
            dep_col = f'legs{leg}_departureAt'
            arr_col = f'legs{leg}_arrivalAt'
            
            if dep_col in df.columns:
                df[dep_col] = pd.to_datetime(df[dep_col], errors='coerce')
                df[f'leg{leg}_departure_hour'] = df[dep_col].dt.hour
                df[f'leg{leg}_departure_day_of_week'] = df[dep_col].dt.dayofweek
                df[f'leg{leg}_is_early_flight'] = (df[dep_col].dt.hour <= 6).astype(np.int8)
                df[f'leg{leg}_is_late_flight'] = (df[dep_col].dt.hour >= 22).astype(np.int8)
                df[f'leg{leg}_is_business_hours'] = ((df[dep_col].dt.hour >= 8) & (df[dep_col].dt.hour <= 18)).astype(np.int8)
                df[f'leg{leg}_is_weekend'] = (df[dep_col].dt.dayofweek >= 5).astype(np.int8)
            
            if arr_col in df.columns:
                df[arr_col] = pd.to_datetime(df[arr_col], errors='coerce')
                df[f'leg{leg}_arrival_hour'] = df[arr_col].dt.hour
        
        return df
    
    def _add_flight_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flight structure and complexity features"""
        
        # Initialize segment counters
        df['total_segments_leg0'] = 0
        df['total_segments_leg1'] = 0
        
        # Count segments for each leg
        for leg in [0, 1]:
            for seg in range(4):  # segments 0-3
                seg_col = f'legs{leg}_segments{seg}_duration'
                if seg_col in df.columns:
                    has_segment = df[seg_col].notna().astype(np.int8)
                    df[f'leg{leg}_has_segment{seg}'] = has_segment
                    df[f'total_segments_leg{leg}'] += has_segment
        
        # Derived structure features
        df['total_segments'] = df['total_segments_leg0'] + df['total_segments_leg1']
        df['has_connections'] = (df['total_segments_leg0'] > 1).astype(np.int8)
        df['is_round_trip'] = (df['total_segments_leg1'] > 0).astype(np.int8)
        df['max_segments_per_leg'] = df[['total_segments_leg0', 'total_segments_leg1']].max(axis=1)
        
        # Calculate layover times (total duration - sum of segment durations)
        for leg in [0, 1]:
            total_duration_col = f'legs{leg}_duration'
            if total_duration_col in df.columns:
                segment_duration_cols = [f'legs{leg}_segments{seg}_duration' for seg in range(4) 
                                       if f'legs{leg}_segments{seg}_duration' in df.columns]
                if segment_duration_cols:
                    # Convert durations to numeric (in case they're strings)
                    try:
                        segment_duration_sum = pd.to_numeric(df[segment_duration_cols], errors='coerce').fillna(0).sum(axis=1)
                        total_duration_numeric = pd.to_numeric(df[total_duration_col], errors='coerce').fillna(0)
                        df[f'leg{leg}_layover_time'] = (total_duration_numeric - segment_duration_sum).clip(lower=0)
                    except:
                        # If conversion fails, skip layover calculation
                        df[f'leg{leg}_layover_time'] = 0
        
        return df
    
    def _add_pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pricing and policy-related features"""
        
        if 'totalPrice' in df.columns:
            # Price efficiency metrics
            if 'legs0_duration' in df.columns:
                try:
                    legs0_dur = pd.to_numeric(df['legs0_duration'], errors='coerce').fillna(0)
                    legs1_dur = pd.to_numeric(df.get('legs1_duration', 0), errors='coerce').fillna(0)
                    total_duration = legs0_dur + legs1_dur
                    df['price_per_hour'] = df['totalPrice'] / (total_duration + 1e-6)  # Add small epsilon
                except:
                    df['price_per_hour'] = df['totalPrice']  # Fallback
            
            # Tax ratio
            if 'taxes' in df.columns:
                df['tax_ratio'] = df['taxes'] / (df['totalPrice'] + 1e-6)
            
            # Price percentiles within each session
            df['price_rank_in_session'] = df.groupby('ranker_id')['totalPrice'].rank(pct=True)
            
        # Policy compliance features
        if 'pricingInfo_isAccessTP' in df.columns:
            df['policy_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(np.int8)
        
        # Penalty features
        cancel_cols = [col for col in df.columns if 'miniRules0' in col and 'monetaryAmount' in col]
        exchange_cols = [col for col in df.columns if 'miniRules1' in col and 'monetaryAmount' in col]
        
        if cancel_cols:
            df['total_cancel_penalty'] = df[cancel_cols].sum(axis=1)
            if 'totalPrice' in df.columns:
                df['cancel_penalty_ratio'] = df['total_cancel_penalty'] / (df['totalPrice'] + 1e-6)
        
        if exchange_cols:
            df['total_exchange_penalty'] = df[exchange_cols].sum(axis=1)
            if 'totalPrice' in df.columns:
                df['exchange_penalty_ratio'] = df['total_exchange_penalty'] / (df['totalPrice'] + 1e-6)
        
        return df
    
    def _add_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add carrier and frequent flyer features"""
        
        # Collect all carrier columns
        carrier_cols = [col for col in df.columns if 'marketingCarrier_code' in col]
        
        if carrier_cols and 'frequentFlyer' in df.columns:
            # Check if user's frequent flyer program matches any carrier
            df['ff_carrier_match'] = 0
            for carrier_col in carrier_cols:
                match = (df['frequentFlyer'] == df[carrier_col]).fillna(False).astype(np.int8)
                df['ff_carrier_match'] = (df['ff_carrier_match'] | match).astype(np.int8)
        
        # Main carrier consistency (same carrier for all segments)
        if len(carrier_cols) > 1:
            first_carrier = df[carrier_cols[0]]
            df['carrier_consistency'] = 1
            for carrier_col in carrier_cols[1:]:
                df['carrier_consistency'] &= (df[carrier_col] == first_carrier) | df[carrier_col].isna()
            df['carrier_consistency'] = df['carrier_consistency'].astype(np.int8)
        
        # User preference features
        user_cols = ['isVip', 'bySelf', 'isAccess3D']
        for col in user_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(np.int8)
        
        return df
    
    def _add_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add service quality and cabin class features"""
        
        # Cabin class analysis
        cabin_cols = [col for col in df.columns if 'cabinClass' in col]
        if cabin_cols:
            df['best_cabin_class'] = df[cabin_cols].max(axis=1)
            df['worst_cabin_class'] = df[cabin_cols].min(axis=1)
            df['cabin_class_consistency'] = (df['best_cabin_class'] == df['worst_cabin_class']).astype(np.int8)
            
            # Create binary indicators for each class
            for leg in [0, 1]:
                for seg in range(4):
                    cabin_col = f'legs{leg}_segments{seg}_cabinClass'
                    if cabin_col in df.columns:
                        df[f'leg{leg}_seg{seg}_is_economy'] = (df[cabin_col] == 1.0).astype(np.int8)
                        df[f'leg{leg}_seg{seg}_is_business'] = (df[cabin_col] == 3.0).astype(np.int8)
                        df[f'leg{leg}_seg{seg}_is_first'] = (df[cabin_col] == 4.0).astype(np.int8)
        
        # Seat availability features
        seat_cols = [col for col in df.columns if 'seatsAvailable' in col]
        if seat_cols:
            df['min_seats_available'] = df[seat_cols].min(axis=1)
            df['max_seats_available'] = df[seat_cols].max(axis=1)
            df['total_seats_available'] = df[seat_cols].sum(axis=1)
            df['avg_seats_available'] = df[seat_cols].mean(axis=1)
        
        # Baggage allowance features
        baggage_cols = [col for col in df.columns if 'baggageAllowance_quantity' in col]
        if baggage_cols:
            df['min_baggage_allowance'] = df[baggage_cols].min(axis=1)
            df['max_baggage_allowance'] = df[baggage_cols].max(axis=1)
            df['total_baggage_allowance'] = df[baggage_cols].sum(axis=1)
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route-specific features"""
        
        # Route direction features
        if 'searchRoute' in df.columns:
            df['is_round_trip_search'] = df['searchRoute'].str.contains('/', na=False).astype(np.int8)
        
        # Airport diversity (count unique airports)
        airport_cols = [col for col in df.columns if 'airport_iata' in col and 'departureFrom' in col or 'arrivalTo' in col]
        if airport_cols:
            # Count unique airports in the journey
            airport_data = df[airport_cols].fillna('MISSING')
            df['unique_airports'] = airport_data.nunique(axis=1)
        
        # Duration efficiency
        duration_cols = [col for col in df.columns if col.endswith('_duration') and 'legs' in col and 'segments' not in col]
        if duration_cols:
            try:
                # Convert to numeric first
                duration_numeric = df[duration_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                df['total_flight_duration'] = duration_numeric.sum(axis=1)
            except:
                df['total_flight_duration'] = 0
        
        # Aircraft consistency
        aircraft_cols = [col for col in df.columns if 'aircraft_code' in col]
        if len(aircraft_cols) > 1:
            first_aircraft = df[aircraft_cols[0]]
            df['aircraft_consistency'] = 1
            for aircraft_col in aircraft_cols[1:]:
                df['aircraft_consistency'] &= (df[aircraft_col] == first_aircraft) | df[aircraft_col].isna()
            df['aircraft_consistency'] = df['aircraft_consistency'].astype(np.int8)
        
        return df
