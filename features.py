"""
Comprehensive feature engineering for FlightRank 2025 competition
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from config import logger, TARGET_ENCODING_SMOOTHING
from utils import safe_divide, target_encode_feature, MemoryManager

class FeatureEngineer:
    """Advanced feature engineering for flight ranking"""
    
    def __init__(self):
        self.target_encoders = {}
        
    def engineer_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline
        """
        logger.info(f"Starting feature engineering for {'train' if is_train else 'test'} data")
        df = df.copy()
        
        with MemoryManager():
            # Time-based features
            df = self._add_time_features(df)
            
            # Route complexity features
            df = self._add_route_features(df)
            
            # Pricing features
            df = self._add_pricing_features(df)
            
            # Carrier and airline features
            df = self._add_carrier_features(df)
            
            # Service class features
            df = self._add_service_features(df)
            
            # User preference features
            df = self._add_user_features(df)
            
            # Policy compliance features
            df = self._add_policy_features(df)
            
            # Advanced aggregation features
            df = self._add_aggregation_features(df)
            
            # Target encoding (only for training data)
            if is_train and 'selected' in df.columns:
                df = self._add_target_encoding(df)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("Adding time-based features...")
        
        # Request time features
        if 'requestDate' in df.columns:
            df['requestDate'] = pd.to_datetime(df['requestDate'])
            df['request_hour'] = df['requestDate'].dt.hour
            df['request_day_of_week'] = df['requestDate'].dt.dayofweek
            df['request_month'] = df['requestDate'].dt.month
            df['request_day_of_year'] = df['requestDate'].dt.dayofyear
            df['request_is_weekend'] = (df['request_day_of_week'] >= 5).astype(int)
            df['request_is_business_hours'] = ((df['request_hour'] >= 9) & 
                                             (df['request_hour'] <= 17)).astype(int)
            df['request_is_morning'] = (df['request_hour'] < 12).astype(int)
            df['request_is_evening'] = (df['request_hour'] >= 18).astype(int)
        
        # Flight timing features for each leg
        for leg in [0, 1]:
            dep_col = f'legs{leg}_departureAt'
            arr_col = f'legs{leg}_arrivalAt'
            
            if dep_col in df.columns and df[dep_col].notna().sum() > 0:
                df[dep_col] = pd.to_datetime(df[dep_col])
                df[f'leg{leg}_departure_hour'] = df[dep_col].dt.hour
                df[f'leg{leg}_departure_day_of_week'] = df[dep_col].dt.dayofweek
                df[f'leg{leg}_departure_month'] = df[dep_col].dt.month
                
                # Categorize departure times
                df[f'leg{leg}_is_early_morning'] = (df[dep_col].dt.hour <= 6).astype(int)
                df[f'leg{leg}_is_morning'] = ((df[dep_col].dt.hour > 6) & 
                                            (df[dep_col].dt.hour <= 12)).astype(int)
                df[f'leg{leg}_is_afternoon'] = ((df[dep_col].dt.hour > 12) & 
                                               (df[dep_col].dt.hour <= 18)).astype(int)
                df[f'leg{leg}_is_evening'] = ((df[dep_col].dt.hour > 18) & 
                                             (df[dep_col].dt.hour <= 22)).astype(int)
                df[f'leg{leg}_is_late_night'] = (df[dep_col].dt.hour > 22).astype(int)
                df[f'leg{leg}_is_red_eye'] = ((df[dep_col].dt.hour >= 22) | 
                                             (df[dep_col].dt.hour <= 6)).astype(int)
                df[f'leg{leg}_is_business_hours'] = ((df[dep_col].dt.hour >= 8) & 
                                                   (df[dep_col].dt.hour <= 18)).astype(int)
            
            if arr_col in df.columns and df[arr_col].notna().sum() > 0:
                df[arr_col] = pd.to_datetime(df[arr_col])
                df[f'leg{leg}_arrival_hour'] = df[arr_col].dt.hour
                df[f'leg{leg}_arrival_day_of_week'] = df[arr_col].dt.dayofweek
        
        # Time to departure (how far in advance booking)
        if 'requestDate' in df.columns and 'legs0_departureAt' in df.columns:
            df['days_to_departure'] = (df['legs0_departureAt'] - df['requestDate']).dt.days
            df['hours_to_departure'] = (df['legs0_departureAt'] - df['requestDate']).dt.total_seconds() / 3600
            df['is_last_minute'] = (df['days_to_departure'] <= 1).astype(int)
            df['is_advance_booking'] = (df['days_to_departure'] >= 14).astype(int)
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add route complexity and connection features"""
        logger.info("Adding route complexity features...")
        
        # Initialize segment counters
        df['total_segments_leg0'] = 0
        df['total_segments_leg1'] = 0
        
        # Count segments for each leg
        for leg in [0, 1]:
            for seg in range(4):  # segments 0-3
                seg_col = f'legs{leg}_segments{seg}_duration'
                if seg_col in df.columns:
                    has_segment = df[seg_col].notna().astype(int)
                    df[f'leg{leg}_has_segment{seg}'] = has_segment
                    df[f'total_segments_leg{leg}'] += has_segment
        
        # Derived route features
        df['total_segments'] = df['total_segments_leg0'] + df['total_segments_leg1']
        df['has_connections_leg0'] = (df['total_segments_leg0'] > 1).astype(int)
        df['has_connections_leg1'] = (df['total_segments_leg1'] > 1).astype(int)
        df['has_any_connections'] = ((df['total_segments_leg0'] > 1) | 
                                   (df['total_segments_leg1'] > 1)).astype(int)
        df['is_round_trip'] = (df['total_segments_leg1'] > 0).astype(int)
        df['is_complex_route'] = (df['total_segments'] > 2).astype(int)
        
        # Layover time calculation
        for leg in [0, 1]:
            duration_col = f'legs{leg}_duration'
            if duration_col in df.columns:
                segment_durations = []
                for seg in range(4):
                    seg_duration_col = f'legs{leg}_segments{seg}_duration'
                    if seg_duration_col in df.columns:
                        segment_durations.append(df[seg_duration_col].fillna(0))
                
                if segment_durations:
                    total_segment_duration = sum(segment_durations)
                    df[f'leg{leg}_layover_time'] = df[duration_col] - total_segment_duration
                    df[f'leg{leg}_layover_time'] = df[f'leg{leg}_layover_time'].clip(lower=0)
                    
                    # Layover categories
                    df[f'leg{leg}_short_layover'] = (df[f'leg{leg}_layover_time'] < 1).astype(int)
                    df[f'leg{leg}_medium_layover'] = ((df[f'leg{leg}_layover_time'] >= 1) & 
                                                    (df[f'leg{leg}_layover_time'] < 4)).astype(int)
                    df[f'leg{leg}_long_layover'] = (df[f'leg{leg}_layover_time'] >= 4).astype(int)
        
        # Route string analysis
        if 'searchRoute' in df.columns:
            df['route_contains_slash'] = df['searchRoute'].str.contains('/', na=False).astype(int)
            df['route_length'] = df['searchRoute'].str.len()
            
        return df
    
    def _add_pricing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pricing and cost-related features"""
        logger.info("Adding pricing features...")
        
        if 'totalPrice' in df.columns:
            # Price per hour of travel time
            if 'legs0_duration' in df.columns:
                total_duration = df['legs0_duration'].fillna(0)
                if 'legs1_duration' in df.columns:
                    total_duration += df['legs1_duration'].fillna(0)
                df['price_per_hour'] = safe_divide(df['totalPrice'], total_duration)
            
            # Tax-related features
            if 'taxes' in df.columns:
                df['tax_ratio'] = safe_divide(df['taxes'], df['totalPrice'])
                df['base_price'] = df['totalPrice'] - df['taxes'].fillna(0)
                df['has_high_taxes'] = (df['tax_ratio'] > 0.2).astype(int)
            
            # Price categories
            df['is_budget_flight'] = (df['totalPrice'] < df['totalPrice'].quantile(0.25)).astype(int)
            df['is_premium_flight'] = (df['totalPrice'] > df['totalPrice'].quantile(0.75)).astype(int)
            df['is_luxury_flight'] = (df['totalPrice'] > df['totalPrice'].quantile(0.9)).astype(int)
        
        # Cancellation and exchange penalties
        cancel_cols = [col for col in df.columns if 'miniRules0' in col and 'monetaryAmount' in col]
        exchange_cols = [col for col in df.columns if 'miniRules1' in col and 'monetaryAmount' in col]
        
        if cancel_cols:
            df['total_cancel_penalty'] = df[cancel_cols].sum(axis=1)
            df['has_cancel_penalty'] = (df['total_cancel_penalty'] > 0).astype(int)
            
        if exchange_cols:
            df['total_exchange_penalty'] = df[exchange_cols].sum(axis=1)
            df['has_exchange_penalty'] = (df['total_exchange_penalty'] > 0).astype(int)
        
        # Percentage penalties
        cancel_pct_cols = [col for col in df.columns if 'miniRules0' in col and 'percentage' in col]
        exchange_pct_cols = [col for col in df.columns if 'miniRules1' in col and 'percentage' in col]
        
        if cancel_pct_cols:
            df['total_cancel_penalty_pct'] = df[cancel_pct_cols].sum(axis=1)
        if exchange_pct_cols:
            df['total_exchange_penalty_pct'] = df[exchange_pct_cols].sum(axis=1)
        
        return df
    
    def _add_carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add airline and carrier features"""
        logger.info("Adding carrier and airline features...")
        
        # Collect all marketing carriers
        marketing_carriers = []
        operating_carriers = []
        
        for leg in [0, 1]:
            for seg in range(4):
                marketing_col = f'legs{leg}_segments{seg}_marketingCarrier_code'
                operating_col = f'legs{leg}_segments{seg}_operatingCarrier_code'
                
                if marketing_col in df.columns:
                    marketing_carriers.append(marketing_col)
                if operating_col in df.columns:
                    operating_carriers.append(operating_col)
        
        # Main carrier (first segment of first leg)
        if marketing_carriers:
            first_carrier_col = marketing_carriers[0]
            df['main_carrier'] = df[first_carrier_col]
            
            # Carrier consistency
            if len(marketing_carriers) > 1:
                carrier_consistency = df[marketing_carriers].nunique(axis=1)
                df['carrier_consistency'] = (carrier_consistency == 1).astype(int)
                df['multi_carrier_flight'] = (carrier_consistency > 1).astype(int)
        
        # Frequent flyer alignment
        if 'frequentFlyer' in df.columns and marketing_carriers:
            df['ff_carrier_match'] = 0
            for carrier_col in marketing_carriers:
                if carrier_col in df.columns:
                    match = (df['frequentFlyer'] == df[carrier_col]).fillna(False)
                    df['ff_carrier_match'] = df['ff_carrier_match'] | match
            df['ff_carrier_match'] = df['ff_carrier_match'].astype(int)
        
        # Marketing vs Operating carrier differences
        if marketing_carriers and operating_carriers:
            for i, (marketing_col, operating_col) in enumerate(zip(marketing_carriers, operating_carriers)):
                if marketing_col in df.columns and operating_col in df.columns:
                    df[f'seg{i}_codeshare'] = (df[marketing_col] != df[operating_col]).astype(int)
        
        return df
    
    def _add_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add service class and amenity features"""
        logger.info("Adding service class features...")
        
        # Cabin class analysis
        cabin_cols = [col for col in df.columns if 'cabinClass' in col]
        if cabin_cols:
            df['best_cabin_class'] = df[cabin_cols].max(axis=1)
            df['worst_cabin_class'] = df[cabin_cols].min(axis=1)
            df['cabin_class_consistency'] = (df['best_cabin_class'] == df['worst_cabin_class']).astype(int)
            
            # Service level flags
            df['has_economy'] = (df[cabin_cols] == 1.0).any(axis=1).astype(int)
            df['has_comfort'] = (df[cabin_cols] == 2.0).any(axis=1).astype(int)
            df['has_business'] = (df[cabin_cols] == 3.0).any(axis=1).astype(int)
            df['has_first'] = (df[cabin_cols] == 4.0).any(axis=1).astype(int)
            
            df['is_all_economy'] = (df['best_cabin_class'] == 1.0).astype(int)
            df['is_all_business_or_higher'] = (df['worst_cabin_class'] >= 3.0).astype(int)
            df['has_mixed_classes'] = (~df['cabin_class_consistency']).astype(int)
        
        # Seat availability
        seat_cols = [col for col in df.columns if 'seatsAvailable' in col]
        if seat_cols:
            df['min_seats_available'] = df[seat_cols].min(axis=1)
            df['max_seats_available'] = df[seat_cols].max(axis=1)
            df['total_seats_available'] = df[seat_cols].sum(axis=1)
            df['avg_seats_available'] = df[seat_cols].mean(axis=1)
            
            # Availability categories
            df['very_limited_seats'] = (df['min_seats_available'] <= 2).astype(int)
            df['limited_seats'] = ((df['min_seats_available'] > 2) & 
                                 (df['min_seats_available'] <= 5)).astype(int)
            df['good_availability'] = (df['min_seats_available'] > 5).astype(int)
        
        # Baggage allowance
        baggage_cols = [col for col in df.columns if 'baggageAllowance_quantity' in col]
        if baggage_cols:
            df['min_baggage_allowance'] = df[baggage_cols].min(axis=1)
            df['max_baggage_allowance'] = df[baggage_cols].max(axis=1)
            df['total_baggage_allowance'] = df[baggage_cols].sum(axis=1)
            df['baggage_consistency'] = (df['min_baggage_allowance'] == df['max_baggage_allowance']).astype(int)
        
        return df
    
    def _add_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add user preference and profile features"""
        logger.info("Adding user preference features...")
        
        # User profile flags
        user_flag_cols = ['isVip', 'bySelf', 'isAccess3D']
        for col in user_flag_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        # User type combinations
        if 'isVip' in df.columns and 'bySelf' in df.columns:
            df['vip_independent'] = (df['isVip'] & df['bySelf']).astype(int)
            df['vip_managed'] = (df['isVip'] & ~df['bySelf']).astype(int)
        
        # Nationality vs route analysis
        if 'nationality' in df.columns:
            # Extract departure and arrival countries from IATA codes
            departure_cols = [col for col in df.columns if 'departureFrom_airport_iata' in col]
            arrival_cols = [col for col in df.columns if 'arrivalTo_airport_iata' in col]
            
            if departure_cols:
                # This would require airport IATA to country mapping
                # For now, we'll create a placeholder
                df['domestic_flight'] = 0  # Placeholder
        
        return df
    
    def _add_policy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add corporate policy compliance features"""
        logger.info("Adding policy compliance features...")
        
        # Travel policy compliance
        if 'pricingInfo_isAccessTP' in df.columns:
            df['policy_compliant'] = df['pricingInfo_isAccessTP'].fillna(0).astype(int)
        
        # Corporate tariff analysis
        if 'corporateTariffCode' in df.columns:
            df['has_corporate_tariff'] = df['corporateTariffCode'].notna().astype(int)
        
        # Passenger count
        if 'pricingInfo_passengerCount' in df.columns:
            df['is_single_passenger'] = (df['pricingInfo_passengerCount'] == 1).astype(int)
            df['is_group_booking'] = (df['pricingInfo_passengerCount'] > 1).astype(int)
        
        return df
    
    def _add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on aggregations within sessions"""
        logger.info("Adding aggregation features...")
        
        # Session-level aggregations
        if 'ranker_id' in df.columns:
            session_stats = df.groupby('ranker_id').agg({
                'totalPrice': ['min', 'max', 'mean', 'std'],
                'legs0_duration': ['min', 'max', 'mean'],
            }).fillna(0)
            
            # Flatten column names
            session_stats.columns = ['_'.join(col).strip() for col in session_stats.columns]
            session_stats = session_stats.add_prefix('session_')
            
            # Add rank within session
            df['price_rank_in_session'] = df.groupby('ranker_id')['totalPrice'].rank()
            df['duration_rank_in_session'] = df.groupby('ranker_id')['legs0_duration'].rank()
            
            # Relative to session statistics
            session_stats = session_stats.reset_index()
            df = df.merge(session_stats, on='ranker_id', how='left')
            
            # Price relative to session
            if 'session_totalPrice_mean' in df.columns:
                df['price_vs_session_avg'] = df['totalPrice'] - df['session_totalPrice_mean']
                df['price_ratio_vs_session_avg'] = safe_divide(df['totalPrice'], df['session_totalPrice_mean'])
                df['is_cheapest_in_session'] = (df['price_rank_in_session'] == 1).astype(int)
                df['is_most_expensive_in_session'] = (df['price_rank_in_session'] == df.groupby('ranker_id')['ranker_id'].transform('count')).astype(int)
        
        return df
    
    def _add_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target encoding for high-cardinality categorical features"""
        logger.info("Adding target encoding features...")
        
        if 'selected' not in df.columns:
            return df
        
        # High cardinality categorical features to encode
        cat_features = ['corporateTariffCode', 'main_carrier', 'nationality', 'frequentFlyer']
        
        for feature in cat_features:
            if feature in df.columns and df[feature].nunique() > 10:
                encoded_feature = f'{feature}_target_encoded'
                df[encoded_feature] = target_encode_feature(
                    df, feature, 'selected', TARGET_ENCODING_SMOOTHING
                )
                self.target_encoders[feature] = {
                    'global_mean': df['selected'].mean(),
                    'feature_stats': df.groupby(feature)['selected'].agg(['mean', 'count'])
                }
        
        return df
    
    def apply_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding to test data using training statistics"""
        logger.info("Applying target encoding to test data...")
        
        for feature, encoder_info in self.target_encoders.items():
            if feature in df.columns:
                encoded_feature = f'{feature}_target_encoded'
                global_mean = encoder_info['global_mean']
                feature_stats = encoder_info['feature_stats']
                
                # Calculate smoothed means
                smoothed_means = (
                    feature_stats['mean'] * feature_stats['count'] + 
                    global_mean * TARGET_ENCODING_SMOOTHING
                ) / (feature_stats['count'] + TARGET_ENCODING_SMOOTHING)
                
                df[encoded_feature] = df[feature].map(smoothed_means).fillna(global_mean)
        
        return df

def main():
    """Example usage"""
    from data_loading import DataLoader
    
    # Load data
    loader = DataLoader()
    train_df, test_df, _ = loader.load_main_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Engineer features for training data
    train_fe = engineer.engineer_features(train_df, is_train=True)
    
    # Engineer features for test data (without target encoding)
    test_fe = engineer.engineer_features(test_df, is_train=False)
    
    # Apply target encoding to test data
    test_fe = engineer.apply_target_encoding(test_fe)
    
    logger.info(f"Original train shape: {train_df.shape}")
    logger.info(f"Enhanced train shape: {train_fe.shape}")
    logger.info(f"Original test shape: {test_df.shape}")
    logger.info(f"Enhanced test shape: {test_fe.shape}")
    
    # Show new features
    new_features = [col for col in train_fe.columns if col not in train_df.columns]
    logger.info(f"Created {len(new_features)} new features")
    logger.info(f"Sample new features: {new_features[:10]}")

if __name__ == "__main__":
    main()
