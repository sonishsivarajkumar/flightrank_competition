"""
Simplified Feature Engineering for FlightRank 2025
Robust handling of mixed data types from JSON conversion
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SimpleFeatureEngineer:
    """Simplified feature engineering for JSON-converted data"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create robust features from the flight data"""
        print(f"🔧 Engineering features... Input shape: {df.shape}")
        print(f"🔍 Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        df = df.copy()
        
        # Basic numeric features
        df = self._add_basic_features(df)
        
        # Time features
        df = self._add_time_features(df)
        
        # Categorical features
        df = self._add_categorical_features(df)
        
        # Route features
        df = self._add_simple_route_features(df)
        
        # Clean up and select features
        df = self._clean_features(df)
        
        print(f"✓ Created features, output shape: {df.shape}")
        print(f"✓ Final columns: {list(df.columns)}")
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic numeric features"""
        
        # Price features
        if 'totalPrice' in df.columns:
            df['price_log'] = np.log1p(df['totalPrice'].fillna(0))
            
        if 'taxes' in df.columns and 'totalPrice' in df.columns:
            df['tax_ratio'] = df['taxes'].fillna(0) / (df['totalPrice'].fillna(1) + 1e-6)
        
        # Boolean features
        bool_cols = ['isVip', 'bySelf', 'isAccess3D', 'pricingInfo_isAccessTP']
        for col in bool_cols:
            if col in df.columns:
                df[f'{col}_flag'] = df[col].fillna(False).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if 'requestDate' in df.columns:
            try:
                request_dt = pd.to_datetime(df['requestDate'], errors='coerce')
                df['request_hour'] = request_dt.dt.hour.fillna(12)
                df['request_day_of_week'] = request_dt.dt.dayofweek.fillna(0)
                df['is_weekend'] = (df['request_day_of_week'] >= 5).astype(int)
                df['is_business_hours'] = ((df['request_hour'] >= 9) & (df['request_hour'] <= 17)).astype(int)
            except:
                df['request_hour'] = 12
                df['request_day_of_week'] = 0
                df['is_weekend'] = 0
                df['is_business_hours'] = 1
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical features with label encoding"""
        
        # High-cardinality categoricals to encode
        cat_cols = ['corporateTariffCode', 'profileId', 'companyID', 'nationality']
        
        for col in cat_cols:
            if col in df.columns:
                # Fill missing values
                df[col] = df[col].fillna('missing').astype(str)
                
                # Label encode
                if col not in self.label_encoders:
                    # First time - fit on all unique values including 'missing'
                    unique_vals = df[col].unique()
                    if 'missing' not in unique_vals:
                        unique_vals = np.append(unique_vals, 'missing')
                    
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(unique_vals)
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    # Handle unseen categories by mapping them to 'missing'
                    known_cats = set(self.label_encoders[col].classes_)
                    unknown_mask = ~df[col].isin(known_cats)
                    df.loc[unknown_mask, col] = 'missing'
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _add_simple_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple route-based features"""
        
        # Count segments
        segment_cols = [col for col in df.columns if 'segments' in col and 'duration' in col]
        if segment_cols:
            df['total_segments'] = (df[segment_cols].notna()).sum(axis=1)
            df['has_connections'] = (df['total_segments'] > 1).astype(int)
        
        # Route complexity from searchRoute
        if 'searchRoute' in df.columns:
            df['route_length'] = df['searchRoute'].fillna('').astype(str).str.len()
            df['is_round_trip'] = df['searchRoute'].fillna('').astype(str).str.contains('/').astype(int)
        
        # Carrier features
        carrier_cols = [col for col in df.columns if 'Carrier_code' in col]
        if carrier_cols:
            # Count unique carriers
            carrier_data = df[carrier_cols].fillna('missing')
            df['unique_carriers'] = carrier_data.nunique(axis=1)
            
            # Most common carrier
            if len(carrier_cols) > 0:
                first_carrier_col = carrier_cols[0]
                df['primary_carrier'] = df[first_carrier_col].fillna('missing').astype(str)
                
                if 'primary_carrier' not in self.label_encoders:
                    # First time - fit including 'missing'
                    unique_carriers = df['primary_carrier'].unique()
                    if 'missing' not in unique_carriers:
                        unique_carriers = np.append(unique_carriers, 'missing')
                    
                    self.label_encoders['primary_carrier'] = LabelEncoder()
                    self.label_encoders['primary_carrier'].fit(unique_carriers)
                    df['primary_carrier_encoded'] = self.label_encoders['primary_carrier'].transform(df['primary_carrier'])
                else:
                    # Handle unseen categories
                    known_carriers = set(self.label_encoders['primary_carrier'].classes_)
                    unknown_mask = ~df['primary_carrier'].isin(known_carriers)
                    df.loc[unknown_mask, 'primary_carrier'] = 'missing'
                    df['primary_carrier_encoded'] = self.label_encoders['primary_carrier'].transform(df['primary_carrier'])
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and select final features"""
        
        # Select numeric features for model
        feature_cols = []
        
        # Add basic numeric features
        numeric_features = [
            'totalPrice', 'taxes', 'price_log', 'tax_ratio',
            'request_hour', 'request_day_of_week', 'is_weekend', 'is_business_hours',
            'total_segments', 'has_connections', 'route_length', 'is_round_trip',
            'unique_carriers'
        ]
        
        # Add boolean flags
        flag_features = [col for col in df.columns if col.endswith('_flag')]
        
        # Add encoded features
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        all_features = numeric_features + flag_features + encoded_features
        
        # Keep only features that exist and are numeric
        for col in all_features:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    feature_cols.append(col)
                except:
                    pass
        
        # Keep essential columns
        essential_cols = ['Id', 'ranker_id']
        if 'selected' in df.columns:
            essential_cols.append('selected')
        
        # Return cleaned dataframe
        final_cols = essential_cols + feature_cols
        final_cols = [col for col in final_cols if col in df.columns]
        
        # Ensure we don't lose the selected column
        result_df = df[final_cols].copy()
        
        return result_df

def create_simple_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Create features for both train and test sets"""
    
    fe = SimpleFeatureEngineer()
    
    # Fit on train and transform both
    train_fe = fe.engineer_features(train_df)
    test_fe = fe.engineer_features(test_df)
    
    # Ensure same feature columns (but keep selected for train)
    feature_cols = [col for col in train_fe.columns if col not in ['Id', 'ranker_id', 'selected']]
    common_feature_cols = [col for col in feature_cols if col in test_fe.columns]
    
    # Create final datasets
    train_final_cols = ['Id', 'ranker_id', 'selected'] + common_feature_cols
    test_final_cols = ['Id', 'ranker_id'] + common_feature_cols
    
    train_fe_final = train_fe[train_final_cols]
    test_fe_final = test_fe[test_final_cols]
    
    print(f"✓ Final train columns: {len(train_fe_final.columns)}")
    print(f"✓ Final test columns: {len(test_fe_final.columns)}")
    
    return train_fe_final, test_fe_final
