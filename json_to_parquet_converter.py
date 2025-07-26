"""
JSON to Parquet Converter for FlightRank 2025 Competition
Converts raw JSON files to the parquet format expected by the solution
"""

import pandas as pd
import numpy as np
import json
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class JSONToParquetConverter:
    """Convert JSON files to parquet format for the competition"""
    
    def __init__(self, json_dir: str = "json_samples", output_dir: str = "."):
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        
    def parse_single_json(self, json_file: Path) -> List[Dict]:
        """Parse a single JSON file and extract flight options"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Skip files with list structure for now
                return []
            
            ranker_id = data.get('ranker_id', json_file.stem)
            request_date = data.get('routeData', {}).get('requestDate') if data.get('routeData') else None
            search_route = data.get('routeData', {}).get('searchRoute') if data.get('routeData') else None
            
            # Personal data
            personal = data.get('personalData', {})
            profile_id = personal.get('profileId') if personal else None
            sex = personal.get('sex') if personal else None
            nationality = personal.get('nationality') if personal else None
            company_id = personal.get('companyID') if personal else None
            is_vip = personal.get('isVip', False) if personal else False
            by_self = personal.get('bySelf', True) if personal else True
            is_access_3d = personal.get('isAccess3D', False) if personal else False
            
            # Extract flight options
            flight_options = []
            flight_data = data.get('data', {}).get('$values', [])
            
            for idx, flight_option in enumerate(flight_data):
                # Extract legs information
                legs = flight_option.get('legs', [])
                
                # Extract pricing information  
                pricings = flight_option.get('pricings', [])
                if not pricings:
                    continue
                    
                for pricing_idx, pricing in enumerate(pricings):
                    row = {
                        'Id': f"{ranker_id}_{idx}_{pricing_idx}",
                        'ranker_id': ranker_id,
                        'profileId': profile_id,
                        'companyID': company_id,
                        'sex': sex,
                        'nationality': nationality,
                        'isVip': is_vip,
                        'bySelf': by_self,
                        'isAccess3D': is_access_3d,
                        'searchRoute': search_route,
                        'requestDate': request_date,
                        'totalPrice': pricing.get('totalPrice'),
                        'taxes': pricing.get('taxes'),
                        'selected': 1 if pricing_idx == 0 else 0  # Assume first pricing is selected
                    }
                    
                    # Extract corporate tariff code
                    row['corporateTariffCode'] = pricing.get('corporateTariffCode')
                    
                    # Extract frequent flyer (simplified)
                    row['frequentFlyer'] = None  # Not easily extractable from this JSON structure
                    
                    # Extract pricing info
                    pricing_info = pricing.get('pricingInfo', [])
                    if pricing_info:
                        first_pricing = pricing_info[0]
                        row['pricingInfo_isAccessTP'] = first_pricing.get('isAccessTP', False)
                        row['pricingInfo_passengerCount'] = first_pricing.get('passengerCount', 1)
                    
                    # Extract legs information
                    for leg_idx, leg in enumerate(legs[:2]):  # Max 2 legs (legs0, legs1)
                        leg_prefix = f'legs{leg_idx}'
                        
                        row[f'{leg_prefix}_departureAt'] = leg.get('departureAt')
                        row[f'{leg_prefix}_arrivalAt'] = leg.get('arrivalAt')
                        row[f'{leg_prefix}_duration'] = leg.get('duration')
                        
                        # Extract segments
                        segments = leg.get('segments', [])
                        for seg_idx, segment in enumerate(segments[:4]):  # Max 4 segments
                            seg_prefix = f'{leg_prefix}_segments{seg_idx}'
                            
                            # Airport codes
                            dept_airport = segment.get('departureFrom', {}).get('airport', {})
                            arr_airport = segment.get('arrivalTo', {}).get('airport', {})
                            
                            row[f'{seg_prefix}_departureFrom_airport_iata'] = dept_airport.get('iata')
                            row[f'{seg_prefix}_arrivalTo_airport_iata'] = arr_airport.get('iata')
                            row[f'{seg_prefix}_arrivalTo_airport_city_iata'] = arr_airport.get('city', {}).get('iata')
                            
                            # Carrier information
                            marketing_carrier = segment.get('marketingCarrier', {})
                            operating_carrier = segment.get('operatingCarrier', {})
                            
                            row[f'{seg_prefix}_marketingCarrier_code'] = marketing_carrier.get('code')
                            row[f'{seg_prefix}_operatingCarrier_code'] = operating_carrier.get('code')
                            row[f'{seg_prefix}_flightNumber'] = segment.get('flightNumber')
                            row[f'{seg_prefix}_duration'] = segment.get('duration')
                            
                            # Aircraft and service info
                            aircraft = segment.get('aircraft', {})
                            row[f'{seg_prefix}_aircraft_code'] = aircraft.get('code')
                            
                            # Extract baggage and cabin info from pricing
                            if pricing_info and pricing_info[0].get('faresInfo'):
                                fare_info = pricing_info[0]['faresInfo'][0]
                                baggage = fare_info.get('baggageAllowance', {})
                                row[f'{seg_prefix}_baggageAllowance_quantity'] = baggage.get('quantity')
                                row[f'{seg_prefix}_baggageAllowance_weightMeasurementType'] = baggage.get('weightMeasurementType')
                                row[f'{seg_prefix}_cabinClass'] = fare_info.get('cabinClass')
                                row[f'{seg_prefix}_seatsAvailable'] = fare_info.get('seatsAvailable')
                    
                    # Extract mini rules (cancellation and exchange)
                    mini_rules = pricing.get('miniRules', [])
                    for rule in mini_rules:
                        category = rule.get('category')
                        if category == 31:  # Cancellation
                            row['miniRules0_monetaryAmount'] = rule.get('monetaryAmount')
                            row['miniRules0_percentage'] = rule.get('percentage')
                            row['miniRules0_statusInfos'] = rule.get('statusInfos')
                        elif category == 33:  # Exchange
                            row['miniRules1_monetaryAmount'] = rule.get('monetaryAmount')
                            row['miniRules1_percentage'] = rule.get('percentage')
                            row['miniRules1_statusInfos'] = rule.get('statusInfos')
                    
                    flight_options.append(row)
            
            return flight_options
            
        except Exception as e:
            print(f"Error parsing {json_file}: {e}")
            return []
    
    def convert_all_jsons(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """Convert all JSON files to a single DataFrame"""
        print("🔄 Converting JSON files to DataFrame...")
        
        json_files = list(self.json_dir.glob("*.json"))
        if max_files:
            json_files = json_files[:max_files]
        
        all_flight_options = []
        processed = 0
        
        for json_file in json_files:
            flight_options = self.parse_single_json(json_file)
            all_flight_options.extend(flight_options)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"   Processed {processed:,} files ({len(all_flight_options):,} rows)...")
                gc.collect()
        
        print(f"✅ Converted {processed:,} JSON files to {len(all_flight_options):,} flight options")
        
        df = pd.DataFrame(all_flight_options)
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, test_ratio: float = 0.3):
        """Create train/test split by ranker_id"""
        unique_rankers = df['ranker_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_rankers)
        
        split_idx = int(len(unique_rankers) * (1 - test_ratio))
        train_rankers = unique_rankers[:split_idx]
        test_rankers = unique_rankers[split_idx:]
        
        train_df = df[df['ranker_id'].isin(train_rankers)].copy()
        test_df = df[df['ranker_id'].isin(test_rankers)].copy()
        
        # Remove 'selected' column from test set
        if 'selected' in test_df.columns:
            test_df = test_df.drop('selected', axis=1)
        
        return train_df, test_df
    
    def create_sample_submission(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Create sample submission file"""
        submission = test_df[['Id', 'ranker_id']].copy()
        
        # Create random ranks for each ranker_id group
        def assign_ranks(group):
            n = len(group)
            ranks = np.arange(1, n + 1)
            np.random.shuffle(ranks)
            return ranks
        
        submission['selected'] = test_df.groupby('ranker_id').apply(
            lambda x: pd.Series(assign_ranks(x), index=x.index)
        ).values
        
        return submission
    
    def run_conversion(self, max_files: Optional[int] = None):
        """Run the complete conversion process"""
        print("🚀 STARTING JSON TO PARQUET CONVERSION")
        print("=" * 50)
        
        # Convert JSON files
        df = self.convert_all_jsons(max_files=max_files)
        
        # Memory optimization
        print("🔧 Optimizing memory usage...")
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                except:
                    pass
        
        # Create train/test split
        print("🎯 Creating train/test split...")
        train_df, test_df = self.create_train_test_split(df)
        
        # Save files
        print("💾 Saving parquet files...")
        train_df.to_parquet(self.output_dir / 'train.parquet', index=False)
        test_df.to_parquet(self.output_dir / 'test.parquet', index=False)
        
        # Create sample submission
        sample_submission = self.create_sample_submission(test_df)
        sample_submission.to_parquet(self.output_dir / 'sample_submission.parquet', index=False)
        
        print("✅ CONVERSION COMPLETED!")
        print(f"📊 Train: {len(train_df):,} rows, {len(train_df['ranker_id'].unique()):,} sessions")
        print(f"📊 Test: {len(test_df):,} rows, {len(test_df['ranker_id'].unique()):,} sessions")
        print(f"📄 Files saved: train.parquet, test.parquet, sample_submission.parquet")
        
        return train_df, test_df, sample_submission

if __name__ == "__main__":
    converter = JSONToParquetConverter()
    
    # For quick testing, convert first 1000 files
    # For full conversion, remove max_files parameter
    train_df, test_df, submission = converter.run_conversion(max_files=1000)
