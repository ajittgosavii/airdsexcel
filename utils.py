import pandas as pd
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import html

def parse_uploaded_file(uploaded_file):
    """Debug version - replace the existing function in utils.py with this temporarily"""
    import pandas as pd
    
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, ["Unsupported file format. Please upload CSV or Excel file."]
        
        # Debug: Print original data
        print(f"=== DEBUG: Original DataFrame ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        
        # Required columns mapping
        required_columns = {
            'database_engine': 'engine',
            'aws_region': 'region',
            'cpu_cores': 'cores',
            'cpu_utilization': 'cpu_util',
            'ram_gb': 'ram',
            'ram_utilization': 'ram_util',
            'storage_gb': 'storage',
            'iops': 'iops'
        }
        
        # Check for required columns
        missing_columns = [col for col in required_columns.keys() if col not in df.columns]
        if missing_columns:
            return None, [f"Missing required columns: {', '.join(missing_columns)}"]
        
        # Create a copy and rename columns
        df_processed = df.copy()
        df_processed.rename(columns=required_columns, inplace=True)
        
        # Add optional columns with defaults
        optional_columns = {
            'growth_rate': ('growth', 15),
            'backup_days': ('backup_days', 7),
            'projection_years': ('years', 3),
            'data_transfer_gb': ('data_transfer_gb', 100)
        }
        
        for col, (new_name, default) in optional_columns.items():
            if col in df.columns:
                df_processed[new_name] = df[col]
            else:
                df_processed[new_name] = default
        
        # Add database name
        if 'database_name' in df.columns:
            df_processed['db_name'] = df['database_name']
        else:
            df_processed['db_name'] = [f"Database {i+1}" for i in range(len(df_processed))]
        
        print(f"=== DEBUG: Processed DataFrame ===")
        print(f"Shape: {df_processed.shape}")
        print(f"Columns: {list(df_processed.columns)}")
        print(f"First processed row:\n{df_processed.iloc[0].to_dict()}")
        
        # Convert to list of dictionaries
        inputs_list = df_processed.to_dict(orient='records')
        
        print(f"=== DEBUG: Conversion to list ===")
        print(f"Number of dictionaries: {len(inputs_list)}")
        print(f"Sample dictionary keys: {list(inputs_list[0].keys()) if inputs_list else 'None'}")
        
        # Simple validation - just check for basic required fields
        valid_inputs = []
        errors = []
        
        for idx, input_data in enumerate(inputs_list):
            # Very basic validation
            basic_fields = ['cores', 'ram', 'storage', 'cpu_util', 'ram_util']
            row_errors = []
            
            for field in basic_fields:
                if field not in input_data:
                    row_errors.append(f"Missing {field}")
                elif pd.isna(input_data[field]) or input_data[field] is None:
                    row_errors.append(f"{field} is null/empty")
                elif not isinstance(input_data[field], (int, float)) or input_data[field] <= 0:
                    row_errors.append(f"{field} is not a positive number: {input_data[field]}")
            
            if not row_errors:
                valid_inputs.append(input_data)
                print(f"DEBUG: Row {idx+1} ({input_data.get('db_name', 'Unknown')}) - VALID")
            else:
                db_name = input_data.get('db_name', f"Row {idx+1}")
                error_msg = f"{db_name}: {', '.join(row_errors)}"
                errors.append(error_msg)
                print(f"DEBUG: Row {idx+1} ({db_name}) - INVALID: {row_errors}")
        
        print(f"=== DEBUG: Final Results ===")
        print(f"Valid inputs: {len(valid_inputs)}")
        print(f"Errors: {len(errors)}")
        if errors:
            print("Error details:", errors)
        
        return valid_inputs, errors
        
    except Exception as e:
        import traceback
        print(f"=== DEBUG: Exception occurred ===")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, [f"Error processing file: {str(e)}"]