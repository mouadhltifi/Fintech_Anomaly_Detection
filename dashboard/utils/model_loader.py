import os
import joblib
import pandas as pd
import numpy as np
from glob import glob

class ModelManager:
    """
    Class to load and manage models for the financial crisis early warning system dashboard
    """
    def __init__(self, models_dir='/Users/mouadh/Fintech_Projects/Business_Case_4/notebooks/models'):
        self.models_dir = models_dir
        self.models_info = []
        self.model_names = []
        self.datasets = {}
        self.load_models()
        
    def load_models(self):
        """Load all models from the models directory"""
        # Find all best model directories
        model_dirs = []
        for metric in ['f1_score', 'precision', 'recall', 'roc_auc', 'accuracy']:
            dirs = glob(os.path.join(self.models_dir, f'best_{metric}_model_*'))
            model_dirs.extend(dirs)
        
        print(f"Found {len(model_dirs)} model directories")
        
        # Load each model
        for model_dir in model_dirs:
            model_path = os.path.join(model_dir, 'model.joblib')
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            imputer_path = os.path.join(model_dir, 'imputer.joblib')
            metadata_path = os.path.join(model_dir, 'metadata.json')
            
            if not os.path.exists(model_path):
                print(f"Warning: No model file in {model_dir}")
                continue
            
            try:
                # Load model
                model = joblib.load(model_path)
                
                # Load scaler if it exists
                scaler = None
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    
                # Load imputer if it exists
                imputer = None
                if os.path.exists(imputer_path):
                    imputer = joblib.load(imputer_path)
                    
                # Load metadata if it exists
                import json
                metadata = {}
                features = []
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    features = metadata.get('features', [])
                
                # Create readable model name
                model_type = type(model).__name__
                dataset_name = metadata.get('dataset', 'Unknown')
                metric_name = metadata.get('metric', 'Unknown')
                metric_score = metadata.get('score', 0)
                
                display_name = f"{model_type} - {dataset_name} (Best {metric_name}: {metric_score:.4f})"
                
                # Create model info dictionary
                model_info = {
                    'model': model,
                    'scaler': scaler,
                    'imputer': imputer,
                    'features': features,
                    'metadata': metadata,
                    'model_dir': model_dir,
                    'display_name': display_name,
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                }
                
                self.models_info.append(model_info)
                self.model_names.append(display_name)
                
                print(f"Loaded model: {display_name}")
                
            except Exception as e:
                print(f"Error loading model from {model_dir}: {str(e)}")
    
    def load_datasets(self, data_dir='/Users/mouadh/Fintech_Projects/Business_Case_4/data/processed/'):
        """Load all required datasets for predictions"""
        datasets_needed = set()
        for model_info in self.models_info:
            dataset_name = model_info.get('dataset_name')
            if dataset_name:
                datasets_needed.add(dataset_name)
        
        print(f"Loading {len(datasets_needed)} datasets...")
        for dataset_name in datasets_needed:
            if dataset_name == 'interpolated_full':
                file_path = os.path.join(data_dir, 'interpolated_data_daily_full.csv')
                print(f"Loading {dataset_name} from {file_path}")
                try:
                    self.datasets[dataset_name] = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    print(f"  Loaded {dataset_name} with shape {self.datasets[dataset_name].shape}")
                except Exception as e:
                    print(f"  Error loading {dataset_name}: {str(e)}")
            else:
                file_path = os.path.join(data_dir, f'feature_set_{dataset_name}.csv')
                print(f"Loading {dataset_name} from {file_path}")
                try:
                    self.datasets[dataset_name] = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    print(f"  Loaded {dataset_name} with shape {self.datasets[dataset_name].shape}")
                except Exception as e:
                    print(f"  Error loading {dataset_name}: {str(e)}")
    
    def get_model_by_name(self, display_name):
        """Get model info by display name"""
        for model_info in self.models_info:
            if model_info['display_name'] == display_name:
                return model_info
        return None
    
    def predict(self, model_info, start_date, end_date):
        """
        Make predictions using a specific model for a given date range
        
        Parameters:
        -----------
        model_info : dict
            Model info dictionary
        start_date : str
            Start date for predictions
        end_date : str
            End date for predictions
            
        Returns:
        --------
        DataFrame
            DataFrame with dates, predictions, and probabilities
        """
        dataset_name = model_info['dataset_name']
        
        if dataset_name not in self.datasets:
            print(f"Error: Dataset {dataset_name} not loaded")
            return None
        
        # Get the data slice for the date range
        df = self.datasets[dataset_name]
        df_slice = df[start_date:end_date]
        
        if df_slice.empty:
            print(f"No data available for the selected date range")
            return None
        
        # Extract features
        features = model_info['features']
        X = df_slice.drop(columns=['Y'] + (['pre_crisis'] if 'pre_crisis' in df_slice.columns else []))
        
        # Filter to relevant features if specified
        if features and len(features) > 0:
            common_features = [feat for feat in features if feat in X.columns]
            if not common_features:
                print(f"Error: No common features for this model")
                return None
            X = X[common_features].copy()
        
        # Apply imputer if available
        imputer = model_info.get('imputer')
        if imputer is not None:
            try:
                X = pd.DataFrame(
                    imputer.transform(X), 
                    columns=X.columns,
                    index=X.index
                )
            except Exception as e:
                print(f"Error applying imputer: {str(e)}")
        
        # Apply scaler if available
        scaler = model_info.get('scaler')
        if scaler is not None:
            try:
                X = pd.DataFrame(
                    scaler.transform(X), 
                    columns=X.columns,
                    index=X.index
                )
            except Exception as e:
                print(f"Error applying scaler: {str(e)}")
        
        # Make predictions
        model = model_info['model']
        try:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)[:, 1]
                preds = (probas >= 0.5).astype(int)
            else:
                preds = model.predict(X)
                probas = preds.astype(float)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'date': X.index,
                'prediction': preds,
                'probability': probas,
                'actual': df_slice['Y'] if 'Y' in df_slice.columns else np.nan
            })
            
            return result
        
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return None
    
    def get_vix_data(self, start_date, end_date):
        """
        Get VIX data for the selected date range
        
        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        DataFrame
            DataFrame with VIX data
        """
        # First try the interpolated dataset as it should have VIX
        if 'interpolated_full' in self.datasets:
            df = self.datasets['interpolated_full']
            df_slice = df[start_date:end_date]
            
            # Look for VIX column - could be called VIX, vix, or similar
            vix_cols = [col for col in df_slice.columns if 'vix' in col.lower()]
            
            if vix_cols:
                return df_slice[vix_cols + ['Y']].copy()
        
        # If not found, try other datasets
        for dataset_name, df in self.datasets.items():
            df_slice = df[start_date:end_date]
            vix_cols = [col for col in df_slice.columns if 'vix' in col.lower()]
            
            if vix_cols:
                return df_slice[vix_cols + (['Y'] if 'Y' in df_slice.columns else [])].copy()
        
        print("VIX data not found in any dataset")
        return None

    def predict_for_day(self, model_info, specific_date):
        """
        Make a prediction using a specific model for a single date
        
        Parameters:
        -----------
        model_info : dict
            Model info dictionary
        specific_date : str
            Specific date for prediction (YYYY-MM-DD)
            
        Returns:
        --------
        dict
            Dictionary with prediction information or None if no data
        """
        dataset_name = model_info['dataset_name']
        
        if dataset_name not in self.datasets:
            print(f"Error: Dataset {dataset_name} not loaded")
            return None
        
        # Get the data for the specific date
        df = self.datasets[dataset_name]
        
        try:
            # Convert to pandas timestamp
            specific_ts = pd.Timestamp(specific_date)
            
            # Try to get exact date match
            if specific_ts in df.index:
                data_point = df.loc[specific_ts:specific_ts]
            else:
                # Find nearest available date (not future)
                available_dates = df.index[df.index <= specific_ts]
                if len(available_dates) == 0:
                    print(f"No data available on or before {specific_date}")
                    return None
                    
                nearest_date = available_dates[-1]  # Get the most recent date
                data_point = df.loc[nearest_date:nearest_date]
                print(f"Using nearest available date: {nearest_date.strftime('%Y-%m-%d')}")
            
            if data_point.empty:
                print(f"No data available for the selected date")
                return None
            
            # Extract features
            features = model_info['features']
            X = data_point.drop(columns=['Y'] + (['pre_crisis'] if 'pre_crisis' in data_point.columns else []))
            
            # Filter to relevant features if specified
            if features and len(features) > 0:
                common_features = [feat for feat in features if feat in X.columns]
                if not common_features:
                    print(f"Error: No common features for this model")
                    return None
                X = X[common_features].copy()
            
            # Apply imputer if available
            imputer = model_info.get('imputer')
            if imputer is not None:
                try:
                    X = pd.DataFrame(
                        imputer.transform(X), 
                        columns=X.columns,
                        index=X.index
                    )
                except Exception as e:
                    print(f"Error applying imputer: {str(e)}")
            
            # Apply scaler if available
            scaler = model_info.get('scaler')
            if scaler is not None:
                try:
                    X = pd.DataFrame(
                        scaler.transform(X), 
                        columns=X.columns,
                        index=X.index
                    )
                except Exception as e:
                    print(f"Error applying scaler: {str(e)}")
            
            # Make prediction
            model = model_info['model']
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0, 1]  # Get probability of class 1
                pred = int(proba >= 0.5)
            else:
                pred = int(model.predict(X)[0])
                proba = float(pred)
            
            # Get actual value if available
            actual = float(data_point['Y'].iloc[0]) if 'Y' in data_point.columns else None
            
            # Get VIX value if available
            vix_value = None
            vix_cols = [col for col in data_point.columns if 'vix' in col.lower()]
            if vix_cols:
                vix_value = float(data_point[vix_cols[0]].iloc[0])
            
            # Return prediction info
            return {
                'date': data_point.index[0].strftime('%Y-%m-%d'),
                'prediction': pred,
                'probability': proba,
                'actual': actual,
                'vix': vix_value
            }
            
        except Exception as e:
            print(f"Error making prediction for specific date: {str(e)}")
            return None 