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
        
        # After loading datasets, optimize thresholds for each model
        self._optimize_model_thresholds()
        
    def _optimize_model_thresholds(self):
        """
        Determine optimal threshold for each model based on validation data
        This method should be called after datasets are loaded
        
        Additionally calculates all performance metrics for each model and stores them in the model metadata
        """
        print("Optimizing thresholds for each model...")
        
        for i, model_info in enumerate(self.models_info):
            dataset_name = model_info['dataset_name']
            if dataset_name not in self.datasets:
                print(f"Cannot optimize threshold for {model_info['display_name']}: Dataset {dataset_name} not loaded")
                continue
                
            # Get the dataset
            df = self.datasets[dataset_name]
            
            # Check if dataset has target variable
            if 'Y' not in df.columns:
                print(f"Cannot optimize threshold for {model_info['display_name']}: No target variable in dataset")
                continue
            
            # Use data from 2000 to 2018 as validation set
            val_data = df.loc['2000-01-01':'2018-12-31']
            if val_data.empty:
                print(f"No validation data available for {model_info['display_name']}")
                continue
            
            # Extract features and target
            X = val_data.drop(columns=['Y'] + (['pre_crisis'] if 'pre_crisis' in val_data.columns else []))
            y = val_data['Y']
            
            # Filter to relevant features if specified
            features = model_info['features']
            if features and len(features) > 0:
                common_features = [feat for feat in features if feat in X.columns]
                if len(common_features) == 0:
                    print(f"No common features for {model_info['display_name']}")
                    continue
                X = X[common_features]
            
            # Apply preprocessing
            try:
                # Apply imputer if available
                imputer = model_info.get('imputer')
                if imputer is not None:
                    X = pd.DataFrame(
                        imputer.transform(X), 
                        columns=X.columns,
                        index=X.index
                    )
                
                # Apply scaler if available
                scaler = model_info.get('scaler')
                if scaler is not None:
                    X = pd.DataFrame(
                        scaler.transform(X), 
                        columns=X.columns,
                        index=X.index
                    )
            except Exception as e:
                print(f"Error in preprocessing for {model_info['display_name']}: {str(e)}")
                continue
            
            # Get predictions
            model = model_info['model']
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]
                else:
                    # Skip models that don't output probabilities
                    print(f"Model {model_info['display_name']} doesn't output probabilities, skipping threshold optimization")
                    continue
            except Exception as e:
                print(f"Error predicting with {model_info['display_name']}: {str(e)}")
                continue
            
            # Calculate metrics for multiple thresholds
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                # Get the metric the model was optimized for
                metric_name = model_info['metadata'].get('metric', 'f1_score')
                
                # Try different thresholds
                thresholds = np.linspace(0.01, 0.99, 99)
                
                # Store all metrics for each threshold
                all_metrics = {
                    'thresholds': thresholds,
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1_score': [],
                }
                
                # Calculate roc_auc once since it doesn't depend on threshold
                roc_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0
                
                # Calculate metrics for each threshold
                for threshold in thresholds:
                    preds = (y_proba >= threshold).astype(int)
                    all_metrics['accuracy'].append(accuracy_score(y, preds))
                    all_metrics['precision'].append(precision_score(y, preds, zero_division=0))
                    all_metrics['recall'].append(recall_score(y, preds, zero_division=0))
                    all_metrics['f1_score'].append(f1_score(y, preds, zero_division=0))
                
                # Find best threshold according to the model's primary metric
                best_idx = np.argmax(all_metrics[metric_name])
                best_threshold = thresholds[best_idx]
                
                # Store all metrics at the best threshold in the model metadata
                model_info['metadata']['threshold'] = float(best_threshold)
                model_info['metadata']['accuracy'] = float(all_metrics['accuracy'][best_idx])
                model_info['metadata']['precision'] = float(all_metrics['precision'][best_idx])
                model_info['metadata']['recall'] = float(all_metrics['recall'][best_idx])
                model_info['metadata']['f1_score'] = float(all_metrics['f1_score'][best_idx])
                model_info['metadata']['roc_auc'] = float(roc_auc)
                
                # Also evaluate on test data (2019-2023) to see out-of-sample performance
                try:
                    test_data = df.loc['2019-01-01':'2023-12-31'].copy()
                    
                    if not test_data.empty and 'Y' in test_data.columns:
                        # Process test data
                        X_test = test_data.drop(columns=['Y'] + (['pre_crisis'] if 'pre_crisis' in test_data.columns else []))
                        y_test = test_data['Y']
                        
                        # Filter features
                        if features and len(features) > 0:
                            X_test = X_test[common_features].copy()
                            
                        # Apply preprocessing
                        if imputer is not None:
                            X_test = pd.DataFrame(
                                imputer.transform(X_test),
                                columns=X_test.columns,
                                index=X_test.index
                            )
                            
                        if scaler is not None:
                            X_test = pd.DataFrame(
                                scaler.transform(X_test),
                                columns=X_test.columns,
                                index=X_test.index
                            )
                        
                        # Get predictions on test data
                        y_test_proba = model.predict_proba(X_test)[:, 1]
                        y_test_pred = (y_test_proba >= best_threshold).astype(int)
                        
                        # Calculate and store test metrics
                        model_info['metadata']['test_metrics'] = {
                            'accuracy': float(accuracy_score(y_test, y_test_pred)),
                            'precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
                            'recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
                            'f1_score': float(f1_score(y_test, y_test_pred, zero_division=0)),
                            'roc_auc': float(roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0)
                        }
                except Exception as e:
                    print(f"Error evaluating on test data for {model_info['display_name']}: {str(e)}")
                
                print(f"Optimized threshold for {model_info['display_name']}: {best_threshold:.4f}")
                print(f"  Metrics at this threshold: Acc={all_metrics['accuracy'][best_idx]:.4f}, "
                      f"Prec={all_metrics['precision'][best_idx]:.4f}, "
                      f"Rec={all_metrics['recall'][best_idx]:.4f}, "
                      f"F1={all_metrics['f1_score'][best_idx]:.4f}, "
                      f"AUC={roc_auc:.4f}")
                
                if 'test_metrics' in model_info['metadata']:
                    test_metrics = model_info['metadata']['test_metrics']
                    print(f"  Test metrics: Acc={test_metrics['accuracy']:.4f}, "
                          f"Prec={test_metrics['precision']:.4f}, "
                          f"Rec={test_metrics['recall']:.4f}, "
                          f"F1={test_metrics['f1_score']:.4f}, "
                          f"AUC={test_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"Error optimizing threshold for {model_info['display_name']}: {str(e)}")
                # Use default threshold or one from metadata if optimization fails
                if 'threshold' not in model_info['metadata']:
                    model_info['metadata']['threshold'] = 0.5
                    print(f"Using default threshold 0.5")

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
        
        # Get model-specific threshold from metadata
        # Use the thresholds from the model metadata
        model_type = model_info.get('model_type', '')
        model_metric = model_info['metadata'].get('metric', '')
        
        # Default threshold is 0.5
        threshold = 0.5
        
        # Use specific thresholds from model metadata
        if 'threshold' in model_info['metadata']:
            threshold = model_info['metadata']['threshold']
            
        try:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)[:, 1]
                preds = (probas >= threshold).astype(int)
            else:
                preds = model.predict(X)
                probas = preds.astype(float)
            
            # Create result DataFrame
            result = pd.DataFrame({
                'date': X.index,
                'prediction': preds,
                'probability': probas,
                'threshold': [threshold] * len(X),  # Add the threshold used
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
    
    def get_market_data(self, start_date, end_date, indicator_name=None):
        """
        Get market data for the selected date range and indicator
        
        Parameters:
        -----------
        start_date : str
            Start date
        end_date : str
            End date
        indicator_name : str, optional
            Specific indicator column name to get
            
        Returns:
        --------
        DataFrame
            DataFrame with market data
        """
        # Set up a market data collection if we don't have one yet
        if 'market_data' not in self.datasets:
            self.datasets['market_data'] = pd.DataFrame()
            
            # Collect common market indicators from all datasets
            for dataset_name, df in self.datasets.items():
                if dataset_name != 'market_data':
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(term in col_lower for term in ['vix', 'index', 'price', 'market', 'sp500', 's&p', 'dji', 'djia', 'nasdaq']):
                            self.datasets['market_data'][col] = df[col]
            
            # Add Y column if available in any dataset
            for dataset_name, df in self.datasets.items():
                if 'Y' in df.columns:
                    self.datasets['market_data']['Y'] = df['Y']
                    break
        
        # Get the data slice for the date range
        df_slice = self.datasets['market_data'][start_date:end_date]
        
        # If a specific indicator is requested, filter to just that column (if it exists)
        if indicator_name and indicator_name in df_slice.columns:
            return df_slice[[indicator_name] + (['Y'] if 'Y' in df_slice.columns else [])]
        elif indicator_name:
            # Try to find similar column
            similar_cols = [col for col in df_slice.columns if indicator_name.lower() in col.lower()]
            if similar_cols:
                return df_slice[similar_cols + (['Y'] if 'Y' in df_slice.columns else [])]
        
        # If no specific indicator or not found, return all market data
        return df_slice
    
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
                # Get model-specific threshold from metadata if available, otherwise use default 0.5
                threshold = model_info['metadata'].get('threshold', 0.5)
                pred = int(proba >= threshold)
            else:
                pred = int(model.predict(X)[0])
                proba = float(pred)
                
            # Store the threshold used for this prediction
            threshold_used = model_info['metadata'].get('threshold', 0.5)
            
            # Get actual value if available
            actual = float(data_point['Y'].iloc[0]) if 'Y' in data_point.columns else None
            
            # Get market indicator values
            # First collect various market indicators
            market_indicators = {}
            
            # Check for VIX
            vix_cols = [col for col in data_point.columns if 'vix' in col.lower()]
            if vix_cols:
                market_indicators['vix'] = float(data_point[vix_cols[0]].iloc[0])
            
            # Check for other important indicators
            for indicator_type in ['index', 'price', 'sp500', 'dji', 'nasdaq']:
                indicator_cols = [col for col in data_point.columns 
                                 if indicator_type in col.lower() and col not in vix_cols]
                if indicator_cols:
                    market_indicators[indicator_type] = float(data_point[indicator_cols[0]].iloc[0])
            
            # Now also check market_data collection for more indicators
            if 'market_data' in self.datasets:
                market_df = self.datasets['market_data']
                if specific_ts in market_df.index:
                    market_row = market_df.loc[specific_ts]
                    for col in market_row.index:
                        if col != 'Y' and not pd.isna(market_row[col]):
                            market_indicators[col] = float(market_row[col])
            
            # Return prediction info
            result = {
                'date': data_point.index[0].strftime('%Y-%m-%d'),
                'prediction': pred,
                'probability': proba,
                'actual': actual,
                'threshold': threshold_used,
            }
            
            # Add all market indicators to the result
            result.update(market_indicators)
            
            return result
            
        except Exception as e:
            print(f"Error making prediction for specific date: {str(e)}")
            return None 