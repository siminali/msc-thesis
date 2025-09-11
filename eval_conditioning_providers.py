#!/usr/bin/env python3
"""
Evaluation-Time Conditioning Providers

Provides causal, day-by-day conditioning for evaluation without refitting on evaluation data.
All providers use exact transforms saved in conditioning_spec.json and handle missing data gracefully.

Features:
- Strictly causal computation (no look-ahead bias)
- Spec-consistent transforms (never refit on evaluation data)
- Graceful fallback handling for incomplete specs
- Comprehensive logging and error handling
- Support for all model types (zero, explicit, LLM)

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseEvalProvider:
    """Base class for evaluation-time conditioning providers."""
    
    def __init__(self, conditioning_spec: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        self.conditioning_spec = conditioning_spec
        self.checkpoint_dir = checkpoint_dir
        self.conditioning_type = conditioning_spec.get('type', 'unknown')
        self.conditioning_dim = conditioning_spec.get('conditioning_dim', 0)
        
        # Validation flags
        self.spec_complete = True
        self.warnings = []
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.conditioning_type} conditioning")
    
    def validate_spec(self) -> bool:
        """Validate the conditioning specification."""
        required_fields = ['type', 'conditioning_dim']
        
        for field in required_fields:
            if field not in self.conditioning_spec:
                self.spec_complete = False
                warning_msg = f"Missing required field '{field}' in conditioning spec"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        return self.spec_complete
    
    def generate_conditioning(self, returns_data: pd.DataFrame, 
                            target_dates: List[pd.Timestamp]) -> Optional[np.ndarray]:
        """Generate conditioning vectors for target dates."""
        raise NotImplementedError("Subclasses must implement generate_conditioning")
    
    def get_warnings(self) -> List[str]:
        """Get all warnings accumulated during initialization and computation."""
        return self.warnings.copy()

class NoneProvider(BaseEvalProvider):
    """Provider for zero conditioning (no conditioning)."""
    
    def __init__(self, conditioning_spec: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        super().__init__(conditioning_spec, checkpoint_dir)
        
        # Validate zero conditioning spec
        if self.conditioning_type != 'zero':
            warning_msg = f"Expected zero conditioning, got {self.conditioning_type}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        if self.conditioning_dim != 0:
            warning_msg = f"Zero conditioning should have dim=0, got {self.conditioning_dim}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    def generate_conditioning(self, returns_data: pd.DataFrame, 
                            target_dates: List[pd.Timestamp]) -> None:
        """Zero conditioning always returns None."""
        logger.info(f"Zero conditioning: returning None for {len(target_dates)} target dates")
        return None

class ExplicitEvalProvider(BaseEvalProvider):
    """Provider for explicit conditioning with causal regime classification and financial features."""
    
    def __init__(self, conditioning_spec: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        super().__init__(conditioning_spec, checkpoint_dir)
        
        # Validate explicit conditioning spec
        if self.conditioning_type != 'explicit':
            warning_msg = f"Expected explicit conditioning, got {self.conditioning_type}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Extract feature specifications from saved spec
        self.features = conditioning_spec.get('features', {})
        self.vol_threshold = conditioning_spec.get('vol_threshold', None)
        self.vol_window = conditioning_spec.get('vol_window', 20)
        self.trend_window = conditioning_spec.get('trend_window', 60)
        
        # Extract scaler parameters (never refit on evaluation data)
        vol_info = self.features.get('z_vol', {})
        trend_info = self.features.get('trend', {})
        
        self.vol_scaler_mean = vol_info.get('scaler_mean', None)
        self.vol_scaler_scale = vol_info.get('scaler_scale', None)
        self.trend_scaler_mean = trend_info.get('scaler_mean', None)
        self.trend_scaler_scale = trend_info.get('scaler_scale', None)
        
        # Validate completeness of spec
        self._validate_explicit_spec()
        
        logger.info(f"Explicit conditioning initialized:")
        logger.info(f"  Vol threshold: {self.vol_threshold}")
        logger.info(f"  Vol window: {self.vol_window}, Trend window: {self.trend_window}")
        logger.info(f"  Vol scaler: mean={self.vol_scaler_mean}, scale={self.vol_scaler_scale}")
        logger.info(f"  Trend scaler: mean={self.trend_scaler_mean}, scale={self.trend_scaler_scale}")
    
    def _validate_explicit_spec(self):
        """Validate the explicit conditioning specification."""
        missing_params = []
        
        if self.vol_threshold is None:
            missing_params.append('vol_threshold')
        if self.vol_scaler_mean is None:
            missing_params.append('vol_scaler_mean')
        if self.vol_scaler_scale is None:
            missing_params.append('vol_scaler_scale')
        if self.trend_scaler_mean is None:
            missing_params.append('trend_scaler_mean')
        if self.trend_scaler_scale is None:
            missing_params.append('trend_scaler_scale')
        
        if missing_params:
            self.spec_complete = False
            warning_msg = f"Incomplete explicit conditioning spec, missing: {missing_params}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    def _compute_causal_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute causal rolling volatility (no look-ahead)."""
        # Use expanding window for early periods, then rolling window
        volatility = pd.Series(index=returns.index, dtype=float)
        
        for i in range(len(returns)):
            if i < window - 1:
                # Use expanding window for early periods
                vol_data = returns.iloc[:i+1]
            else:
                # Use rolling window
                vol_data = returns.iloc[i-window+1:i+1]
            
            if len(vol_data) > 1:
                volatility.iloc[i] = vol_data.std()
            else:
                volatility.iloc[i] = 0.0
        
        return volatility
    
    def _compute_causal_trend(self, returns: pd.Series, window: int) -> pd.Series:
        """Compute causal rolling trend (cumulative sum, no look-ahead)."""
        # Use expanding window for early periods, then rolling window
        trend = pd.Series(index=returns.index, dtype=float)
        
        for i in range(len(returns)):
            if i < window - 1:
                # Use expanding window for early periods
                trend_data = returns.iloc[:i+1]
            else:
                # Use rolling window
                trend_data = returns.iloc[i-window+1:i+1]
            
            trend.iloc[i] = trend_data.sum()
        
        return trend
    
    def _create_fallback_parameters(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Create fallback parameters from early data when spec is incomplete."""
        logger.warning("Creating fallback parameters from evaluation data (SUSPECT)")
        
        # Use first 100 days or available data for fallback estimation
        early_data = returns_data.head(min(100, len(returns_data)))
        early_returns = early_data['returns'] if 'returns' in early_data.columns else early_data.iloc[:, 0]
        
        # Compute fallback volatility
        fallback_vol = self._compute_causal_volatility(early_returns, self.vol_window)
        fallback_trend = self._compute_causal_trend(early_returns, self.trend_window)
        
        fallback_params = {
            'vol_threshold': fallback_vol.median(),
            'vol_scaler_mean': fallback_vol.mean(),
            'vol_scaler_scale': fallback_vol.std() if fallback_vol.std() > 0 else 1.0,
            'trend_scaler_mean': fallback_trend.mean(),
            'trend_scaler_scale': fallback_trend.std() if fallback_trend.std() > 0 else 1.0
        }
        
        logger.warning(f"Fallback parameters: {fallback_params}")
        return fallback_params
    
    def generate_conditioning(self, returns_data: pd.DataFrame, 
                            target_dates: List[pd.Timestamp]) -> np.ndarray:
        """Generate explicit conditioning vectors causally for target dates."""
        logger.info(f"Generating explicit conditioning for {len(target_dates)} target dates")
        
        # Ensure we have returns column
        if 'returns' not in returns_data.columns:
            if len(returns_data.columns) == 1:
                returns_data = returns_data.rename(columns={returns_data.columns[0]: 'returns'})
            else:
                raise ValueError("Returns data must have 'returns' column or be single-column")
        
        # Handle incomplete spec with fallbacks
        params = {}
        if not self.spec_complete:
            fallback_params = self._create_fallback_parameters(returns_data)
            params.update(fallback_params)
            
            # Log as suspect
            self.warnings.append("Using fallback parameters derived from evaluation data (SUSPECT)")
            logger.warning("SUSPECT: Using evaluation-derived fallback parameters")
        
        # Use saved parameters or fallbacks
        vol_threshold = self.vol_threshold if self.vol_threshold is not None else params.get('vol_threshold', 0.01)
        vol_mean = self.vol_scaler_mean if self.vol_scaler_mean is not None else params.get('vol_scaler_mean', 0.0)
        vol_scale = self.vol_scaler_scale if self.vol_scaler_scale is not None else params.get('vol_scaler_scale', 1.0)
        trend_mean = self.trend_scaler_mean if self.trend_scaler_mean is not None else params.get('trend_scaler_mean', 0.0)
        trend_scale = self.trend_scaler_scale if self.trend_scaler_scale is not None else params.get('trend_scaler_scale', 1.0)
        
        # Compute features causally for all available data
        returns_series = returns_data['returns']
        
        # Causal volatility computation
        volatility = self._compute_causal_volatility(returns_series, self.vol_window)
        
        # Causal trend computation  
        trend = self._compute_causal_trend(returns_series, self.trend_window)
        
        # Scale features using saved parameters (never refit)
        z_vol = (volatility - vol_mean) / vol_scale
        z_trend = (trend - trend_mean) / trend_scale
        
        # Create regime classification (Up/Down × Low/High volatility)
        is_up = returns_series > 0
        is_high_vol = volatility > vol_threshold
        
        # Generate conditioning vectors for target dates
        conditioning_vectors = []
        
        for target_date in target_dates:
            # Find the closest available date (causal - must be <= target_date)
            available_dates = returns_data.index[returns_data.index <= target_date]
            
            if len(available_dates) == 0:
                # No causal data available, use zero conditioning
                logger.warning(f"No causal data for {target_date}, using zero conditioning")
                conditioning_vector = np.zeros(6)  # 4 regime + vol + trend
            else:
                # Use the latest available causal date
                causal_date = available_dates[-1]
                
                # Get causal features
                up = is_up.loc[causal_date]
                high_vol = is_high_vol.loc[causal_date]
                vol_scaled = z_vol.loc[causal_date]
                trend_scaled = z_trend.loc[causal_date]
                
                # Create regime one-hot encoding
                regime_onehot = np.zeros(4)
                if up and not high_vol:
                    regime_onehot[0] = 1  # Up-Low
                elif up and high_vol:
                    regime_onehot[1] = 1  # Up-High
                elif not up and not high_vol:
                    regime_onehot[2] = 1  # Down-Low
                else:  # not up and high_vol
                    regime_onehot[3] = 1  # Down-High
                
                # Create conditioning vector: [regime_onehot(4), vol_scaled(1), trend_scaled(1)]
                conditioning_vector = np.concatenate([
                    regime_onehot,
                    [vol_scaled],
                    [trend_scaled]
                ])
            
            conditioning_vectors.append(conditioning_vector)
        
        conditioning_vectors = np.array(conditioning_vectors)
        logger.info(f"Generated explicit conditioning: {conditioning_vectors.shape}")
        
        return conditioning_vectors

class LLMEvalProvider(BaseEvalProvider):
    """Provider for LLM conditioning with PCA from saved specification."""
    
    def __init__(self, conditioning_spec: Dict[str, Any], checkpoint_dir: Optional[str] = None):
        super().__init__(conditioning_spec, checkpoint_dir)
        
        # Validate LLM conditioning spec
        if self.conditioning_type != 'llm':
            warning_msg = f"Expected llm conditioning, got {self.conditioning_type}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Extract PCA specifications
        self.pca_components = conditioning_spec.get('pca_components', 16)
        self.original_embedding_dim = conditioning_spec.get('original_embedding_dim', 768)
        self.explained_variance_ratio = conditioning_spec.get('explained_variance_ratio', 0.0)
        self.train_cutoff = conditioning_spec.get('train_cutoff', '2019-12-31')
        
        # Load PCA model
        self.pca = None
        self.pca_loaded = False
        
        if checkpoint_dir:
            self._load_pca_model()
        
        # If no PCA available and this is pre-COVID, we'll need to fit on ≤ 2019-12-31
        self.need_pca_fitting = not self.pca_loaded
        
        logger.info(f"LLM conditioning initialized:")
        logger.info(f"  PCA components: {self.pca_components}")
        logger.info(f"  Original embedding dim: {self.original_embedding_dim}")
        logger.info(f"  PCA loaded: {self.pca_loaded}")
    
    def _load_pca_model(self):
        """Load PCA model from checkpoint directory."""
        if not self.checkpoint_dir:
            return
        
        pca_path = Path(self.checkpoint_dir) / 'pca_model.pkl'
        
        if pca_path.exists():
            try:
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                self.pca_loaded = True
                self.need_pca_fitting = False
                logger.info(f"Loaded PCA model from {pca_path}")
                logger.info(f"PCA components: {self.pca.n_components_}, explained variance: {self.explained_variance_ratio:.4f}")
            except Exception as e:
                warning_msg = f"Failed to load PCA model: {e}"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
        else:
            warning_msg = f"PCA model not found at {pca_path}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
    
    def _fit_pca_on_precovid(self, embeddings_data: pd.DataFrame) -> bool:
        """Fit PCA on pre-COVID data if needed."""
        if self.pca_loaded:
            return True
        
        logger.warning("spec_missing_pca: Fitting PCA on ≤ 2019-12-31")
        self.warnings.append("spec_missing_pca: Fitting PCA on evaluation data")
        
        # Filter to pre-COVID data
        train_cutoff = pd.Timestamp(self.train_cutoff)
        precovid_data = embeddings_data[embeddings_data.index <= train_cutoff]
        
        if len(precovid_data) == 0:
            warning_msg = "No pre-COVID data available for PCA fitting"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            return False
        
        # Extract embeddings (assume all columns except date are embedding dimensions)
        embedding_columns = [col for col in precovid_data.columns if col != 'date']
        embeddings = precovid_data[embedding_columns].values
        
        # Remove any rows with NaN values
        valid_rows = ~np.isnan(embeddings).any(axis=1)
        embeddings = embeddings[valid_rows]
        
        if len(embeddings) == 0:
            warning_msg = "No valid embeddings for PCA fitting"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            return False
        
        # Fit PCA
        try:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(embeddings)
            self.pca_loaded = True
            
            # Update explained variance ratio
            self.explained_variance_ratio = self.pca.explained_variance_ratio_.sum()
            
            logger.info(f"Fitted PCA on {len(embeddings)} pre-COVID embeddings")
            logger.info(f"Explained variance ratio: {self.explained_variance_ratio:.4f}")
            
            return True
            
        except Exception as e:
            warning_msg = f"Failed to fit PCA: {e}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            return False
    
    def _discover_embedding_files(self) -> List[Dict[str, Any]]:
        """Discover available embedding files in cache directory."""
        cache_dir = Path("cache/news_embeddings")
        if not cache_dir.exists():
            logger.warning(f"Embedding cache directory not found: {cache_dir}")
            return []
        
        embedding_files = []
        for pkl_file in cache_dir.glob("*.pkl"):
            filename = pkl_file.name
            # Extract date range from filename: embeddings_YYYYMMDD_YYYYMMDD.pkl
            if filename.startswith("embeddings_") and filename.endswith(".pkl"):
                date_part = filename.replace("embeddings_", "").replace(".pkl", "")
                parts = date_part.split("_")
                if len(parts) == 2:
                    try:
                        start_str, end_str = parts
                        start_date = pd.Timestamp(start_str)
                        end_date = pd.Timestamp(end_str)
                        
                        embedding_files.append({
                            'path': pkl_file,
                            'start_date': start_date,
                            'end_date': end_date,
                            'filename': filename
                        })
                    except Exception as e:
                        logger.warning(f"Could not parse date range from {filename}: {e}")
        
        # Sort by start date
        embedding_files.sort(key=lambda x: x['start_date'])
        return embedding_files
    
    def _select_embedding_file(self, target_dates: List[pd.Timestamp]) -> Optional[Dict[str, Any]]:
        """Select the best embedding file that covers the target date range."""
        embedding_files = self._discover_embedding_files()
        
        if not embedding_files:
            logger.warning("No embedding files found in cache")
            return None
        
        min_target = min(target_dates)
        max_target = max(target_dates)
        
        # Find files that cover the entire target range
        covering_files = []
        for file_info in embedding_files:
            if file_info['start_date'] <= min_target and file_info['end_date'] >= max_target:
                covering_files.append(file_info)
        
        if covering_files:
            # Use the file with the smallest date range (most specific)
            best_file = min(covering_files, key=lambda x: (x['end_date'] - x['start_date']).days)
            logger.info(f"Selected embedding file: {best_file['filename']} "
                       f"({best_file['start_date'].date()} to {best_file['end_date'].date()})")
            return best_file
        
        # Find files with best partial coverage
        partial_files = []
        for file_info in embedding_files:
            # Check if there's any overlap
            overlap_start = max(file_info['start_date'], min_target)
            overlap_end = min(file_info['end_date'], max_target)
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                partial_files.append((file_info, overlap_days))
        
        if partial_files:
            # Use the file with maximum overlap
            best_file, overlap_days = max(partial_files, key=lambda x: x[1])
            logger.warning(f"Selected partial coverage file: {best_file['filename']} "
                          f"(covers {overlap_days} days of {(max_target - min_target).days} requested)")
            return best_file
        
        logger.error(f"No embedding files cover target range {min_target.date()} to {max_target.date()}")
        return None

    def _load_llm_embeddings(self, target_dates: List[pd.Timestamp]) -> Optional[pd.DataFrame]:
        """Load real LLM embeddings from cached files."""
        logger.info(f"Loading LLM embeddings for {len(target_dates)} target dates")
        
        # Create date range covering all target dates plus buffer for causal computation
        start_date = min(target_dates) - timedelta(days=100)
        end_date = max(target_dates)
        
        # Ensure we have pandas Timestamps
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date) - timedelta(days=100)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Extend range to include buffer dates
        extended_dates = [start_date] + target_dates + [end_date]
        
        # Select appropriate embedding file
        embedding_file = self._select_embedding_file(extended_dates)
        if not embedding_file:
            logger.error("No suitable embedding file found")
            return None
        
        # Load embeddings from pickle file
        try:
            with open(embedding_file['path'], 'rb') as f:
                embeddings_dict = pickle.load(f)
            
            logger.info(f"Loaded embeddings from {embedding_file['filename']}")
            logger.info(f"Contains {len(embeddings_dict)} dates")
            
            # Determine actual embedding dimension
            sample_embedding = next(iter(embeddings_dict.values()))
            actual_embedding_dim = len(sample_embedding)
            
            if actual_embedding_dim != self.original_embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.original_embedding_dim}, "
                              f"got {actual_embedding_dim}. Updating specification.")
                self.original_embedding_dim = actual_embedding_dim
            
            # Convert to DataFrame format expected by the rest of the code
            all_dates = sorted(embeddings_dict.keys())
            embedding_matrix = np.array([embeddings_dict[date] for date in all_dates])
            
            # Create column names
            embedding_columns = [f'emb_{i}' for i in range(actual_embedding_dim)]
            
            # Create DataFrame
            embeddings_df = pd.DataFrame(
                embedding_matrix, 
                index=pd.DatetimeIndex(all_dates), 
                columns=embedding_columns
            )
            
            # Filter to the required date range
            date_mask = (embeddings_df.index >= start_date) & (embeddings_df.index <= end_date)
            embeddings_df = embeddings_df[date_mask]
            
            logger.info(f"Filtered to {len(embeddings_df)} embedding records for date range "
                       f"{start_date.date()} to {end_date.date()}")
            
            # Identify actually missing dates vs available dates
            available_dates = len(embeddings_df)
            expected_dates = (end_date - start_date).days + 1
            missing_count = expected_dates - available_dates
            
            if missing_count > 0:
                logger.info(f"Note: {missing_count} dates in range have no embeddings (normal for weekends/holidays)")
            
            return embeddings_df
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embedding_file['path']}: {e}")
            return None
    
    def generate_conditioning(self, returns_data: pd.DataFrame, 
                            target_dates: List[pd.Timestamp]) -> np.ndarray:
        """Generate LLM conditioning vectors for target dates."""
        logger.info(f"Generating LLM conditioning for {len(target_dates)} target dates")
        
        # Load embeddings
        embeddings_data = self._load_llm_embeddings(target_dates)
        
        if embeddings_data is None:
            warning_msg = "Failed to load LLM embeddings, using random conditioning"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            
            # Return random conditioning as fallback
            return np.random.normal(0, 1, (len(target_dates), self.conditioning_dim))
        
        # Fit PCA if needed
        if self.need_pca_fitting:
            if not self._fit_pca_on_precovid(embeddings_data):
                warning_msg = "PCA fitting failed, using random conditioning"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
                return np.random.normal(0, 1, (len(target_dates), self.conditioning_dim))
        
        # Generate conditioning for each target date
        conditioning_vectors = []
        missing_dates = []
        
        for target_date in target_dates:
            # Find causal embeddings (must be <= target_date)
            available_dates = embeddings_data.index[embeddings_data.index <= target_date]
            
            if len(available_dates) == 0:
                # No causal data available
                missing_dates.append(target_date)
                conditioning_vector = np.zeros(self.conditioning_dim)
            else:
                # Use the latest available causal date
                causal_date = available_dates[-1]
                embedding = embeddings_data.loc[causal_date].values
                
                # Check if embedding is valid (not all NaN)
                if np.isnan(embedding).all():
                    missing_dates.append(target_date)
                    conditioning_vector = np.zeros(self.conditioning_dim)
                else:
                    # Replace NaN values with zeros
                    embedding = np.nan_to_num(embedding)
                    
                    # Apply PCA transformation
                    try:
                        embedding_2d = embedding.reshape(1, -1)
                        pca_embedding = self.pca.transform(embedding_2d)[0]
                        conditioning_vector = pca_embedding
                    except Exception as e:
                        logger.warning(f"PCA transform failed for {causal_date}: {e}")
                        missing_dates.append(target_date)
                        conditioning_vector = np.zeros(self.conditioning_dim)
            
            conditioning_vectors.append(conditioning_vector)
        
        # Log missing dates once
        if missing_dates:
            logger.warning(f"Missing embeddings for {len(missing_dates)} dates: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")
            self.warnings.append(f"Missing embeddings for {len(missing_dates)} target dates")
        
        conditioning_vectors = np.array(conditioning_vectors)
        logger.info(f"Generated LLM conditioning: {conditioning_vectors.shape}")
        
        return conditioning_vectors

class EvalProviderFactory:
    """Factory for creating evaluation conditioning providers."""
    
    @staticmethod
    def create_provider(conditioning_spec: Dict[str, Any], 
                       checkpoint_dir: Optional[str] = None) -> BaseEvalProvider:
        """Create an evaluation conditioning provider based on the specification."""
        conditioning_type = conditioning_spec.get('type', 'unknown')
        
        if conditioning_type == 'zero':
            return NoneProvider(conditioning_spec, checkpoint_dir)
        
        elif conditioning_type == 'explicit':
            return ExplicitEvalProvider(conditioning_spec, checkpoint_dir)
        
        elif conditioning_type == 'llm':
            return LLMEvalProvider(conditioning_spec, checkpoint_dir)
        
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")

def load_conditioning_spec(checkpoint_dir: str) -> Dict[str, Any]:
    """Load conditioning specification from checkpoint directory."""
    spec_path = Path(checkpoint_dir) / 'conditioning_spec.json'
    
    if not spec_path.exists():
        raise FileNotFoundError(f"Conditioning spec not found: {spec_path}")
    
    try:
        with open(spec_path, 'r') as f:
            conditioning_spec = json.load(f)
        
        logger.info(f"Loaded conditioning spec: {conditioning_spec.get('type', 'unknown')}")
        return conditioning_spec
        
    except Exception as e:
        raise RuntimeError(f"Failed to load conditioning spec: {e}")

def generate_eval_conditioning(checkpoint_dir: str, returns_data: pd.DataFrame,
                             target_dates: List[Union[str, pd.Timestamp]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to generate evaluation conditioning.
    
    Args:
        checkpoint_dir: Path to checkpoint directory with conditioning_spec.json
        returns_data: DataFrame with returns data (index as dates)
        target_dates: List of target dates for conditioning generation
        
    Returns:
        conditioning_vectors: Array of conditioning vectors [n_dates, conditioning_dim]
        warnings: List of warning messages
    """
    # Convert dates to timestamps
    if isinstance(target_dates[0], str):
        target_dates = [pd.Timestamp(date) for date in target_dates]
    
    # Load conditioning specification
    conditioning_spec = load_conditioning_spec(checkpoint_dir)
    
    # Create provider
    provider = EvalProviderFactory.create_provider(conditioning_spec, checkpoint_dir)
    
    # Generate conditioning
    conditioning = provider.generate_conditioning(returns_data, target_dates)
    
    # Get warnings
    warnings = provider.get_warnings()
    
    return conditioning, warnings

def main():
    """Example usage of evaluation conditioning providers."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluation Conditioning Providers')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--returns-file', type=str, default='sp500_data.csv',
                       help='Path to returns data file')
    parser.add_argument('--target-dates', nargs='+', 
                       default=['2020-03-01', '2020-06-01', '2020-12-31'],
                       help='Target dates for conditioning generation')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for conditioning vectors')
    
    args = parser.parse_args()
    
    try:
        # Load returns data
        if os.path.exists(args.returns_file):
            returns_data = pd.read_csv(args.returns_file, index_col=0, parse_dates=True)
            if 'log_returns' in returns_data.columns:
                returns_data['returns'] = returns_data['log_returns']
            logger.info(f"Loaded returns data: {returns_data.shape}")
        else:
            logger.warning(f"Returns file not found: {args.returns_file}, using synthetic data")
            # Create synthetic returns data
            date_range = pd.date_range('2010-01-01', '2021-12-31', freq='D')
            np.random.seed(42)
            returns = np.random.normal(0.0004, 0.01, len(date_range))
            returns_data = pd.DataFrame({'returns': returns}, index=date_range)
        
        # Generate conditioning
        logger.info("Generating evaluation conditioning...")
        conditioning, warnings = generate_eval_conditioning(
            args.checkpoint_dir,
            returns_data,
            args.target_dates
        )
        
        # Print results
        if conditioning is not None:
            logger.info(f"Generated conditioning shape: {conditioning.shape}")
            logger.info(f"Conditioning statistics: mean={conditioning.mean():.6f}, std={conditioning.std():.6f}")
        else:
            logger.info("Generated conditioning: None (zero conditioning)")
        
        # Print warnings
        if warnings:
            logger.info("Warnings:")
            for warning in warnings:
                logger.info(f"  - {warning}")
        
        # Save conditioning if requested
        if args.output_file and conditioning is not None:
            np.save(args.output_file, conditioning)
            logger.info(f"Saved conditioning to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate evaluation conditioning: {e}")
        raise

if __name__ == "__main__":
    main()
