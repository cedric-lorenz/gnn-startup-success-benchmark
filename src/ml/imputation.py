"""Missing value imputation strategies with split-aware fitting to prevent data leakage."""
import torch
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import warnings
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class BaseImputer(ABC):
    """Base class for all imputers with train-test leakage prevention."""

    def __init__(self, **kwargs):
        self.imputer = None
        self.kwargs = kwargs
        self.is_fitted = False

    @abstractmethod
    def create_imputer(self):
        """Create the specific imputer instance."""
        pass

    def fit(self, X: np.ndarray) -> "BaseImputer":
        """Fit the imputer on training data only."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if np.all(np.isnan(X), axis=0).any():
            print(
                f"Warning: A feature column in the training data is all NaN. Imputation might not be meaningful."
            )

        self.imputer = self.create_imputer()
        self.imputer.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted imputer."""
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.imputer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class CategoricalEncoderImputer(BaseImputer):
    """
    Imputer for categorical string columns using OrdinalEncoder + SimpleImputer.
    Handles encoding and decoding.
    """

    def __init__(self, strategy="most_frequent", **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.imputer = SimpleImputer(strategy=strategy, **kwargs)
        self.categories_ = None

    def create_imputer(self):
        return SimpleImputer(strategy=self.strategy, **self.kwargs)

    def fit(self, X: np.ndarray) -> "CategoricalEncoderImputer":
        X_reshaped = X if X.ndim == 2 else X.reshape(-1, 1)
        self.encoder.fit(X_reshaped)
        encoded = self.encoder.transform(X_reshaped)
        self.imputer.fit(encoded)
        self.categories_ = self.encoder.categories_
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        X_reshaped = X if X.ndim == 2 else X.reshape(-1, 1)
        encoded = self.encoder.transform(X_reshaped)
        imputed_encoded = self.imputer.transform(encoded)
        decoded = self.encoder.inverse_transform(imputed_encoded)
        return decoded


# --- Imputer Implementations ---
class MeanImputer(BaseImputer):
    def create_imputer(self):
        return SimpleImputer(strategy="mean", **self.kwargs)


class MedianImputer(BaseImputer):
    def create_imputer(self):
        return SimpleImputer(strategy="median", **self.kwargs)


class MostFrequentImputer(BaseImputer):
    def create_imputer(self):
        return SimpleImputer(strategy="most_frequent", **self.kwargs)


class ConstantImputer(BaseImputer):
    def __init__(self, fill_value=0, **kwargs):
        super().__init__(fill_value=fill_value, **kwargs)

    def create_imputer(self):
        return SimpleImputer(strategy="constant", **self.kwargs)


class KNNImputerWrapper(BaseImputer):
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)

    def create_imputer(self):
        return KNNImputer(**self.kwargs)


class IterativeImputerWrapper(BaseImputer):
    def __init__(self, random_state=42, max_iter=10, **kwargs):
        super().__init__(random_state=random_state, max_iter=max_iter, **kwargs)

    def create_imputer(self):
        return IterativeImputer(**self.kwargs)


class ZeroImputer(BaseImputer):
    def create_imputer(self):
        return SimpleImputer(strategy="constant", fill_value=0, **self.kwargs)


class GraphImputer:
    """
    Main imputation class for heterogeneous graphs with streamlined configuration
    and robust train-test leakage prevention.
    """

    IMPUTER_REGISTRY = {
        "mean": MeanImputer,
        "median": MedianImputer,
        "most_frequent": MostFrequentImputer,
        "constant": ConstantImputer,
        "knn": KNNImputerWrapper,
        "iterative": IterativeImputerWrapper,
        "zero": ZeroImputer,
    }

    def __init__(
        self,
        imputation_config: Dict[str, Any],
        feature_column_map: Dict[str, List[str]],
    ):
        """
        Initialize GraphImputer.
        Args:
            imputation_config: Dict with imputation settings per node type.
            feature_column_map: Dict mapping node type to its definitive list of feature columns.
        """
        self.imputation_config = imputation_config
        self.feature_column_map = feature_column_map
        self.fitted_imputers = {}
        self.missing_masks = {}

    def _get_imputers_for_node(
        self, node_type: str
    ) -> List[Tuple[BaseImputer, List[int]]]:
        config = self.imputation_config.get(node_type)
        if not config:
            config = self.imputation_config.get('default')

        all_feature_columns = self.feature_column_map.get(node_type)
        if all_feature_columns is None:
            return []

        imputers_with_cols = []
        all_col_indices = set(range(len(all_feature_columns)))
        categorical_indices = set()

        if "categorical_columns" in config:
            cat_cols = config["categorical_columns"]
            indices = [
                all_feature_columns.index(c)
                for c in cat_cols
                if c in all_feature_columns
            ]
            if indices:
                categorical_indices.update(indices)
                cat_params = config.get("categorical_params", {})
                imputer = CategoricalEncoderImputer(**cat_params)
                imputers_with_cols.append((imputer, sorted(indices)))

        if "numerical_method" in config:
            numerical_indices = sorted(list(all_col_indices - categorical_indices))
            if numerical_indices:
                num_method = config["numerical_method"]
                num_params = config.get("numerical_params", {})
                imputer = self.IMPUTER_REGISTRY[num_method](**num_params)
                imputers_with_cols.append((imputer, numerical_indices))

        return imputers_with_cols

    def _detect_missing_values(self, features: torch.Tensor) -> np.ndarray:
        features_np = (
            features.numpy() if isinstance(features, torch.Tensor) else features
        )
        return np.isnan(features_np) | np.isinf(features_np) | np.isneginf(features_np)

    def _prepare_features(
        self, features: torch.Tensor, column_names: List[str]
    ) -> pd.DataFrame:
        features_np = (
            features.numpy() if isinstance(features, torch.Tensor) else features
        )

        num_features = features_np.shape[1] if features_np.ndim > 1 else 0

        # If column names are missing or mismatched, generate default names
        if not column_names or len(column_names) != num_features:
            column_names = [f"feature_{i}" for i in range(num_features)]

        features_df = pd.DataFrame(features_np, columns=column_names)

        # Replace infinities with NaN
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return features_df

    def fit_node_imputers(self, graph_data, train_mask_key: str = "train_mask"):
        print("Fitting imputers...")
        for node_type in graph_data.node_types:
            # Skip if node has no features (e.g. ablated)
            if not hasattr(graph_data[node_type], 'x'):
                continue
            features = graph_data[node_type].x
            if features.numel() == 0:
                continue

            column_names = self.feature_column_map.get(node_type, [])
            features_df = self._prepare_features(features, column_names)
            missing_mask = features_df.isna().to_numpy()

            if not missing_mask.any():
                continue

            print(f"Found {missing_mask.sum()} missing values in {node_type} features")

            imputers_to_fit = self._get_imputers_for_node(node_type)
            if not imputers_to_fit:
                continue

            if node_type == "startup" and hasattr(graph_data[node_type], train_mask_key):
                train_mask = graph_data[node_type][train_mask_key]
                if train_mask.any():
                    fit_df = features_df.iloc[train_mask.nonzero(as_tuple=True)[0]]
                    print(
                        f"Fitting {node_type} imputer on {len(fit_df)} training samples"
                    )
                else:
                    print(f"No training samples for startup nodes, cannot fit imputer.")
                    continue
            else:
                print(
                    f"Fitting {node_type} imputer on all {len(features_df)} samples"
                )
                fit_df = features_df

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for imputer, col_indices in imputers_to_fit:
                    col_names = [column_names[i] for i in col_indices]
                    imputer.fit(fit_df[col_names].to_numpy())

            self.fitted_imputers[node_type] = imputers_to_fit
            self.missing_masks[node_type] = missing_mask

    def transform_node_features(self, graph_data):
        print("Transforming node features...")
        for node_type, fitted_imputers in self.fitted_imputers.items():
            # Identify all feature tensors to transform for this node type
            tensors_to_transform = [("x", graph_data[node_type].x)]
            
            # Check for split-specific tensors (only relevant for startup nodes usually)
            for split in ['val_mask', 'test_mask', 'test_mask_original']:
                split_key = f"x_{split}"
                if hasattr(graph_data[node_type], split_key):
                    tensors_to_transform.append((split_key, getattr(graph_data[node_type], split_key)))

            column_names = self.feature_column_map.get(node_type, [])

            for name, tensor in tensors_to_transform:
                features_df = self._prepare_features(tensor, column_names)
                imputed_df = features_df.copy()
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for imputer, col_indices in fitted_imputers:
                        col_names = [column_names[i] for i in col_indices]
                        imputed_values = imputer.transform(imputed_df[col_names])
                        imputed_df[col_names] = imputed_values

                # Convert back to torch
                new_tensor = torch.tensor(
                    imputed_df.astype(np.float32).to_numpy(), dtype=torch.float32
                )
                
                if name == "x":
                    graph_data[node_type].x = new_tensor
                else:
                    setattr(graph_data[node_type], name, new_tensor)
                
                print(
                    f"Imputed {self.missing_masks[node_type].sum()} values in {node_type} features (tensor: {name})"
                )

    def fit_transform(self, graph_data, train_mask_key: str = "train_mask"):
        self.fit_node_imputers(graph_data, train_mask_key)
        self.transform_node_features(graph_data)
        return graph_data
