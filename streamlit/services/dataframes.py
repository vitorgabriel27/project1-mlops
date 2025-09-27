import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List

class BaseDataFrame(ABC):
    path = None

    @classmethod
    def mount(cls) -> pd.DataFrame:
        if cls.path is None:
            raise NotImplementedError("Subclass should define `path` attribute.")
        return pd.read_csv(cls.path)

    @classmethod
    @abstractmethod
    def filter(cls, filters: dict) -> pd.DataFrame:
        return cls.mount()

class AirbnbRioDataFrame(BaseDataFrame):
    """
    Classe para gerenciar dados do Airbnb Rio de Janeiro
    Herda de BaseDataFrame e implementa os métodos abstratos
    """
    path = Path(__file__).resolve().parent.parent / "data" / "airbnb_rio_cleaned.csv" 
    
    # Cache para melhor performance
    _data = None
    
    @classmethod
    def mount(cls) -> pd.DataFrame:
        """Implementação com cache para melhor performance"""
        if cls._data is None:
            if cls.path.exists():
                cls._data = pd.read_csv(cls.path)
                print(f"✅ Dados carregados: {cls._data.shape[0]} linhas, {cls._data.shape[1]} colunas")
            else:
                # Lista arquivos CSV disponíveis
                csv_files = list(Path('.').glob('airbnb_rio_*.csv'))
                if csv_files:
                    print(f"📁 Arquivos encontrados: {[f.name for f in csv_files]}")
                    # Usa o primeiro arquivo encontrado
                    cls.path = csv_files[0]
                    cls._data = pd.read_csv(cls.path)
                    print(f"✅ Usando arquivo: {cls.path.name}")
                else:
                    raise FileNotFoundError(f"Nenhum arquivo Airbnb encontrado. Esperado: {cls.path}")
        return cls._data.copy()

    @classmethod
    def filter(cls, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Implementação do método abstrato filter
        Filtra o dataframe baseado nos critérios fornecidos
        """
        df = cls.mount()
        
        # Começa com máscara verdadeira para todos
        mask = pd.Series(True, index=df.index)
        
        # Aplica filtros sequencialmente
        if "neighbourhood" in filters and filters["neighbourhood"]:
            neighbourhoods = filters["neighbourhood"]
            if isinstance(neighbourhoods, list):
                mask &= df["neighbourhood_cleansed"].isin(neighbourhoods)
            else:
                mask &= df["neighbourhood_cleansed"] == neighbourhoods
        
        if "room_type" in filters and filters["room_type"]:
            room_types = filters["room_type"]
            if isinstance(room_types, list):
                mask &= df["room_type"].isin(room_types)
            else:
                mask &= df["room_type"] == room_types
        
        if "price_min" in filters:
            mask &= df["price"] >= filters["price_min"]
        
        if "price_max" in filters:
            mask &= df["price"] <= filters["price_max"]
        
        if "bedrooms_min" in filters:
            mask &= df["bedrooms"] >= filters["bedrooms_min"]
        
        if "bedrooms_max" in filters:
            mask &= df["bedrooms"] <= filters["bedrooms_max"]
        
        if "accommodates_min" in filters:
            mask &= df["accommodates"] >= filters["accommodates_min"]
        
        if "accommodates_max" in filters:
            mask &= df["accommodates"] <= filters["accommodates_max"]
        
        if "bathrooms_min" in filters:
            mask &= df["bathrooms"] >= filters["bathrooms_min"]
        
        # Filtro por categoria de preço (se existir a coluna)
        if "price_category" in filters and "price_category" in df.columns:
            price_categories = filters["price_category"]
            if isinstance(price_categories, list):
                mask &= df["price_category"].isin(price_categories)
            else:
                mask &= df["price_category"] == price_categories
        
        # Retorna dataframe filtrado
        filtered_df = df[mask].copy()
        print(f"✅ Filtro aplicado: {len(filtered_df)} de {len(df)} listings")
        
        return filtered_df

    @classmethod
    def list_available_filters(cls) -> List[str]:
        """Retorna lista de filtros disponíveis"""
        return [
            "neighbourhood", "room_type", "price_min", "price_max",
            "bedrooms_min", "bedrooms_max", "accommodates_min", "accommodates_max",
            "bathrooms_min", "price_category"
        ]

    @classmethod
    def get_unique_values(cls, column: str) -> List:
        """Retorna valores únicos de uma coluna"""
        df = cls.mount()
        if column in df.columns:
            return sorted(df[column].dropna().unique().tolist())
        else:
            available_cols = df.columns.tolist()
            raise ValueError(f"Coluna '{column}' não encontrada. Colunas disponíveis: {available_cols}")

    @classmethod
    def get_column_stats(cls, column: str) -> Dict[str, Any]:
        """Retorna estatísticas de uma coluna"""
        df = cls.mount()
        if column not in df.columns:
            raise ValueError(f"Coluna '{column}' não encontrada")
        
        col_data = df[column]
        stats = {
            "dtype": str(col_data.dtype),
            "non_null_count": col_data.count(),
            "null_count": col_data.isnull().sum(),
            "unique_values": col_data.nunique()
        }
        
        # Estatísticas específicas para numéricas
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                "min": col_data.min(),
                "max": col_data.max(),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std()
            })
        
        return stats

# Classe especializada para análise de preços
class AirbnbPriceAnalysis(AirbnbRioDataFrame):
    """
    Classe especializada em análise de preços
    Herda de AirbnbRioDataFrame
    """
    
    @classmethod
    def get_price_summary(cls, filters: Dict[str, Any] = None) -> Dict[str, float]:
        """Retorna resumo estatístico dos preços"""
        if filters:
            df = cls.filter(filters)
        else:
            df = cls.mount()
        
        return {
            "count": len(df),
            "mean": df["price"].mean(),
            "median": df["price"].median(),
            "std": df["price"].std(),
            "min": df["price"].min(),
            "max": df["price"].max(),
            "q25": df["price"].quantile(0.25),
            "q75": df["price"].quantile(0.75)
        }
    
    @classmethod
    def compare_price_by_category(cls, category_column: str, filters: Dict[str, Any] = None) -> Dict[str, Dict]:
        """Compara preços por categoria (bairro, tipo de quarto, etc)"""
        if filters:
            df = cls.filter(filters)
        else:
            df = cls.mount()
        
        if category_column not in df.columns:
            raise ValueError(f"Coluna '{category_column}' não encontrada")
        
        results = {}
        for category in df[category_column].unique():
            category_data = df[df[category_column] == category]
            results[category] = {
                "count": len(category_data),
                "mean_price": category_data["price"].mean(),
                "median_price": category_data["price"].median(),
                "min_price": category_data["price"].min(),
                "max_price": category_data["price"].max()
            }
        
        return results

    @classmethod
    def get_price_per_accommodate(cls, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Calcula preço por pessoa"""
        if filters:
            df = cls.filter(filters)
        else:
            df = cls.mount()
        
        df_result = df.copy()
        df_result["price_per_person"] = df_result["price"] / df_result["accommodates"]
        return df_result[["neighbourhood_cleansed", "room_type", "price", "accommodates", "price_per_person"]]

# Classe para análise geográfica
class AirbnbGeographicAnalysis(AirbnbRioDataFrame):
    """
    Classe especializada em análise geográfica
    """
    
    @classmethod
    def get_neighbourhood_boundaries(cls) -> Dict[str, Dict[str, float]]:
        """Retorna limites geográficos de cada bairro"""
        df = cls.mount()
        
        boundaries = {}
        for neighbourhood in df["neighbourhood_cleansed"].unique():
            neigh_data = df[df["neighbourhood_cleansed"] == neighbourhood]
            boundaries[neighbourhood] = {
                "min_lat": neigh_data["latitude"].min(),
                "max_lat": neigh_data["latitude"].max(),
                "min_lon": neigh_data["longitude"].min(),
                "max_lon": neigh_data["longitude"].max(),
                "center_lat": neigh_data["latitude"].mean(),
                "center_lon": neigh_data["longitude"].mean()
            }
        
        return boundaries

    @classmethod
    def get_listings_by_region(cls, lat_range: tuple, lon_range: tuple) -> pd.DataFrame:
        """Filtra listings por região geográfica"""
        df = cls.mount()
        mask = (df["latitude"].between(lat_range[0], lat_range[1])) & \
               (df["longitude"].between(lon_range[0], lon_range[1]))
        return df[mask]