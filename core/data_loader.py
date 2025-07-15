# data_loader.py
# Data loading and import functionality

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
import csv
import json
import requests
from typing import Optional, Dict, Any, Tuple
import urllib.parse
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Handles all data loading operations with enhanced error handling"""

    def __init__(self):
        self.supported_formats = {
            '.csv': self.load_csv,
            '.xlsx': self.load_excel,
            '.xls': self.load_excel,
            '.parquet': self.load_parquet,
            '.json': self.load_json,
            '.jsonl': self.load_jsonlines,
            '.tsv': self.load_tsv,
            '.txt': self.load_text
        }

    def detect_delimiter(self, file_path: str, sample_size: int = 1024) -> str:
        """Detect CSV delimiter automatically with fallback"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(sample_size)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                return delimiter
        except Exception:
            # Fallback to comma
            return ','

    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection and error handling"""
        try:
            # Detect delimiter
            delimiter = kwargs.get('delimiter') or self.detect_delimiter(file_path)

            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    # Try with polars first for better performance
                    df_pl = pl.read_csv(
                        str(file_path),
                        separator=delimiter,
                        try_parse_dates=True,
                        ignore_errors=True,
                        encoding=encoding
                    )
                    return df_pl.to_pandas()
                except:
                    try:
                        # Fallback to pandas
                        return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, **kwargs)
                    except:
                        continue

            # If all encodings fail, try with error handling
            return pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8',
                               errors='ignore', **kwargs)

        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def load_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Excel file with error handling"""
        try:
            sheet_name = kwargs.get('sheet_name', 0)
            engine = kwargs.get('engine', None)

            # Try different engines
            engines = [engine] if engine else ['openpyxl', 'xlrd']

            for eng in engines:
                try:
                    return pd.read_excel(file_path, sheet_name=sheet_name, engine=eng, **kwargs)
                except:
                    continue

            # Final attempt without specifying engine
            return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

        except Exception as e:
            raise Exception(f"Error loading Excel: {str(e)}")

    def load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load Parquet file with error handling"""
        try:
            # Try polars first
            try:
                df_pl = pl.read_parquet(str(file_path))
                return df_pl.to_pandas()
            except:
                # Fallback to pandas
                return pd.read_parquet(file_path, **kwargs)

        except Exception as e:
            raise Exception(f"Error loading Parquet: {str(e)}")

    def load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to normalize nested JSON
                try:
                    return pd.json_normalize(data)
                except:
                    return pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")

        except Exception as e:
            raise Exception(f"Error loading JSON: {str(e)}")

    def load_jsonlines(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSONL (JSON Lines) file with error handling"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                        continue

            if not data:
                raise ValueError("No valid JSON data found")

            return pd.DataFrame(data)

        except Exception as e:
            raise Exception(f"Error loading JSONL: {str(e)}")

    def load_tsv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load TSV (Tab-separated values) file with error handling"""
        try:
            return pd.read_csv(file_path, delimiter='\t', **kwargs)
        except Exception as e:
            raise Exception(f"Error loading TSV: {str(e)}")

    def load_text(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load plain text file with custom delimiter and error handling"""
        try:
            delimiter = kwargs.get('delimiter', '\t')
            return pd.read_csv(file_path, delimiter=delimiter, **kwargs)
        except Exception as e:
            raise Exception(f"Error loading text file: {str(e)}")

    def load_from_url(self, url: str, file_format: str = 'csv', **kwargs) -> pd.DataFrame:
        """Load data from URL with error handling"""
        try:
            # Add timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            if file_format.lower() == 'csv':
                return pd.read_csv(url, headers=0, **kwargs)
            elif file_format.lower() == 'json':
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported URL format: {file_format}")

        except Exception as e:
            raise Exception(f"Error loading from URL: {str(e)}")

    def load_file(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main file loading function with comprehensive error handling"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                raise ValueError("File is empty")

            # Get file extension
            ext = file_path.suffix.lower()

            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {ext}")

            # Load the file
            df = self.supported_formats[ext](str(file_path), **kwargs)

            # Validate loaded data
            if df.empty:
                raise ValueError("Loaded data is empty")

            # Calculate file info
            file_size_mb = file_size / (1024 * 1024)
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

            info = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'rows': len(df),
                'columns': len(df.columns),
                'memory': f"{memory_usage:.1f} MB",
                'file_size': f"{file_size_mb:.1f} MB",
                'extension': ext,
                'load_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return df, info

        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    def get_sample_data(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load sample datasets for testing with FIXED Boston Housing"""
        try:
            if dataset_name.lower() == 'iris':
                # Load Iris dataset
                try:
                    from sklearn.datasets import load_iris
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['species'] = data.target
                    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                except ImportError:
                    # Create synthetic iris data if sklearn not available
                    np.random.seed(42)
                    df = self._create_synthetic_iris()

            elif dataset_name.lower() == 'titanic':
                # Create synthetic Titanic dataset
                np.random.seed(42)
                df = self._create_synthetic_titanic()

            elif dataset_name.lower() == 'boston housing':
                # Create synthetic Boston Housing dataset (replacing deprecated load_boston)
                np.random.seed(42)
                df = self._create_synthetic_boston_housing()

            elif dataset_name.lower() == 'wine quality':
                # Create synthetic wine quality dataset
                np.random.seed(42)
                df = self._create_synthetic_wine_quality()

            elif dataset_name.lower() == 'stock market':
                # Create synthetic stock market data
                np.random.seed(42)
                df = self._create_synthetic_stock_market()

            else:
                raise ValueError(f"Unknown sample dataset: {dataset_name}")

            # Calculate info
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            info = {
                'filename': f"{dataset_name} (Sample Data)",
                'file_path': f"sample://{dataset_name}",
                'rows': len(df),
                'columns': len(df.columns),
                'memory': f"{memory_usage:.1f} MB",
                'file_size': 'N/A',
                'extension': '.sample',
                'load_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return df, info

        except Exception as e:
            raise Exception(f"Error loading sample data: {str(e)}")

    def _create_synthetic_iris(self) -> pd.DataFrame:
        """Create synthetic Iris dataset"""
        n_samples = 150

        # Create realistic flower measurements
        data = []

        # Setosa (smaller flowers)
        for _ in range(50):
            data.append([
                np.random.normal(5.0, 0.3),  # sepal_length
                np.random.normal(3.4, 0.3),  # sepal_width
                np.random.normal(1.5, 0.2),  # petal_length
                np.random.normal(0.2, 0.1),  # petal_width
                'setosa'
            ])

        # Versicolor (medium flowers)
        for _ in range(50):
            data.append([
                np.random.normal(6.0, 0.4),  # sepal_length
                np.random.normal(2.8, 0.3),  # sepal_width
                np.random.normal(4.3, 0.4),  # petal_length
                np.random.normal(1.3, 0.2),  # petal_width
                'versicolor'
            ])

        # Virginica (larger flowers)
        for _ in range(50):
            data.append([
                np.random.normal(6.6, 0.4),  # sepal_length
                np.random.normal(3.0, 0.3),  # sepal_width
                np.random.normal(5.6, 0.4),  # petal_length
                np.random.normal(2.0, 0.3),  # petal_width
                'virginica'
            ])

        df = pd.DataFrame(data, columns=[
            'sepal_length_cm', 'sepal_width_cm',
            'petal_length_cm', 'petal_width_cm', 'species'
        ])

        return df

    def _create_synthetic_titanic(self) -> pd.DataFrame:
        """Create synthetic Titanic dataset"""
        n_samples = 891

        # Define survival probabilities by class and sex
        survival_probs = {
            (1, 'female'): 0.97,
            (1, 'male'): 0.37,
            (2, 'female'): 0.92,
            (2, 'male'): 0.16,
            (3, 'female'): 0.50,
            (3, 'male'): 0.14
        }

        data = []
        for i in range(n_samples):
            pclass = np.random.choice([1, 2, 3], p=[0.24, 0.21, 0.55])
            sex = np.random.choice(['male', 'female'], p=[0.65, 0.35])
            age = max(0, np.random.normal(29.7, 14.5))
            sibsp = np.random.poisson(0.52)
            parch = np.random.poisson(0.38)

            # Fare based on class
            if pclass == 1:
                fare = np.random.lognormal(4.0, 0.8)
            elif pclass == 2:
                fare = np.random.lognormal(2.8, 0.6)
            else:
                fare = np.random.lognormal(2.1, 0.7)

            embarked = np.random.choice(['C', 'Q', 'S'], p=[0.19, 0.09, 0.72])

            # Determine survival based on class and sex
            survival_prob = survival_probs.get((pclass, sex), 0.3)
            survived = np.random.random() < survival_prob

            data.append([
                i + 1,  # PassengerId
                int(survived),  # Survived
                pclass,  # Pclass
                sex,  # Sex
                age,  # Age
                sibsp,  # SibSp
                parch,  # Parch
                fare,  # Fare
                embarked  # Embarked
            ])

        df = pd.DataFrame(data, columns=[
            'PassengerId', 'Survived', 'Pclass', 'Sex', 'Age',
            'SibSp', 'Parch', 'Fare', 'Embarked'
        ])

        return df

    def _create_synthetic_boston_housing(self) -> pd.DataFrame:
        """Create synthetic Boston Housing dataset (replaces deprecated load_boston)"""
        n_samples = 506

        # Create realistic housing features
        data = []

        for _ in range(n_samples):
            # Crime rate
            crim = np.random.exponential(3.6)

            # Proportion of residential land zoned for lots over 25,000 sq.ft
            zn = np.random.choice([0, 12.5, 25, 50, 75], p=[0.7, 0.1, 0.1, 0.05, 0.05])

            # Proportion of non-retail business acres
            indus = np.random.exponential(6.8)

            # Charles River dummy variable
            chas = np.random.choice([0, 1], p=[0.93, 0.07])

            # Nitric oxides concentration
            nox = np.random.normal(0.55, 0.12)

            # Average number of rooms per dwelling
            rm = np.random.normal(6.3, 0.7)

            # Proportion of owner-occupied units built prior to 1940
            age = np.random.uniform(2.9, 100)

            # Weighted distances to employment centers
            dis = np.random.exponential(3.8)

            # Index of accessibility to radial highways
            rad = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 24],
                                   p=[0.2, 0.1, 0.15, 0.11, 0.12, 0.05, 0.05, 0.12, 0.1])

            # Property tax rate per $10,000
            tax = np.random.choice([187, 222, 250, 279, 307, 330, 398, 437, 469, 666, 711],
                                   p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

            # Pupil-teacher ratio
            ptratio = np.random.normal(18.5, 2.2)

            # Black proportion index
            b = np.random.normal(356.7, 91.3)

            # Lower status of population
            lstat = np.random.exponential(7.1)

            # Calculate median home value based on features (realistic relationship)
            medv = (
                    50 - 0.8 * crim + 0.1 * zn - 0.2 * indus + 3 * chas
                    - 15 * nox + 4 * rm - 0.06 * age + 1.2 * dis
                    - 0.3 * rad - 0.01 * tax - 0.9 * ptratio
                    + 0.004 * b - 0.5 * lstat
                    + np.random.normal(0, 3)  # Add noise
            )
            medv = max(5, min(50, medv))  # Constrain to realistic range

            data.append([
                crim, zn, indus, chas, nox, rm, age, dis,
                rad, tax, ptratio, b, lstat, medv
            ])

        df = pd.DataFrame(data, columns=[
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
            'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
        ])

        return df

    def _create_synthetic_wine_quality(self) -> pd.DataFrame:
        """Create synthetic wine quality dataset"""
        n_samples = 1599

        data = []
        for _ in range(n_samples):
            # Create correlated wine features
            fixed_acidity = np.random.normal(8.32, 1.74)
            volatile_acidity = np.random.normal(0.53, 0.18)
            citric_acid = np.random.normal(0.27, 0.19)
            residual_sugar = np.random.exponential(2.54)
            chlorides = np.random.normal(0.087, 0.047)
            free_sulfur_dioxide = np.random.normal(15.87, 10.46)
            total_sulfur_dioxide = np.random.normal(46.47, 32.9)
            density = np.random.normal(0.9967, 0.0019)
            pH = np.random.normal(3.31, 0.15)
            sulphates = np.random.normal(0.66, 0.17)
            alcohol = np.random.normal(10.42, 1.07)

            # Calculate quality based on features (realistic relationship)
            quality = (
                    3 + 0.1 * fixed_acidity - 2 * volatile_acidity + 0.5 * citric_acid
                    + 0.05 * residual_sugar - 5 * chlorides + 0.01 * free_sulfur_dioxide
                    - 0.005 * total_sulfur_dioxide - 2 * (density - 0.996) + 0.5 * pH
                    + 0.8 * sulphates + 0.3 * alcohol + np.random.normal(0, 0.5)
            )
            quality = max(3, min(8, round(quality)))  # Constrain to 3-8 range

            data.append([
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                pH, sulphates, alcohol, quality
            ])

        df = pd.DataFrame(data, columns=[
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'quality'
        ])

        return df

    def _create_synthetic_stock_market(self) -> pd.DataFrame:
        """Create synthetic stock market data"""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_samples = len(dates)

        # Generate realistic stock prices with trends and volatility
        base_price = 100
        returns = np.random.normal(0.0005, 0.015, n_samples)  # Daily returns

        prices = [base_price]
        for i in range(1, n_samples):
            # Add some trend and mean reversion
            trend = 0.0001 * np.sin(i / 252)  # Annual cycle
            volatility = 0.015 * (1 + 0.5 * np.sin(i / 50))  # Varying volatility

            daily_return = np.random.normal(trend, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(10, new_price))  # Minimum price of $10

        # Generate other OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC
            volatility = close * 0.02  # 2% daily volatility

            high = close + abs(np.random.normal(0, volatility))
            low = close - abs(np.random.normal(0, volatility))

            # Ensure low < close < high
            low = min(low, close * 0.95)
            high = max(high, close * 1.05)

            # Open is close to previous close
            if i == 0:
                open_price = close
            else:
                open_price = prices[i - 1] * (1 + np.random.normal(0, 0.005))

            volume = int(np.random.lognormal(14, 0.5))  # Random volume
            adj_close = close  # Simplified

            data.append([
                date, open_price, high, low, close, volume, adj_close
            ])

        df = pd.DataFrame(data, columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close'
        ])

        return df

    def get_excel_sheets(self, file_path: str) -> list:
        """Get list of sheet names in Excel file"""
        try:
            excel_file = pd.ExcelFile(file_path)
            return excel_file.sheet_names
        except Exception as e:
            raise Exception(f"Error reading Excel sheets: {str(e)}")

    def preview_file(self, file_path: str, n_rows: int = 5) -> pd.DataFrame:
        """Preview first few rows of a file without loading completely"""
        try:
            file_path = Path(file_path)
            ext = file_path.suffix.lower()

            if ext == '.csv':
                delimiter = self.detect_delimiter(str(file_path))
                return pd.read_csv(file_path, delimiter=delimiter, nrows=n_rows)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, nrows=n_rows)
            elif ext == '.parquet':
                # For parquet, load all and take head
                df = pd.read_parquet(file_path)
                return df.head(n_rows)
            elif ext == '.json':
                # Load and preview JSON
                df = self.load_json(str(file_path))
                return df.head(n_rows)
            else:
                return pd.DataFrame()  # Empty for unsupported

        except Exception as e:
            raise Exception(f"Error previewing file: {str(e)}")

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate file before loading"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                return False, "File does not exist"

            if file_path.stat().st_size == 0:
                return False, "File is empty"

            if file_path.suffix.lower() not in self.supported_formats:
                return False, f"Unsupported file format: {file_path.suffix}"

            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except PermissionError:
                return False, "Permission denied"
            except Exception as e:
                return False, f"File read error: {str(e)}"

            return True, "File validation passed"

        except Exception as e:
            return False, f"Validation error: {str(e)}"