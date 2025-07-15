# database_manager.py
# Database connection and management functionality

import pandas as pd
import sqlite3
import duckdb
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json


class DatabaseManager:
    """Handles database connections and operations"""

    def __init__(self):
        self.connections = {}
        self.active_connection = None
        self.connection_history = []

        # Supported database types
        self.supported_databases = {
            'sqlite': self._connect_sqlite,
            'duckdb': self._connect_duckdb,
            'postgresql': self._connect_postgresql,
            'mysql': self._connect_mysql,
            'mssql': self._connect_mssql
        }

    def _connect_sqlite(self, connection_params: Dict[str, Any]) -> Any:
        """Connect to SQLite database"""
        try:
            database_path = connection_params.get('database', ':memory:')

            # Create directory if it doesn't exist for file-based databases
            if database_path != ':memory:':
                Path(database_path).parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(database_path)
            return conn

        except Exception as e:
            raise Exception(f"Error connecting to SQLite: {str(e)}")

    def _connect_duckdb(self, connection_params: Dict[str, Any]) -> Any:
        """Connect to DuckDB database"""
        try:
            database_path = connection_params.get('database', ':memory:')
            conn = duckdb.connect(database_path)
            return conn

        except Exception as e:
            raise Exception(f"Error connecting to DuckDB: {str(e)}")

    def _connect_postgresql(self, connection_params: Dict[str, Any]) -> Any:
        """Connect to PostgreSQL database"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host=connection_params.get('host', 'localhost'),
                port=connection_params.get('port', 5432),
                database=connection_params.get('database'),
                user=connection_params.get('username'),
                password=connection_params.get('password')
            )
            return conn

        except ImportError:
            raise Exception("psycopg2 library required for PostgreSQL connections")
        except Exception as e:
            raise Exception(f"Error connecting to PostgreSQL: {str(e)}")

    def _connect_mysql(self, connection_params: Dict[str, Any]) -> Any:
        """Connect to MySQL database"""
        try:
            import mysql.connector

            conn = mysql.connector.connect(
                host=connection_params.get('host', 'localhost'),
                port=connection_params.get('port', 3306),
                database=connection_params.get('database'),
                user=connection_params.get('username'),
                password=connection_params.get('password')
            )
            return conn

        except ImportError:
            raise Exception("mysql-connector-python library required for MySQL connections")
        except Exception as e:
            raise Exception(f"Error connecting to MySQL: {str(e)}")

    def _connect_mssql(self, connection_params: Dict[str, Any]) -> Any:
        """Connect to Microsoft SQL Server database"""
        try:
            import pyodbc

            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={connection_params.get('host', 'localhost')};"
                f"DATABASE={connection_params.get('database')};"
                f"UID={connection_params.get('username')};"
                f"PWD={connection_params.get('password')}"
            )

            conn = pyodbc.connect(connection_string)
            return conn

        except ImportError:
            raise Exception("pyodbc library required for SQL Server connections")
        except Exception as e:
            raise Exception(f"Error connecting to SQL Server: {str(e)}")

    def connect_database(self, db_type: str, connection_params: Dict[str, Any],
                         connection_name: str = None) -> str:
        """Connect to a database"""
        try:
            if db_type.lower() not in self.supported_databases:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Create connection
            conn = self.supported_databases[db_type.lower()](connection_params)

            # Generate connection name if not provided
            if connection_name is None:
                connection_name = f"{db_type}_{len(self.connections) + 1}"

            # Store connection
            self.connections[connection_name] = {
                'connection': conn,
                'type': db_type.lower(),
                'params': connection_params,
                'created_at': pd.Timestamp.now()
            }

            # Set as active connection
            self.active_connection = connection_name

            # Add to history
            self.connection_history.append({
                'name': connection_name,
                'type': db_type.lower(),
                'timestamp': pd.Timestamp.now(),
                'status': 'connected'
            })

            return connection_name

        except Exception as e:
            # Add failed connection to history
            self.connection_history.append({
                'name': connection_name or f"{db_type}_failed",
                'type': db_type.lower(),
                'timestamp': pd.Timestamp.now(),
                'status': 'failed',
                'error': str(e)
            })
            raise Exception(f"Error connecting to database: {str(e)}")

    def disconnect_database(self, connection_name: str) -> bool:
        """Disconnect from a database"""
        try:
            if connection_name not in self.connections:
                raise ValueError(f"Connection {connection_name} not found")

            conn_info = self.connections[connection_name]
            conn = conn_info['connection']

            # Close connection based on type
            if hasattr(conn, 'close'):
                conn.close()

            # Remove from connections
            del self.connections[connection_name]

            # Update active connection
            if self.active_connection == connection_name:
                self.active_connection = list(self.connections.keys())[0] if self.connections else None

            # Add to history
            self.connection_history.append({
                'name': connection_name,
                'type': conn_info['type'],
                'timestamp': pd.Timestamp.now(),
                'status': 'disconnected'
            })

            return True

        except Exception as e:
            raise Exception(f"Error disconnecting from database: {str(e)}")

    def execute_query(self, query: str, connection_name: str = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                raise ValueError("No active database connection")

            conn_info = self.connections[connection_name]
            conn = conn_info['connection']
            db_type = conn_info['type']

            # Execute query based on database type
            if db_type in ['sqlite', 'postgresql', 'mysql', 'mssql']:
                df = pd.read_sql_query(query, conn)
            elif db_type == 'duckdb':
                df = conn.execute(query).fetchdf()
            else:
                raise ValueError(f"Query execution not supported for {db_type}")

            return df

        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")

    def get_table_list(self, connection_name: str = None) -> List[str]:
        """Get list of tables in the database"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                raise ValueError("No active database connection")

            conn_info = self.connections[connection_name]
            db_type = conn_info['type']

            # Query to get table names based on database type
            if db_type == 'sqlite':
                query = "SELECT name FROM sqlite_master WHERE type='table'"
            elif db_type == 'duckdb':
                query = "SHOW TABLES"
            elif db_type == 'postgresql':
                query = "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            elif db_type == 'mysql':
                query = "SHOW TABLES"
            elif db_type == 'mssql':
                query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
            else:
                return []

            df = self.execute_query(query, connection_name)
            return df.iloc[:, 0].tolist()

        except Exception as e:
            raise Exception(f"Error getting table list: {str(e)}")

    def get_table_schema(self, table_name: str, connection_name: str = None) -> pd.DataFrame:
        """Get schema information for a table"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                raise ValueError("No active database connection")

            conn_info = self.connections[connection_name]
            db_type = conn_info['type']

            # Query to get table schema based on database type
            if db_type == 'sqlite':
                query = f"PRAGMA table_info({table_name})"
            elif db_type == 'duckdb':
                query = f"DESCRIBE {table_name}"
            elif db_type == 'postgresql':
                query = f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                """
            elif db_type == 'mysql':
                query = f"DESCRIBE {table_name}"
            elif db_type == 'mssql':
                query = f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}'
                """
            else:
                return pd.DataFrame()

            return self.execute_query(query, connection_name)

        except Exception as e:
            raise Exception(f"Error getting table schema: {str(e)}")

    def load_table_to_dataframe(self, table_name: str, limit: int = None,
                                connection_name: str = None) -> pd.DataFrame:
        """Load entire table or limited rows to DataFrame"""
        try:
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"

            return self.execute_query(query, connection_name)

        except Exception as e:
            raise Exception(f"Error loading table to DataFrame: {str(e)}")

    def insert_dataframe_to_table(self, df: pd.DataFrame, table_name: str,
                                  if_exists: str = 'replace', connection_name: str = None) -> bool:
        """Insert DataFrame into database table"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                raise ValueError("No active database connection")

            conn_info = self.connections[connection_name]
            conn = conn_info['connection']
            db_type = conn_info['type']

            if db_type == 'duckdb':
                # For DuckDB, use register and create table
                conn.register('temp_df', df)

                if if_exists == 'replace':
                    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
                conn.unregister('temp_df')
            else:
                # For other databases, use pandas to_sql
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)

            return True

        except Exception as e:
            raise Exception(f"Error inserting DataFrame to table: {str(e)}")

    def test_connection(self, db_type: str, connection_params: Dict[str, Any]) -> bool:
        """Test database connection without storing it"""
        try:
            if db_type.lower() not in self.supported_databases:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Create temporary connection
            conn = self.supported_databases[db_type.lower()](connection_params)

            # Test with a simple query
            if db_type.lower() == 'sqlite':
                test_query = "SELECT 1"
            elif db_type.lower() == 'duckdb':
                test_query = "SELECT 1"
            elif db_type.lower() in ['postgresql', 'mysql', 'mssql']:
                test_query = "SELECT 1"

            # Execute test query
            if hasattr(conn, 'execute'):
                conn.execute(test_query)
            else:
                cursor = conn.cursor()
                cursor.execute(test_query)
                cursor.close()

            # Close test connection
            if hasattr(conn, 'close'):
                conn.close()

            return True

        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")

    def get_connection_info(self, connection_name: str = None) -> Dict[str, Any]:
        """Get information about a connection"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                return {}

            conn_info = self.connections[connection_name]

            # Get basic info
            info = {
                'name': connection_name,
                'type': conn_info['type'],
                'created_at': conn_info['created_at'],
                'is_active': connection_name == self.active_connection
            }

            # Add safe connection parameters (without sensitive data)
            safe_params = {}
            for key, value in conn_info['params'].items():
                if key.lower() not in ['password', 'pwd']:
                    safe_params[key] = value
                else:
                    safe_params[key] = '***'

            info['parameters'] = safe_params

            # Try to get additional database info
            try:
                tables = self.get_table_list(connection_name)
                info['table_count'] = len(tables)
                info['tables'] = tables[:10]  # First 10 tables
            except:
                info['table_count'] = 'Unknown'
                info['tables'] = []

            return info

        except Exception as e:
            raise Exception(f"Error getting connection info: {str(e)}")

    def list_connections(self) -> List[Dict[str, Any]]:
        """List all active connections"""
        connections_list = []

        for name, conn_info in self.connections.items():
            connections_list.append({
                'name': name,
                'type': conn_info['type'],
                'created_at': conn_info['created_at'],
                'is_active': name == self.active_connection
            })

        return connections_list

    def set_active_connection(self, connection_name: str) -> bool:
        """Set active connection"""
        try:
            if connection_name not in self.connections:
                raise ValueError(f"Connection {connection_name} not found")

            self.active_connection = connection_name
            return True

        except Exception as e:
            raise Exception(f"Error setting active connection: {str(e)}")

    def export_connection_config(self, connection_name: str, filepath: str) -> bool:
        """Export connection configuration to file (without password)"""
        try:
            if connection_name not in self.connections:
                raise ValueError(f"Connection {connection_name} not found")

            conn_info = self.connections[connection_name]

            # Create safe config (without sensitive data)
            config = {
                'name': connection_name,
                'type': conn_info['type'],
                'parameters': {}
            }

            for key, value in conn_info['params'].items():
                if key.lower() not in ['password', 'pwd']:
                    config['parameters'][key] = value

            # Save to file
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2, default=str)

            return True

        except Exception as e:
            raise Exception(f"Error exporting connection config: {str(e)}")

    def import_connection_config(self, filepath: str) -> Dict[str, Any]:
        """Import connection configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            return config

        except Exception as e:
            raise Exception(f"Error importing connection config: {str(e)}")

    def get_connection_history(self) -> List[Dict[str, Any]]:
        """Get connection history"""
        return self.connection_history.copy()

    def close_all_connections(self):
        """Close all database connections"""
        try:
            for connection_name in list(self.connections.keys()):
                self.disconnect_database(connection_name)

            self.active_connection = None

        except Exception as e:
            print(f"Error closing connections: {e}")

    def get_database_stats(self, connection_name: str = None) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if connection_name is None:
                connection_name = self.active_connection

            if connection_name is None or connection_name not in self.connections:
                return {}

            conn_info = self.connections[connection_name]
            db_type = conn_info['type']

            stats = {
                'connection_name': connection_name,
                'database_type': db_type,
                'tables': []
            }

            # Get table list
            tables = self.get_table_list(connection_name)
            stats['table_count'] = len(tables)

            # Get stats for each table
            for table in tables[:10]:  # Limit to first 10 tables
                try:
                    # Get row count
                    count_df = self.execute_query(f"SELECT COUNT(*) as count FROM {table}", connection_name)
                    row_count = count_df.iloc[0, 0] if not count_df.empty else 0

                    # Get column count
                    schema_df = self.get_table_schema(table, connection_name)
                    column_count = len(schema_df) if not schema_df.empty else 0

                    stats['tables'].append({
                        'name': table,
                        'rows': row_count,
                        'columns': column_count
                    })

                except Exception as e:
                    stats['tables'].append({
                        'name': table,
                        'rows': 'Error',
                        'columns': 'Error',
                        'error': str(e)
                    })

            return stats

        except Exception as e:
            raise Exception(f"Error getting database stats: {str(e)}")