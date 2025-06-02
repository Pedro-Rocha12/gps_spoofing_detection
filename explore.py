# explore.py

import os
import logging
import pandas as pd
from connect_opensky import get_cursor

logger = logging.getLogger(__name__)

def list_catalogs():
    cur = get_cursor()
    cur.execute("SHOW CATALOGS")
    return [row[0] for row in cur.fetchall()]

def list_schemas(catalog: str):
    cur = get_cursor()
    cur.execute(f"SHOW SCHEMAS FROM {catalog}")
    return [row[0] for row in cur.fetchall()]

def list_tables(catalog: str, schema: str):
    cur = get_cursor()
    cur.execute(f"SHOW TABLES FROM {catalog}.{schema}")
    return [row[0] for row in cur.fetchall()]

def describe_table(catalog: str, schema: str, table: str):
    cur = get_cursor()
    cur.execute(f"DESCRIBE {catalog}.{schema}.{table}")
    return cur.fetchall()  # lista de tuplas (column_name, data_type, ...)

def explore_to_csv(
    output_path: str = "estrutura_opensky.csv",
    skip_schemas: dict = None
) -> pd.DataFrame:
    """
    Explora a estrutura do banco Trino/OpenSky e salva em CSV.
    Retorna um DataFrame com colunas: ['catalog', 'schema', 'table', 'column', 'type'].
    
    :param output_path: caminho onde escrever o CSV
    :param skip_schemas: dict opcional {catalog: [schema1, schema2]} para pular schemas
    """
    skip_schemas = skip_schemas or {}
    records = []
    
    logger.info("Iniciando exploração da estrutura do OpenSky...")
    for catalog in list_catalogs():
        schemas = [s for s in list_schemas(catalog) 
                   if s not in skip_schemas.get(catalog, [])]
        for schema in schemas:
            try:
                tables = list_tables(catalog, schema)
                for table in tables:
                    cols = describe_table(catalog, schema, table)
                    for col_name, col_type, *rest in cols:
                        records.append({
                            "catalog": catalog,
                            "schema": schema,
                            "table": table,
                            "column": col_name,
                            "type": col_type
                        })
            except Exception as e:
                logger.warning(f"Não foi possível descrever {catalog}.{schema}: {e}")
    
    # Monta DataFrame e salva
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Estrutura salva em {output_path}")
    return df
