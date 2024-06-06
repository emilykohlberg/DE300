from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import boto3
import pandas as pd
from io import StringIO
import tomli
import pathlib
from sqlalchemy import create_engine
import subprocess
from pyspark.sql import SparkSession

def start_spark_func(spark_name):
    print(spark_name)
    
    spark = SparkSession.builder \
                        .appName(spark_name) \
                        .config("spark.driver.memory", "6g") \
                        .config("spark.executor.memory", "6g") \
                        .getOrCreate()
    return spark

# read the parameters from toml
CONFIG_BUCKET = "de300spring2024-emilykohlberg"
CONFIG_FILE = "hw4_config.toml"

TABLE_NAMES = {
    "original_data": "heart_disease",
    "clean_data_1": "hd_clean_data_1",
    "clean_data_2": "hd_clean_data_2",
    "train_data_1": "hd_train_data_1",
    "train_data_2": "hd_train_data_2",
    "test_data_1": "hd_test_data_1",
    "test_data_2": "hd_test_data_2",
    "normalization_data_1": "normalization_values_1",
    "normalization_data_2": "normalization_values_2",
    "fe_high_risk_1": "fe_high_risk_features_1",
    "product_fe_1": "product_fe_features_1",
    "fe_high_risk_2": "fe_high_risk_features_2",
    "product_fe_2": "product_fe_features_2",
    "per_by_gender": "per_by_gender",
    "per_by_age_1": "per_by_age_1",
    "per_by_age_2": "per_by_age_2",
    "scrape_merged": "scrape_merged"
}


ENCODED_SUFFIX = "_encoded"
NORMALIZATION_TABLE_COLUMN_NAMES = ["name", "data_min", "data_max", "scale", "min"]

# Define the default args dictionary for DAG
default_args = {
    'owner': 'emilykohlberg',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 2,
}

def read_config_from_s3() -> dict:
    # Create a boto3 S3 client
    s3_client = boto3.client('s3')

    try:
        # Fetch the file from S3
        response = s3_client.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_FILE)
        file_content = response['Body'].read()

        # Load the TOML content
        params = tomli.loads(file_content.decode('utf-8'))
        return params
    except Exception as e:
        print(f"Failed to read from S3: {str(e)}")
        return {}

# Usage
PARAMS = read_config_from_s3()

def create_db_connection():
    """
    Create a database connection to the PostgreSQL RDS instance using SQLAlchemy.
    """
   
    conn_uri = f"{PARAMS['db']['db_alchemy_driver']}://{PARAMS['db']['username']}:{PARAMS['db']['password']}@{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}"


    # Create a SQLAlchemy engine and connect
    engine = create_engine(conn_uri)
    connection = engine.connect()

    return connection



def from_table_to_df(input_table_names: list[str], output_table_names: list[str]):
    """
    Decorator to open a list of tables input_table_names, load them in df and pass the dataframe to the function; on exit, it deletes tables in output_table_names
    The function has key = dfs with the value corresponding the list of the dataframes 

    The function must return a dictionary with key dfs; the values must be a list of dictionaries with keys df and table_name; Each df is written to table table_name
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import pandas as pd
            """
            load tables to dataframes
            """
            if input_table_names is None:
                raise ValueError('input_table_names cannot be None')
            
            _input_table_names = None
            if isinstance(input_table_names, str):
                _input_table_names = [input_table_names]
            else:
                _input_table_names = input_table_names

            import pandas as pd
            
            print(f'Loading input tables to dataframes: {_input_table_names}')

            # open the connection
            conn = create_db_connection()

            # read tables and convert to dataframes
            dfs = []
            for table_name in _input_table_names:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                dfs.append(df)

            if isinstance(input_table_names, str):
                dfs = dfs[0]

            """
            call the main function
            """

            kwargs['dfs'] = dfs
            kwargs['output_table_names'] = output_table_names

            result = func(*args, **kwargs)

            """
            delete tables
            """

            print(f'Deleting tables: {output_table_names}')
            if output_table_names is None:
                _output_table_names = []
            elif isinstance(output_table_names, str):
                _output_table_names = [output_table_names]
            else:
                _output_table_names = output_table_names
            
            print(f"Dropping tables {_output_table_names}")
            for table_name in _output_table_names:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            """
            write dataframes in result to tables
            """

            for pairs in result['dfs']:
                df = pairs['df']
                table_name = pairs['table_name']

                print("table name:", table_name)
                print("returned df:", df) 
                    
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Wrote to table {table_name}")

            conn.close()
            result.pop('dfs')

            return result
        return wrapper
    return decorator


def pandas_to_spark(dfs, spark_name):
    from pyspark.sql.types import StructType, StructField, StringType

    spark = start_spark_func(spark_name)
    
    spark_dfs = []
    for df in dfs:
        # Create Spark DataFrame with inferred schema
        try:
            spark_df = spark.createDataFrame(df)
            spark_dfs.append(spark_df)
        except Exception as e:
            string_schema = StructType([
                StructField(col, StringType(), True) for col in df.columns
            ])
            spark_df = spark.createDataFrame(df, schema=string_schema)
            spark_dfs.append(spark_df)
            
    return spark, spark_dfs

def process_with_spark(func):
    def wrapper(*args, **kwargs):
        # Convert dfs to a list if it's not already
        dfs = kwargs['dfs']
        if not isinstance(dfs, list):
            dfs = [dfs]
        
        # Check if dfs is a list of pandas DataFrames
        if all(isinstance(df, pd.DataFrame) for df in dfs):
            # Convert pandas DataFrames to Spark DataFrames
            spark, spark_dfs = pandas_to_spark(dfs, func.__name__)
            
            # Pass the Spark DataFrames to the decorated function
            kwargs['spark'] = spark
            kwargs['dfs'] = spark_dfs
            try:
                result = func(*args, **kwargs)
            finally:
                # Convert Spark DataFrame back to Pandas DataFrame
                result['dfs'] = [{"df": pair['df'].toPandas(), "table_name": pair["table_name"]} for pair in result['dfs']]

                # Ensure Spark session is stopped even if an exception occurs
                spark.stop()
            
            return result
        else:
            raise ValueError("dfs is not a list of pandas DataFrames")
    return wrapper

def add_data_to_table_func(**kwargs):
    """
    Insert data from a CSV file stored in S3 to a database table.
    """
    # Create a database connection
    conn = create_db_connection()
    
    # Set the S3 bucket and file key
    s3_bucket = PARAMS['files']['s3_bucket']
    s3_key = PARAMS['files']['s3_file_key']
    
    # Create an S3 client
    s3_client = boto3.client('s3')
    
    # Get the object from the S3 bucket
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    
    # Read the CSV file directly from the S3 object's byte stream into a DataFrame
    csv_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    
    # Write the DataFrame to SQL
    df.to_sql(TABLE_NAMES['original_data'], con=conn, if_exists="replace", index=False)

    # Close the database connection
    conn.close()

    return {'status': 1}

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_and_impute_data_1_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    import pandas as pd

    data = kwargs['dfs']

    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 
                     'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 
                     'exang', 'oldpeak', 'slope', 'target']

    data = data[columns_to_retain]

    # prediction does not work when target is NaN
    data = data.dropna(subset=['target'])

    data['age'] = pd.to_numeric(data['age'], errors='coerce').astype('float')

    # impute
    for col in columns_to_retain:
        if col != 'slope':
            m = data[col].mean()
            data.loc[data[col].isna(), col] = m

    mode_slope = data['slope'].mode()
    data.loc[data['slope'].isna(), 'slope'] = mode_slope[0]
    
    data.loc[data['slope'] < 1, 'slope'] = mode_slope[0]
    data.loc[data['slope'] > 3, 'slope'] = mode_slope[0]

    # Out of range
    data.loc[data['painloc'] < 0, 'painloc'] = 0
    data.loc[data['painloc'] > 1, 'painloc'] = 1
    
    data.loc[data['painexer'] < 0, 'painexer'] = 0
    data.loc[data['painexer'] > 1, 'painexer'] = 1
    
    data.loc[data['trestbps'] < 100, 'trestbps'] = 100
    
    data.loc[data['oldpeak'] < 0, 'oldpeak'] = 0
    data.loc[data['oldpeak'] > 4, 'oldpeak'] = 4
    
    data.loc[data['fbs'] < 0, 'fbs'] = 0
    data.loc[data['fbs'] > 1, 'fbs'] = 1
    
    data.loc[data['prop'] < 0, 'prop'] = 0
    data.loc[data['prop'] > 1, 'prop'] = 1
    
    data.loc[data['nitr'] < 0, 'nitr'] = 0
    data.loc[data['nitr'] > 1, 'nitr'] = 1
    
    data.loc[data['pro'] < 0, 'pro'] = 0
    data.loc[data['pro'] > 1, 'pro'] = 1
    
    data.loc[data['diuretic'] < 0, 'diuretic'] = 0
    data.loc[data['diuretic'] > 1, 'diuretic'] = 1
    
    data.loc[data['exang'] < 0, 'exang'] = 0
    data.loc[data['exang'] > 1, 'exang'] = 1

    print(data)

    return {
        'dfs': [
            {'df': data, 
             'table_name': TABLE_NAMES['clean_data_1']
             }]
        }

@from_table_to_df(TABLE_NAMES['original_data'], None)
@process_with_spark
def clean_and_impute_data_2_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    from pyspark.sql.functions import col, when
    from pyspark.ml.feature import Imputer
    from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

    data = kwargs['dfs'][0]

    print("Columns in the DataFrame:", data.columns)


    # retain cols
    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 
                         'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 
                         'exang', 'oldpeak', 'slope', 'target']
    
    data = data.select(columns_to_retain)

    # replace out of range
    data = data.withColumn('painloc', when(col('painloc') < 0, 0).when(col('painloc') > 1, 1).otherwise(col('painloc')))
    data = data.withColumn('painexer', when(col('painexer') < 0, 0).when(col('painexer') > 1, 1).otherwise(col('painexer')))
    data = data.withColumn('trestbps', when(col('trestbps') < 100, 100).otherwise(col('trestbps')))
    data = data.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))
    data = data.withColumn('fbs', when(col('fbs') < 0, 0).when(col('fbs') > 1, 1).otherwise(col('fbs')))
    data = data.withColumn('prop', when(col('prop') < 0, 0).when(col('prop') > 1, 1).otherwise(col('prop')))
    data = data.withColumn('nitr', when(col('nitr') < 0, 0).when(col('nitr') > 1, 1).otherwise(col('nitr')))
    data = data.withColumn('pro', when(col('pro') < 0, 0).when(col('pro') > 1, 1).otherwise(col('pro')))
    data = data.withColumn('diuretic', when(col('diuretic') < 0, 0).when(col('diuretic') > 1, 1).otherwise(col('diuretic')))
    data = data.withColumn('exang', when(col('exang') < 0, 0).when(col('exang') > 1, 1).otherwise(col('exang')))
    data = data.withColumn('slope', when(col('slope') < 1, None).when(col('slope') > 3, None).otherwise(col('slope')))

    # drop null targets
    data = data.dropna(subset=["target"])

    # make features numeric
    for feature in columns_to_retain:   
        data = data.withColumn(feature, data[feature].cast(DoubleType()))

    print("Data: ", data)

    # Identify numeric features
    numeric_features = [f.name for f in data.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
    numeric_features.remove("target")
    
    # numeric columns
    imputed_columns = [f"Imputed{v}" for v in numeric_features]
    imputer_numeric = Imputer(inputCols=numeric_features, outputCols=imputed_columns, strategy = "mean")

    print("imputed_columns: ", imputed_columns)
    
    # Apply Imputer transformation
    model = imputer_numeric.fit(data)
    imputed_data = model.transform(data)

    print("imputed data", imputed_data)

    for original_col, imputed_col in zip(numeric_features, imputed_columns):
        imputed_data = imputed_data.withColumn(original_col, imputed_data[imputed_col]).drop(imputed_col)

    # drop null targets
    imputed_data = imputed_data.dropna(subset=["target"])
    
    print("imputed data", imputed_data)

    return {
        'dfs': [
            {'df': imputed_data, 
             'table_name': TABLE_NAMES['clean_data_2']
             }]
        }


@from_table_to_df(TABLE_NAMES['clean_data_1'], None)
def normalize_data_1_func(**kwargs):
    """
    normalization
    split to train/test
    """
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=PARAMS['sklearn']['train_test_ratio'], random_state=42)

    # Normalize numerical columns
    normalization_values = [] # 
    for column in [v for v in df_train.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['sklearn']['labels']]:
        scaler = MinMaxScaler()
        df_train[column] = scaler.fit_transform(df_train[column].values.reshape(-1, 1))
        normalization_values.append((column, scaler.data_min_[0], scaler.data_max_[0], scaler.scale_[0]))
    normalization_df = pd.DataFrame(data=normalization_values, columns=["name", "min", "max", "scale"])

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['train_data_1']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['test_data_1']   
            },
            {
                'df': normalization_df,
                'table_name': TABLE_NAMES['normalization_data_1']
            }
            ]
        }

@from_table_to_df(TABLE_NAMES['clean_data_2'], None)
@process_with_spark
def normalize_data_2_func(**kwargs):
    """
    normalization
    split to train/test
    """

    from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType, StructType, StructField
    import pyspark.sql.functions as F
    from pyspark.sql.functions import when, col
    from pyspark.ml.feature import MinMaxScaler, VectorAssembler
    
    df = kwargs['dfs'][0]
    spark = kwargs['spark']

    # make features numeric
    for col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # Split the data into training and test sets
    df_train, df_test = df.randomSplit([PARAMS['spark']['train_test_split'], 1-PARAMS['spark']['train_test_split']], seed=PARAMS['spark']['split_seed'])

    # Identify numerical columns to be normalized
    numeric_columns = [f.name for f in df_train.schema.fields]
    numeric_columns.remove(PARAMS['spark']['labels'])
    
    # Normalize numerical columns
    normalization_values = []
    for column in [col_name for col_name in df.columns if col_name != PARAMS['spark']['labels']]:
        # Assemble the column into a vector
        assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}_vec")
        df_vec = assembler.transform(df)

        # Apply MinMaxScaler
        scaler = MinMaxScaler(inputCol=f"{column}_vec", outputCol=f"{column}_scaled")
        scaler_model = scaler.fit(df_vec)
        df = scaler_model.transform(df_vec)

        # Store normalization values
        normalization_values.append((column, scaler_model.getMin(), scaler_model.getMax()))

    # Create a schema for the normalization values DataFrame
    schema = StructType([
        StructField("name", StringType(), True),
        StructField("min", DoubleType(), True),
        StructField("max", DoubleType(), True)
    ])

    # Create Spark DataFrame from normalization values
    normalization_spark_df = spark.createDataFrame(normalization_values, schema)

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['train_data_2']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['test_data_2']   
            },
            {
                'df': normalization_spark_df,
                'table_name': TABLE_NAMES['normalization_data_2']
            }
            ]
        }
    
@from_table_to_df(TABLE_NAMES['clean_data_1'], None)
def eda_1_func(**kwargs):
    """
    print basic statistics
    """

    import pandas as pd

    df = kwargs['dfs']
    
    print(df.describe())

    return { 'dfs': [] }


@from_table_to_df(TABLE_NAMES['clean_data_2'], None)
@process_with_spark
def eda_2_func(**kwargs):
    """
    print basic statistics
    """
    from pyspark.sql.types import StructField, StructType, StringType
    
    # Extract the DataFrame from the list
    pdf = kwargs['dfs'][0]
    spark = kwargs['spark']
    
    # Extract column names and data types
    fields = [StructField(col, StringType(), True) for col in pdf.columns]
    
    # Create StructType schema
    schema = StructType(fields)
    
    # Create DataFrame with the specified schema
    df = spark.createDataFrame(pdf.rdd, schema=schema)
    
    # Print descriptive statistics
    df.describe().show()
    
    # Return an empty DataFrame as specified
    empty_schema = StructType()
    empty_df = spark.createDataFrame([], empty_schema)
    
    return {'dfs': empty_df}


@from_table_to_df(TABLE_NAMES['train_data_1'], None)
def fe_high_risk_1_func(**kwargs):
    """
    Add additional features including high-risk indicators, and polynomial features
    """

    import pandas as pd
    import numpy as np

    df = kwargs['dfs']

    new_features_df = pd.DataFrame()
    
    # High-risk condition indicators
    new_features_df['high_trestbps'] = np.where(df['trestbps'] > 130, 1, 0)
    new_features_df['high_cholesterol'] = np.where(df['prop'] > 200, 1, 0)
    new_features_df['high_oldpeak'] = np.where(df['oldpeak'] > 2, 1, 0)

    # Polynomial features
    new_features_df['age_squared'] = df['age'] ** 2
    new_features_df['thalach_squared'] = df['thalach'] ** 2
    new_features_df['oldpeak_squared'] = df['oldpeak'] ** 2

    new_features_df['age_cubed'] = df['age'] ** 3
    new_features_df['thalach_cubed'] = df['thalach'] ** 3
    new_features_df['oldpeak_cubed'] = df['oldpeak'] ** 3

    # Handle any potential NaN values
    new_features_df.fillna(0, inplace=True)


    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['fe_high_risk_1']
             }]
        }

@from_table_to_df(TABLE_NAMES['train_data_1'], None)
def fe_product_1_func(**kwargs):
    """
    Add additional features including interaction features, log transformations, and binning using Pandas DataFrame.
    """
    import pandas as pd
    import math

    # Extract the Pandas DataFrame
    df = kwargs['dfs']

    # Interaction features
    df['age_trestbps'] = df['age'] * df['trestbps']
    df['age_thalach'] = df['age'] * df['thalach']
    df['trestbps_thalach'] = df['trestbps'] * df['thalach']

    # Log transformations (adding 1 to avoid log(0) which is undefined)
    df['log_age'] = df['age'].apply(lambda x: math.log(x + 1))
    df['log_trestbps'] = df['trestbps'].apply(lambda x: math.log(x + 1))
    df['log_thalach'] = df['thalach'].apply(lambda x: math.log(x + 1))
    df['log_oldpeak'] = df['oldpeak'].apply(lambda x: math.log(x + 1))

    # Bin features
    df['age_bin'] = (df['age'] // 10) * 10
    df['trestbps_bin'] = (df['trestbps'] // 10) * 10
    df['thalach_bin'] = (df['thalach'] // 10) * 10
    df['oldpeak_bin'] = (df['oldpeak'] // 1) * 1

    # Boolean features
    df['painloc_bool'] = df['painloc'].apply(lambda x: 1 if x > 0 else 0)
    df['painexer_bool'] = df['painexer'].apply(lambda x: 1 if x > 0 else 0)
    df['cp_bool'] = df['cp'].apply(lambda x: 1 if x > 0 else 0)

    # Handle any potential NaN values by filling them with 0
    df = df.fillna(0)

    # Collect all new columns
    new_cols = ['age_trestbps', 'age_thalach', 'trestbps_thalach',
                'log_age', 'log_trestbps', 'log_thalach', 'log_oldpeak',
                'age_bin', 'trestbps_bin', 'thalach_bin', 'oldpeak_bin',
                'painloc_bool', 'painexer_bool', 'cp_bool']

    df_new = df[new_cols]

    return {
        'dfs': [
            {'df': df_new, 
             'table_name': TABLE_NAMES['product_fe_1']
             }]
        }


@from_table_to_df(TABLE_NAMES['train_data_2'], None)
@process_with_spark
def fe_high_risk_2_func(**kwargs):
    """
    Add additional features including high-risk indicators, and polynomial features using Spark DataFrame.
    """
    from pyspark.sql.functions import when, col, pow
    from pyspark.sql.types import DoubleType

    # Extract the Spark DataFrame
    df = kwargs['dfs'][0]

    # make features numeric
    for col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # High-risk condition indicators
    df = df.withColumn('high_trestbps', when(col('trestbps') > 130, 1).otherwise(0))
    df = df.withColumn('high_cholesterol', when(col('prop') > 200, 1).otherwise(0))
    df = df.withColumn('high_oldpeak', when(col('oldpeak') > 2, 1).otherwise(0))

    # Polynomial features
    df = df.withColumn('age_squared', pow(col('age'), 2))
    df = df.withColumn('thalach_squared', pow(col('thalach'), 2))
    df = df.withColumn('oldpeak_squared', pow(col('oldpeak'), 2))

    df = df.withColumn('age_cubed', pow(col('age'), 3))
    df = df.withColumn('thalach_cubed', pow(col('thalach'), 3))
    df = df.withColumn('oldpeak_cubed', pow(col('oldpeak'), 3))

    # Handle any potential NaN values by filling them with 0
    df = df.fillna(0)

    # Filter the DataFrame to include only the newly added columns
    new_cols = ['high_trestbps', 'high_cholesterol', 'high_oldpeak',
                'age_squared', 'thalach_squared', 'oldpeak_squared',
                'age_cubed', 'thalach_cubed', 'oldpeak_cubed']
    df_new = df.select(*new_cols)

    return {
        'dfs': [
            {'df': df_new, 
             'table_name': TABLE_NAMES['fe_high_risk_2']
             }]
        }

@from_table_to_df(TABLE_NAMES['train_data_2'], None)
@process_with_spark
def fe_product_2_func(**kwargs):
    """
    Add additional features including interaction features, log transformations, and binning using Spark DataFrame.
    """
    from pyspark.sql.functions import col, pow, log, when
    from pyspark.sql.types import DoubleType

    # Extract the Spark DataFrame
    df = kwargs['dfs'][0]

    # Ensure all columns are numeric
    for col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))

    # Interaction features
    df = df.withColumn('age_trestbps', col('age') * col('trestbps'))
    df = df.withColumn('age_thalach', col('age') * col('thalach'))
    df = df.withColumn('trestbps_thalach', col('trestbps') * col('thalach'))

    # Log transformations (adding 1 to avoid log(0) which is undefined)
    df = df.withColumn('log_age', log(col('age') + 1))
    df = df.withColumn('log_trestbps', log(col('trestbps') + 1))
    df = df.withColumn('log_thalach', log(col('thalach') + 1))
    df = df.withColumn('log_oldpeak', log(col('oldpeak') + 1))

    # Bin features
    df = df.withColumn('age_bin', (col('age') / 10).cast('int') * 10)
    df = df.withColumn('trestbps_bin', (col('trestbps') / 10).cast('int') * 10)
    df = df.withColumn('thalach_bin', (col('thalach') / 10).cast('int') * 10)
    df = df.withColumn('oldpeak_bin', (col('oldpeak') / 1).cast('int') * 1)

    # Boolean features
    df = df.withColumn('painloc_bool', when(col('painloc') > 0, 1).otherwise(0))
    df = df.withColumn('painexer_bool', when(col('painexer') > 0, 1).otherwise(0))
    df = df.withColumn('cp_bool', when(col('cp') > 0, 1).otherwise(0))

    # Handle any potential NaN values by filling them with 0
    df = df.fillna(0)

    # Collect all new columns
    new_cols = ['age_trestbps', 'age_thalach', 'trestbps_thalach',
                'log_age', 'log_trestbps', 'log_thalach', 'log_oldpeak',
                'age_bin', 'trestbps_bin', 'thalach_bin', 'oldpeak_bin',
                'painloc_bool', 'painexer_bool', 'cp_bool']

    df_new = df.select(*new_cols)

    return {
        'dfs': [
            {'df': df_new, 
             'table_name': TABLE_NAMES['product_fe_2']
             }]
        }


def scrape_smoke_func(**kwargs):
    """
    Insert data from webscraping to a database table.
    """

    import pandas as pd
    import requests
    from scrapy import Selector
    import re
    from typing import List
        
    url1 = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    response = requests.get(url1)
    # get the HTML file as a string
    html_content = response.content
    # create a selector object
    full_sel = Selector(text=html_content)

    # select all tables in page -> returns a SelectorList object
    tables = full_sel.xpath('//table')
    smokers_by_age = tables[1]
    # get the rows
    rows = smokers_by_age.xpath('./tbody//tr')

    def parse_row_1(row:Selector) -> List[str]:
        '''
        Parses a html row into a list of individual elements
        '''
        cells = row.xpath('.//th | .//td')
        row_data = []
        
        for i, cell in enumerate(cells):
            if i == 0 or i == 10:
                cell_text = cell.xpath('normalize-space(.)').get()
                cell_text = re.sub(r'<.*?>', ' ', cell_text)  # Remove remaining HTML tags
                # if there are br tags, there will be some binary characters
                cell_text = cell_text.replace('\xa0', '')  # Remove \xa0 characters
                row_data.append(cell_text)
        
        return row_data
            
    table_data = [parse_row_1(row) for row in rows]
    per_by_age_1_df = pd.DataFrame(table_data, columns=['age', 'rate'])


    # smoke 2
    url2 = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'
    response = requests.get(url2)

    # Create a scrapy Selector from the response content
    selector = Selector(text=response.content)

    ul_sel_list = selector.xpath('//ul[@class="block-list"]')
    genders = ul_sel_list[0]
    ages = ul_sel_list[1]

    def clean_gender_percents(rows):
        dict = {}
        for row in rows:
            gender = 'woman' if 'women' in row.split('(')[0] else 'man'
            percent = float(row.split('(')[1].split('%')[0])
            dict[gender] = float(percent)
        return dict

    def clean_age_percents(rows):
        for i, row in enumerate(rows):
            if i < len(rows) - 1:
                age = int(row.split('–')[1].split(' ')[0])
            else:
                age = int(row.split(' ')[7])
                
            percent = float(row.split('(')[1].split('%')[0])
            rows[i] = [age, percent]
        return rows

    def parse_row_2(row:Selector) -> List[str]:
        '''
        Parses a html row into a list of individual elements
        '''
        cells = row.xpath('./li')
        row_data = []
        
        for i, cell in enumerate(cells):
            cell_text = cell.xpath('normalize-space(.)').get()
            cell_text = re.sub(r'<.*?>', ' ', cell_text)  # Remove remaining HTML tags
            # if there are br tags, there will be some binary characters
            cell_text = cell_text.replace('\xa0', '')  # Remove \xa0 characters
            row_data.append(cell_text)
        
        return row_data

    per_by_gender = clean_gender_percents(parse_row_2(genders))
    per_by_gender_df = pd.DataFrame.from_dict(per_by_gender, orient='index', columns=['values'])

    per_by_age = clean_age_percents(parse_row_2(ages))
    per_by_age_2_df = pd.DataFrame(per_by_age, columns=['age', 'rate'])

     # Create a database connection
    conn = create_db_connection()
    
    # Write the DataFrame to SQL
    per_by_age_1_df.to_sql(TABLE_NAMES['per_by_age_1'], con=conn, if_exists="replace", index=False)
    per_by_age_2_df.to_sql(TABLE_NAMES['per_by_age_2'], con=conn, if_exists="replace", index=False)
    per_by_gender_df.to_sql(TABLE_NAMES['per_by_gender'], con=conn, if_exists="replace", index=False)

    # Close the database connection
    conn.close()

    return {'status': 1}

@from_table_to_df([TABLE_NAMES['fe_high_risk_1'], TABLE_NAMES['product_fe_2'], TABLE_NAMES['train_data_1'], TABLE_NAMES['per_by_age_1'], TABLE_NAMES['per_by_age_2'], TABLE_NAMES['per_by_gender']], [TABLE_NAMES['per_by_age_1'], TABLE_NAMES['per_by_age_2'], TABLE_NAMES['per_by_gender']])
def merge_smoke_func(**kwargs):

    dfs = kwargs['dfs']
    data = pd.concat([dfs[0], dfs[1], dfs[2]], axis=1)
    data = data.dropna()
    per_by_age_1 = dfs[3]
    per_by_age_2 = dfs[4]
    per_by_gender = dfs[5]

    data['smoke_1'] = data['smoke']
    data['smoke_2'] = data['smoke']

    def get_rate_1(age):
        age = float(age)
        for i, row in per_by_age_1.iterrows():
            if i < len(table_data) - 1:
                cutoff = row[0].split('–')[1]
                if age <= float(cutoff):
                    return float(row[1])
            else:
                return float(row[1])

    
    def add_rates_1(data):
        for index, row in data.iterrows():
            data.loc[index, 'smoke_1'] = get_rate_1(row['age'])/100
        return data

    def get_rate_2(sex, age):
        if sex == 0:
            age = float(age)
            for i, row in per_by_age_2.iterrows():
                if i < len(per_by_age_2) - 1:
                    if age <= row[0]:
                        return row[1]
                else:
                    return row[1]
        else:
            age = float(age)
            for i, row in per_by_age_2.iterrows():
                if i < len(per_by_age_2) - 1:
                    if age <= row[0]:
                        return row[1] * per_by_gender.loc['man', 'rate'] / per_by_gender.loc['woman', 'rate']
                else:
                    return row[1] * per_by_gender.loc['man', 'rate'] / per_by_gender.loc['woman', 'rate']
      
    
    def add_rates_2(data):
        for index, row in data.iterrows():
            data.loc[index, 'smoke_2'] = get_rate_2(row['sex'], row['age'])/100
        return data
    
    smoke_mean = data['smoke_1'].mean()
    
    # Select the rows where 'smoke_1' is equal to the mean
    mean_smoke_1_data = data[data['smoke_1'] == smoke_mean]
    # Apply the add_rates_1 function to these rows
    mean_smoke_1_data = add_rates_1(mean_smoke_1_data)
    # Update the original DataFrame with the processed rows
    data.update(mean_smoke_1_data)

    # Select the rows where 'smoke_1' is equal to the mean
    mean_smoke_2_data = data[data['smoke_2'] == smoke_mean]
    # Apply the add_rates_1 function to these rows
    mean_smoke_2_data = add_rates_2(mean_smoke_2_data)
    # Update the original DataFrame with the processed rows
    data.update(mean_smoke_2_data)

    return {
            'dfs': [
                {
                    'df': data, 
                    'table_name': TABLE_NAMES['scrape_merged']
                }]
            }

def lr_1(dfs):
    """
    train logistic regression
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    org_df = dfs[1]
    fe_df = dfs[0]

    # combine dataframes
    df = pd.concat([org_df, fe_df], axis=1)

    # Split the data into training and validation sets
    string_columns = [v for v in df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['sklearn']['labels']]
    df = df.drop(string_columns, axis=1)

    Y = df[PARAMS['sklearn']['labels']]
    X = df.drop(PARAMS['sklearn']['labels'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=PARAMS['sklearn']['train_test_ratio'], random_state=42)

    # Create an instance of Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the val set
    y_pred = model.predict(X_val)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)        

    return accuracy

def lr_2(dfs):
    """
    Train logistic regression using Spark DataFrame and evaluate accuracy.
    """
    
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import monotonically_increasing_id

    # Extract the Spark DataFrames
    if len(dfs) < 2:
        df = dfs[0]
        print(df.columns)
    else:
        org_df = dfs[1]
        fe_df = dfs[0]
    
        print(org_df.columns)
        print(fe_df.columns)
        
        # Add a unique identifier to each DataFrame
        org_df_with_id = org_df.withColumn("row_id", monotonically_increasing_id())
        fe_df_with_id = fe_df.withColumn("row_id", monotonically_increasing_id())
    
        # Perform the join operation based on the row_id
        df = org_df_with_id.join(fe_df_with_id, on="row_id", how="inner")
    
        # Drop the row_id column after joining
        df = df.drop("row_id")

    # Define label
    label_col = PARAMS['spark']['labels']
    
    feature_cols = [col for col in df.columns if col != label_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Split the data into training and validation sets
    train_data, val_data = df.randomSplit([PARAMS['spark']['train_test_split'], 1-PARAMS['spark']['train_test_split']], seed=PARAMS['spark']['split_seed']) 

    # Define the Logistic Regression model
    lr = LogisticRegression(labelCol=label_col, featuresCol="features")

    # Define parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .build()

    # Define evaluator for accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")

    pipeline = Pipeline(stages=[assembler, lr])  

    # Define cross-validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=param_grid,
                              evaluator=evaluator,
                              numFolds=PARAMS['spark']['number_of_folds'],
                              parallelism=2)

    # Fit the model
    cv_model = crossval.fit(train_data)

    # Make predictions on the validation set
    predictions = cv_model.transform(val_data)

    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)

    print("accuracy", accuracy)

    return accuracy

def svm_1(dfs):
    """
    Train Support Vector Machine
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    # Extract the DataFrames
    org_df = dfs[1]
    fe_df = dfs[0]

    # combine dataframes
    df = pd.concat([org_df, fe_df], axis=1)

    # Split the data into training and validation sets
    string_columns = [v for v in df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['sklearn']['labels']]
    df = df.drop(string_columns, axis=1)

    Y = df[PARAMS['sklearn']['labels']]
    X = df.drop(PARAMS['sklearn']['labels'], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=PARAMS['sklearn']['train_test_ratio'], random_state=42)
    
    # Create an instance of SVM model
    model = SVC(kernel='linear')  # You can change the kernel to 'rbf', 'poly', etc. based on your needs

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy

def svm_2(dfs):
    """
    Train Support Vector Machine using Spark DataFrame and evaluate accuracy.
    """

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LinearSVC
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import monotonically_increasing_id

    # Extract the Spark DataFrames
    if len(dfs) < 2:
        df = dfs[0]
        print(df.columns)
    else:
        org_df = dfs[1]
        fe_df = dfs[0]
    
        print(org_df.columns)
        print(fe_df.columns)
    
        # Add a unique identifier to each DataFrame
        org_df_with_id = org_df.withColumn("row_id", monotonically_increasing_id())
        fe_df_with_id = fe_df.withColumn("row_id", monotonically_increasing_id())
    
        # Perform the join operation based on the row_id
        df = org_df_with_id.join(fe_df_with_id, on="row_id", how="inner")
    
        # Drop the row_id column after joining
        df = df.drop("row_id")
    
    # Define label
    label_col = PARAMS['spark']['labels']
    
    feature_cols = [col for col in df.columns if col != label_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    # Split the data into training and validation sets
    train_data, val_data = df.randomSplit([PARAMS['spark']['train_test_split'], 1-PARAMS['spark']['train_test_split']], seed=PARAMS['spark']['split_seed']) 

    # Define the SVM model
    svm = LinearSVC(labelCol=label_col, featuresCol="features", regParam=0.1)  # Adjust regParam here

    pipeline = Pipeline(stages=[assembler, svm])  

    # Define evaluator for accuracy
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")

    # Fit the model
    model = pipeline.fit(train_data)

    # Make predictions on the validation set
    predictions = model.transform(val_data)

    # Evaluate the model
    accuracy = evaluator.evaluate(predictions)

    print("accuracy", accuracy)

    return accuracy

@from_table_to_df([TABLE_NAMES['scrape_merged']], None)
def merge_lr_func(**kwargs):
    """
    train logistic regression on product features
    """
    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = lr_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['scrape_merged']], None)
def merge_svm_func(**kwargs):
    """
    train svm regression on product features
    """
    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = svm_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['product_fe_1'], TABLE_NAMES['train_data_1']], None)
def product_lr_1_func(**kwargs):
    """
    train logistic regression on product features
    """

    dfs = kwargs['dfs']
 
    accuracy = lr_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['product_fe_1'], TABLE_NAMES['train_data_1']], None)
def product_svm_1_func(**kwargs):
    """
    train svm regression on product features
    """

    dfs = kwargs['dfs']
 
    accuracy = svm_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data_1']], None)
def production_lr_1_func(**kwargs):
    """
    train logistic regression on the production model which is not using any additional features
    """

    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = lr_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data_1']], None)
def production_svm_1_func(**kwargs):
    """
    train svm regression on the production model which is not using any additional features
    """

    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = svm_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['fe_high_risk_1'], TABLE_NAMES['train_data_1']], None)
def high_risk_lr_1_func(**kwargs):
    """
    train logistic regression on max features
    """

    dfs = kwargs['dfs']
 
    accuracy = lr_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['fe_high_risk_1'], TABLE_NAMES['train_data_1']], None)
def high_risk_svm_1_func(**kwargs):
    """
    train logistic regression on max features
    """

    dfs = kwargs['dfs']
    accuracy = svm_1(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }


@from_table_to_df([TABLE_NAMES['product_fe_2'], TABLE_NAMES['train_data_2']], None)
@process_with_spark
def product_lr_2_func(**kwargs):
    """
    Train logistic regression on product features.
    """
    dfs = kwargs['dfs']
    print(dfs)
    accuracy = lr_2(dfs)

    print("accuracy", accuracy)
    
    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['product_fe_2'], TABLE_NAMES['train_data_2']], None)
@process_with_spark
def product_svm_2_func(**kwargs):
    """
    Train SVM regression on product features.
    """
    dfs = kwargs['dfs']
    print(dfs)
    accuracy = svm_2(dfs)
    
    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data_2']], None)
@process_with_spark
def production_lr_2_func(**kwargs):
    """
    Train logistic regression on the production model which is not using any additional features.
    """

    from pyspark.sql.types import StructType

    dfs = kwargs['dfs']
    print(dfs)
    accuracy = lr_2(dfs)    
    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data_2']], None)
@process_with_spark
def production_svm_2_func(**kwargs):
    """
    Train SVM regression on the production model which is not using any additional features.
    """
    
    from pyspark.sql.types import StructType

    dfs = kwargs['dfs']
    print(dfs)
    accuracy = svm_2(dfs)
    
    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['fe_high_risk_2'], TABLE_NAMES['train_data_2']], None)
@process_with_spark
def high_risk_lr_2_func(**kwargs):
    """
    Train logistic regression on max features.
    """
    dfs = kwargs['dfs']
    print(dfs)
    accuracy = lr_2(dfs)
    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['fe_high_risk_2'], TABLE_NAMES['train_data_2']], None)
@process_with_spark
def high_risk_svm_2_func(**kwargs):
    """
    Train SVM regression on max features.
    """
    dfs = kwargs['dfs']
    print(dfs)
    accuracy = svm_2(dfs)
    return {
        "accuracy": accuracy,
        'dfs': []
    }

feature_operations = ["high_risk_lr_1", "high_risk_svm_1", "product_lr_1", "product_svm_1", "high_risk_lr_2", "high_risk_svm_2", "product_lr_2", "product_svm_2", "merge_lr", "merge_svm"] 

def encode_task_id(feature_operation: str):
    return f'{feature_operation}_evaluation'

def decide_which_model(**kwargs):
    """
    Perform testing on the best model; if the best model is not better than the production model, do nothing
    """

    ti = kwargs['ti']


    # Get the maximum accuracies for each model type and version
    accuracies = {}
    for model_type in ['high_risk_lr', 'product_lr', 'high_risk_svm', 'product_svm']:
        for version in ['_1', '_2']:
            task_id = model_type + version
            print(task_id)
            accuracy = ti.xcom_pull(task_ids=task_id)['accuracy']
            accuracies[task_id] = accuracy
    for model_type in ['merge_lr', 'merge_svm']:
            accuracy = ti.xcom_pull(task_ids=task_id)['accuracy']
            accuracies[model_type] = accuracy
    
    # Get the production accuracies
    production_lr_1_return_value = ti.xcom_pull(task_ids='production_lr_1')
    production_lr_2_return_value = ti.xcom_pull(task_ids='production_lr_2')
    production_svm_1_return_value = ti.xcom_pull(task_ids='production_svm_1')
    production_svm_2_return_value = ti.xcom_pull(task_ids='production_svm_2')
    production_accuracy_1 = max(production_lr_1_return_value['accuracy'], production_svm_1_return_value['accuracy'])
    production_accuracy_2 = max(production_lr_2_return_value['accuracy'], production_svm_2_return_value['accuracy'])
    production_accuracy = max(production_accuracy_1, production_accuracy_2)

    # Find max accuracy
    max_accuracy_task_id, max_accuracy = max(accuracies.items(), key=lambda x: x[1])
    
    print("Maximum Accuracy Task ID:", max_accuracy_task_id)
    print("Maximum Accuracy:", max_accuracy)
    
    # Decide what to do based on accuracy comparisons
    if max_accuracy - production_accuracy < -PARAMS['ml']['tolerance']:
        return "do_nothing"
    
    return encode_task_id(max_accuracy_task_id)
    
               
def extensive_evaluation_lr_func(train_df, test_df, fe_type: str, **kwargs):
    """
    train the model on the entire validation data set
    test the final model on test; evaluation also on perturbed test data set
    """
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    from scipy.stats import norm

    model = LogisticRegression()

    # Train the model
    y_train = train_df[PARAMS['ml']['labels']]
    X_train = train_df.drop(PARAMS['ml']['labels'], axis=1)
    model.fit(X_train, y_train)

    def accuracy_on_test(perturb:str = True) -> float:
        X_test = test_df.drop(PARAMS['ml']['labels'], axis=1)
        y_test = test_df[PARAMS['ml']['labels']]

        if perturb == True:
            # we are also perturbing categorical features which is fine since the perturbation is small and thus should not have affect on such features
            X_test = X_test.apply(lambda x: x + np.random.normal(0, PARAMS['ml']['perturbation_std'], len(x)))

        y_pred = model.predict(X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    accuracy = accuracy_on_test(perturb=False)
    print(f"Accuracy on test {accuracy}")

    # we stop when given confidence in accuracy is achieved
    accuracies = []
    for i in range(PARAMS['ml']['max_perturbation_iterations']):
        # Make predictions on the test set
        accuracy = accuracy_on_test()
        accuracies.append(accuracy)

        # compute the confidence interval; break if in the range
        average = np.mean(accuracies)
        std_error = np.std(accuracies) / np.sqrt(len(accuracies))
        confidence_interval = norm.interval(PARAMS['ml']['confidence_level'], loc=average, scale=std_error)
        confidence = confidence_interval[1] - confidence_interval[0]
        if confidence <= 2 * std_error:
            break
    else:
        print(f"Max number of trials reached. Average accuracy on perturbed test {average} with confidence {confidence} and std error of {2 * std_error}")

    print(f"Average accuracy on perturbed test {average}")

def extensive_evaluation_svm_func(train_df, test_df, fe_type: str, **kwargs):
    """
    Train the model on the entire validation data set
    Test the final model on test; evaluation also on perturbed test data set
    """
    
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import numpy as np
    from scipy.stats import norm

    model = SVC(kernel='linear')  # You can change the kernel type based on your needs

    # Train the model
    y_train = train_df[PARAMS['ml']['labels']]
    X_train = train_df.drop(PARAMS['ml']['labels'], axis=1)
    model.fit(X_train, y_train)

    def accuracy_on_test(perturb: str = True) -> float:
        X_test = test_df.drop(PARAMS['ml']['labels'], axis=1)
        y_test = test_df[PARAMS['ml']['labels']]

        if perturb:
            # Perturbing the test data
            X_test = X_test.apply(lambda x: x + np.random.normal(0, PARAMS['ml']['perturbation_std'], len(x)))

        y_pred = model.predict(X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    accuracy = accuracy_on_test(perturb=False)
    print(f"Accuracy on test: {accuracy}")

    # Stop when given confidence in accuracy is achieved
    accuracies = []
    for i in range(PARAMS['ml']['max_perturbation_iterations']):
        # Make predictions on the perturbed test set
        accuracy = accuracy_on_test()
        accuracies.append(accuracy)

        # Compute the confidence interval; break if within the range
        average = np.mean(accuracies)
        std_error = np.std(accuracies) / np.sqrt(len(accuracies))
        confidence_interval = norm.interval(PARAMS['ml']['confidence_level'], loc=average, scale=std_error)
        confidence = confidence_interval[1] - confidence_interval[0]
        if confidence <= 2 * std_error:
            break
    else:
        print(f"Max number of trials reached. Average accuracy on perturbed test: {average} with confidence: {confidence} and std error of {2 * std_error}")

    print(f"Average accuracy on perturbed test: {average}")
    
@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def high_risk_lr_1_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_lr_func(dfs[0], dfs[1], "high_risk")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def product_lr_1_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_lr_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def high_risk_svm_1_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_svm_func(dfs[0], dfs[1], "high_risk")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def product_svm_1_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_svm_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def high_risk_lr_2_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_lr_func(dfs[0], dfs[1], "high_risk")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def product_lr_2_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_lr_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def high_risk_svm_2_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_svm_func(dfs[0], dfs[1], "high_risk")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def product_svm_2_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_svm_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def merge_lr_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_lr_func(dfs[0], dfs[1], "merge")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def merge_svm_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_svm_func(dfs[0], dfs[1], "merge")

    return {'dfs': []}


# Instantiate the DAG
dag = DAG(
    'Emily_HW4',
    default_args=default_args,
    description='HW4',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

drop_tables = PostgresOperator(
    task_id="drop_tables",
    postgres_conn_id=PARAMS['db']['db_connection'],
    sql=f"""
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO {PARAMS['db']['username']};
    GRANT ALL ON SCHEMA public TO public;
    COMMENT ON SCHEMA public IS 'standard public schema';
    """,
    dag=dag
)


add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    provide_context=True,
    dag=dag
)

clean_and_impute_data_1 = PythonOperator(
    task_id='clean_and_impute_data_1',
    python_callable=clean_and_impute_data_1_func,
    provide_context=True,
    dag=dag
)

clean_and_impute_data_2 = PythonOperator(
    task_id='clean_and_impute_data_2',
    python_callable=clean_and_impute_data_2_func,
    provide_context=True,
    dag=dag
)

normalize_data_1 = PythonOperator(
    task_id='normalize_data_1',
    python_callable=normalize_data_1_func,
    provide_context=True,
    dag=dag
)

normalize_data_2 = PythonOperator(
    task_id='normalize_data_2',
    python_callable=normalize_data_2_func,
    provide_context=True,
    dag=dag
)

eda_1 = PythonOperator(
    task_id='EDA_1',
    python_callable=eda_1_func,
    provide_context=True,
    dag=dag
)

eda_2 = PythonOperator(
    task_id='EDA_2',
    python_callable=eda_2_func,
    provide_context=True,
    dag=dag
)

fe_high_risk_1 = PythonOperator(
    task_id='fe_high_risk_1',
    python_callable=fe_high_risk_1_func,
    provide_context=True,
    dag=dag
)

fe_product_1 = PythonOperator(
    task_id='add_product_features_1',
    python_callable=fe_product_1_func,
    provide_context=True,
    dag=dag
)

fe_high_risk_2 = PythonOperator(
    task_id='fe_high_risk_2',
    python_callable=fe_high_risk_2_func,
    provide_context=True,
    dag=dag
)

fe_product_2 = PythonOperator(
    task_id='add_product_features_2',
    python_callable=fe_product_2_func,
    provide_context=True,
    dag=dag
)

product_lr_1 = PythonOperator(
    task_id='product_lr_1',
    python_callable=product_lr_1_func,
    provide_context=True,
    dag=dag
)

high_risk_lr_1 = PythonOperator(
    task_id='high_risk_lr_1',
    python_callable=high_risk_lr_1_func,
    provide_context=True,
    dag=dag
)

production_lr_1 = PythonOperator(
    task_id='production_lr_1',
    python_callable=production_lr_1_func,
    provide_context=True,
    dag=dag
)

product_svm_1 = PythonOperator(
    task_id='product_svm_1',
    python_callable=product_svm_1_func,
    provide_context=True,
    dag=dag
)

high_risk_svm_1 = PythonOperator(
    task_id='high_risk_svm_1',
    python_callable=high_risk_svm_1_func,
    provide_context=True,
    dag=dag
)

production_svm_1 = PythonOperator(
    task_id='production_svm_1',
    python_callable=production_svm_1_func,
    provide_context=True,
    dag=dag
)

product_lr_2 = PythonOperator(
    task_id='product_lr_2',
    python_callable=product_lr_2_func,
    provide_context=True,
    dag=dag
)

high_risk_lr_2 = PythonOperator(
    task_id='high_risk_lr_2',
    python_callable=high_risk_lr_2_func,
    provide_context=True,
    dag=dag
)

production_lr_2 = PythonOperator(
    task_id='production_lr_2',
    python_callable=production_lr_2_func,
    provide_context=True,
    dag=dag
)

product_svm_2 = PythonOperator(
    task_id='product_svm_2',
    python_callable=product_svm_2_func,
    provide_context=True,
    dag=dag
)

high_risk_svm_2 = PythonOperator(
    task_id='high_risk_svm_2',
    python_callable=high_risk_svm_2_func,
    provide_context=True,
    dag=dag
)

production_svm_2 = PythonOperator(
    task_id='production_svm_2',
    python_callable=production_svm_2_func,
    provide_context=True,
    dag=dag
)


model_selection = BranchPythonOperator(
    task_id='model_selection',
    python_callable=decide_which_model,
    provide_context=True,
    dag=dag,
)

dummy_task = DummyOperator(
    task_id='do_nothing',
    dag=dag,
)

scrape_smoke = PythonOperator(
    task_id='scrape_smoke',
    python_callable=scrape_smoke_func,
    provide_context=True,
    dag=dag
)

merge_smoke = PythonOperator(
    task_id='merge_smoke',
    python_callable=merge_smoke_func,
    provide_context=True,
    dag=dag
)

merge_lr = PythonOperator(
    task_id='merge_lr',
    python_callable=merge_lr_func,
    provide_context=True,
    dag=dag
)

merge_svm = PythonOperator(
    task_id='merge_svm',
    python_callable=merge_svm_func,
    provide_context=True,
    dag=dag
)

evaluation_tasks = []
for feature_type in feature_operations:
    encoding = encode_task_id(feature_type)
    evaluation_tasks.append(PythonOperator(
        task_id=encoding,
        python_callable=locals()[f'{encoding}_func'],
        provide_context=True,
        dag=dag
    ))


drop_tables >> add_data_to_table >> clean_and_impute_data_1 >> normalize_data_1
clean_and_impute_data_1 >> eda_1
add_data_to_table >> clean_and_impute_data_2 >> normalize_data_2
clean_and_impute_data_2 >> eda_2 

normalize_data_1 >> [fe_high_risk_1, fe_product_1]
normalize_data_2 >> [fe_high_risk_2, fe_product_2]

fe_product_1 >> [product_lr_1, product_svm_1]
fe_high_risk_1 >> [high_risk_lr_1, high_risk_svm_1]
normalize_data_1 >> [production_lr_1, production_svm_1]

fe_product_2 >> [product_lr_2, product_svm_2]
fe_high_risk_2 >> [high_risk_lr_2, high_risk_svm_2]
normalize_data_2 >> [production_lr_2, production_svm_2]

[fe_high_risk_1, fe_product_2, scrape_smoke] >> merge_smoke >> [merge_lr, merge_svm]

[product_lr_1, product_svm_1, high_risk_lr_1, high_risk_svm_1, production_lr_1, production_svm_1, product_lr_2, product_svm_2, high_risk_lr_2, high_risk_svm_2, production_lr_2, production_svm_2, merge_lr, merge_svm] >> model_selection

model_selection >> [dummy_task, *evaluation_tasks] 