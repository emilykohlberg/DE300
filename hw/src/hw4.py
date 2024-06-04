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
    "max_fe_1": "max_fe_features_1",
    "product_fe_1": "product_fe_features_1"
}

ENCODED_SUFFIX = "_encoded"
NORMALIZATION_TABLE_COLUMN_NAMES = ["name", "data_min", "data_max", "scale", "min"]

# Define the default args dictionary for DAG
default_args = {
    'owner': 'emilykohlberg',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
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
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Wrote to table {table_name}")

            conn.close()
            result.pop('dfs')

            return result
        return wrapper
    return decorator



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
def clean_data_1_func(**kwargs):
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

    # impute
    m_painloc = data['painloc'].mean()
    data.loc[data['painloc'].isna(), 'painloc'] = m_painloc
    
    m_painexer = data['painexer'].mean()
    data.loc[data['painexer'].isna(), 'painexer'] = m_painexer
    
    mean_trestbps = data['trestbps'].mean()
    data.loc[data['trestbps'].isna(), 'trestbps'] = mean_trestbps

    mean_oldpeak = data['oldpeak'].mean()
    data.loc[data['oldpeak'].isna(), 'oldpeak'] = mean_oldpeak

    mean_thaldur = data['thaldur'].mean()
    data.loc[data['thaldur'].isna(), 'thaldur'] = mean_thaldur
    
    mean_thalach = data['thalach'].mean()
    data.loc[data['thalach'].isna(), 'thalach'] = mean_thalach

    m_fbs = data['fbs'].mean()
    data.loc[data['fbs'].isna(), 'fbs'] = m_fbs
    
    m_prop = data['prop'].mean()
    data.loc[data['prop'].isna(), 'prop'] = m_prop
    
    m_nitr = data['nitr'].mean()
    data.loc[data['nitr'].isna(), 'nitr'] = m_nitr
    
    m_pro = data['pro'].mean()
    data.loc[data['pro'].isna(), 'pro'] = m_pro
    
    m_diuretic = data['diuretic'].mean()
    data.loc[data['diuretic'].isna(), 'diuretic'] = m_diuretic

    m_exang = data['exang'].mean()
    data.loc[data['exang'].isna(), 'exang'] = m_exang
    
    mode_slope = data['slope'].mean()
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


    return {
        'dfs': [
            {'df': data, 
             'table_name': TABLE_NAMES['clean_data_1']
             }]
        }

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_data_2_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    from pyspark.sql import SparkSession
    from pyspark.sql.dataframe import DataFrame
    from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType
    from pyspark.ml.feature import Imputer
    import pyspark.sql.functions as F
    from pyspark.sql.functions import when, col

    data = kwargs['dfs']

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

    # make age an int
    data = data.withColumn("age", data["age"].cast(IntegerType()))


    # Identify numeric features
    numeric_features = [f.name for f in data.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
    numeric_features.remove("target")
    
    # numeric columns
    imputed_columns = [f"Imputed{v}" for v in numeric_features]
    imputer_numeric = Imputer(inputCols=numeric_features, outputCols=imputed_columns, strategy = "mean")
    
    # Apply Imputer transformation
    model = imputer_numeric.fit(data)
    imputed_data = model.transform(data)
    
    # Select and rename columns to replace original columns with imputed ones
    for original_col, imputed_col in zip(numeric_features, imputed_columns):
        imputed_data = imputed_data.withColumnRenamed(imputed_col, original_col)

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
def normalize_data_2_func(**kwargs):
    """
    normalization
    split to train/test
    """

    from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType
    import pyspark.sql.functions as F
    from pyspark.sql.functions import when, col
    from pyspark.ml.feature import MinMaxScaler, VectorAssembler
    import pandas as pd
    

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = data.randomSplit([PARAMS['spark']['train_test_split'], 1-PARAMS['spark']['train_test_split']], seed=PARAMS['spark']['split_seed'])

    # Identify numerical columns to be normalized
    numeric_columns = [f.name for f in df_train.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
    numeric_columns.remove(PARAMS['sklearn']['labels'])
    
    normalization_values = []
    
    # Apply MinMaxScaler to each numerical column
    for column in numeric_columns:
        # Assemble the column into a vector
        assembler = VectorAssembler(inputCols=[column], outputCol=f"{column}_vec")
        df_train_vec = assembler.transform(df_train)
        
        # Initialize MinMaxScaler
        scaler = MinMaxScaler(inputCol=f"{column}_vec", outputCol=f"{column}_scaled")
        
        # Fit the scaler on the data
        scaler_model = scaler.fit(df_train_vec)
        
        # Transform the data
        df_train_scaled = scaler_model.transform(df_train_vec)
        
        # Extract and store normalization values
        normalization_values.append((column, float(scaler_model.originalMin[0]), float(scaler_model.originalMax[0]), float(scaler_model.surrogateScaler.scale[0])))
        
        # Replace the original column with the scaled column
        df_train_scaled = df_train_scaled.withColumn(column, col(f"{column}_scaled")[0])
        
        # Drop the intermediate columns
        df_train_scaled = df_train_scaled.drop(f"{column}_vec", f"{column}_scaled")
        
        # Update df_train
        df_train = df_train_scaled
    
    # Convert normalization values to a pandas DataFrame
    normalization_df = pd.DataFrame(data=normalization_values, columns=["name", "min", "max", "scale"])


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
                'df': normalization_df,
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
def eda_2_func(**kwargs):
    """
    print basic statistics
    """

    from pyspark.sql.types import StructType

    pdf = kwargs['dfs']
    schema = StructType.fromJson(eval(pdf.to_json(orient="records"))[0])
    df = spark.createDataFrame(pdf, schema=schema)
    
    # Print descriptive statistics
    df.describe().show()
    
    # Return an empty DataFrame as specified
    empty_schema = StructType()
    empty_df = spark.createDataFrame([], empty_schema)
    
    return {'dfs': empty_df}


@from_table_to_df(TABLE_NAMES['train_data_1'], None)
def fe_max_1_func(**kwargs):
    """
    add features that are max of two features 
    """

    import pandas as pd

    df = kwargs['dfs']

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['sklearn']['labels']]
    new_features_df = pd.DataFrame()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            new_features_df['max_'+features[i]+'_'+features[j]] = df[[features[i], features[j]]].max(axis=1)

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['max_fe_1']
             }]
        }

@from_table_to_df(TABLE_NAMES['train_data_1'], None)
def fe_product_1_func(**kwargs):
    """
    add features that are products of two features
    """
    
    import pandas as pd

    df = kwargs['dfs']

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['sklearn']['labels']]
    new_features_df = pd.DataFrame()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            new_features_df[features[i]+'*'+features[j]] = df[features[i]] * df[features[j]]

    # NOTE: normalization should be done

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['product_fe_1']
             }]
        }

def train_model_1(dfs):
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

@from_table_to_df([TABLE_NAMES['product_fe_1'], TABLE_NAMES['train_data_1']], TABLE_NAMES["product_fe_1"])
def product_train_1_func(**kwargs):
    """
    train logistic regression on product features
    """

    dfs = kwargs['dfs']
 
    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data_1']], None)
def production_train_1_func(**kwargs):
    """
    train logistic regression on the production model which is not using any additional features
    """

    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['max_fe_1'], TABLE_NAMES['train_data_1']], TABLE_NAMES["max_fe_1"])
def max_train_1_func(**kwargs):
    """
    train logistic regression on max features
    """

    dfs = kwargs['dfs']
 
    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

feature_operations = ["max", "product"] # used when we automatically create tasks
def encode_task_id(feature_operation: str):
    return f'{feature_type}_evaluation_1'

def decide_which_model(**kwargs):
    """
    perform testing on the best model; if the best model not better than the production model, do nothing
    """
    
    ti = kwargs['ti']
    max_train_return_value = ti.xcom_pull(task_ids='max_train')
    product_train_return_value = ti.xcom_pull(task_ids='product_train')
    production_train_return_value = ti.xcom_pull(task_ids='production_train')
    
    print(f"Accuracies (product, max, production) {product_train_return_value['accuracy']}, {max_train_return_value['accuracy']}, {production_train_return_value['accuracy']}")

    if max(max_train_return_value['accuracy'], product_train_return_value['accuracy']) - production_train_return_value['accuracy'] <  -PARAMS['ml']['tolerance']:
        return "do_nothing"
    elif max_train_return_value['accuracy'] > product_train_return_value['accuracy']:
        return encode_task_id("max")
    else:
        return encode_task_id("product")

def extensive_evaluation_func(train_df, test_df, fe_type: str, **kwargs):
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
    
@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def max_evaluation_1_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_func(dfs[0], dfs[1], "max")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data_1'], TABLE_NAMES['test_data_1']], None)
def product_evaluation_1_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

# Instantiate the DAG
dag = DAG(
    'HW4',
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

clean_data_1 = PythonOperator(
    task_id='clean_data_1',
    python_callable=clean_data_1_func,
    provide_context=True,
    dag=dag
)

clean_data_2 = PythonOperator(
    task_id='clean_data_2',
    python_callable=clean_data_2_func,
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

fe_max_1 = PythonOperator(
    task_id='add_max_features_1',
    python_callable=fe_max_1_func,
    provide_context=True,
    dag=dag
)

fe_product_1 = PythonOperator(
    task_id='add_product_features_1',
    python_callable=fe_product_1_func,
    provide_context=True,
    dag=dag
)

product_train_1 = PythonOperator(
    task_id='product_train_1',
    python_callable=product_train_1_func,
    provide_context=True,
    dag=dag
)

max_train_1 = PythonOperator(
    task_id='max_train_1',
    python_callable=max_train_1_func,
    provide_context=True,
    dag=dag
)

production_train_1 = PythonOperator(
    task_id='production_train_1',
    python_callable=production_train_1_func,
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

evaluation_tasks = []
for feature_type in feature_operations:
    encoding = encode_task_id(feature_type)
    evaluation_tasks.append(PythonOperator(
        task_id=encoding,
        python_callable=locals()[f'{encoding}_func'],
        provide_context=True,
        dag=dag
    ))

drop_tables >> add_data_to_table >> clean_data_1 >> normalize_data_1
clean_data_1 >> eda_1
add_data_to_table >> clean_data_2 >> normalize_data_2
clean_data_2 >> eda_2
normalize_data_1 >> [fe_max_1, fe_product_1]
fe_product_1 >> product_train_1
fe_max_1 >> max_train_1
normalize_data_1 >> production_train_1
[product_train_1, max_train_1, production_train_1] >> model_selection
model_selection >> [dummy_task, *evaluation_tasks]