from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
from airflow.operators.subdag_operator import SubDagOperator
import tomli
import pathlib
import boto3

# read the parameters from toml
CONFIG_BUCKET = "de300spring2024-emilykohlberg"
CONFIG_FILE = "config.toml"

TABLE_NAMES = {
    "original_data": "wine",
    "clean_data": "wine_clean_data",
    "train_data": "wine_train_data",
    "test_data": "wine_test_data",
    "normalization_data": "normalization_values",
    "max_fe": "max_fe_features",
    "product_fe": "product_fe_features"
}

ENCODED_SUFFIX = "_encoded"
NORMALIZATION_TABLE_COLUMN_NAMES = ["name", "data_min", "data_max", "scale", "min"]

# Define the default args dictionary for DAG
default_args = {
    'owner': 'emilykohlberg',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
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
    create a db connection to the postgres connection

    return the connection
    """
    
    import re
    from sqlalchemy import create_engine

    conn = BaseHook.get_connection(PARAMS['db']['db_connection'])
    conn_uri = conn.get_uri()

    # replace the driver; airflow connections use postgres which needs to be replaced
    conn_uri= re.sub('^[^:]*://', PARAMS['db']['db_alchemy_driver']+'://', conn_uri)

    engine = create_engine(conn_uri)
    conn = engine.connect()

    return conn

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
    insert data from local csv to a db table
    """

    import pandas as pd

    conn = create_db_connection()

    df = pd.read_csv(PARAMS['files']['local_file'], header=0)
    df.to_sql(TABLE_NAMES['original_data'], conn, if_exists="replace", index=False)

    conn.close()

    return {'status': 1}

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_data_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_df = kwargs['dfs']

    # Drop rows with missing values
    data_df = data_df.dropna()

    # Remove outliers using Z-score 
    numeric_columns = [v for v in data_df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for column in numeric_columns:
        values = (data_df[column] - data_df[column].mean()).abs() / data_df[column].std() - PARAMS['ml']['outliers_std_factor']
        data_df = data_df[values < PARAMS['ml']['tolerance']]

    # label encoding
    label_encoder = LabelEncoder()
    string_columns = [v for v in data_df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for v in string_columns:
        data_df[v + ENCODED_SUFFIX] = label_encoder.fit_transform(data_df[v])

    return {
        'dfs': [
            {'df': data_df, 
             'table_name': TABLE_NAMES['clean_data']
             }]
        }

"""
normalization related functions; if normalization algorithm is changed, only these functions must change
"""

def normalize(df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(df)
    return scaler

def normalize_column(df, column: str):
    """
    normalize df[column]

    return tuple that can be directly inserted into the normalization table
    """

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return (column, scaler.data_min_[0], scaler.data_max_[0], scaler.scale_[0], scaler.min_[0])

def normalization_transform_column(df, column: str, values: dict):
    # must be equivalent to MinMaxScaler.transform
    df[column] = (df[column] - values['min'])/values['scale']

    return df

def denormalize_columns(df, normalization_values):
    """
    denormalize columns in df; based on the code in MinMaxScaler in sklearn

    column is a column name in df; normalization is the dataframe with normalization values

    must be equivalant to MinMaxScaler.inverse_transform
    """

    for column in df.columns:
        values = normalization_values[normalization_values['name'] == column]
        if values.empty and column != PARAMS['ml']['labels']:
            print('Column {column} not found in the normalization data table.')
        else:
            values = values.iloc[0].to_dict()
            df[column] = df[column] * values['scale'] + values['min']

    return df

"""
end of normalization functions
"""

@from_table_to_df(TABLE_NAMES['clean_data'], None)
def normalize_data_func(**kwargs):
    """
    normalization
    split to train/test
    """
    
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=PARAMS['ml']['train_test_ratio'], random_state=42)

    # Normalize numerical columns
    normalization_values = [] # 
    for column in [v for v in df_train.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]:
        normalization_values.append(normalize_column(df_train, column))
    
    normalization_df = pd.DataFrame(data=normalization_values, columns=NORMALIZATION_TABLE_COLUMN_NAMES)

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['train_data']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['test_data']   
            },
            {
                'df': normalization_df,
                'table_name': TABLE_NAMES['normalization_data']
            }
            ]
        }

@from_table_to_df(TABLE_NAMES['clean_data'], None)
def eda_func(**kwargs):
    """
    print basic statistics
    """

    import pandas as pd

    df = kwargs['dfs']
    
    print(df.describe())

    return { 'dfs': [] }

def add_to_table_normalization_values(normalization_values):
    """
    append to the normalization table new values (engineered features)
    """

    import pandas as pd

    normalization_df = pd.DataFrame(data=normalization_values, columns=NORMALIZATION_TABLE_COLUMN_NAMES)
    conn = create_db_connection()
    normalization_df.to_sql(TABLE_NAMES['normalization_data'], conn, if_exists="append", index=False)
    conn.close()

def _fe_max_func(df=None, normalization=True):
    import pandas as pd

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    new_features_df = pd.DataFrame()
    normalization_values = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            column_name = 'max_'+features[i]+'_'+features[j]
            new_features_df[column_name] = df[[features[i], features[j]]].max(axis=1)
            # normalize the new feature
            if normalization == True:
                normalization_values.append(normalize_column(new_features_df, column_name))

    return normalization_values, new_features_df

@from_table_to_df(TABLE_NAMES['train_data'], None)
def fe_max_func(**kwargs):
    """
    add features that are max of two features 
    """

    import pandas as pd

    df = kwargs['dfs']

    normalization_values, new_features_df = _fe_max_func(df=df)

    # insert the new normalization values to the table
    add_to_table_normalization_values(normalization_values)

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['max_fe']
             }]
        }

def _fe_product_func(df=None, normalization=True):
    import pandas as pd

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    new_features_df = pd.DataFrame()
    normalization_values = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            column_name = f'{features[i]}*{features[j]}'
            new_features_df[column_name] = df[features[i]] * df[features[j]]
            if normalization == True:
                normalization_values.append(normalize_column(new_features_df, column_name))

    return normalization_values, new_features_df

@from_table_to_df(TABLE_NAMES['train_data'], None)
def fe_product_func(**kwargs):
    """
    add features that are products of two features
    """
    
    import pandas as pd

    df = kwargs['dfs']

    normalization_values, new_features_df = _fe_product_func(df=df)

    # insert the new normalization values to the table
    add_to_table_normalization_values(normalization_values)

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['product_fe']
             }]
        }

def train_model(dfs):
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
    string_columns = [v for v in df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    df = df.drop(string_columns, axis=1)

    Y = df[PARAMS['ml']['labels']]
    X = df.drop(PARAMS['ml']['labels'], axis=1)

    # data has been normalized with respect to train/test split; we need to denormalize it and then renormalize it
    conn = create_db_connection()
    normalization_values = pd.read_sql(f"SELECT * FROM {TABLE_NAMES['normalization_data']}", conn)
    conn.close()
    X = denormalize_columns(X, normalization_values)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=PARAMS['ml']['train_test_ratio'], random_state=42)

    scaler = normalize(X_train)
    X_train = scaler.transform(X_train)

    # Create an instance of Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the val set
    X_val = scaler.transform(X_val)
    y_pred = model.predict(X_val)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)        

    return accuracy

@from_table_to_df([TABLE_NAMES['product_fe'], TABLE_NAMES['train_data']], None)
def product_train_func(**kwargs):
    """
    train logistic regression on product features
    """

    dfs = kwargs['dfs']
 
    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data']], None)
def production_train_func(**kwargs):
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

@from_table_to_df([TABLE_NAMES['max_fe'], TABLE_NAMES['train_data']], None)
def max_train_func(**kwargs):
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
    return f'{feature_type}_evaluation'

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

def extensive_evaluation_func(train_df=None, test_df=None, train_fe_df=None, normalization_df=None, fe_func=None, **kwargs):
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

     # combine dataframes
    train_df = pd.concat([train_df, train_fe_df], axis=1)

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

        # compute engineered features
        _, new_features = fe_func(df = X_test, normalization=False)
        X_test = pd.concat([X_test, new_features], axis=1)

        # test data must be scaled
        for column in X_test.columns:
            values = normalization_df[normalization_df['name'] == column]
            if values.empty:
                print('Column {column} not found in the normalization data table.')
            else:
                values = values.iloc[0].to_dict()
                X_test = normalization_transform_column(X_test, column, values)
        
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
    
@from_table_to_df([TABLE_NAMES['train_data'], TABLE_NAMES['test_data'], TABLE_NAMES['max_fe'], TABLE_NAMES["normalization_data"]], TABLE_NAMES['product_fe'])
def max_evaluation_func(**kwargs):
    dfs = kwargs['dfs']

    extensive_evaluation_func(train_df=dfs[0], test_df=dfs[1], train_fe_df=dfs[2], normalization_df=dfs[3], fe_func=_fe_max_func)

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data'], TABLE_NAMES['test_data'], TABLE_NAMES['product_fe'], TABLE_NAMES["normalization_data"]], TABLE_NAMES['max_fe'])
def product_evaluation_func(**kwargs):
    dfs = kwargs['dfs']

    extensive_evaluation_func(train_df=dfs[0], test_df=dfs[1], train_fe_df=dfs[2], normalization_df=dfs[3], fe_func=_fe_product_func)
    
    return {'dfs': []}

# Instantiate the DAG
dag = DAG(
    'Pro-Classify',
    default_args=default_args,
    description='Classify with feature engineering and model selection',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

drop_tables = PostgresOperator(
    task_id="drop_tables",
    postgres_conn_id=PARAMS['db']['db_connection'],
    queue=PARAMS['workflow']['default_queue'],
    sql=f"""
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;
    COMMENT ON SCHEMA public IS 'standard public schema';
    """,
    dag=dag
)

download_data = S3ToLocalOperator(
    task_id="download_data",
    aws_conn_id=PARAMS['files']['s3_connection'],
    bucket_name=PARAMS['files']['s3_bucket'],
    object_name=PARAMS['files']['remote_file'],
    filename=PARAMS['files']['local_file'],
    task_concurrency=PARAMS['workflow']['sequential_queue'],
    dag=dag
)

# download_data = SFTPOperator(
#     task_id="download_data",
#     ssh_hook = SSHHook(ssh_conn_id=PARAMS['files']['sftp_connection']),
#     remote_filepath=PARAMS['files']['remote_file'], 
#     local_filepath=PARAMS['files']['local_file'],
#     operation="get",
#     create_intermediate_dirs=True,
#     queue=PARAMS['workflow']['sequential_queue'],
#     dag=dag
# )

download_data = S3ToLocalOperator(
    task_id="download_data",
    aws_conn_id=PARAMS['files']['s3_connection'],  # Assuming this holds the S3 connection ID
    bucket_name=PARAMS['files']['s3_bucket'],
    object_name=PARAMS['files']['s3_file_key'],  # S3 key (path) to the file
    filename=PARAMS['files']['local_file'],  # Local path where the file will be saved
    task_concurrency=PARAMS['workflow']['sequential_queue'],  # If you meant to control task concurrency
    dag=dag
)

add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    provide_context=True,
    queue=PARAMS['workflow']['sequential_queue'],
    dag=dag
)

clean_data = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

normalize_data = PythonOperator(
    task_id='normalize_data',
    python_callable=normalize_data_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

eda = PythonOperator(
    task_id='EDA',
    python_callable=eda_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

fe_max = PythonOperator(
    task_id='add_max_features',
    python_callable=fe_max_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

fe_product = PythonOperator(
    task_id='add_product_features',
    python_callable=fe_product_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

product_train = PythonOperator(
    task_id='product_train',
    python_callable=product_train_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

max_train = PythonOperator(
    task_id='max_train',
    python_callable=max_train_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

production_train = PythonOperator(
    task_id='production_train',
    python_callable=production_train_func,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag
)

model_selection = BranchPythonOperator(
    task_id='model_selection',
    python_callable=decide_which_model,
    provide_context=True,
    queue=PARAMS['workflow']['default_queue'],
    dag=dag,
)

dummy_task = DummyOperator(
    task_id='do_nothing',
    queue=PARAMS['workflow']['default_queue'],
    dag=dag,
)

evaluation_tasks = []
for feature_type in feature_operations:
    encoding = encode_task_id(feature_type)
    evaluation_tasks.append(PythonOperator(
        task_id=encoding,
        python_callable=locals()[f'{encoding}_func'],
        provide_context=True,
        queue=PARAMS['workflow']['default_queue'],
        dag=dag
    ))

drop_tables >> download_data >> add_data_to_table >> clean_data >> normalize_data
clean_data >> eda
normalize_data >> [fe_max, fe_product]
fe_product >> product_train
fe_max >> max_train
normalize_data >> production_train
[product_train, max_train, production_train] >> model_selection
model_selection >> [dummy_task, *evaluation_tasks]
