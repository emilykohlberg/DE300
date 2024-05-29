from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago

import random

WORKFLOW_SCHEDULE_INTERVAL = "@hourly"

# Define the default args dictionary
default_args = {
    'owner': 'johndoe',
    'depends_on_past': False,
    'start_date': days_ago(1), # should be a historical date
    'retries': 1,
}

# Define some simple Python functions to use as tasks
def task1_func(**kwargs):
    print('Running task 1')
    # Returns one string with 50% probability and a different string otherwise
    value = 'success' if random.random() < 0.5 else 'failure'
    print(f'Task 1 output: {value}')

    return {'status': value}

def task2_func(**kwargs):
    print('Running task 2')
    # Returns a random integer between 0 and 10
    value = random.randint(0, 10)
    print(f'Task 2 output: {value}')

    return {'value': value}

def task3_func(**kwargs):
    print('Running task 3')
    # Returns two random integers between 0 and 10
    value_one = random.randint(0, 10)
    value_two = random.randint(0, 10)
    print(f'Task 3 output: {value_one} {value_two}')

    return {'value1': value_one, 'value2': value_two}

def task4_func(**kwargs):
    print('Running task 4')
    
    ti = kwargs['ti']
    task1_return_value = ti.xcom_pull(task_ids='task1')
    task2_return_value = ti.xcom_pull(task_ids='task2')
    task3_return_value = ti.xcom_pull(task_ids='task3')
    
    print("Task 1 returned: ", task1_return_value)
    print("Task 2 returned: ", task2_return_value)
    print("Task 3 returned: ", task3_return_value)

    return_value = ""
    if task3_return_value['value1'] > task3_return_value['value2'] and task1_return_value['status'] == "success":
        return_value = "do-task5"
    
    return return_value

def decide_which_path(**kwargs):
    ti = kwargs['ti']
    task4_return_value = ti.xcom_pull(task_ids='task4')
    if task4_return_value == "do-task5":
        return 'task5'
    else:
        return 'dummy_task'

def task5_func(**kwargs):
    print('Running task 5')
    return {'task': 'task5', 'status': 'completed'}

# Instantiate the DAG
dag = DAG(
    'example_dag',
    default_args=default_args,
    description='An example DAG with dependencies',
    schedule_interval=WORKFLOW_SCHEDULE_INTERVAL, # it can be @daily, ...\
    tags=["de300"]
)

# Define the tasks
task1 = PythonOperator(
    task_id='task1',
    python_callable=task1_func,
    provide_context=True,
    dag=dag,
)

task2 = PythonOperator(
    task_id='task2',
    python_callable=task2_func,
    provide_context=True,
    dag=dag,
)

task3 = PythonOperator(
    task_id='task3',
    python_callable=task3_func,
    provide_context=True,
    dag=dag,
)

task4 = PythonOperator(
    task_id='task4',
    python_callable=task4_func,
    provide_context=True,
    dag=dag,
)

decide = BranchPythonOperator(
    task_id='branch_task',
    python_callable=decide_which_path,
    provide_context=True,
    dag=dag,
)

task5 = PythonOperator(
    task_id='task5',
    python_callable=task5_func,
    provide_context=True,
    dag=dag,
)

dummy_task = DummyOperator(
    task_id='dummy_task',
    dag=dag,
)

# Define the dependencies
[task1, task2, task3] >> task4 >> decide
decide >> task5
decide >> dummy_task

