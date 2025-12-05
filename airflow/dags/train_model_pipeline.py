from datetime import datetime, timedelta
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from refactor.train_test_model import main

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_training():
    return main()

@dag(
    dag_id='train_model_pipeline',
    description='Pipeline for training football match prediction model',
    start_date=datetime(2025, 12, 1),
    schedule=timedelta(days=7),
    default_args=default_args,
    catchup=False
)

def create_dag():
    """Create the DAG with tasks."""
    start = EmptyOperator(task_id='start')
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=run_training
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> train_task >> end
    
dag = create_dag()