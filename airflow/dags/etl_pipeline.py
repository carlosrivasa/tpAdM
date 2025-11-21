from refactor.etl import main

from datetime import datetime, timedelta
from airflow.decorators import dag
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_etl():
    return main()

@dag(
    dag_id='etl_pipeline',
    description='Pipeline de etl con MLflow',
    start_date=datetime(2025, 11, 17),
    schedule=timedelta(days=7),
    default_args=default_args,
    catchup=False
)
def create_dag():
    """Create the DAG with tasks."""
    start = EmptyOperator(task_id='start')
    
    # Una task de etl, o podríamos como next step, divider etl, varias tasks como
    # split, preprocess, encode, standardize 
    etl_task = PythonOperator(
        task_id='etl',
        python_callable=run_etl
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> etl_task >> end
    
dag = create_dag()    