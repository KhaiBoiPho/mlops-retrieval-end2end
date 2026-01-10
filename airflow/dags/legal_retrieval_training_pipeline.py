import os
import sys
import pendulum
from pathlib import Path
from datetime import timedelta
import subprocess

from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Go up 3 levels

# Add to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.configuration import ConfigurationManager  # noqa: E402
from src.train.bi_encoder.trainer import BiEncoderTrainer  # noqa: E402
from src.eval.bi_encoder_evaluator import BiEncoderEvaluator  # noqa: E402
from src.common.notify.telegram import send_telegram_message  # noqa: E402
from src.features.hard_negative_mining import HardNegativeMiner  # noqa: E402
from src.train.cross_encoder.trainer import CrossEncoderTrainer  # noqa: E402
from src.eval.cross_encoder_evaluator import CrossEncoderEvaluator  # noqa: E402
from src.features.build_corpus_embeddings import CorpusEmbeddingBuilder  # noqa: E402
from src.common.model_gates.retrieval_candidate_gate import RetrievalModelGate  # noqa: E402


def run_preprocess(**context):
    """Run DVC data preprocessing pipeline"""
    os.chdir(PROJECT_ROOT)
    
    # Run all DVC preprocessing stages
    stages = ['preprocess', 'tokenize', 'split_data', 'validate']
    
    for stage in stages:
        print(f"Running DVC stage: {stage}")
        result = subprocess.run(
            ['dvc', 'repro', stage],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error in stage {stage}:")
            print(result.stderr)
            raise Exception(f"DVC stage {stage} failed")
        
        print(result.stdout)

def run_train_bi_encoder(**context):
    config_manager = ConfigurationManager()
    config = config_manager.get_bi_encoder_training_config()
    trainer = BiEncoderTrainer(config=config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()

def run_eval_bi_encoder(**context):
    config_manager = ConfigurationManager()
    config = config_manager.get_bi_encoder_eval_config()
    evaluator = BiEncoderEvaluator(config=config)
    evaluator.run()

def run_build_corpus_embeddings(**context):
    config_manager = ConfigurationManager()
    config = config_manager.get_corpus_embeddings_config()
    builder = CorpusEmbeddingBuilder(config=config)
    builder.build()

def run_mine_hard_negatives(**context):
    config_manager = ConfigurationManager()
    splits = ['train', 'val', 'test']
    for split in splits:
        print(f"Mining hard negatives for {split}...")
        config = config_manager.get_hard_negative_mining_config(split=split)
        miner = HardNegativeMiner(config)
        miner.run()

def run_train_cross_encoder(**context):
    config_manager = ConfigurationManager()
    config = config_manager.get_cross_encoder_training_config()
    trainer = CrossEncoderTrainer(config=config)
    trainer.train()

def run_eval_cross_encoder(**context):
    config_manager = ConfigurationManager()
    config = config_manager.get_cross_encoder_eval_config()
    evaluator = CrossEncoderEvaluator(config)
    evaluator.run()

def notify_model_candidate(**context):
    gate = RetrievalModelGate(
        bi_metrics_path="artifacts/bi-encoder/eval_results/metrics.json",
        cross_metrics_path="artifacts/cross-encoder/eval_results/metrics.json",
    )

    passed, details = gate.evaluate()

    if passed:
        status = "🟢 PASSED"
        header = "🚀 *RETRIEVAL MODEL CANDIDATE*"
    else:
        status = "🔴 FAILED"
        header = "⚠️ *RETRIEVAL MODEL FAILED GATE*"

    message = f"""
{header}

🧠 *Bi-Encoder*
📊 MRR: `{details['bi_mrr']:.4f}`
🎯 Recall@10: `{details['bi_recall@10']:.4f}`
🚦 Gate: `{"PASS ✅" if details['bi_encoder_pass'] else "FAIL ❌"}`

⚖️ *Cross-Encoder*
📈 AUC: `{details['cross_auc']:.4f}`
🎯 Recall: `{details['cross_recall']:.4f}`
🧮 True Positives: `{details['cross_tp']}`
🚦 Gate: `{"PASS ✅" if details['cross_encoder_pass'] else "FAIL ❌"}`

📌 *Overall Status*: **{status}**
"""

    send_telegram_message(
        token=Variable.get("TELEGRAM_BOT_TOKEN"),
        chat_id=Variable.get("TELEGRAM_CHAT_ID"),
        text=message,
    )


# DAG definition
default_args = {
    'owner': 'khai_mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='legal_retrieval_training_pipeline',
    default_args=default_args,
    description='End-to-end Legal Retrieval MLOps Pipeline',
    schedule=None,
    start_date=pendulum.datetime(2023, 1, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    tags=['legal-retrieval', 'training'],
) as dag:
    
    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # 1. Preprocessing (DVC pipeline)
    preprocess_task = PythonOperator(
        task_id='preprocess_and_upload',
        python_callable=run_preprocess
    )

    # 2. Bi-encoder Workflow
    train_bi_task = PythonOperator(
        task_id="train_bi_encoder",
        python_callable=run_train_bi_encoder
    )

    eval_bi_task = PythonOperator(
        task_id="eval_bi_encoder",
        python_callable=run_eval_bi_encoder
    )

    build_emb_task = PythonOperator(
        task_id="build_corpus_embeddings",
        python_callable=run_build_corpus_embeddings
    )

    mine_neg_task = PythonOperator(
        task_id="mine_hard_negatives",
        python_callable=run_mine_hard_negatives
    )

    # 3. Cross-Encoder Workflow
    train_cross_task = PythonOperator(
        task_id='train_cross_encoder',
        python_callable=run_train_cross_encoder
    )

    eval_cross_task = PythonOperator(
        task_id='eval_cross_encoder',
        python_callable=run_eval_cross_encoder
    )

    # 4. Notification Developer
    notify_task = PythonOperator(
        task_id="notify_telegram",
        python_callable=notify_model_candidate
    )

    # Define dependencies
    start >> preprocess_task

    # Bi-Encoder chain
    preprocess_task >> train_bi_task >> build_emb_task >> eval_bi_task

    # Connect to Cross-Encoder (need embeddings to mine negative)
    eval_bi_task >> mine_neg_task

    # Cross-Encoder chain
    mine_neg_task >> train_cross_task >> eval_cross_task >> notify_task >> end