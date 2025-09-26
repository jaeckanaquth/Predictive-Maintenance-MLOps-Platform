# Project: Predictive Maintenance MLOps Platform

**Business goal:** Predict breakdowns for wind turbines using sensor data, and automatically retrain when drift is detected.


***

## Adjusted Plan for Local Execution

- **Data Storage:** Use SQLite to simulate RDS, and a local JSON or SQLite table to mimic DynamoDB.
- **ETL Jobs:** Use Python scripts run via CLI or crontab (or subprocess for "batch" jobs) to simulate Lambda and Batch workflows.
- **Model Training & Inference:** Train with PyTorch/TensorFlow locally. Deploy with FastAPI/Flask for RESTful APIs (can be tested with curl or Swagger UI).
- **Message Queue:** Use Python's `queue` module or Redis to simulate SQS.
- **Observability:** Integrate with Prometheus and Grafana for local dashboards. Log events to local files or use SQLite as log storage. Use NannyML/Evidently for model drift.
- **Pipeline Orchestration:** Simulate Airflow/Kubeflow with Apache Airflow in Docker or even just Python scripts/Makefiles for workflow management.
- **Configuration:** All settings (paths, ports, etc) in local `.env` or config files.


***

## Repository/Directory Structure

```
predictive-maintenance-mlops/
│
├─ data/                     # CSVs/JSONs as incoming sensor data
├─ storage/                  # SQLite DB for features, logs, DynamoDB/RDS simulation
├─ etl/                      # Python scripts for ingestion, transformation (simulated Lambda/Batch)
├─ model/                    # PyTorch/TensorFlow scripts, saved models
├─ api/                      # FastAPI/Flask prediction and admin endpoints
├─ monitoring/               # Evidently/NannyML notebooks/scripts, Prometheus config
├─ orchestration/            # Airflow DAGs or Python workflow scripts
├─ notebooks/                # Jupyter for EDA, interactive drift checks
├─ README.md                 # Run instructions, requirements

```

***

## What This Covers

- End-to-end model lifecycle: ingest, process, train, deploy, monitor, alert, retrain.[3][2][1]
- REST APIs for prediction and observability.
- Full-stack AWS/MLOps practices: Lambda, Batch+Fargate, RDS, DynamoDB, SQS, SageMaker, API Gateway, CloudWatch, Grafana.
- Model drift analysis and automation for production ML systems.
- Advanced monitoring (Grafana/CloudWatch) and open-source observability (Evidently/NannyML).
- Orchestrated retraining using Airflow/MLflow/Kubeflow.

This project design is modern, practical, and directly aligned with industry-standard approaches and your JD requirements.[2][3][1]

[1](https://github.com/aws-samples/amazon-sagemaker-drift-detection)
[2](https://aws.amazon.com/blogs/machine-learning/build-an-end-to-end-mlops-pipeline-using-amazon-sagemaker-pipelines-github-and-github-actions/)
[3](https://github.com/giuseppeporcelli/end-to-end-ml-sm)
[4](https://github.com/aws-samples/sagemaker-end-to-end-workshop)
[5](https://github.com/aws-samples/sagemaker-ml-workflow-with-apache-airflow)
[6](https://github.com/aws/amazon-sagemaker-examples)
[7](https://github.com/CharlieSergeant/sagemaker-fastapi)
[8](https://github.com/ManyamSanjayKumarReddy/End-To-End-Machine-Learning-Project-Implementation-Using-AWS-Sagemaker)