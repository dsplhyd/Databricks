# Install required libraries (run in Databricks notebook)
%pip install sagemaker-feature-store-pyspark boto3 sagemaker mlflow

# Import necessary libraries
import os
import time
import boto3
from datetime import datetime, timezone
import pandas as pd
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, to_timestamp
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sagemaker import Session
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureDefinition, FeatureTypeEnum
from feature_store_pyspark.FeatureStoreManager import FeatureStoreManager

# Set up Spark session
spark = SparkSession.builder.appName("ML with PySpark, SageMaker FS, MLflow").getOrCreate()

# Set up AWS credentials (assume configured in Databricks or set via secrets)
# os.environ['AWS_ACCESS_KEY_ID'] = dbutils.secrets.get(scope="aws", key="access_key")
# os.environ['AWS_SECRET_ACCESS_KEY'] = dbutils.secrets.get(scope="aws", key="secret_key")
# os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

# SageMaker session
boto_session = boto3.Session()
sagemaker_session = Session(boto_session=boto_session)

# Define variables
bucket = 'your-s3-bucket'  # Replace with your S3 bucket
role_arn = 'arn:aws:iam::your-account:role/your-sagemaker-role'  # Replace with your IAM role
feature_group_name = 'demo-feature-group'

# Create sample data with PySpark (Iris-like binary classification for simplicity)
data = [
    (0, 5.1, 3.5, 1.4, 0.2, 0),
    (1, 4.9, 3.0, 1.4, 0.2, 0),
    (2, 7.0, 3.2, 4.7, 1.4, 1),
    (3, 6.4, 3.2, 4.5, 1.5, 1),
    # Add more rows as needed
]
columns = ["id", "sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
df = spark.createDataFrame(data, columns)

# Add required columns for Feature Store
current_time = int(round(time.time()))
df = df.withColumn("event_time", lit(current_time).cast("double"))
df = df.withColumn("record_id", df.id.cast("string"))

# Define feature definitions
feature_definitions = [
    FeatureDefinition(feature_name='record_id', feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name='sepal_length', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='sepal_width', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='petal_length', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='petal_width', feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name='label', feature_type=FeatureTypeEnum.INTEGRAL),
    FeatureDefinition(feature_name='event_time', feature_type=FeatureTypeEnum.FRACTIONAL),
]

# Create Feature Group if not exists
feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
try:
    feature_group.describe()
    print(f"Feature group {feature_group_name} already exists.")
except:
    feature_group.create(
        s3_uri=f"s3://{bucket}/offline-store",
        record_identifier_name='record_id',
        event_time_feature_name='event_time',
        role_arn=role_arn,
        enable_online_store=True,
        feature_definitions=feature_definitions
    )
    # Wait for creation
    time.sleep(60)

feature_group_arn = feature_group.arn

# Ingest data to SageMaker Feature Store
feature_store_manager = FeatureStoreManager()
feature_store_manager.ingest_data(input_data_frame=df, feature_group_arn=feature_group_arn, direct_offline_store=True)

# Wait for data to be available in offline store (may take a few minutes)
time.sleep(300)  # Adjust based on data size

# Retrieve features from offline store for training
offline_store_s3_uri = feature_group.describe()['OfflineStoreConfig']['S3StorageConfig']['S3Uri']
features_df = spark.read.parquet(f"{offline_store_s3_uri}/*/*/*/*.parquet")  # Read all partitions

# Prepare data for ML
feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_df = assembler.transform(features_df)

train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Enable MLflow autologging
mlflow.pyspark.ml.autolog()

# Train, evaluate, and tune model with MLflow
with mlflow.start_run(run_name="LogisticRegression_Tuning"):
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    
    paramGrid = (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.1, 0.01])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5])
                 .build())
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)
    
    cvModel = cv.fit(train_df)
    
    # Evaluate on test data
    predictions = cvModel.transform(test_df)
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc}")
    
    # Log best model manually if needed
    mlflow.spark.log_model(cvModel.bestModel, "model")

print("Script completed: Model created, evaluated, tuned, and logged with MLflow.")