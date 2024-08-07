import sys
import os
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from random import randrange
import urllib
from urllib.parse import urlencode

import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Workspace, AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.sweep import (
    Choice,
    Uniform
)

workspace_name="azureml-workspace"
cluster_name="cpu-cluster"

# NOTE:  for local runs, I'm using the Azure CLI credential
# For production runs as part of an MLOps configuration using
# Azure DevOps or GitHub Actions, I recommend using the DefaultAzureCredential

ml_client=MLClient.from_config(DefaultAzureCredential())
# ml_client=MLClient.from_config(AzureCliCredential())
ws=ml_client.workspaces.get(workspace_name)

# Make sure the compute cluster exists already
try:
    cpu_cluster=ml_client.compute.get(cluster_name)
    print(
        f"You already have a cluster named {cluster_name}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    cpu_cluster=AmlCompute(
        name=cluster_name,
        type="amlcompute",
        # size="STANDARD_DS3_V2",
        size="Standard_D4s_v3",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=180,
        tier="LowPriority",
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    cpu_cluster=ml_client.compute.begin_create_or_update(cpu_cluster)

parent_dir="./config"
# Perform data preparation
feature_engineering=load_component(source=os.path.join(parent_dir, "feature_engineering.yml"))
train_model=load_component(source=os.path.join(parent_dir, "train_model.yml"))
register_model=load_component(source=os.path.join(parent_dir, "register_model.yml"))

@pipeline(name="training_pipeline", description="Build a training pipeline")
def build_pipeline(raw_data):
    step_feature_engineering=feature_engineering(input_data=raw_data)

    train_model_data=train_model(train_data=step_feature_engineering.outputs.output_data_train,
                                   test_data=step_feature_engineering.outputs.output_data_test,
                                   max_leaf_nodes=128,
                                   min_samples_leaf=32,
                                   max_depth=12,
                                   learning_rate=0.1,
                                   n_estimators=100)
    register_model(model=train_model_data.outputs.model_output, test_report=train_model_data.outputs.test_report)
    return { "model": train_model_data.outputs.model_output,
             "report": train_model_data.outputs.test_report }

def prepare_pipeline_job(cluster_name):
    cpt_asset = ml_client.data.get("ChicagoParkingTickets", version="1")
    raw_data=Input(type='uri_file', path=cpt_asset.path)
    pipeline_job=build_pipeline(raw_data)
    pipeline_job.settings.default_compute=cluster_name
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=True
    pipeline_job.display_name="train_pipeline"
    return pipeline_job

prepped_job=prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job, experiment_name="Chicago Parking Tickets")

print("Now look in the Azure ML Jobs UI to see the status of the pipeline job.  This will be in the 'Chicago Parking Tickets' experiment.")