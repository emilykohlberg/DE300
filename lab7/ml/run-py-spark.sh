#!/bin/bash

# Variables
CLUSTER_ID="j-23JEXHTO2U99W"
STEP_NAME="Lab 7"
ARCHIVE_PATH="s3://de300spring2024/emily_kohlberg/lab7/demos.tar.gz#demos"
SCRIPT_PATH="s3://de300spring2024/emily_kohlberg/lab7/classify.py"

# Add the Spark step
aws emr add-steps \
  --cluster-id $CLUSTER_ID \
  --steps Type=Spark,Name="$STEP_NAME",ActionOnFailure=CONTINUE,\
Jar=command-runner.jar,Args=[\
"--deploy-mode",\
"cluster",\
"--master",\
"yarn",\
$SCRIPT_PATH]
