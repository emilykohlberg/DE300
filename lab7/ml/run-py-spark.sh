#!/bin/bash

# Variables
CLUSTER_ID="j-30VPB97D4OCD6"
STEP_NAME="Lab 7"
ARCHIVE_PATH="s3://de300spring2024/emily_kohlberg/lab7/demos.tar.gz#demos"
SCRIPT_PATH="s3://de300spring2024/emily_kohlberg/lab7/classify.py"

# Add the Spark step
aws emr add-steps \
  --cluster-id $CLUSTER_ID \
  --steps Type=Spark,Name="$STEP_NAME",ActionOnFailure=CONTINUE,\
Jar=command-runner.jar,Args=[\
"spark-submit",\
"--archives",$ARCHIVE_PATH,\
$SCRIPT_PATH]
