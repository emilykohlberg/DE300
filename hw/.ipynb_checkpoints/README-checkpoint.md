# HW1

## Instruction for deploying:

   - log into EC2
   - run the commands: 'aws sso login --sso-session nu-sso' and 'sudo chmod 666 /var/run/docker.sock' 
   - navigate to folder DE300/hw
   - run the command: docker exec -it etl-container /bin/bash
   - start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'.
   - go to src/heart_disease_eda.ipynb
   - run every cell in order EXCEPT the collapsed section: Unused, but this how I would fill the missing values
   - The explanations are in the file, and the cleaned data is in the staging data folder

# HW2

## Instructions for deploying:

    - log into EC2 (Emily EC2)
    - run the commands 'aws sso login --sso-session nu-sso' and 'sudo chmod 666 /var/run/docker.sock'
    - navigate to the folder DE/300/hw
    - run the command: docker start etl-container
    - run the command: docker exec -it etl-container /bin/bash
    - start the jupyter notebook by running the command: jupyter notebook --ip=0.0.0.0
    - go to src/hw2.ipynb
    - run every cell in order, you can either load the data from s3 by changing the secrets, or from the rawdata folder
    - simple imputation is commented out since I chose to use a more advanced method with the sklearn module
    - my final result is that RandomForestClassifier, with the hyperparameters specified in the jupyter notebook is the best classifier

# HW3
## Instructions for deploying:
    - log into EC2 (Emily EC2)
    - run the commands 'aws sso login --sso-session nu-sso' and 'sudo chmod 666 /var/run/docker.sock'
    - paste aws credentials into the terminal
    - navigate to the folder DE300/hw
    - run 'bash hw3.sh' in the terminal
    - this will create an EMR cluster and a step that runs the spark job
    - The best model is LogisticRegression with metrics - AUC: 0.8853, Accuracy: 0.8118, Precision: 0.8118, Recall: 0.8118, F1-Score: 0.8118 The selected regularization parameter was 0.1.
    - These results can be seen from the EMR job or in hw3.ipynb
## To run the notebook (not necessary):
    - If you want to run hw3.ipynb, run the commands 'docker restart spark-sql-container' and 'docker exec -it spark-sql-container /bin/bash' in the terminal and open 'http://localhost:8888/tree/hw3', find the file, and run every cell in the notebook


# HW4
## Instructions:

    - log into EC2 (Emily EC2)
    - use de300spring2024-airflow-demo MWAA
    - my DAG is Emily_HW4
    - some spark tasks need to rerun because of limited memory, but they have all successfully run, which can be seen on the Emily_HW4 DAG on de300spring2024-airflow-demo
