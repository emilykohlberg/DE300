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

