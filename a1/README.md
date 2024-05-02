# HW1

## Instruction for deploying:

   - log into EC2
   - run the commands: 
	- aws sso login --sso-session nu-sso 
	- sudo chmod 666 /var/run/docker.sock 
   - navigate to folder DE300/a1
   - run the command: docker exec -it etl-container /bin/bash
   - start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'.
   - go to src/heart_disease_eda.ipynb
   - run every cell in order EXCEPT the collapsed section: Unused, but this how I would fill the missing values
   - The explanations are in the file, and the cleaned data is in the staging data folder
