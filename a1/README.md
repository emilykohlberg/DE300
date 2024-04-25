Steps to run the code
1. log into EC2
2. run the commands:
   aws sso login --sso-session nu-sso
   sudo chmod 666 /var/run/docker.sock
   docker run -p 8888:8888 -v ~/DE300:/home/jovyan/ my_jupyter
3. go to a1/heart_disease_eda.ipynb
4. run every cell in order EXCEPT the collapsed sections, which are:
        DB Connection
        Unused, but this how I would fill the missing values
5. The explanations are in the file, and the cleaned data ends in a1/cleaned_data.csv