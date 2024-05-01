# Lab5


The purpose of this lab is to show you the proper way to organize files in a homework/lab folder. Your are recommended to follow this format for your future homework/lab submission.


## Files in this folder

   - data -- folder that contains the raw data. When using containers, it's usually ideal to keep the raw data intact. For this reason, use the 'COPY ...' in the dockerfile (for example, see ./dockerfiles/dockerfile-jupyter in this folder), which makes the raw data static and will not be modified. In case the raw data keep coming into this data folder, you may need to synchronize the data to your container. For such scenario, you will need to mount the data folder to your container by using the '-v' argument when 'docker run'.

   - dockerfiles -- Put all dockerfiles for the project in this folder. In this folder, 'dockerfile-postgresql' is aimed at creating the image that solely for the postgresql database server. 'dockerfile-jupyter' is used for creating a python image with required package that relates to the project.
   
   - run.sh -- the bash file contains all comands that (1) create the docker volume(this is a space for keeping database existed even you delete the database container) (2) create docker network that connects the database container and your analysis container (3)build the image for the database container (4) build the image for the analysis container (5)run the database container (5)run the analysis container
   
   - src -- the folder contains source files and staging data. (Check the dockerfile ./dockerfiles/dockerfile-jupyter, you can see this folder is synchronized with path /app/src in the container)
   

## Instruction for deploying: (please write this section for you future homework/lab submission on behalf of the grading)

   - Run the run.sh file to create initiate all required containers(etl-container and postgres-container) and enter the shell of etl-container. 
   - Start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'. 
   - Run all cells in the jupyter notebook etl.ipynb
