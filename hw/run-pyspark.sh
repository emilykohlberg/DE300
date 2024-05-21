echo "Starting Pyspark container"
docker run -v ~/DE300/hw:/tmp/hw3 -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image

