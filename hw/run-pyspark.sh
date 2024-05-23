echo "Starting Pyspark container"
docker run -v ~/DE300/hw/src:/tmp/hw3 \
	   -v ~/DE300/hw/data:/tmp/data -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image

