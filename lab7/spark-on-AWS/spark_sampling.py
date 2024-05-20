from pyspark.sql import SparkSession

print("Hello!")

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("randomSample") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("Reading data")
    textFile = spark.read.csv("s3://de300spring2024/emily_kohlberg/lab7/data/data.csv", header = True)
    print("Read data!")
    samples = textFile.sample(.01, False, 42)
    print("Took sample!")
    samples.write.csv("s3://de300spring2024/emily_kohlberg/lab7/output/")
    print("Wrote all partitions to a directory")
    df = samples.toPandas()
    print("Made pandas")
    df.to_csv("s3://de300spring2024/emily_kohlberg/lab7/output.csv", index = False)
    print("Wrote file")

    spark.sparkContext.stop()
