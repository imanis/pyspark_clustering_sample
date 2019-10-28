from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import coalesce
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
import ConfigParser


# Spark session
spark = SparkSession.builder.appName('Clustering for Brisbane_CityBike').getOrCreate()

# get config 
config = ConfigParser.RawConfigParser()
config.read('config.ini')
INPUT_PATH = config.get('Data', 'input_path')
OUTPUT_PATH = config.get('Data', 'output_path')
KMEANS_K = int(config.get('Clustering', 'kmeans_k'))
KMEANS_SEED = int(config.get('Clustering', 'kmeans_seed'))


# Loads data
dataset = spark.read.option("multiline","true").json( INPUT_PATH)

# Data preparation
stations = dataset.withColumn(
					"latitude", coalesce(
								dataset["coordinates"]["latitude"],
								dataset["latitude"].cast(DoubleType()))
					).withColumn(
					"longitude", coalesce(
								dataset["coordinates"]["longitude"],
								dataset["longitude"].cast(DoubleType()))
					).drop("coordinates")
# Debug
print stations.show()
stations.printSchema()


# Selecting feature 
assembler = VectorAssembler(
    inputCols=["latitude", "longitude"],
    outputCol="features").setHandleInvalid("skip")


# Initializing the KMeans clustering model
kmeans = KMeans().setK(KMEANS_K).setSeed(KMEANS_SEED).setFeaturesCol("features").setPredictionCol("cluster")

# Building the Pipeline for clustering
# Step 1 : assembling features
# Step 2 : Running the clustering on data
pipeline = Pipeline(stages=[assembler, kmeans])

# Running the clustering pipeline
model = pipeline.fit(stations)

# Make predictions
clusters = model.transform(stations)


# Saving the dataset with cluster
clusters.drop("features").repartition(1).write.format("json").mode('overwrite').save(OUTPUT_PATH)

# Debug
print clusters.show()
clusters.printSchema()


