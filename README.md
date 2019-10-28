# Pyspark Clustering Sample
This code is showing a simple usage of Kmeans clustering algorithm on top of Spark. 
The Json input file is scanned and then loaded to the Spark application, some data preparation is performed to unify data and select features.
Then the model is trained with **pyspark.ml.clustering.KMeans** algorithm and used to predict data clusters. The data will then be stored with the predicted cluster as json.

### Configuration

Before executing the code, you have to complete the configuration file  **config.ini**.

    [Data]
    input_path=/Users/macbookpro/Downloads/Brisbane_CityBike.json
    output_path=/Users/macbookpro/Downloads/Brisbane_CityBike_output
    [Clustering]
    kmeans_k=2
    kmeans_seed=1
    
* **input_patha** Path to the file Brisbane_CityBike.json
* **output_path** The directory that will be used to store clustered data
* **kmeans_k** number of cluster(s) used by Kmeans clustering algorithm 
* **kmeans_seed** number of seed(s) used by Kmeans clustering algorithm 


### Launching Application with spark-submit

The application can be launched using the bin/spark-submit script. This script takes care of setting up the classpath with Spark and its dependencies, and can support different cluster managers and deploy modes that Spark supports.
The program arguments are the main python script and the config.ini file needed to lead script configurations.

```ruby
./bin/spark-submit \
  --class main \
  --master local[*] \
  --executor-memory 8G \
  --deploy-mode cluster \
  /path/to/brisbane_citybike.py
  -py-files /path/to/config.ini
```