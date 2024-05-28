from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType
from pyspark.sql.dataframe import DataFrame
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, Evaluator
import pyspark.sql.functions as F
from pyspark.sql.functions import when, col
from itertools import combinations
import os

DATA_FOLDER = "../data"

NUMBER_OF_FOLDS = 5
SPLIT_SEED = 7576
TRAIN_TEST_SPLIT = 0.9

def read_data(spark: SparkSession) -> DataFrame:
    """
    read data; since the data has the header we let spark guess the schema
    """
    
    data = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("s3a://de300spring2024/emily_kohlberg/hw/heart_disease.csv")

    return data

def retain_cols(data: DataFrame) -> DataFrame:
    columns_to_retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 
                         'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 
                         'exang', 'oldpeak', 'slope', 'target']
    
    filtered_data = data.select(columns_to_retain)
    return filtered_data
    
def replace_out_of_range(data: DataFrame) -> DataFrame:
    data = data.withColumn('painloc', when(col('painloc') < 0, 0).when(col('painloc') > 1, 1).otherwise(col('painloc')))
    data = data.withColumn('painexer', when(col('painexer') < 0, 0).when(col('painexer') > 1, 1).otherwise(col('painexer')))
    data = data.withColumn('trestbps', when(col('trestbps') < 100, 100).otherwise(col('trestbps')))
    data = data.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))
    data = data.withColumn('fbs', when(col('fbs') < 0, 0).when(col('fbs') > 1, 1).otherwise(col('fbs')))
    data = data.withColumn('prop', when(col('prop') < 0, 0).when(col('prop') > 1, 1).otherwise(col('prop')))
    data = data.withColumn('nitr', when(col('nitr') < 0, 0).when(col('nitr') > 1, 1).otherwise(col('nitr')))
    data = data.withColumn('pro', when(col('pro') < 0, 0).when(col('pro') > 1, 1).otherwise(col('pro')))
    data = data.withColumn('diuretic', when(col('diuretic') < 0, 0).when(col('diuretic') > 1, 1).otherwise(col('diuretic')))
    data = data.withColumn('exang', when(col('exang') < 0, 0).when(col('exang') > 1, 1).otherwise(col('exang')))
    data = data.withColumn('slope', when(col('slope') < 1, None).when(col('slope') > 3, None).otherwise(col('slope')))
    return data

def clean_data(data: DataFrame) -> DataFrame:
    # clean
    data = retain_cols(data)
    data = replace_out_of_range(data)

    # drop null targets
    data = data.dropna(subset=["target"])

    # make age an int
    data = data.withColumn("age", data["age"].cast(IntegerType()))
    return data

# Custom CompositeEvaluator
class CompositeEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.auc_evaluator = BinaryClassificationEvaluator(labelCol="target", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        self.accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
        self.precision_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedPrecision")
        self.recall_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="weightedRecall")
        self.f1_evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="f1")

        self.weights = {
            "AUC": 0.6,
            "accuracy": 0.1,
            "precision": 0.1,
            "recall": 0.1,
            "f1": 0.1
        }

    def isLargerBetter(self):
        return True

    def _evaluate(self, dataset):
        auc = self.auc_evaluator.evaluate(dataset)
        accuracy = self.accuracy_evaluator.evaluate(dataset)
        precision = self.precision_evaluator.evaluate(dataset)
        recall = self.recall_evaluator.evaluate(dataset)
        f1 = self.f1_evaluator.evaluate(dataset)

        composite_score = (self.weights["AUC"] * auc +
                           self.weights["accuracy"] * accuracy +
                           self.weights["precision"] * precision +
                           self.weights["recall"] * recall +
                           self.weights["f1"] * f1)

        return composite_score

def pipeline(data: DataFrame):

    numeric_features = [f.name for f in data.schema.fields if isinstance(f.dataType, (DoubleType, FloatType, IntegerType, LongType))]
    numeric_features.remove("target")

    # numeric columns
    imputed_columns = [f"Imputed{v}" for v in numeric_features]
    imputer_numeric = Imputer(inputCols=numeric_features, outputCols=imputed_columns, strategy = "mean")
    
    # Assemble feature columns into a single feature vector
    assembler = VectorAssembler(
        inputCols=imputed_columns, 
        outputCol="features"
        )

    # Define binary classification models
    classifiers = {
        "RandomForest": RandomForestClassifier(labelCol="target", featuresCol="features"),
        "LogisticRegression": LogisticRegression(labelCol="target", featuresCol="features"),
        "GBTClassifier": GBTClassifier(labelCol="target", featuresCol="features"),
        "DecisionTree": DecisionTreeClassifier(labelCol="target", featuresCol="features")

    }

    # Define parameter grids for each classifier
    param_grids = {
        "RandomForest": ParamGridBuilder() \
            .addGrid(classifiers["RandomForest"].maxDepth, [4, 6, 8]) \
            .addGrid(classifiers["RandomForest"].numTrees, [50, 100]) \
            .build(),
        "LogisticRegression": ParamGridBuilder() \
            .addGrid(classifiers["LogisticRegression"].regParam, [0.01, 0.1]) \
            .build(),
        "GBTClassifier": ParamGridBuilder() \
            .addGrid(classifiers["GBTClassifier"].maxDepth, [2, 4]) \
            .addGrid(classifiers["GBTClassifier"].maxIter, [10, 20]) \
            .build(),
        "DecisionTree": ParamGridBuilder() \
            .addGrid(classifiers["DecisionTree"].maxDepth, [4, 6, 8]) \
            .addGrid(classifiers["DecisionTree"].minInstancesPerNode, [1, 2, 4]) \
            .build()
    }
    
    # Set up the composite evaluator
    composite_evaluator = CompositeEvaluator()

    # Split the data into training and test sets
    train_data, test_data = data.randomSplit([TRAIN_TEST_SPLIT, 1-TRAIN_TEST_SPLIT], seed=SPLIT_SEED)

    # Cache the training data to improve performance
    train_data.cache()

    best_model = None
    best_model_name = ""
    best_composite_score = float('-inf')
    
    # Iterate through each classifier
    for model_name, classifier in classifiers.items():
        # Create the pipeline
        pipeline = Pipeline(stages=[imputer_numeric, assembler, classifier])  

        # Set up the cross-validator
        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grids[model_name],
            evaluator=composite_evaluator,
            numFolds=NUMBER_OF_FOLDS,
            seed=SPLIT_SEED)

        # Train the cross-validated pipeline model
        cvModel = crossval.fit(train_data)

        # Make predictions on the test data
        predictions = cvModel.transform(test_data)

        # Evaluate the model using the composite score
        composite_score = composite_evaluator.evaluate(predictions)
        print(f"{model_name} Composite Score: {composite_score:.4f}")

        metrics = {
                "composite_score": composite_score,
                "AUC": composite_evaluator.auc_evaluator.evaluate(predictions),
                "accuracy": composite_evaluator.accuracy_evaluator.evaluate(predictions),
                "precision": composite_evaluator.precision_evaluator.evaluate(predictions),
                "recall": composite_evaluator.recall_evaluator.evaluate(predictions),
                "f1": composite_evaluator.f1_evaluator.evaluate(predictions)
            }
        print(f"Metrics - AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}")


        # Update the best model if current model is better
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_model = cvModel.bestModel.stages[-1]
            best_model_name = model_name
            best_metrics = {
                "composite_score": composite_score,
                "AUC": composite_evaluator.auc_evaluator.evaluate(predictions),
                "accuracy": composite_evaluator.accuracy_evaluator.evaluate(predictions),
                "precision": composite_evaluator.precision_evaluator.evaluate(predictions),
                "recall": composite_evaluator.recall_evaluator.evaluate(predictions),
                "f1": composite_evaluator.f1_evaluator.evaluate(predictions)
            }
            
    # Print the best model information
    print(f"Best Model: {best_model_name}")
    print(f"Best Model Composite Score: {best_composite_score:.4f}")
    print(f"Best Model Metrics - AUC: {best_metrics['AUC']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1-Score: {best_metrics['f1']:.4f}")


    # Retrieve and print the best model parameters
    if best_model_name == "RandomForest":
        selected_max_depth = best_model.getOrDefault(best_model.getParam("maxDepth"))
        selected_num_trees = best_model.getOrDefault(best_model.getParam("numTrees"))
        print(f"Selected Maximum Tree Depth: {selected_max_depth}")
        print(f"Selected Number of Trees: {selected_num_trees}")
    elif best_model_name == "LogisticRegression":
        selected_reg_param = best_model.getOrDefault(best_model.getParam("regParam"))
        print(f"Selected Regularization Parameter: {selected_reg_param}")
    elif best_model_name == "GBTClassifier":
        selected_max_depth = best_model.getOrDefault(best_model.getParam("maxDepth"))
        selected_max_iter = best_model.getOrDefault(best_model.getParam("maxIter"))
        print(f"Selected Maximum Tree Depth: {selected_max_depth}")
        print(f"Selected Maximum Iterations: {selected_max_iter}")
    elif best_model_name == "DecisionTree":
        selected_max_depth = best_model.getOrDefault(best_model.getParam("maxDepth"))
        selected_min_instances_per_node = best_model.getOrDefault(best_model.getParam("minInstancesPerNode"))
        print(f"Selected Maximum Tree Depth: {selected_max_depth}")
        print(f"Selected Minimum Instances Per Node: {selected_min_instances_per_node}")


def main():
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Predict Heart Disease") \
        .getOrCreate()


    data = read_data(spark)
    data = clean_data(data)
   
    pipeline(data)

    spark.stop()

main()