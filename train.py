#Loading the libraries
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession	
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Starting the spark session

app_name = 'wine_quality_train'

spark = (
        pyspark.sql.SparkSession.builder.appName(app_name)
        .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0")
        .getOrCreate()
    )

#Loading the dataset
df = spark.read.format("csv").load("s3://winequalityaws/train/TrainingDataset.csv" , header = True ,sep =";")
print("Reading completed, printing dataframe")
df.show(truncate=False)

#changing the 'quality' column name to 'label'
for column in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(column, col(column).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "label")


#getting the features and label seperately and converting it to numpy array
features =np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('label').collect())

#creating the feature vector
VectorAssembler = VectorAssembler(inputCols = df.columns[1:-1] , outputCol = 'features')
df_train = VectorAssembler.transform(df)
df_train = df_train.select(['features','label'])

#The following function creates the labeledpoint and parallelize it to convert it into RDD
def to_rdd(spark, features, labels):
    labeled_points = []
    for x, y in zip(features, labels):        
        lp = LabeledPoint(y, x)
        labeled_points.append(lp)
    sparkContext=spark.sparkContext
    return sparkContext.parallelize(labeled_points) 

#rdd converted dataset
dataset = to_rdd(spark, features, label)

#Splitting the dataset into train and test
training, test = dataset.randomSplit([0.7, 0.3],seed =11)


#Creating a random forest training classifier
RFmodel = RandomForest.trainClassifier(training, numClasses=10, categoricalFeaturesInfo={},
                                     numTrees=21, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=30, maxBins=32)

#predictions
predictions = RFmodel.predict(test.map(lambda x: x.features))
#predictionAndLabels = test.map(lambda x: (float(model.predict(x.features)), x.label))

#getting a RDD of label and predictions
labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)

labelsAndPredictions_df = labelsAndPredictions.toDF()
#cpnverting rdd ==> spark dataframe ==> pandas dataframe 
labelpred = labelsAndPredictions.toDF(["label", "Prediction"])
labelpred.show(truncate=False)
labelpred_df = labelpred.toPandas()


#Calculating the F1score
F1score = f1_score(labelpred_df['label'], labelpred_df['Prediction'], average='micro')
print("F1- score: ", F1score)
print(confusion_matrix(labelpred_df['label'],labelpred_df['Prediction']))
print(classification_report(labelpred_df['label'],labelpred_df['Prediction']))
print("Accuracy" , accuracy_score(labelpred_df['label'], labelpred_df['Prediction']))

#calculating the test error
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(test.count())    
print('Test Error = ' + str(testErr))

#save training model
RFmodel.save(spark.sparkContext, 's3://winequalityaws/model/trainingmodel.model')



