from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F
import findspark
findspark.init()

import warnings
warnings.filterwarnings('ignore')


# Khởi tạo SparkSession
spark = SparkSession.builder\
    .master('local[1]')\
    .appName('Setdata.com')\
    .getOrCreate()

schema = StructType([
    StructField("Name", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Gender", StringType(), True),
    StructField("Blood Type", StringType(), True),
    StructField("Medical Condition", StringType(), True),
    StructField("Date of Admission", StringType(), True),
    StructField("Doctor", StringType(), True),
    StructField("Hospital", StringType(), True),
    StructField("Insurance Provider", StringType(), True),
    StructField("Billing Amount", IntegerType(), True),
    StructField("Room Number", IntegerType(), True),
    StructField("Admission Type", StringType(), True),
    StructField("Discharge Date", StringType(), True),
    StructField("Medication", StringType(), True),
    StructField("Test Results", StringType(), True)

])

df = spark.read.csv("dataset.csv", header=True, schema=schema)

# Xử lý dữ liệu category sang dạng số
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) 
            for column in ["Gender", "Blood Type", "Medical Condition"]]
for indexer in indexers:
    df = indexer.transform(df)

# Tạo vectơ features
feature_cols = ["Age", "Gender_index", "Blood Type_index", "Medical Condition_index"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=123)

# Huấn luyện mô hình Logistic Regression
lr = LogisticRegression(labelCol="Medical Condition_index", featuresCol="features", maxIter=10)
lr_model = lr.fit(train_data)
# Dự đoán trên tập kiểm tra
predictions = lr_model.transform(test_data)
# Lưu mô hình
lr_model.write().overwrite().save("logistic_regression_model")