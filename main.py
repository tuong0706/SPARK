from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import functions as F
import findspark
findspark.init()

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

age = int(input("Nhap tuoi: "))
gender = input("Gioi tinh (Male/Female): ").capitalize()
blood_type = input("Nhom mau: ").upper()
medical_condition = input("Tinh trang benh: ")

def suggest_disease():
    df = spark.read.csv("dataset.csv", header=True, schema=schema)

    # Lọc ra các bản ghi giống với thông tin người dùng nhập vào
    df = df.filter((col("Medical Condition") == medical_condition))

    # Drop các cột không cần thiết
    columns_to_drop = ["Date of Admission", "Hospital", "Insurance Provider", 
                    "Billing Amount", "Room Number", "Admission Type", "Discharge Date"]
    df = df.drop(*columns_to_drop)

    # Xử lý dữ liệu category sang dạng số
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) 
                for column in ["Gender", "Blood Type", "Medical Condition"]]
    for indexer in indexers:
        df = indexer.transform(df)

    # Tải mô hình trong hàm dự đoán
    lr_model = LogisticRegressionModel.load("logistic_regression_model")

    # Tạo vectơ features cho dữ liệu đầu vào mới
    feature_cols = ["Age", "Gender_index", "Blood Type_index", "Medical Condition_index"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Dự đoán trên dữ liệu đầu vào mới
    predictions = lr_model.transform(df)

    # Hiển thị kết quả dự đoán
    predictions.select("Age", "Gender", "Blood Type", "Medical Condition", "Doctor", "Medication", "Test Results").show()

    medication_counts = predictions.groupBy("Medication").count()
    test_results_counts = predictions.groupBy("Test Results").count()

    # Tổng số lượng bản ghi
    total_records = predictions.count()

    # Tính phần trăm xuất hiện
    medication_percentage = medication_counts.withColumn("Percentage", F.col("count") / total_records * 100).orderBy("Percentage", ascending=False)
    test_results_percentage = test_results_counts.withColumn("Percentage", F.col("count") / total_records * 100).orderBy("Percentage", ascending=False)

    # Hiển thị kết quả
    print("Phần trăm xuất hiện của từng Medication:")
    medication_percentage.show(truncate=False)

    print("Phần trăm xuất hiện của từng Test Results:")
    test_results_percentage.show(truncate=False)

suggest_disease()