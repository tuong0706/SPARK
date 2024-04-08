from pyspark.sql import SparkSession
import findspark
from functools import reduce
findspark.init()
import csv

# Khởi tạo SparkSession
spark = SparkSession.builder\
    .master('local[1]')\
    .appName('Setdata.com')\
    .getOrCreate()


df = spark.read.csv("dataset.csv", header=True, inferSchema=True)

dfs = [df] * 100

# Hàm nối hai DataFrame
def union_all(df1, df2):
    return df1.union(df2)

# Nối lặp lại 100 lần
df_combined = reduce(union_all, dfs)

# Lưu file mới
df_combined.coalesce(1).write.mode("overwrite").format("csv").option("header", "true").save("dataset_1m.csv")

# df.write.csv("dataset_modified.csv", header=True, mode="overwrite")
# df.write.mode("overwrite").csv("dataset_modified.csv", header=True)
# df.coalesce(1).write.mode("overwrite").csv("dataset_modified.csv", header=True)
# df = spark.read.csv("C:/dataset_modified.csv", header=True, inferSchema=True)