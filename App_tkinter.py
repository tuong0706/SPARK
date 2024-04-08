import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
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

def predict_disease():
    age = int(age_entry.get())
    gender = gender_combobox.get().capitalize()
    blood_type = blood_type_entry.get().upper()
    medical_condition = medical_condition_entry.get()

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
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, predictions.select("Age", "Gender", "Blood Type", "Medical Condition", "Doctor", "Medication", 
                                                  "Test Results").toPandas().to_string(index=False))

    medication_counts = predictions.groupBy("Medication").count()
    test_results_counts = predictions.groupBy("Test Results").count()

    # Tổng số lượng bản ghi
    total_records = predictions.count()

    # Tính phần trăm xuất hiện
    medication_percentage = medication_counts.withColumn("Percentage", F.col("count") / total_records * 100).orderBy("Percentage", ascending=False)
    test_results_percentage = test_results_counts.withColumn("Percentage", F.col("count") / total_records * 100).orderBy("Percentage", ascending=False)

    # Hiển thị kết quả
    medication_text.delete(1.0, tk.END)
    medication_text.insert(tk.END, "Phần trăm xuất hiện của từng Medication:\n" + medication_percentage.toPandas().to_string(index=False))

    test_results_text.delete(1.0, tk.END)
    test_results_text.insert(tk.END, "Phần trăm xuất hiện của từng Test Results:\n" + test_results_percentage.toPandas().to_string(index=False))

def reset_inputs():
    # Xóa nội dung các trường nhập liệu
    age_entry.delete(0, tk.END)
    gender_combobox.set('')
    blood_type_entry.delete(0, tk.END)
    medical_condition_entry.delete(0, tk.END)
    # Xóa nội dung trường hiển thị
    result_text.delete(1.0, tk.END)
    medication_text.delete(1.0, tk.END)
    test_results_text.delete(1.0, tk.END)

# Tạo giao diện Tkinter
root = tk.Tk()
root.title("Healthcare data analytics")

# Frame chứa các widget
input_frame = ttk.Frame(root, padding="10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

result_frame = ttk.Frame(root, padding="10")
result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Nhập thông tin
age_label = ttk.Label(input_frame, text="Age:")
age_label.grid(row=0, column=0, sticky=tk.W)
age_entry = ttk.Entry(input_frame)
age_entry.grid(row=0, column=1, sticky=tk.W)

gender_label = ttk.Label(input_frame, text="Gender:")
gender_label.grid(row=1, column=0, sticky=tk.W)
gender_combobox = ttk.Combobox(input_frame, values=["Male", "Female"])
gender_combobox.grid(row=1, column=1, sticky=tk.W)

blood_type_label = ttk.Label(input_frame, text="Blood Type:")
blood_type_label.grid(row=2, column=0, sticky=tk.W)
blood_type_entry = ttk.Combobox(input_frame, values=["O-", "O+", "AB+", "AB-", "A+", "A-", "B+", "B-"])
blood_type_entry.grid(row=2, column=1, sticky=tk.W)

medical_condition_label = ttk.Label(input_frame, text="Medical Condition:")
medical_condition_label.grid(row=3, column=0, sticky=tk.W)
medical_condition_entry = ttk.Entry(input_frame)
medical_condition_entry.grid(row=3, column=1, sticky=tk.W)

predict_button = ttk.Button(input_frame, text="Predict", command=predict_disease)
predict_button.grid(row=4, column=0, columnspan=3, pady=10)

cancel_button = ttk.Button(input_frame, text="Cancel", command=reset_inputs)
cancel_button.grid(row=4, column=3, columnspan=3, pady=10)

# Hiển thị kết quả
result_label = ttk.Label(result_frame, text="Predictions:")
result_label.grid(row=0, column=0, sticky=tk.W)
result_text = tk.Text(result_frame, height=10, width=120)
result_text.grid(row=1, column=0,columnspan=2, sticky=tk.W)

medication_label = ttk.Label(result_frame, text="Medication Percentage:")
medication_label.grid(row=2, column=0, sticky=tk.W)
medication_text = tk.Text(result_frame, height=10, width=40)
medication_text.grid(row=3, column=0, sticky=tk.W)

test_results_label = ttk.Label(result_frame, text="Test Results Percentage:")
test_results_label.grid(row=2, column=1, sticky=tk.W)
test_results_text = tk.Text(result_frame, height=10, width=40)
test_results_text.grid(row=3, column=1, sticky=tk.W)

root.mainloop()
