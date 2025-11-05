from pyspark.sql import SparkSession
from pyspark.sql.types import *
import re
import os
from decimal import Decimal, InvalidOperation

# my comments


spark = SparkSession.builder.appName("SmartSchemaInferenceWithConfidence").getOrCreate()

# --- User settings ---
file_path = '/kaggle/input/covid19-tweets-morning-27032020/all_location_tweets.json'   # CSV or JSON path
sample_size = 5000  # number of rows to sample
export_confidence_path = None  # set to a path like '/tmp/confidence.csv' to export

# --- Detect file type ---
ext = os.path.splitext(file_path)[1].lower()
if ext == '.csv':
    file_type = 'csv'
elif ext == '.json':
    file_type = 'json'
else:
    raise ValueError(f"Unsupported file extension: {ext}. Please use .csv or .json.")

print(f"Detected file type: {file_type}")


def infer_type_with_confidence(values):
    """Infer Spark SQL data type from a list of values and compute confidence."""
    checks = {
        "boolean": 0,
        "date": 0,
        "timestamp": 0,
        "numeric": 0,
        "string": 0
    }
    total = 0

    date_patterns = [
        re.compile(r"^\d{4}-\d{2}-\d{2}$"),
        re.compile(r"^\d{4}/\d{2}/\d{2}$")
    ]
    timestamp_patterns = [
        re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?$")
    ]

    for v in values:
        if v is None or str(v).strip() == "":
            continue
        s = str(v).strip()
        total += 1
        if s.lower() in ["true", "false", "0", "1", "t", "f", "yes", "no"]:
            checks["boolean"] += 1
        elif any(p.match(s) for p in date_patterns):
            checks["date"] += 1
        elif any(p.match(s) for p in timestamp_patterns):
            checks["timestamp"] += 1
        else:
            try:
                Decimal(s)
                checks["numeric"] += 1
            except InvalidOperation:
                checks["string"] += 1

    inferred = max(checks, key=checks.get)
    confidence = (checks[inferred] / total) if total > 0 else 0

    # Map to Spark type
    type_map = {
        "boolean": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType(),
        "numeric": DoubleType(),
        "string": StringType()
    }
    return type_map[inferred], confidence


def recommend_schema(file_path, file_type="csv", sample_size=1000):
    """Infer schema for CSV or JSON file with confidence metrics."""
    if file_type.lower() == "csv":
        df = spark.read.option("header", True).csv(file_path)
    elif file_type.lower() == "json":
        df = spark.read.option("multiline", True).json(file_path)
    else:
        raise ValueError("file_type must be 'csv' or 'json'")

    cols = df.columns
    samples = df.limit(sample_size).collect()

    col_values = {c: [] for c in cols}
    for row in samples:
        for c in cols:
            col_values[c].append(row[c])

    fields = []
    print("\nInferred Spark Schema Recommendation\n")
    print(f"{'Column':25} | {'Type':12} | {'Confidence'}")
    print("-" * 55)

    confidence_scores = {}

    for c in cols:
        dtype, conf = infer_type_with_confidence(col_values[c])
        fields.append(StructField(c, dtype, True))
        confidence_scores[c] = conf
        print(f"{c:25} | {dtype.simpleString():12} | {conf:6.2%}")

    struct = StructType(fields)

    print("\nPasteable StructType schema for PySpark:\n")
    print("schema = StructType([")
    for f in fields:
        print(f"    StructField('{f.name}', {type(f.dataType).__name__}(), True),")
    print("])")

    return struct, confidence_scores


# --- Run schema recommendation ---
schema, confidence_scores = recommend_schema(file_path, file_type=file_type, sample_size=sample_size)

# --- Automatically read the file with inferred schema ---
if file_type.lower() == "csv":
    df = spark.read.schema(schema).option("header", True).csv(file_path)
elif file_type.lower() == "json":
    df = spark.read.schema(schema).option("multiline", True).json(file_path)

# --- Optional correlation analysis for numeric columns ---
numeric_cols = [
    f.name for f in schema.fields
    if isinstance(f.dataType, (DoubleType, LongType, IntegerType, DecimalType))
]

if len(numeric_cols) >= 2:
    print("\nNumeric Column Correlations:")
    corr_df = df.select(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            try:
                corr_val = corr_df.stat.corr(c1, c2)
                print(f"  {c1} ↔ {c2}: {corr_val:.3f}")
            except Exception:
                pass

# --- Show a sample of the dataframe ---
print("\nSample data:")
df.show(5, truncate=False)
