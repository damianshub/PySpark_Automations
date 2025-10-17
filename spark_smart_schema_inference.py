from pyspark.sql import SparkSession
from pyspark.sql.types import *
import re
import os
from decimal import Decimal, InvalidOperation

spark = SparkSession.builder.appName("SmartSchemaInference").getOrCreate()

# --- User settings ---
file_path = '/kaggle/input/covid19-tweets-morning-27032020/all_location_tweets.json'   # CSV or JSON path

# Pick: JSON or CSV file type
# file_type = "csv"                    # choose "csv"
file_type = "json"                    # choose "json"
sample_size = 5000                   # number of rows to sample

# --------------------------
def infer_type(values):
    """
    Infer Spark SQL data type from a list of sample values.
    """
    numeric = True
    has_decimal = False
    has_large = False
    is_boolean = True
    is_date = True
    is_timestamp = True

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

        # Boolean check
        if s.lower() not in ["true", "false", "0", "1", "t", "f", "yes", "no"]:
            is_boolean = False

        # Date check
        if not any(p.match(s) for p in date_patterns):
            is_date = False

        # Timestamp check
        if not any(p.match(s) for p in timestamp_patterns):
            is_timestamp = False

        # Numeric check
        try:
            d = Decimal(s)
            if d % 1 != 0:
                has_decimal = True
            if abs(d) > Decimal(9223372036854775807):
                has_large = True
        except InvalidOperation:
            numeric = False

    # Decision logic
    if is_boolean:
        return BooleanType()
    if is_date:
        return DateType()
    if is_timestamp:
        return TimestampType()
    if numeric:
        if has_decimal:
            return DoubleType() if not has_large else DecimalType(38, 10)
        else:
            if has_large:
                return DecimalType(38, 0)
            return LongType()
    return StringType()


def recommend_schema(file_path, file_type="csv", sample_size=1000):
    """
    Infer schema for CSV or JSON file.
    """
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
    for c in cols:
        dtype = infer_type(col_values[c])
        fields.append(StructField(c, dtype, True))
        print(f"  {c:25} → {dtype.simpleString()}")

    struct = StructType(fields)

    print("\nPasteable StructType schema for PySpark:\n")
    print("schema = StructType([")
    for f in fields:
        print(f"    StructField('{f.name}', {type(f.dataType).__name__}(), True),")
    print("])")

    return struct


# # --- Run schema recommendation ---
# schema = recommend_schema(file_path, file_type=file_type, sample_size=sample_size)

# # Example usage:
# # df = spark.read.schema(schema).csv(file_path)   # for CSV
# # df = spark.read.schema(schema).json(file_path)  # for JSON


# Run schema recommendation
schema = recommend_schema(file_path, file_type=file_type, sample_size=sample_size)

# Automatically read the file with the inferred schema
if file_type.lower() == "csv":
    df = spark.read.schema(schema).option("header", True).csv(file_path)
elif file_type.lower() == "json":
    df = spark.read.schema(schema).option("multiline", True).json(file_path)
else:
    raise ValueError("file_type must be 'csv' or 'json'")

# Show a sample of the dataframe
df.show(5, truncate=False)


