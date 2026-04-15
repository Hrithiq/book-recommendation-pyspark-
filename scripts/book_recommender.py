#!/usr/bin/env python3
"""
Single-file comprehensive Spark recommender pipeline (hybrid HDFS fallback).
Place at: goodbooks_recommender/scripts/recommender.py
Run:
  spark-submit --master local[*] scripts/recommender.py
or on YARN:
  spark-submit --master yarn --deploy-mode client scripts/recommender.py
"""

import os
import sys
import json
import subprocess
import logging
import getpass
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F, types as T, Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors as MLibVectors

# Optional MLflow
try:
    import mlflow
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False

    # ----------------------------
# HDFS auto-detection & configuration helpers
# ----------------------------
import xml.etree.ElementTree as ET

def detect_hdfs_uri(default_ip="10.86.2.87", default_port="9000"):
    """
    Try to detect the HDFS URI from Hadoop's core-site.xml.
    Fallback to hdfs://<default_ip>:<default_port> if not found.
    """
    conf_dir = os.environ.get("HADOOP_CONF_DIR", "/usr/local/hadoop/etc/hadoop")
    core_site = os.path.join(conf_dir, "core-site.xml")
    try:
        if os.path.exists(core_site):
            tree = ET.parse(core_site)
            for prop in tree.findall("property"):
                name = prop.find("name").text.strip()
                if name == "fs.defaultFS":
                    val = prop.find("value").text.strip()
                    if val.startswith("hdfs://"):
                        return val
    except Exception as e:
        logger.warning(f"Failed to parse core-site.xml for fs.defaultFS: {e}")
    return f"hdfs://{default_ip}:{default_port}"

def safe_int_from_env(name, default):
    try:
        v = os.environ.get(name, None)
        return int(v) if v is not None else int(default)
    except Exception:
        return int(default)

# ----------------------------
#  USER-SPECIFIC / DEFAULT CONFIG
# ----------------------------
# HDFS base (user/provided)
HDFS_BASE = detect_hdfs_uri() + "/user/hadoop/goodbooks_recommender"
  # <- uses 'hadoop' as username as requested

# Local layout (expected)
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # parent of scripts/
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOCAL_MODELS_DIR = LOCAL_OUTPUT_DIR / "models"
LOCAL_RECS_DIR = LOCAL_OUTPUT_DIR / "recommendations"
LOCAL_METRICS_DIR = LOCAL_OUTPUT_DIR / "metrics"
LOCAL_PREPROC_DIR = LOCAL_OUTPUT_DIR / "preprocessed"
LOCAL_LOG_DIR = PROJECT_ROOT / "logs"

for p in [LOCAL_OUTPUT_DIR, LOCAL_MODELS_DIR, LOCAL_RECS_DIR, LOCAL_METRICS_DIR, LOCAL_PREPROC_DIR, LOCAL_LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Ratings bounds & misc
MIN_RATING = 1.0
MAX_RATING = 5.0

# Experiment / algorithm params
TOP_K = 10
RELEVANT_THRESHOLD = 3.0
TRAIN_FRAC = 0.8
HOLDOUT_PER_USER = True
RANDOM_SEED = 42
SAMPLE_MAX_RATINGS = None  # set to int to limit dataset for quick local runs

# ALS CV grid
ALS_RANKS = [40, 80]
ALS_REGS = [0.05, 0.1]
ALS_ITERS = [8, 12]
ALS_MAX_PARALLELISM = 8

# SVD / kNN params
SVD_K = 50
KNN_NEIGHBORS = 50
KNN_AGG_TOP = TOP_K

# Safety / cluster hints (you told me 20 cores, 16GB)
CLUSTER_CORES = 20
CLUSTER_MEM_GB = 16

# Logging
LOG_FILE = LOCAL_LOG_DIR / "run.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(str(LOG_FILE)), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Global flag set at runtime
USE_HDFS = False

# ----------------------------
# Shell helpers for HDFS ops
# ----------------------------
def run_cmd(cmd):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 1, "", f"command-not-found: {e}"

def detect_hdfs():
    # quick check whether `hdfs` CLI works and root is listable
    code, out, err = run_cmd(["hdfs", "dfs", "-ls", "/"])
    if code == 0:
        logger.info("HDFS CLI detected; will attempt HDFS operations at %s", HDFS_BASE)
        return True
    logger.info("HDFS CLI not available; running in local-only mode.")
    return False

def hdfs_mkdir(path):
    run_cmd(["hdfs", "dfs", "-mkdir", "-p", path])

def hdfs_put(local_path, hdfs_dir):
    # put (overwrite)
    run_cmd(["hdfs", "dfs", "-mkdir", "-p", hdfs_dir])
    run_cmd(["hdfs", "dfs", "-put", "-f", str(local_path), hdfs_dir])

def hdfs_exists(hpath):
    code, _, _ = run_cmd(["hdfs", "dfs", "-test", "-e", hpath])
    return code == 0

def hdfs_rm(path):
    run_cmd(["hdfs", "dfs", "-rm", "-r", "-f", path])

# ----------------------------
# Metrics helpers
# ----------------------------
def precision_at_k(recs, truth, k):
    total, users = 0.0, 0
    for uid, rlist in recs.items():
        gt = set(truth.get(uid, []))
        if not gt: continue
        users += 1
        hits = sum(1 for i in rlist[:k] if i in gt)
        total += hits / k
    return float(total / users) if users else 0.0

def recall_at_k(recs, truth, k):
    total, users = 0.0, 0
    for uid, rlist in recs.items():
        gt = set(truth.get(uid, []))
        if not gt: continue
        users += 1
        hits = sum(1 for i in rlist[:k] if i in gt)
        total += hits / len(gt)
    return float(total / users) if users else 0.0

def ndcg_at_k(recs, truth, k):
    import math
    total, users = 0.0, 0
    for uid, rlist in recs.items():
        gt = set(truth.get(uid, []))
        if not gt: continue
        users += 1
        dcg = 0.0
        for i, item in enumerate(rlist[:k]):
            rel = 1.0 if item in gt else 0.0
            dcg += (2**rel - 1) / math.log2(i + 2)
        idcg = sum((2**1 - 1) / math.log2(i + 2) for i in range(min(len(gt), k)))
        total += (dcg / idcg) if idcg > 0 else 0.0
    return float(total / users) if users else 0.0

# ----------------------------
# Save helpers that write to HDFS via Spark or use CLI when needed
# ----------------------------
def save_dict_as_parquet_via_spark(spark, dic, path):
    """dic: {user_id: [book_ids,...]} -> write parquet to path (hdfs or local)"""
    rows = [{"user_id": int(k), "recommendations": dic[k]} for k in dic]
    if not rows:
        return
    df = spark.createDataFrame(pd.DataFrame(rows))
    df.write.mode("overwrite").parquet(path)

def save_metrics_json_locally(metrics):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = LOCAL_METRICS_DIR / f"comparison_{ts}.json"
    with open(p, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics JSON locally: %s", p)
    return str(p)

# ----------------------------
# Spark session
# ----------------------------
def create_spark():
    """
    Create SparkSession with safer memory/serialization settings tuned for local runs.
    Uses env vars (if set) or sensible defaults.
    """
    # Allow environment overrides
    drv_mem = os.environ.get("SPARK_DRIVER_MEMORY", "10g")   # default driver heap
    exec_mem = os.environ.get("SPARK_EXECUTOR_MEMORY", "10g")  # executor heap
    max_result = os.environ.get("SPARK_DRIVER_MAXRESULTSIZE", "3g")
    shuffle_partitions = safe_int_from_env("SPARK_SHUFFLE_PARTITIONS", max(4, min(200, CLUSTER_CORES // 2)))
    parallelism = safe_int_from_env("SPARK_DEFAULT_PARALLELISM", max(2, CLUSTER_CORES))
    kryo_buf = os.environ.get("SPARK_KRYO_BUFFER_MAX", "1024m")

    builder = SparkSession.builder.appName("Goodbooks_Recommender")
    # Basic memory settings
    builder = builder.config("spark.driver.memory", "10g")
    builder = builder.config("spark.executor.memory", "10g")
    builder = builder.config("spark.driver.maxResultSize", "3g")
    # parallelism & partitions
    builder = builder.config("spark.default.parallelism", "80")
    builder = builder.config("spark.sql.shuffle.partitions", "80")
    # serialization
    builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    builder = builder.config("spark.kryoserializer.buffer.max", kryo_buf)
    # other helpful settings
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    builder = builder.config("spark.local.dir", "/tmp/spark-local")
    # Reduce memory pressure from broadcast joins by making threshold small (avoid huge broadcasts)
    builder = builder.config("spark.sql.autoBroadcastJoinThreshold", -1)
    builder = builder.config("spark.memory.offHeap.enabled", "true")
    builder = builder.config("spark.memory.offHeap.size", "512m")


    # Create
    spark = builder.getOrCreate()

    # Ensure Hadoop fs.defaultFS is set if you have detect_hdfs_uri()
    try:
        if "detect_hdfs_uri" in globals():
            hdfs_uri = detect_hdfs_uri()
            spark.sparkContext._jsc.hadoopConfiguration().set("fs.defaultFS", hdfs_uri)
            logger.info("Using HDFS endpoint (fs.defaultFS) = %s", hdfs_uri)
    except Exception as e:
        logger.warning("Could not set fs.defaultFS via Spark: %s", e)

    logger.info("Spark created: driver.memory=%s executor.memory=%s driver.maxResultSize=%s shuffle.partitions=%s parallelism=%s",
                drv_mem, exec_mem, max_result, shuffle_partitions, parallelism)

    return spark


# ----------------------------
# Data prep: upload CSVs to HDFS and convert to parquet
# ----------------------------
def upload_local_csvs_to_hdfs_if_needed(spark):
    """
    If USE_HDFS True: upload local CSVs to HDFS base path under /data and
    create a ratings.parquet there (if not present).
    Otherwise just ensure local CSV exists.
    """
    global USE_HDFS
    USE_HDFS = detect_hdfs()
    if USE_HDFS:
        hdfs_data_dir = HDFS_BASE.rstrip("/") + "/data"
        hdfs_mkdir(hdfs_data_dir)
        # upload CSVs present locally
        for name in ["ratings.csv", "books.csv", "book_tags.csv", "tags.csv", "to_read.csv"]:
            lp = LOCAL_DATA_DIR / name
            if lp.exists():
                logger.info("Uploading %s to HDFS %s", lp, hdfs_data_dir)
                hdfs_put(lp, hdfs_data_dir)
            else:
                logger.info("Local data file not present: %s (skipping upload)", lp)
        # create parquet for ratings on HDFS if not exists
        hdfs_ratings_parquet = HDFS_BASE.rstrip("/") + "/data/ratings.parquet"
        if not hdfs_exists(hdfs_ratings_parquet):
            hdfs_ratings_csv = hdfs_data_dir + "/ratings.csv"
            logger.info("Creating Parquet on HDFS from %s", hdfs_ratings_csv)
            try:
                df = spark.read.csv(hdfs_ratings_csv, header=True, inferSchema=True)
                df.write.mode("overwrite").parquet(hdfs_ratings_parquet)
                logger.info("Wrote ratings.parquet to HDFS: %s", hdfs_ratings_parquet)
            except Exception as e:
                logger.warning("Failed to convert ratings.csv to parquet on HDFS: %s", e)
        else:
            logger.info("Ratings parquet already exists on HDFS: %s", hdfs_ratings_parquet)
        return {"use_hdfs": True, "hdfs_data_dir": hdfs_data_dir, "hdfs_ratings_parquet": hdfs_ratings_parquet}
    else:
        logger.info("HDFS not available: staying local; ensure ratings.csv present at %s", LOCAL_DATA_DIR)
        local_ratings = LOCAL_DATA_DIR / "ratings.csv"
        if not local_ratings.exists():
            raise FileNotFoundError(f"ratings.csv not found locally at {local_ratings}")
        return {"use_hdfs": False, "local_ratings": str(local_ratings)}

# ----------------------------
# Preprocessing pipeline (Spark)
# ----------------------------
def preprocess(spark, ratings_path):
    """
    Read ratings_path (csv or parquet), clean, compute book/user stats, optional metadata join,
    impute numeric features, encode top authors, standardize, and save preprocessed parquet.
    """
    logger.info("Reading ratings from %s", ratings_path)
    if str(ratings_path).endswith(".parquet"):
        df = spark.read.parquet(ratings_path)
    else:
        df = spark.read.csv(ratings_path, header=True, inferSchema=True)
    df = df.select(F.col("user_id").cast("int"), F.col("book_id").cast("int"), F.col("rating").cast("double")).na.drop()
    df = df.filter((F.col("rating") >= MIN_RATING) & (F.col("rating") <= MAX_RATING)).dropDuplicates()
    logger.info("Loaded %d ratings", df.count())

    # book/user stats
    book_stats = df.groupBy("book_id").agg(F.count("*").alias("book_rating_count"), F.avg("rating").alias("book_rating_mean"))
    user_stats = df.groupBy("user_id").agg(F.count("*").alias("user_rating_count"), F.avg("rating").alias("user_rating_mean"))
    joined = df.join(book_stats, "book_id", "left").join(user_stats, "user_id", "left")

    # optional metadata join (authors)
    local_books = LOCAL_DATA_DIR / "books.csv"
    if local_books.exists():
        try:
            books_df = spark.read.csv(str(local_books), header=True, inferSchema=True)
            if "book_id" in books_df.columns:
                sel = [c for c in ["book_id", "authors", "original_publication_year", "average_rating"] if c in books_df.columns]
                joined = joined.join(books_df.select(*sel), on="book_id", how="left")
        except Exception as e:
            logger.warning("Failed to read local books.csv: %s", e)

    # authors top-K collapse
    cat_cols = []
    if "authors" in joined.columns:
        top_auth = [r.authors for r in joined.groupBy("authors").count().orderBy(F.desc("count")).limit(200).collect() if r.authors]
        top_auth_set = set(top_auth)
        bc = spark.sparkContext.broadcast(top_auth_set)
        @F.udf(returnType=T.StringType())
        def topk_author(a):
            if a is None: return "unknown"
            return a if a in bc.value else "other"
        joined = joined.withColumn("authors_top", topk_author(F.col("authors")))
        cat_cols.append("authors_top")

    numeric_cols = [c for c in ["book_rating_count", "book_rating_mean", "user_rating_count", "user_rating_mean", "original_publication_year", "average_rating"] if c in joined.columns]
    imputer_output = [c + "_imp" for c in numeric_cols]
    stages = []
    if numeric_cols:
        imputer = Imputer(inputCols=numeric_cols, outputCols=imputer_output).setStrategy("median")
        stages.append(imputer)
    indexers = []
    encoders = []
    for c in cat_cols:
        si = StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe", handleInvalid="keep")
        indexers.append(si); encoders.append(ohe)
    stages += indexers + encoders
    assembler_inputs = (imputer_output if imputer_output else []) + [c + "_ohe" for c in cat_cols]
    if not assembler_inputs:
        joined = joined.withColumn("dummy_feat", F.lit(0.0))
        assembler_inputs = ["dummy_feat"]
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_assembled", handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_assembled", outputCol="features", withStd=True, withMean=False)
    stages += [assembler, scaler]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(joined)
    transformed = model.transform(joined)
    # save preproc local
    local_preproc = LOCAL_PREPROC_DIR / "ratings_preprocessed.parquet"
    transformed.select("user_id", "book_id", "rating", "features").write.mode("overwrite").parquet(str(local_preproc))
    logger.info("Saved preprocessed parquet locally: %s", local_preproc)
    if USE_HDFS:
        hdfs_preproc = HDFS_BASE.rstrip("/") + "/outputs/preprocessed/ratings_preprocessed.parquet"
        transformed.select("user_id", "book_id", "rating", "features").write.mode("overwrite").parquet(hdfs_preproc)
        logger.info("Saved preprocessed parquet to HDFS: %s", hdfs_preproc)
    return transformed

# ----------------------------
# Splitting helpers
# ----------------------------
def holdout_per_user(df):
    w = Window.partitionBy("user_id").orderBy(F.rand(seed=RANDOM_SEED))
    numbered = df.withColumn("rn", F.row_number().over(w))
    counts = df.groupBy("user_id").count().withColumnRenamed("count", "cnt")
    numbered = numbered.join(counts, "user_id", "left")
    test = numbered.filter((F.col("cnt") >= 2) & (F.col("rn") == 1)).select("user_id", "book_id", "rating")
    train = numbered.join(test, on=["user_id", "book_id", "rating"], how="left_anti").select("user_id", "book_id", "rating")
    logger.info("Train rows: %d, Test rows: %d", train.count(), test.count())
    return train, test

def random_split(df, frac=TRAIN_FRAC):
    train, test = df.randomSplit([frac, 1.0 - frac], seed=RANDOM_SEED)
    logger.info("Random split -> train: %d, test: %d", train.count(), test.count())
    return train, test

# ----------------------------
# ALS CV training & recommendation
# ----------------------------
def train_als_cv(spark, train_df, test_df):
    """
    Train ALS with CV but with safety guards to avoid Java heap OOM.
    - adaptively reduces grid size if default grid + parallelism would blow memory.
    - retries once with a safe small-grid config on OOM.
    """
    # Read memory settings from spark conf if present
    driver_mem_cfg = spark.conf.get("spark.driver.memory", "10g")
    # convert '6g' or '512m' to MB integer for heuristic
    def parse_mem_to_mb(s):
        try:
            s = s.lower().strip()
            if s.endswith("g"):
                return int(float(s[:-1]) * 1024)
            if s.endswith("m"):
                return int(float(s[:-1]))
            return int(float(s) / (1024*1024))
        except Exception:
            return 6144
    driver_mb = parse_mem_to_mb(driver_mem_cfg)

    # Heuristic: if driver < 4096MB, be conservative
    safe_mode = driver_mb < 4096

    # Build ALS and param grid — but shrink grid when safe_mode True
    als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
              coldStartStrategy="drop", nonnegative=True, seed=RANDOM_SEED)

    # decide grid
    if safe_mode:
        logger.warning("Detected low driver memory (%d MB) — using reduced ALS grid and serial CV to avoid OOM", driver_mb)
        paramGrid = ParamGridBuilder().addGrid(als.rank, [20, 40]).addGrid(als.regParam, [0.1]).addGrid(als.maxIter, [6]).build()
        parallelism = 1
        numFolds = 2
    else:
        paramGrid = ParamGridBuilder().addGrid(als.rank, ALS_RANKS).addGrid(als.regParam, ALS_REGS).addGrid(als.maxIter, ALS_ITERS).build()
        parallelism = min(ALS_MAX_PARALLELISM, safe_int_from_env("ALS_PARALLELISM", ALS_MAX_PARALLELISM))
        numFolds = 3

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    cv = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator,
                       numFolds=numFolds, parallelism=parallelism)

    # Fit with a try/except; on OOM try once with very small model
    try:
        cv_model = cv.fit(train_df)
    except Exception as e:
        tb = str(e)
        logger.exception("ALS CV initial fit failed: %s", e)
        # If it's OOM or serialization related, retry with fallback tiny config
        if ("OutOfMemoryError" in tb or "Java heap space" in tb or "Task serialization failed" in tb):
            logger.warning("Retrying ALS with fallback small config to avoid OOM")
            try:
                fallback_als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
                                   rank=20, maxIter=5, regParam=0.1, coldStartStrategy="drop", nonnegative=True, seed=RANDOM_SEED)
                fallback_model = fallback_als.fit(train_df)
                best = fallback_model
                # Evaluate
                preds = best.transform(test_df).na.drop()
                rmse_val = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction").evaluate(preds)
                logger.info("ALS fallback model RMSE: %s", rmse_val)
                # recommendations
                users_test = test_df.select("user_id").distinct()
                recs_df = best.recommendForUserSubset(users_test, TOP_K)
                # Save recs to parquet (avoid large collect)
                tmp_recs_local = LOCAL_RECS_DIR / "als_recs_fallback.parquet"
                recs_df.write.mode("overwrite").parquet(str(tmp_recs_local))
                # Collect limited subset to memory for evaluation (do NOT collect all if huge)
                sample_recs = {int(r.user_id): [int(x.book_id) for x in r.recommendations] for r in recs_df.limit(1000).collect()}
                truth = build_truth_map_from_df(test_df)
                prec = precision_at_k(sample_recs, truth, TOP_K)
                rec = recall_at_k(sample_recs, truth, TOP_K)
                ndcg = ndcg_at_k(sample_recs, truth, TOP_K)
                metrics = {"rmse": float(rmse_val), "precision@{}".format(TOP_K): prec, "recall@{}".format(TOP_K): rec, "ndcg@{}".format(TOP_K): ndcg}
                return metrics, sample_recs
            except Exception as ee:
                logger.exception("ALS fallback failed as well: %s", ee)
                return {"error": str(e)}, {}
        else:
            return {"error": str(e)}, {}

    # If fit succeeded normally:
    best = cv_model.bestModel
    # Evaluate
    try:
        preds = best.transform(test_df).na.drop()
        rmse_val = evaluator.evaluate(preds)
        logger.info("ALS test RMSE: %s", rmse_val)
    except Exception as e:
        logger.warning("Evaluation failed after ALS fit: %s", e)
        rmse_val = None

    # recommendations
    try:
        users_test = test_df.select("user_id").distinct()
        recs_df = best.recommendForUserSubset(users_test, TOP_K)
        # write recs to parquet to avoid memory blow from collect
        recs_parquet = LOCAL_RECS_DIR / "als_recs.parquet"
        recs_df.write.mode("overwrite").parquet(str(recs_parquet))
        # for immediate in-memory mapping use a capped collect
        recs_sample_map = {int(r.user_id): [int(x.book_id) for x in r.recommendations] for r in recs_df.limit(5000).collect()}
    except Exception as e:
        logger.warning("Failed to create or persist ALS recommendations: %s", e)
        recs_sample_map = {}

    # save model (best)
    try:
        local_model_path = LOCAL_MODELS_DIR / "als_best"
        best.write().overwrite().save(str(local_model_path))
        logger.info("Saved ALS model locally: %s", local_model_path)
    except Exception as e:
        logger.warning("Failed to save ALS locally: %s", e)
    if USE_HDFS:
        try:
            hdfs_model_path = HDFS_BASE.rstrip("/") + "/outputs/models/als_best"
            best.write().overwrite().save(hdfs_model_path)
            logger.info("Saved ALS model to HDFS: %s", hdfs_model_path)
        except Exception as e:
            logger.warning("Failed to save ALS to HDFS: %s", e)

    truth = build_truth_map_from_df(test_df)
    prec = precision_at_k(recs_sample_map, truth, TOP_K)
    rec = recall_at_k(recs_sample_map, truth, TOP_K)
    ndcg = ndcg_at_k(recs_sample_map, truth, TOP_K)
    metrics = {"rmse": float(rmse_val) if rmse_val is not None else None, "precision@{}".format(TOP_K): prec, "recall@{}".format(TOP_K): rec, "ndcg@{}".format(TOP_K): ndcg}
    return metrics, recs_sample_map

# ----------------------------
# Distributed SVD (RowMatrix) -> recs + item factors saved
# ----------------------------
def run_svd(spark, train_df, test_df, ratings_df):
    # build mappings
    users = ratings_df.select("user_id").distinct().orderBy("user_id").withColumn("u_idx", F.row_number().over(Window.orderBy("user_id")) - 1)
    items = ratings_df.select("book_id").distinct().orderBy("book_id").withColumn("i_idx", F.row_number().over(Window.orderBy("book_id")) - 1)
    n_users = users.count(); n_items = items.count()
    logger.info("SVD indices: n_users=%d n_items=%d", n_users, n_items)
    if n_users <= 1 or n_items <= 1:
        return {"skipped": True}, {}
    train_idx = train_df.join(users, "user_id", "left").join(items, "book_id", "left").select("u_idx", "i_idx", "rating")
    test_idx = test_df.join(users, "user_id", "left").join(items, "book_id", "left").select("u_idx", "i_idx", "rating", "user_id", "book_id")
    grouped = train_idx.rdd.map(lambda r: (int(r.u_idx), (int(r.i_idx), float(r.rating)))).groupByKey().mapValues(list).sortByKey()
    def to_sparse(pairs):
        pairs = sorted(pairs, key=lambda x: x[0])   # ensures increasing indices
        idxs = [int(x[0]) for x in pairs]
        vals = [float(x[1]) for x in pairs]
        return MLibVectors.sparse(n_items, idxs, vals)

    vectors_rdd = grouped.map(lambda x: to_sparse(x[1]))
    rowmat = RowMatrix(vectors_rdd)
    k = min(SVD_K, min(n_users - 1, n_items - 1))
    if k <= 0:
        return {"skipped": True}, {}
    svd = rowmat.computeSVD(k, computeU=True)
    V = np.array(svd.V.toArray()).T  # (n_items, k)
    # collect U rows
    U_rows = svd.U.rows.zipWithIndex().map(lambda x: (int(x[1]), np.array(x[0].toArray()))).collect()
    user_factors = {u_idx: vec for u_idx, vec in U_rows}
    user_idx_to_id = {int(r.u_idx): int(r.user_id) for r in users.select("u_idx", "user_id").collect()}
    item_idx_to_id = {int(r.i_idx): int(r.book_id) for r in items.select("i_idx", "book_id").collect()}
    # train items per user
    train_user_items = train_df.groupBy("user_id").agg(F.collect_set("book_id").alias("train_items")).rdd.map(lambda r: (int(r.user_id), [int(x) for x in r.train_items])).collectAsMap()
    recs = {}
    for r in test_df.select("user_id").distinct().collect():
        uid = int(r.user_id)
        u_idx_row = users.filter(F.col("user_id") == uid).select("u_idx").collect()
        if not u_idx_row: continue
        u_idx = int(u_idx_row[0].u_idx)
        if u_idx not in user_factors: continue
        uvec = user_factors[u_idx]
        scores = V.dot(uvec)
        ranked = np.argsort(-scores)
        seen = set(train_user_items.get(uid, []))
        out = []
        for idx in ranked:
            bid = int(item_idx_to_id[int(idx)])
            if bid in seen: continue
            out.append(bid)
            if len(out) >= TOP_K: break
        recs[uid] = out
    # evaluate
    truth = build_truth_map_from_df(test_df)
    prec = precision_at_k(recs, truth, TOP_K)
    rec_val = recall_at_k(recs, truth, TOP_K)
    ndcg = ndcg_at_k(recs, truth, TOP_K)
    # RMSE on test where factors available
    y_true = []; y_pred = []
    for r in test_idx.collect():
        u_idx = int(r.u_idx); i_idx = int(r.i_idx)
        if u_idx in user_factors and i_idx < V.shape[0]:
            pred = float(np.dot(user_factors[u_idx], V[i_idx]))
            y_true.append(float(r.rating)); y_pred.append(float(min(MAX_RATING, max(MIN_RATING, pred))))
    rmse_val = float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))) if y_true else None
    # save item factors as parquet via Spark
    rows = [{"book_id": int(item_idx_to_id[i]), "features": list(map(float, V[i]))} for i in range(V.shape[0])]
    try:
        sdf = spark.createDataFrame(pd.DataFrame(rows))
        local_path = LOCAL_MODELS_DIR / "svd_item_factors.parquet"
        sdf.write.mode("overwrite").parquet(str(local_path))
        logger.info("Saved SVD item factors locally: %s", local_path)
        if USE_HDFS:
            hdfs_path = HDFS_BASE.rstrip("/") + "/outputs/models/svd_item_factors.parquet"
            sdf.write.mode("overwrite").parquet(hdfs_path)
            logger.info("Saved SVD item factors to HDFS: %s", hdfs_path)
    except Exception as e:
        logger.warning("Failed saving item factors: %s", e)
    save_dict_as_parquet_via_spark(spark, recs, str(LOCAL_RECS_DIR / "svd_recs.parquet"))
    if USE_HDFS:
        save_dict_as_parquet_via_spark(spark, recs, HDFS_BASE.rstrip("/") + "/outputs/recommendations/svd")
    metrics = {"rmse": float(rmse_val) if rmse_val is not None else None, "precision@{}".format(TOP_K): prec, "recall@{}".format(TOP_K): rec_val, "ndcg@{}".format(TOP_K): ndcg}
    return metrics, recs

# ----------------------------
# Item-kNN using item-factors
# ----------------------------
def run_item_knn(spark, ratings_df, train_df, test_df):
    local_factors = LOCAL_MODELS_DIR / "svd_item_factors.parquet"
    if not local_factors.exists():
        logger.info("SVD item factors not found locally; computing SVD first.")
        run_svd(spark, train_df, test_df, ratings_df)
    if not local_factors.exists():
        logger.warning("Item factors still missing; skipping kNN.")
        return {"skipped": True}, {}
    item_df = spark.read.parquet(str(local_factors)).select("book_id", "features")
    items = item_df.collect()
    book_ids = [int(r.book_id) for r in items]
    V = np.array([list(map(float, r.features)) for r in items])
    n_items = V.shape[0]
    # compute neighbors via dot-product (simple) — for huge item sets use LSH / approximate method
    top_neighbors = {}
    logger.info("Precomputing top-%d neighbors per item (dot-product)", KNN_NEIGHBORS)
    for i in range(n_items):
        vec = V[i]
        scores = V.dot(vec)
        scores[i] = -np.inf
        idxs = np.argsort(-scores)[:KNN_NEIGHBORS]
        top_neighbors[book_ids[i]] = [book_ids[j] for j in idxs]
    # aggregate neighbors for each user based on their train liked items
    train_map = train_df.groupBy("user_id").agg(F.collect_set("book_id").alias("liked")).rdd.map(lambda r: (int(r.user_id), [int(x) for x in r.liked])).collectAsMap()
    recs = {}
    for r in test_df.select("user_id").distinct().collect():
        uid = int(r.user_id)
        liked = train_map.get(uid, [])
        if not liked:
            recs[uid] = []
            continue
        counts = Counter()
        for lb in liked:
            for nb in top_neighbors.get(lb, []):
                if nb in liked: continue
                counts[nb] += 1
        recs[uid] = [bid for bid, _ in counts.most_common(TOP_K)]
    truth = build_truth_map_from_df(test_df)
    prec = precision_at_k(recs, truth, TOP_K)
    rec_val = recall_at_k(recs, truth, TOP_K)
    ndcg = ndcg_at_k(recs, truth, TOP_K)
    save_dict_as_parquet_via_spark(spark, recs, str(LOCAL_RECS_DIR / "knn_recs.parquet"))
    if USE_HDFS:
        save_dict_as_parquet_via_spark(spark, recs, HDFS_BASE.rstrip("/") + "/outputs/recommendations/knn")
    metrics = {"precision@{}".format(TOP_K): prec, "recall@{}".format(TOP_K): rec_val, "ndcg@{}".format(TOP_K): ndcg}
    return metrics, recs

# ----------------------------
# Build truth map for evaluation (test DataFrame)
# ----------------------------
def build_truth_map_from_df(test_df):
    truth = {}
    for r in test_df.collect():
        if r.rating >= RELEVANT_THRESHOLD:
            uid = int(r.user_id); bid = int(r.book_id)
            truth.setdefault(uid, []).append(bid)
    return truth

# ----------------------------
# Plots (matplotlib) saved locally and uploaded to HDFS if available
# ----------------------------
def create_and_upload_plots(spark, ratings_df):
    # take small sample to plot to avoid collecting entire dataset
    sample_frac = 0.05
    sample = ratings_df.sample(fraction=sample_frac, seed=RANDOM_SEED).toPandas()
    if sample.empty:
        logger.warning("Plot sample empty, skipping plots.")
        return
    # rating distribution
    fig1 = plt.figure(figsize=(6,4)); sample['rating'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Rating"); plt.ylabel("Count"); plt.title("Rating distribution (sample)")
    p1 = LOCAL_OUTPUT_DIR / "rating_distribution.png"; plt.tight_layout(); fig1.savefig(str(p1)); plt.close(fig1)
    # ratings per book hist
    fig2 = plt.figure(figsize=(6,4)); sample.groupby('book_id').size().hist(bins=50)
    plt.xlabel("Ratings per book (sample)"); plt.ylabel("Count"); plt.title("Ratings per book (sample)")
    p2 = LOCAL_OUTPUT_DIR / "ratings_per_book_hist.png"; plt.tight_layout(); fig2.savefig(str(p2)); plt.close(fig2)
    # ratings per user hist
    fig3 = plt.figure(figsize=(6,4)); sample.groupby('user_id').size().hist(bins=50)
    plt.xlabel("Ratings per user (sample)"); plt.ylabel("Count"); plt.title("Ratings per user (sample)")
    p3 = LOCAL_OUTPUT_DIR / "ratings_per_user_hist.png"; plt.tight_layout(); fig3.savefig(str(p3)); plt.close(fig3)
    logger.info("Saved plots locally: %s, %s, %s", p1, p2, p3)
    if USE_HDFS:
        hdfs_plots_dir = HDFS_BASE.rstrip("/") + "/outputs/plots"
        run_cmd(["hdfs", "dfs", "-mkdir", "-p", hdfs_plots_dir])
        for localp in [str(p1), str(p2), str(p3)]:
            run_cmd(["hdfs", "dfs", "-put", "-f", localp, hdfs_plots_dir])
        logger.info("Uploaded plots to HDFS: %s", hdfs_plots_dir)

# ----------------------------
# Save aggregated metrics (local + HDFS + MLflow)
# ----------------------------
def save_and_log_metrics(spark, all_metrics):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # local JSON
    local_json = LOCAL_METRICS_DIR / f"comparison_{ts}.json"
    with open(local_json, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Saved metrics JSON locally: %s", local_json)
    # HDFS write via Spark (JSON)
    if USE_HDFS:
        rows = []
        for m, metrics in all_metrics.items():
            row = {"model": m}
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    row[k] = v
            rows.append(row)
        sdf = spark.createDataFrame(pd.DataFrame(rows))
        hdfs_metrics_dir = HDFS_BASE.rstrip("/") + f"/outputs/metrics/comparison_{ts}"
        sdf.write.mode("overwrite").json(hdfs_metrics_dir)
        logger.info("Saved metrics JSON to HDFS: %s", hdfs_metrics_dir)
    # MLflow logging
    if MLFLOW_OK:
        mlflow.set_experiment("goodbooks_recommender")
        with mlflow.start_run(run_name=f"comparison_{ts}"):
            for model, metrics in all_metrics.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        try:
                            mlflow.log_metric(f"{model}_{k}", float(v) if v is not None else -1.0)
                        except Exception:
                            pass
            mlflow.log_artifact(str(local_json), artifact_path="metrics")
            logger.info("Logged metrics to MLflow")

# ----------------------------
# MAIN orchestrator
# ----------------------------
def run_pipeline():
    logger.info("=== Starting Goodbooks recommender pipeline ===")
    spark = create_spark()
    # prepare HDFS and upload CSVs if possible
    prep_info = upload_local_csvs_to_hdfs_if_needed(spark)
    # choose ratings input path (prefer HDFS parquet)
    if USE_HDFS:
        hdfs_parquet = HDFS_BASE.rstrip("/") + "/data/ratings.parquet"
        ratings_path = hdfs_parquet if hdfs_exists(hdfs_parquet) else HDFS_BASE.rstrip("/") + "/data/ratings.csv"
    else:
        ratings_path = str(LOCAL_DATA_DIR / "ratings.csv")
    # preprocess
    transformed = preprocess(spark, ratings_path)
    # split
    if HOLDOUT_PER_USER:
        train_df, test_df = holdout_per_user(transformed.select("user_id", "book_id", "rating"))
    else:
        train_df, test_df = random_split(transformed.select("user_id", "book_id", "rating"))
    # create plots
    create_and_upload_plots(spark, transformed)
    # ALS
    try:
        als_metrics, als_recs = train_als_cv(spark, train_df, test_df)
    except Exception as e:
        logger.exception("ALS error: %s", e); als_metrics, als_recs = {"error": str(e)}, {}
    # SVD
    try:
        svd_metrics, svd_recs = run_svd(spark, train_df, test_df, transformed)
    except Exception as e:
        logger.exception("SVD error: %s", e); svd_metrics, svd_recs = {"error": str(e)}, {}
    # kNN
    try:
        knn_metrics, knn_recs = run_item_knn(spark, transformed, train_df, test_df)
    except Exception as e:
        logger.exception("kNN error: %s", e); knn_metrics, knn_recs = {"error": str(e)}, {}
    # aggregate metrics
    all_metrics = {"ALS": als_metrics, "SVD": svd_metrics, "kNN": knn_metrics}
    save_and_log_metrics(spark, all_metrics)
    logger.info("=== Pipeline finished. Local outputs under %s. HDFS base: %s (USE_HDFS=%s) ===", LOCAL_OUTPUT_DIR, HDFS_BASE, USE_HDFS)
    spark.stop()

if __name__ == "__main__":
    run_pipeline()
