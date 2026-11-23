# Databricks notebook source

# STEP 17.1 — PRODUCTION DEPLOYMENT READINESS
# READ-ONLY METADATA CHECKS
# No inference, writes, or job submissions

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 17.1 — PRODUCTION DEPLOYMENT READINESS")
print("=" * 70)

# --------------------------------------------------
# 1. Define existing project resources
# --------------------------------------------------

deployment_config = {
    "environment": "STAGING_CANDIDATE",

    "model_name":
        "workspace.default.telco_churn_logistic_regression",

    "approved_model_version": "1",

    "source_table":
        "workspace.default.telco_test_raw",

    "prediction_table":
        "workspace.default.telco_15_approved_predictions",

    "registry_table":
        "workspace.default.telco_15_job_batch_registry",

    "protected_job_id": "752436180383454",

    "expected_prediction_rows": 1409,

    "allow_inference": False,
    "allow_writes": False,
    "allow_job_submission": False
}

print("\n1. DEPLOYMENT CONFIGURATION")

for key, value in deployment_config.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 2. Verify required Unity Catalog tables
# --------------------------------------------------

print("\n2. UNITY CATALOG TABLE CHECKS")

required_tables = [
    deployment_config["source_table"],
    deployment_config["prediction_table"],
    deployment_config["registry_table"]
]

table_results = {}

for table_name in required_tables:

    try:
        # Read schema metadata only
        schema = spark.table(table_name).schema

        table_results[table_name] = True

        print(f"\nTable: {table_name}")
        print("Accessible: YES")
        print("Column count:", len(schema.fields))

    except Exception as e:

        table_results[table_name] = False

        print(f"\nTable: {table_name}")
        print("Accessible: NO")
        print("Error:", str(e)[:250])


# --------------------------------------------------
# 3. Verify registered model metadata
# --------------------------------------------------

print("\n3. REGISTERED MODEL CHECK")

model_verified = False

try:
    from mlflow import MlflowClient

    client = MlflowClient()

    model_version = client.get_model_version(
        name=deployment_config["model_name"],
        version=deployment_config["approved_model_version"]
    )

    model_verified = True

    print("Registered model: FOUND")
    print("Model name:", model_version.name)
    print("Model version:", model_version.version)
    print("Model status:", model_version.status)

except Exception as e:

    print("Registered model: NOT VERIFIED")
    print("Error:", str(e)[:300])


# --------------------------------------------------
# 4. Verify protected prediction table metadata
# --------------------------------------------------

print("\n4. PROTECTED PREDICTION TABLE")

try:
    detail = spark.sql(
        f"DESCRIBE DETAIL "
        f"{deployment_config['prediction_table']}"
    ).first()

    print("Table format:", detail["format"])
    print("Table location:", detail["location"])
    print("Table metadata: ACCESSIBLE")

except Exception as e:

    print("Table metadata: NOT VERIFIED")
    print("Error:", str(e)[:250])


# --------------------------------------------------
# 5. Check safety configuration
# --------------------------------------------------

print("\n5. DEPLOYMENT SAFETY CHECK")

assert deployment_config["allow_inference"] is False
assert deployment_config["allow_writes"] is False
assert deployment_config["allow_job_submission"] is False

print("Inference permitted by configuration: False")
print("Writes permitted by configuration: False")
print("Job submission permitted by configuration: False")
print("Safety configuration: PASS")

# These flags are declarative; they are not access controls.


# --------------------------------------------------
# 6. Summarize metadata readiness
# --------------------------------------------------

print("\n6. READINESS SUMMARY")

print(
    "Tables accessible:",
    sum(table_results.values()),
    "/",
    len(required_tables)
)

print(
    "Registered model verified:",
    model_verified
)

metadata_ready = (
    all(table_results.values())
    and model_verified
)

print(
    "Metadata readiness:",
    "PASS" if metadata_ready else "REVIEW REQUIRED"
)

print("Production deployment authorized: False")
print("Scheduled inference enabled: False")


# --------------------------------------------------
# 7. Safety confirmation
# --------------------------------------------------

print("\n7. SAFETY")

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 17.1 COMPLETE")


# COMMAND ----------


# STEP 17.2 — MODEL ARTIFACT AND DEPENDENCY READINESS
# READ-ONLY: No model loading, inference, writes or jobs

import mlflow
from mlflow import MlflowClient

print("=" * 70)
print("STEP 17.2 — MODEL ARTIFACT AND DEPENDENCY READINESS")
print("=" * 70)

model_name = deployment_config["model_name"]
model_version = deployment_config["approved_model_version"]

client = MlflowClient()

# --------------------------------------------------
# 1. Inspect registered model version
# --------------------------------------------------

print("\n1. REGISTERED MODEL METADATA")

mv = client.get_model_version(
    name=model_name,
    version=model_version
)

print("Model name:", mv.name)
print("Model version:", mv.version)
print("Model status:", mv.status)
print("Run ID:", mv.run_id or "NOT RECORDED")
print("Artifact source:", mv.source or "NOT RECORDED")

model_uri = f"models:/{model_name}/{model_version}"

print("Model URI:", model_uri)


# --------------------------------------------------
# 2. Inspect artifact directory
# --------------------------------------------------

print("\n2. MODEL ARTIFACT FILES")

artifact_files = []
artifact_accessible = False

try:
    artifact_files = client.list_artifacts(
        run_id=mv.run_id
    ) if mv.run_id else []

    print("Run artifact listing accessible: YES")
    print("Top-level run artifacts:")

    for item in artifact_files[:30]:
        print(" -", item.path)

    artifact_accessible = True

except Exception as e:
    print("Run artifact listing accessible: NO")
    print("Reason:", str(e)[:300])

# A run artifact listing is not proof that every
# registered-model artifact is accessible.


# --------------------------------------------------
# 3. Inspect MLmodel metadata without loading model
# --------------------------------------------------

print("\n3. MODEL FLAVOR AND DEPENDENCY METADATA")

flavor_verified = False
dependencies_found = False

try:
    from mlflow.models import get_model_info

    info = get_model_info(model_uri)

    flavors = list(info.flavors.keys())

    print("Model flavors:", flavors)
    print("Model signature:", str(info.signature)[:500])

    flavor_verified = len(flavors) > 0

    # Inspect declared environment metadata
    for flavor_name, flavor_config in info.flavors.items():

        print("\nFlavor:", flavor_name)

        for key in [
            "loader_module",
            "python_version",
            "env",
            "code",
            "model_data",
            "pyspark_version"
        ]:
            if key in flavor_config:
                print(f"  {key}:", flavor_config[key])

    # A declared environment reference is evidence
    # of dependency metadata, not environment compatibility.
    dependencies_found = any(
        isinstance(cfg, dict)
        and (
            "env" in cfg
            or "python_env" in cfg
        )
        for cfg in info.flavors.values()
    )

    print("\nEnvironment reference found:", dependencies_found)

except Exception as e:
    print("Model metadata inspection: NOT VERIFIED")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 4. Readiness assessment
# --------------------------------------------------

print("\n4. READINESS ASSESSMENT")

checks = {
    "Registered model READY": mv.status == "READY",
    "Artifact source recorded": bool(mv.source),
    "MLflow run ID recorded": bool(mv.run_id),
    "Run artifact listing accessible": artifact_accessible,
    "Model flavor metadata accessible": flavor_verified,
    "Environment reference declared": dependencies_found
}

for name, passed in checks.items():
    print(
        f"{name}:",
        "PASS" if passed else "REVIEW REQUIRED"
    )

print(
    "\nMetadata checks passed:",
    sum(checks.values()),
    "/",
    len(checks)
)

if all(checks.values()):
    print("Model metadata readiness: PASS")
else:
    print("Model metadata readiness: REVIEW REQUIRED")

print("Runtime dependency compatibility: NOT YET VERIFIED")
print("Model loading: NOT EXECUTED")
print("Inference: NOT EXECUTED")


# --------------------------------------------------
# 5. Safety confirmation
# --------------------------------------------------

print("\n5. SAFETY")

assert deployment_config["allow_inference"] is False
assert deployment_config["allow_writes"] is False
assert deployment_config["allow_job_submission"] is False

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")
print("Deployment activated: False")

print("\nSTEP 17.2 COMPLETE")


# COMMAND ----------


# STEP 17.3 — RUNTIME AND DEPENDENCY COMPATIBILITY
# READ-ONLY: No inference, jobs, or writes

import sys
import platform
import importlib.metadata
from mlflow import MlflowClient
from mlflow.models import get_model_info

print("=" * 70)
print("STEP 17.3 — RUNTIME AND DEPENDENCY COMPATIBILITY")
print("=" * 70)

# --------------------------------------------------
# 1. Current notebook runtime
# --------------------------------------------------

print("\n1. CURRENT NOTEBOOK RUNTIME")

current_python = platform.python_version()
current_spark = spark.version

print("Python version:", current_python)
print("Spark version:", current_spark)

packages = [
    "mlflow",
    "pyspark",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow"
]

installed_packages = {}

for package in packages:

    try:
        version = importlib.metadata.version(package)
        installed_packages[package] = version

    except importlib.metadata.PackageNotFoundError:
        version = "NOT INSTALLED"
        installed_packages[package] = None

    print(f"{package}: {version}")


# --------------------------------------------------
# 2. Registered model requirements
# --------------------------------------------------

print("\n2. REGISTERED MODEL REQUIREMENTS")

model_uri = (
    f"models:/{deployment_config['model_name']}/"
    f"{deployment_config['approved_model_version']}"
)

info = get_model_info(model_uri)

pyfunc = info.flavors.get("python_function", {})
spark_flavor = info.flavors.get("spark", {})

required_python = pyfunc.get("python_version")
required_spark = spark_flavor.get("pyspark_version")

print("Model Python version:", required_python)
print("Model PySpark version:", required_spark)


# --------------------------------------------------
# 3. Compare versions
# --------------------------------------------------

print("\n3. VERSION COMPARISON")

python_exact_match = (
    current_python == required_python
)

spark_exact_match = (
    current_spark == required_spark
)

print(
    "Python exact match:",
    "YES" if python_exact_match else "NO"
)

print(
    "Spark exact match:",
    "YES" if spark_exact_match else "NO"
)

if not python_exact_match:
    print(
        "Python compatibility:",
        "REQUIRES FURTHER VERIFICATION"
    )

if not spark_exact_match:
    print(
        "Spark compatibility:",
        "REQUIRES FURTHER VERIFICATION"
    )


# --------------------------------------------------
# 4. Inspect Databricks runtime metadata
# --------------------------------------------------

print("\n4. DATABRICKS RUNTIME METADATA")

try:
    runtime_version = spark.conf.get(
        "spark.databricks.clusterUsageTags.sparkVersion"
    )

    print("Databricks Runtime:", runtime_version)

except Exception:
    print("Databricks Runtime: NOT AVAILABLE")

try:
    cluster_id = spark.conf.get(
        "spark.databricks.clusterUsageTags.clusterId"
    )

    print("Cluster ID:", cluster_id)

except Exception:
    print("Cluster ID: NOT AVAILABLE")


# --------------------------------------------------
# 5. Compatibility assessment
# --------------------------------------------------

print("\n5. COMPATIBILITY ASSESSMENT")

print("Python exact match:", python_exact_match)
print("Spark exact match:", spark_exact_match)

if python_exact_match and spark_exact_match:
    print("Version alignment: EXACT MATCH")
else:
    print("Version alignment: REVIEW REQUIRED")

print("Runtime compatibility: NOT YET CERTIFIED")
print("Model loading: NOT EXECUTED")
print("Inference: NOT EXECUTED")


# --------------------------------------------------
# 6. Safety confirmation
# --------------------------------------------------

print("\n6. SAFETY")

assert deployment_config["allow_inference"] is False
assert deployment_config["allow_writes"] is False
assert deployment_config["allow_job_submission"] is False

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected tables modified: False")
print("Deployment activated: False")

print("\nSTEP 17.3 COMPLETE")


# COMMAND ----------


# STEP 17.4 — PROTECTED JOB CONFIGURATION INSPECTION
# READ-ONLY: Uses Databricks Jobs GET API only

import json
from databricks.sdk import WorkspaceClient

print("=" * 70)
print("STEP 17.4 — PROTECTED JOB CONFIGURATION")
print("=" * 70)

job_id = int(deployment_config["protected_job_id"])

# --------------------------------------------------
# 1. Retrieve existing Job metadata
# --------------------------------------------------

w = WorkspaceClient()

job = w.jobs.get(job_id=job_id)

settings = job.settings

print("\n1. JOB INFORMATION")
print("Job ID:", job.job_id)
print("Job name:", settings.name)
print("Job metadata: ACCESSIBLE")


# --------------------------------------------------
# 2. Inspect tasks and compute configuration
# --------------------------------------------------

print("\n2. JOB TASKS")

tasks = settings.tasks or []

for task in tasks:

    print("\nTask key:", task.task_key)

    if task.notebook_task:
        print(
            "Notebook path:",
            task.notebook_task.notebook_path
        )

    print(
        "Existing cluster ID:",
        task.existing_cluster_id
    )

    print(
        "Job cluster key:",
        task.job_cluster_key
    )

    if task.new_cluster:
        print(
            "New cluster configuration:",
            json.dumps(
                task.new_cluster.as_dict(),
                indent=2,
                default=str
            )[:3000]
        )

    if task.environment_key:
        print(
            "Serverless environment key:",
            task.environment_key
        )


# --------------------------------------------------
# 3. Inspect Job-level compute definitions
# --------------------------------------------------

print("\n3. JOB-LEVEL COMPUTE")

for cluster in settings.job_clusters or []:

    print("Job cluster key:", cluster.job_cluster_key)

    if cluster.new_cluster:
        print(
            json.dumps(
                cluster.new_cluster.as_dict(),
                indent=2,
                default=str
            )[:3000]
        )

print("\nServerless environments:")

for environment in settings.environments or []:

    print(
        json.dumps(
            environment.as_dict(),
            indent=2,
            default=str
        )[:3000]
    )


# --------------------------------------------------
# 4. Inspect scheduling and execution settings
# --------------------------------------------------

print("\n4. SCHEDULING AND EXECUTION")

if settings.schedule:
    print(
        "Schedule:",
        json.dumps(
            settings.schedule.as_dict(),
            indent=2,
            default=str
        )
    )
else:
    print("Schedule: NOT CONFIGURED")

if settings.trigger:
    print(
        "Trigger:",
        json.dumps(
            settings.trigger.as_dict(),
            indent=2,
            default=str
        )
    )
else:
    print("Trigger: NOT CONFIGURED")

print("Max concurrent runs:", settings.max_concurrent_runs)
print("Timeout seconds:", settings.timeout_seconds)


# --------------------------------------------------
# 5. Inspect task execution safeguards
# --------------------------------------------------

print("\n5. TASK EXECUTION SETTINGS")

for task in tasks:

    print("\nTask:", task.task_key)
    print("Max retries:", task.max_retries)
    print("Retry interval:", task.min_retry_interval_millis)
    print("Timeout seconds:", task.timeout_seconds)


# --------------------------------------------------
# 6. Safety confirmation
# --------------------------------------------------

print("\n6. SAFETY")

print("API operation: jobs.get")
print("Job submitted: False")
print("Job configuration modified: False")
print("Schedule enabled: False")
print("Inference executed: False")
print("Delta writes performed: False")
print("Protected tables modified: False")

print("\nSTEP 17.4 COMPLETE")


# COMMAND ----------


# STEP 17.5 — SERVERLESS RUNTIME COMPATIBILITY
# READ-ONLY: No job submissions, writes, or inference

import platform
import importlib.metadata

from databricks.sdk import WorkspaceClient
from mlflow.models import get_model_info

print("=" * 70)
print("STEP 17.5 — SERVERLESS RUNTIME COMPATIBILITY")
print("=" * 70)

# --------------------------------------------------
# 1. Inspect protected Job environment
# --------------------------------------------------

w = WorkspaceClient()

job = w.jobs.get(
    job_id=int(deployment_config["protected_job_id"])
)

print("\n1. PROTECTED JOB SERVERLESS ENVIRONMENT")

environments = job.settings.environments or []

environment_details = {}

for env in environments:

    details = env.as_dict()
    environment_details[env.environment_key] = details

    print("Environment key:", env.environment_key)
    print(
        "Environment version:",
        details.get("spec", {}).get("environment_version")
    )

    dependencies = (
        details.get("spec", {}).get("dependencies", [])
    )

    print("Configured dependencies:", dependencies)


# --------------------------------------------------
# 2. Inspect current notebook environment
# --------------------------------------------------

print("\n2. CURRENT NOTEBOOK ENVIRONMENT")

current_python = platform.python_version()

try:
    current_pyspark = importlib.metadata.version("pyspark")
except importlib.metadata.PackageNotFoundError:
    current_pyspark = "NOT INSTALLED"

try:
    current_mlflow = importlib.metadata.version("mlflow")
except importlib.metadata.PackageNotFoundError:
    current_mlflow = "NOT INSTALLED"

print("Python:", current_python)
print("PySpark:", current_pyspark)
print("MLflow:", current_mlflow)

print(
    "Spark session version:",
    spark.version
)


# --------------------------------------------------
# 3. Inspect registered model requirements
# --------------------------------------------------

print("\n3. REGISTERED MODEL REQUIREMENTS")

model_uri = (
    f"models:/{deployment_config['model_name']}/"
    f"{deployment_config['approved_model_version']}"
)

info = get_model_info(model_uri)

pyfunc = info.flavors.get("python_function", {})
spark_flavor = info.flavors.get("spark", {})

model_python = pyfunc.get("python_version")
model_pyspark = spark_flavor.get("pyspark_version")

print("Model Python:", model_python)
print("Model PySpark:", model_pyspark)
print("Model flavors:", list(info.flavors.keys()))


# --------------------------------------------------
# 4. Evaluate metadata alignment
# --------------------------------------------------

print("\n4. COMPATIBILITY ASSESSMENT")

python_match = current_python == model_python
spark_match = spark.version == model_pyspark

print("Notebook Python matches model:", python_match)
print("Notebook Spark matches model:", spark_match)

task_environment_keys = {
    task.environment_key
    for task in (job.settings.tasks or [])
    if task.environment_key
}

defined_environment_keys = set(environment_details)

environment_keys_valid = (
    bool(task_environment_keys)
    and task_environment_keys.issubset(
        defined_environment_keys
    )
)

print(
    "Task environment references valid:",
    environment_keys_valid
)

print(
    "Serverless environment version:",
    [
        details.get("spec", {}).get("environment_version")
        for details in environment_details.values()
    ]
)

print(
    "Serverless runtime compatibility:",
    "NOT YET VERIFIED"
)

print(
    "Reason:",
    "Job runtime and model artifact dependencies "
    "have not been tested together."
)


# --------------------------------------------------
# 5. Deployment safety
# --------------------------------------------------

print("\n5. SAFETY")

assert deployment_config["allow_inference"] is False
assert deployment_config["allow_writes"] is False
assert deployment_config["allow_job_submission"] is False

print("Protected Job modified: False")
print("Job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")
print("Schedule enabled: False")

print("\nSTEP 17.5 COMPLETE")


# COMMAND ----------


# STEP 17.5A — ENVIRONMENT DIAGNOSTICS
# READ-ONLY: No inference, jobs, or writes

import importlib.metadata
import platform

print("=" * 65)
print("STEP 17.5A — ENVIRONMENT DIAGNOSTICS")
print("=" * 65)

print("\n1. PYTHON")
print(platform.python_version())

print("\n2. SPARK")
print("Spark session:", spark.version)

print("\n3. PYSPARK DISTRIBUTION")

try:
    print(
        importlib.metadata.version("pyspark")
    )
except Exception as e:
    print("Error:", str(e)[:200])

print("\n4. MLFLOW IMPORT")

try:
    import mlflow

    print("MLflow import: SUCCESS")
    print(
        "MLflow version:",
        getattr(mlflow, "__version__", "UNKNOWN")
    )
    print(
        "MLflow module location:",
        getattr(mlflow, "__file__", "UNKNOWN")
    )

except Exception as e:
    print("MLflow import: FAILED")
    print("Error:", str(e)[:300])

print("\n5. INSTALLED MLFLOW DISTRIBUTIONS")

for package in [
    "mlflow",
    "mlflow-skinny",
    "mlflow-tracing"
]:
    try:
        print(
            package,
            importlib.metadata.version(package)
        )
    except importlib.metadata.PackageNotFoundError:
        print(package, "NOT FOUND")

print("\n6. SAFETY")
print("Model loading: NOT EXECUTED")
print("Inference: NOT EXECUTED")
print("Job submission: NOT EXECUTED")
print("Protected assets modified: False")

print("\nSTEP 17.5A COMPLETE")


# COMMAND ----------


# STEP 17.6 — MODEL DEPENDENCY INSPECTION
# READ-ONLY: Inspect registered model requirements

import mlflow
from mlflow.models import get_model_info
from mlflow.pyfunc import get_model_dependencies

print("=" * 70)
print("STEP 17.6 — MODEL DEPENDENCY INSPECTION")
print("=" * 70)

model_uri = (
    f"models:/{deployment_config['model_name']}/"
    f"{deployment_config['approved_model_version']}"
)

# --------------------------------------------------
# 1. Registered model metadata
# --------------------------------------------------

print("\n1. REGISTERED MODEL")

info = get_model_info(model_uri)

print("Model URI:", model_uri)
print("Flavors:", list(info.flavors.keys()))

# --------------------------------------------------
# 2. Inspect dependency specifications
# --------------------------------------------------

print("\n2. DECLARED DEPENDENCIES")

for dependency_format in ["pip", "conda"]:

    print(f"\nFormat: {dependency_format}")

    try:
        dependency_path = get_model_dependencies(
            model_uri,
            format=dependency_format
        )

        # This API may download artifacts to a local
        # cache. Do not use it under strict no-file-creation
        # requirements.
        print("Dependency metadata path:", dependency_path)

        with open(dependency_path, "r") as f:
            content = f.read()

        print(content[:6000])

    except Exception as e:
        print(
            "Dependency inspection unavailable:",
            str(e)[:300]
        )

# --------------------------------------------------
# 3. Safety
# --------------------------------------------------

print("\n3. SAFETY")

print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta tables modified: False")
print("Protected assets modified: False")

print("\nSTEP 17.6 COMPLETE")


# COMMAND ----------

# STEP 17.6A — REGISTERED MODEL METADATA CHECK
# READ-ONLY: No artifact downloads or inference

from mlflow import MlflowClient

print("=" * 65)
print("STEP 17.6A — MODEL METADATA CHECK")
print("=" * 65)

try:
    client = MlflowClient()

    mv = client.get_model_version(
        name=deployment_config["model_name"],
        version=deployment_config["approved_model_version"]
    )

    print("Model name:", mv.name)
    print("Model version:", mv.version)
    print("Model status:", mv.status)
    print("Run ID:", mv.run_id)
    print("Artifact source:", mv.source)
    print("Metadata inspection: PASS")

except Exception as e:
    print("Metadata inspection: FAILED")
    print("Error type:", type(e).__name__)
    print("Error:", str(e)[:500])

print("\nSAFETY")
print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Protected assets modified: False")

print("\nSTEP 17.6A COMPLETE")

# COMMAND ----------


# STEP 17.6B — MLFLOW RUN METADATA INSPECTION
# READ-ONLY: No artifact downloads or model loading

from mlflow import MlflowClient

print("=" * 70)
print("STEP 17.6B — MLFLOW RUN METADATA")
print("=" * 70)

client = MlflowClient()

run_id = "b45de5f13b7b4aa1aaa2318045b4a03f"

# --------------------------------------------------
# 1. Retrieve training run metadata
# --------------------------------------------------

run = client.get_run(run_id)

print("\n1. TRAINING RUN INFORMATION")

print("Run ID:", run.info.run_id)
print("Run status:", run.info.status)
print("Artifact URI:", run.info.artifact_uri)


# --------------------------------------------------
# 2. Inspect recorded parameters
# --------------------------------------------------

print("\n2. MODEL PARAMETERS")

parameters = run.data.params

if parameters:
    for key, value in sorted(parameters.items()):
        print(f"{key}: {value}")
else:
    print("No parameters recorded")


# --------------------------------------------------
# 3. Inspect recorded metrics
# --------------------------------------------------

print("\n3. MODEL METRICS")

metrics = run.data.metrics

if metrics:
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")
else:
    print("No metrics recorded")


# --------------------------------------------------
# 4. Inspect useful training metadata
# --------------------------------------------------

print("\n4. TRAINING METADATA")

tags = run.data.tags

for key, value in sorted(tags.items()):
    if any(
        keyword in key.lower()
        for keyword in [
            "source",
            "model",
            "runtime",
            "python",
            "spark",
            "version",
            "runname"
        ]
    ):
        print(f"{key}: {str(value)[:300]}")


# --------------------------------------------------
# 5. Readiness summary
# --------------------------------------------------

print("\n5. METADATA SUMMARY")

print("Run metadata accessible: YES")
print("Parameters recorded:", len(parameters))
print("Metrics recorded:", len(metrics))
print("Tags recorded:", len(tags))
print("Dependency compatibility: NOT YET VERIFIED")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n6. SAFETY")

print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Artifact files downloaded: False")
print("Protected assets modified: False")

print("\nSTEP 17.6B COMPLETE")


# COMMAND ----------


# STEP 17.7 — STAGING DEPLOYMENT DESIGN
# READ-ONLY: No inference, writes, or job creation

print("=" * 70)
print("STEP 17.7 — STAGING DEPLOYMENT DESIGN")
print("=" * 70)

staging_config = {
    "environment": "STAGING",
    "model_name": deployment_config["model_name"],
    "model_version": deployment_config["approved_model_version"],
    "source_table": deployment_config["source_table"],

    "proposed_prediction_table":
        "workspace.default.telco_17_staging_predictions",

    "proposed_job_name":
        "Telco_Churn_Staging_Batch_Inference",

    "schedule_enabled": False,
    "allow_inference": False,
    "allow_writes": False,
    "allow_job_submission": False
}

# --------------------------------------------------
# 1. Display proposed configuration
# --------------------------------------------------

print("\n1. PROPOSED STAGING CONFIGURATION")

for key, value in staging_config.items():
    print(f"{key}: {value}")


# --------------------------------------------------
# 2. Protect existing output table
# --------------------------------------------------

print("\n2. OUTPUT ISOLATION CHECK")

protected_table = deployment_config["prediction_table"]
staging_table = staging_config["proposed_prediction_table"]

assert staging_table != protected_table

print("Protected table:", protected_table)
print("Proposed staging table:", staging_table)
print("Different table names: PASS")


# --------------------------------------------------
# 3. Check for existing staging table
# --------------------------------------------------

print("\n3. STAGING TABLE EXISTENCE CHECK")

try:
    staging_exists = spark.catalog.tableExists(
        staging_table
    )

    print("Staging table already exists:", staging_exists)

    if staging_exists:
        print(
            "Status: REVIEW REQUIRED — "
            "Do not overwrite an existing table"
        )
    else:
        print(
            "Status: AVAILABLE NAME — "
            "No table creation performed"
        )

except Exception as e:
    staging_exists = None
    print("Status: NOT VERIFIED")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 4. Verify staging safety settings
# --------------------------------------------------

print("\n4. STAGING SAFETY")

assert staging_config["schedule_enabled"] is False
assert staging_config["allow_inference"] is False
assert staging_config["allow_writes"] is False
assert staging_config["allow_job_submission"] is False

print("Schedule enabled: False")
print("Inference allowed: False")
print("Writes allowed: False")
print("Job submission allowed: False")


# --------------------------------------------------
# 5. Readiness summary
# --------------------------------------------------

print("\n5. DESIGN ASSESSMENT")

if staging_exists is False:
    print("Staging output name availability: PASS")
else:
    print("Staging output name availability: REVIEW REQUIRED")

print("Model dependency compatibility: PENDING")
print("Staging Job creation: NOT AUTHORIZED")
print("Staging inference execution: NOT AUTHORIZED")
print("Production scheduling: NOT AUTHORIZED")


# --------------------------------------------------
# 6. Safety confirmation
# --------------------------------------------------

print("\n6. SAFETY")

print("Job created: False")
print("Job submitted: False")
print("Inference executed: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.7 COMPLETE")


# COMMAND ----------


# STEP 17.8 — STAGING DEPLOYMENT RELEASE GATES
# READ-ONLY: In-memory assessment only

print("=" * 70)
print("STEP 17.8 — STAGING DEPLOYMENT RELEASE GATES")
print("=" * 70)

# --------------------------------------------------
# 1. Define release gates
# --------------------------------------------------

release_gates = {
    "Registered model metadata verified": True,
    "MLflow training run verified": True,
    "Protected Job configuration inspected": True,
    "Staging output isolated": True,
    "Staging table name available": True,

    "Model dependency compatibility verified": False,
    "Staging input schema validated": False,
    "Staging inference notebook reviewed": False,
    "Idempotent batch processing verified": False,
    "Staging output schema validated": False,
    "Staging smoke test completed": False,
    "Rollback procedure validated": False,
    "Staging execution explicitly authorized": False
}

# --------------------------------------------------
# 2. Display release gate status
# --------------------------------------------------

print("\n1. RELEASE GATE STATUS")

for name, passed in release_gates.items():

    status = "PASS" if passed else "PENDING"

    print(f"{name}: {status}")


# --------------------------------------------------
# 3. Calculate readiness
# --------------------------------------------------

print("\n2. RELEASE READINESS")

total_gates = len(release_gates)

passed_gates = sum(release_gates.values())

pending_gates = total_gates - passed_gates

print("Total release gates:", total_gates)
print("Passed:", passed_gates)
print("Pending:", pending_gates)

staging_release_ready = all(release_gates.values())

print(
    "Staging release ready:",
    staging_release_ready
)


# --------------------------------------------------
# 4. Identify next required action
# --------------------------------------------------

print("\n3. NEXT REQUIRED VALIDATION")

if not release_gates[
    "Model dependency compatibility verified"
]:
    print(
        "Priority: Verify model dependency compatibility"
    )

print(
    "Staging Job creation authorized: False"
)

print(
    "Staging inference authorized: False"
)

print(
    "Production scheduling authorized: False"
)


# --------------------------------------------------
# 5. Safety confirmation
# --------------------------------------------------

print("\n4. SAFETY")

assert staging_config["allow_inference"] is False
assert staging_config["allow_writes"] is False
assert staging_config["allow_job_submission"] is False

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.8 COMPLETE")


# COMMAND ----------


# STEP 17.9 — STAGING INPUT VALIDATION
# READ-ONLY: No inference, writes, or job submissions

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 17.9 — STAGING INPUT VALIDATION")
print("=" * 70)

source_table = staging_config["source_table"]

df = spark.table(source_table)

# --------------------------------------------------
# 1. Inspect source schema
# --------------------------------------------------

print("\n1. SOURCE SCHEMA")

print("Source table:", source_table)
print("Column count:", len(df.columns))

for field in df.schema.fields:
    print(
        f"{field.name}: {field.dataType.simpleString()}"
    )


# --------------------------------------------------
# 2. Validate expected columns
# --------------------------------------------------

print("\n2. REQUIRED COLUMN VALIDATION")

required_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

missing_columns = sorted(
    set(required_columns) - set(df.columns)
)

unexpected_columns = sorted(
    set(df.columns) - set(required_columns)
)

print("Expected columns:", len(required_columns))
print("Missing columns:", missing_columns)
print("Unexpected columns:", unexpected_columns)

schema_valid = (
    not missing_columns
    and not unexpected_columns
)

print(
    "Column-name validation:",
    "PASS" if schema_valid else "REVIEW REQUIRED"
)


# --------------------------------------------------
# 3. Validate customer identifiers
# --------------------------------------------------

print("\n3. CUSTOMER IDENTIFIER VALIDATION")

row_count = df.count()

invalid_customer_ids = df.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID").cast("string")) == "")
).count()

duplicate_customer_ids = (
    df.groupBy("customerID")
      .count()
      .filter(F.col("count") > 1)
      .count()
)

print("Total rows:", row_count)
print("Invalid customer IDs:", invalid_customer_ids)
print("Duplicate customer IDs:", duplicate_customer_ids)


# --------------------------------------------------
# 4. Validate numerical features
# --------------------------------------------------

print("\n4. NUMERICAL FEATURE VALIDATION")

numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

for feature in numerical_features:

    if feature not in df.columns:
        print(feature, "MISSING COLUMN")
        continue

    numeric_value = F.expr(
        f"try_cast(`{feature}` AS DOUBLE)"
    )

    invalid_count = df.filter(
        F.col(feature).isNotNull()
        & (F.trim(F.col(feature).cast("string")) != "")
        & numeric_value.isNull()
    ).count()

    missing_count = df.filter(
        F.col(feature).isNull()
        | (F.trim(F.col(feature).cast("string")) == "")
    ).count()

    print(f"\nFeature: {feature}")
    print("Invalid numeric values:", invalid_count)
    print("Missing values:", missing_count)


# --------------------------------------------------
# 5. Validate Contract categories
# --------------------------------------------------

print("\n5. CONTRACT CATEGORY VALIDATION")

allowed_contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

unexpected_contracts = (
    df.filter(
        F.col("Contract").isNotNull()
        & ~F.col("Contract").isin(allowed_contracts)
    )
    .count()
)

missing_contracts = df.filter(
    F.col("Contract").isNull()
).count()

print("Unexpected Contract values:", unexpected_contracts)
print("Missing Contract values:", missing_contracts)


# --------------------------------------------------
# 6. Summarize validation
# --------------------------------------------------

print("\n6. VALIDATION SUMMARY")

checks = {
    "Expected column names": schema_valid,
    "Nonempty source": row_count > 0,
    "Valid customer IDs": invalid_customer_ids == 0,
    "Unique customer IDs": duplicate_customer_ids == 0,
    "Valid Contract categories": (
        unexpected_contracts == 0
        and missing_contracts == 0
    )
}

for name, passed in checks.items():
    print(
        f"{name}:",
        "PASS" if passed else "REVIEW REQUIRED"
    )

print(
    "Summary checks passed:",
    sum(checks.values()),
    "/",
    len(checks)
)

print(
    "Full model input compatibility:",
    "NOT YET VERIFIED"
)


# --------------------------------------------------
# 7. Safety
# --------------------------------------------------

print("\n7. SAFETY")

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.9 COMPLETE")


# COMMAND ----------


# STEP 17.10 — PREPROCESSING ARTIFACT VALIDATION
# READ-ONLY: Metadata inspection only
# No pipeline loading, transformation, inference, or writes

print("=" * 70)
print("STEP 17.10 — PREPROCESSING ARTIFACT VALIDATION")
print("=" * 70)

pipeline_path = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

expected_feature_count = 60

# --------------------------------------------------
# 1. Inspect pipeline storage location
# --------------------------------------------------

print("\n1. PREPROCESSING PIPELINE LOCATION")
print("Pipeline path:", pipeline_path)

pipeline_accessible = False
pipeline_files = []

try:
    pipeline_files = dbutils.fs.ls(pipeline_path)
    pipeline_accessible = True

    print("Pipeline directory accessible: YES")
    print("Top-level entries:")

    for item in pipeline_files[:30]:
        print(" -", item.name)

except Exception as e:
    print("Pipeline directory accessible: NO")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 2. Inspect Spark ML pipeline metadata
# --------------------------------------------------

print("\n2. PIPELINE METADATA")

metadata_path = pipeline_path + "/metadata"

metadata_available = False
pipeline_metadata = None

try:
    metadata_entries = dbutils.fs.ls(metadata_path)

    metadata_files = [
        item.path
        for item in metadata_entries
        if item.name.startswith("part-")
    ]

    if metadata_files:
        import json

        metadata_text = dbutils.fs.head(
            metadata_files[0],
            100000
        )

        pipeline_metadata = json.loads(metadata_text)
        metadata_available = True

        print("Metadata available: YES")
        print(
            "Pipeline class:",
            pipeline_metadata.get("class")
        )
        print(
            "Spark version:",
            pipeline_metadata.get("sparkVersion")
        )
        print(
            "Pipeline UID:",
            pipeline_metadata.get("uid")
        )

    else:
        print("Metadata available: NO")

except Exception as e:
    print("Metadata inspection: NOT VERIFIED")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 3. Inspect saved stage directories
# --------------------------------------------------

print("\n3. SAVED PIPELINE STAGES")

stage_path = pipeline_path + "/stages"

stage_names = []

try:
    stage_entries = dbutils.fs.ls(stage_path)

    stage_names = sorted(
        item.name.rstrip("/")
        for item in stage_entries
    )

    print("Saved stage directories:", len(stage_names))

    for name in stage_names:
        print(" -", name)

except Exception as e:
    print("Stage inspection: NOT VERIFIED")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 4. Feature compatibility assessment
# --------------------------------------------------

print("\n4. FEATURE COMPATIBILITY")

print("Expected model feature count:", expected_feature_count)

print(
    "Saved stage directories:",
    len(stage_names)
)

print(
    "Feature vector length:",
    "NOT VERIFIED FROM METADATA"
)

print(
    "Pipeline transformation:",
    "NOT EXECUTED"
)

print(
    "Model compatibility:",
    "PENDING"
)


# --------------------------------------------------
# 5. Readiness summary
# --------------------------------------------------

print("\n5. READINESS SUMMARY")

print("Pipeline directory accessible:", pipeline_accessible)
print("Pipeline metadata available:", metadata_available)
print("Saved stage count:", len(stage_names))

if pipeline_accessible and metadata_available and stage_names:
    print("Preprocessing artifact metadata: PASS")
else:
    print("Preprocessing artifact metadata: REVIEW REQUIRED")

print("End-to-end feature compatibility: NOT YET VERIFIED")


# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n6. SAFETY")

print("Pipeline loaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.10 COMPLETE")


# COMMAND ----------


# STEP 17.11 — PREPROCESSING FEATURE COMPATIBILITY
# READ-ONLY: Saved metadata inspection only

import json

print("=" * 70)
print("STEP 17.11 — PREPROCESSING FEATURE COMPATIBILITY")
print("=" * 70)

pipeline_path = (
    "/Volumes/workspace/default/telco_data/"
    "telco_preprocessing_pipeline"
)

# --------------------------------------------------
# 1. Helper: read Spark ML stage metadata
# --------------------------------------------------

def read_stage_metadata(stage_directory):

    metadata_directory = (
        pipeline_path
        + "/stages/"
        + stage_directory
        + "/metadata"
    )

    entries = dbutils.fs.ls(metadata_directory)

    part_files = [
        item.path
        for item in entries
        if item.name.startswith("part-")
    ]

    if not part_files:
        raise ValueError(
            f"No metadata part file: {stage_directory}"
        )

    return json.loads(
        dbutils.fs.head(part_files[0], 100000)
    )


# --------------------------------------------------
# 2. Inspect feature-producing stages
# --------------------------------------------------

print("\n1. FEATURE TRANSFORMATION STAGES")

stage_directories = [
    item.name.rstrip("/")
    for item in dbutils.fs.ls(pipeline_path + "/stages")
]

selected_stages = [
    name for name in stage_directories
    if name.startswith((
        "00_",
        "16_",
        "17_",
        "18_",
        "19_"
    ))
]

stage_metadata = {}

for stage_name in sorted(selected_stages):

    try:
        metadata = read_stage_metadata(stage_name)
        stage_metadata[stage_name] = metadata

        params = metadata.get("paramMap", {})

        print("\nStage:", stage_name)
        print("Class:", metadata.get("class"))

        for key in [
            "inputCols",
            "inputCol",
            "outputCols",
            "outputCol",
            "handleInvalid",
            "withMean",
            "withStd"
        ]:
            if key in params:
                print(f"{key}: {params[key]}")

    except Exception as e:
        print("\nStage:", stage_name)
        print("Metadata: NOT VERIFIED")
        print("Reason:", str(e)[:250])


# --------------------------------------------------
# 3. Inspect final feature assembler
# --------------------------------------------------

print("\n2. FINAL FEATURE ASSEMBLER")

final_stage_names = [
    name for name in stage_metadata
    if name.startswith("19_")
]

final_output = None
final_inputs = []

if final_stage_names:

    final_metadata = stage_metadata[
        final_stage_names[0]
    ]

    final_params = final_metadata.get("paramMap", {})

    final_output = final_params.get("outputCol")
    final_inputs = final_params.get("inputCols", [])

    print("Output column:", final_output)
    print("Input columns:", final_inputs)

else:
    print("Final assembler metadata unavailable")


# --------------------------------------------------
# 4. Feature compatibility assessment
# --------------------------------------------------

print("\n3. COMPATIBILITY ASSESSMENT")

print("Expected model feature count: 60")
print("Final assembler output:", final_output)

print(
    "Final output named features:",
    final_output == "features"
)

print(
    "Feature vector length:",
    "NOT VERIFIED"
)

print(
    "Registered model input representation:",
    "Array(double)"
)

print(
    "Spark vector / array compatibility:",
    "NOT VERIFIED"
)

print(
    "End-to-end model compatibility:",
    "PENDING"
)


# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n4. SAFETY")

print("Pipeline loaded: False")
print("Data transformed: False")
print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.11 COMPLETE")


# COMMAND ----------


# STEP 17.12 — MODEL INPUT CONTRACT VALIDATION
# READ-ONLY: Inspect saved metadata and registered model signature

import json
from mlflow import MlflowClient

print("=" * 70)
print("STEP 17.12 — MODEL INPUT CONTRACT VALIDATION")
print("=" * 70)

# --------------------------------------------------
# 1. Read fitted OneHotEncoder metadata
# --------------------------------------------------

print("\n1. ONE-HOT ENCODER DIMENSIONS")

encoder_stage = next(
    name for name in stage_directories
    if name.startswith("16_OneHotEncoder")
)

encoder_metadata = read_stage_metadata(encoder_stage)

encoder_params = encoder_metadata.get("paramMap", {})

print("Encoder class:", encoder_metadata.get("class"))
print("Input columns:", len(encoder_params.get("inputCols", [])))
print("Output columns:", len(encoder_params.get("outputCols", [])))
print("dropLast:", encoder_params.get(
    "dropLast",
    encoder_metadata.get("defaultParamMap", {}).get(
        "dropLast", "NOT RECORDED"
    )
))

# --------------------------------------------------
# 2. Read registered model version metadata
# --------------------------------------------------

print("\n2. REGISTERED MODEL IDENTITY")

client = MlflowClient()

model_version = client.get_model_version(
    name=deployment_config["model_name"],
    version=deployment_config["approved_model_version"]
)

print("Model:", model_version.name)
print("Version:", model_version.version)
print("Run ID:", model_version.run_id)
print("Status:", model_version.status)

# --------------------------------------------------
# 3. Compare available input contract evidence
# --------------------------------------------------

print("\n3. FEATURE CONTRACT")

expected_features = 60

final_metadata = stage_metadata[
    next(
        name for name in stage_metadata
        if name.startswith("19_")
    )
]

final_params = final_metadata.get("paramMap", {})

print("Expected feature count:", expected_features)
print("Final assembler output:", final_params.get("outputCol"))
print("Final assembler input columns:",
      len(final_params.get("inputCols", [])))

print("Actual fitted vector dimension: NOT YET VERIFIED")
print("Registered signature (previously inspected): Array(double)")
print("Spark assembler output representation: Vector")
print("Vector-to-array handling: NOT YET VERIFIED")

# --------------------------------------------------
# 4. Release gate assessment
# --------------------------------------------------

print("\n4. RELEASE GATE ASSESSMENT")

print("Model identity metadata: VERIFIED")
print("Final feature column name: VERIFIED")
print("Fitted encoder output dimensions: PENDING")
print("Exact 60-feature vector: PENDING")
print("Vector/array compatibility: PENDING")
print("End-to-end compatibility: NOT VERIFIED")

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Model loaded: False")
print("Pipeline loaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.12 COMPLETE")


# COMMAND ----------


# STEP 17.13 — FITTED ENCODER DIMENSION INSPECTION
# READ-ONLY: No model loading or inference

print("=" * 70)
print("STEP 17.13 — FITTED ENCODER DIMENSIONS")
print("=" * 70)

encoder_stage = next(
    name for name in stage_directories
    if name.startswith("16_OneHotEncoder")
)

encoder_path = (
    pipeline_path + "/stages/" + encoder_stage
)

# --------------------------------------------------
# 1. Inspect saved encoder directory
# --------------------------------------------------

print("\n1. ENCODER ARTIFACT STRUCTURE")

entries = dbutils.fs.ls(encoder_path)

for item in entries:
    print(item.name)

# --------------------------------------------------
# 2. Inspect fitted category-size data
# --------------------------------------------------

print("\n2. FITTED CATEGORY SIZES")

category_sizes = None

try:
    data_path = encoder_path + "/data"

    # Spark reads the fitted encoder's saved Parquet
    # data. No transformation or writes occur.
    encoder_data = spark.read.parquet(data_path)

    print("Saved data columns:", encoder_data.columns)

    rows = encoder_data.limit(5).collect()

    for row in rows:
        record = row.asDict(recursive=True)

        if "categorySizes" in record:
            category_sizes = record["categorySizes"]

    if category_sizes is not None:
        print("Category sizes:", category_sizes)
        print("Categorical features:", len(category_sizes))
    else:
        print("Category sizes: NOT FOUND")

except Exception as e:
    print("Category-size inspection: NOT VERIFIED")
    print("Reason:", str(e)[:300])

# --------------------------------------------------
# 3. Dimension assessment
# --------------------------------------------------

print("\n3. FEATURE DIMENSION ASSESSMENT")

print("Numerical feature count:", 4)
print("Expected final feature count:", 60)

if category_sizes is not None:

    print("Sum of fitted category sizes:",
          sum(category_sizes))

    print(
        "Naive dropLast dimension estimate:",
        4 + sum(category_sizes) - len(category_sizes)
    )

    print(
        "Actual encoder output dimension:",
        "NOT YET VERIFIED"
    )

else:
    print("Dimension estimate: UNAVAILABLE")

print(
    "Exact final feature vector dimension:",
    "PENDING"
)

# --------------------------------------------------
# 4. Safety
# --------------------------------------------------

print("\n4. SAFETY")

print("Pipeline loaded: False")
print("Model loaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.13 COMPLETE")


# COMMAND ----------


# STEP 17.14 — ENCODER CONFIGURATION REVIEW
# READ-ONLY: Inspect existing metadata only

import json

print("=" * 70)
print("STEP 17.14 — ENCODER CONFIGURATION REVIEW")
print("=" * 70)

# --------------------------------------------------
# 1. Inspect fitted encoder parameters
# --------------------------------------------------

print("\n1. ENCODER PARAMETERS")

encoder_stage = next(
    name for name in stage_directories
    if name.startswith("16_OneHotEncoder")
)

metadata = read_stage_metadata(encoder_stage)

param_map = metadata.get("paramMap", {})
default_map = metadata.get("defaultParamMap", {})

for parameter in [
    "dropLast",
    "handleInvalid",
    "inputCols",
    "outputCols"
]:
    value = param_map.get(
        parameter,
        default_map.get(parameter, "NOT RECORDED")
    )

    print(f"{parameter}: {value}")

# --------------------------------------------------
# 2. Inspect category dimensions
# --------------------------------------------------

print("\n2. FITTED CATEGORY DIMENSIONS")

sizes = category_sizes

print("Category sizes:", sizes)
print("Number of categorical columns:", len(sizes))
print("Sum of category sizes:", sum(sizes))

# --------------------------------------------------
# 3. Calculate dimensions under both assumptions
# --------------------------------------------------

print("\n3. FEATURE DIMENSION SCENARIOS")

numerical_count = 4

drop_last = param_map.get(
    "dropLast",
    default_map.get("dropLast", True)
)

handle_invalid = param_map.get(
    "handleInvalid",
    default_map.get("handleInvalid", "error")
)

# Scenario A: categorySizes already include
# the category handling needed for encoding.

widths_a = [
    size - (1 if drop_last else 0)
    for size in sizes
]

total_a = numerical_count + sum(widths_a)

# Scenario B: an additional invalid-category
# slot is required for every categorical input.

widths_b = [
    size
    + (1 if handle_invalid == "keep" else 0)
    - (1 if drop_last else 0)
    for size in sizes
]

total_b = numerical_count + sum(widths_b)

print("Scenario A total:", total_a)
print("Scenario B total:", total_b)
print("Expected training features:", 60)

print(
    "Scenario A matches 60:",
    total_a == 60
)

print(
    "Scenario B matches 60:",
    total_b == 60
)

# --------------------------------------------------
# 4. Assessment
# --------------------------------------------------

print("\n4. COMPATIBILITY ASSESSMENT")

print("Fitted category sizes inspected: YES")
print("Encoder parameters inspected: YES")
print("Actual transformed vector length: NOT VERIFIED")
print("Registered model internal stages: NOT VERIFIED")
print("End-to-end compatibility: PENDING")

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Pipeline loaded: False")
print("Model loaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.14 COMPLETE")


# COMMAND ----------


# STEP 17.15 — REGISTERED MODEL STRUCTURE REVIEW
# READ-ONLY: MLflow run metadata only
# No artifact downloads, model loading, or inference

import json
from mlflow import MlflowClient

print("=" * 70)
print("STEP 17.15 — REGISTERED MODEL STRUCTURE REVIEW")
print("=" * 70)

client = MlflowClient()

model_name = deployment_config["model_name"]
model_version_number = deployment_config["approved_model_version"]

# --------------------------------------------------
# 1. Confirm registered model identity
# --------------------------------------------------

print("\n1. REGISTERED MODEL")

model_version = client.get_model_version(
    name=model_name,
    version=model_version_number
)

print("Model:", model_version.name)
print("Version:", model_version.version)
print("Status:", model_version.status)
print("Run ID:", model_version.run_id)

# --------------------------------------------------
# 2. Retrieve training run metadata
# --------------------------------------------------

print("\n2. MODEL LOGGING HISTORY")

run = client.get_run(model_version.run_id)

history_text = run.data.tags.get(
    "mlflow.log-model.history"
)

model_history = []

if history_text:
    try:
        model_history = json.loads(history_text)
        print("Logged model entries:", len(model_history))
    except Exception as e:
        print("History parsing: NOT VERIFIED")
        print("Reason:", str(e)[:250])
else:
    print("Model logging history: NOT FOUND")

# --------------------------------------------------
# 3. Inspect model classes and flavors
# --------------------------------------------------

print("\n3. LOGGED MODEL STRUCTURE")

for index, entry in enumerate(model_history, start=1):

    print(f"\nLogged model {index}")
    print("Artifact path:", entry.get("artifact_path"))

    flavors = entry.get("flavors", {})

    spark_flavor = flavors.get("spark", {})
    pyfunc_flavor = flavors.get("python_function", {})

    print(
        "Spark model class:",
        spark_flavor.get("model_class", "NOT RECORDED")
    )

    print(
        "Spark version:",
        spark_flavor.get("pyspark_version", "NOT RECORDED")
    )

    print(
        "Python function loader:",
        pyfunc_flavor.get("loader_module", "NOT RECORDED")
    )

    print(
        "Spark model data path:",
        spark_flavor.get("model_data", "NOT RECORDED")
    )

# --------------------------------------------------
# 4. Assess preprocessing inclusion
# --------------------------------------------------

print("\n4. PREPROCESSING INCLUSION ASSESSMENT")

model_classes = [
    entry.get("flavors", {})
         .get("spark", {})
         .get("model_class")
    for entry in model_history
]

if "pyspark.ml.pipeline.PipelineModel" in model_classes:
    print("Registered model class: PipelineModel")
    print("Internal pipeline stages: NOT VERIFIED")
    print("Preprocessing included: NOT VERIFIED")
else:
    print("Registered PipelineModel class: NOT CONFIRMED")

print("Double preprocessing risk: NOT YET RULED OUT")
print("Deployment model contract: PENDING")

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Model loaded: False")
print("Artifact files downloaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.15 COMPLETE")


# COMMAND ----------


# STEP 17.16 — REGISTERED MODEL STAGE INSPECTION
# READ-ONLY: Directory listing only
# No model loading, inference, or writes

print("=" * 70)
print("STEP 17.16 — REGISTERED MODEL STAGE INSPECTION")
print("=" * 70)

# --------------------------------------------------
# 1. Obtain registered model source
# --------------------------------------------------

from mlflow import MlflowClient

client = MlflowClient()

model_version = client.get_model_version(
    name=deployment_config["model_name"],
    version=deployment_config["approved_model_version"]
)

model_source = model_version.source

print("\n1. REGISTERED MODEL SOURCE")
print("Source:", model_source)

# --------------------------------------------------
# 2. Locate Spark ML model directories
# --------------------------------------------------

print("\n2. SPARK MODEL STRUCTURE")

spark_model_path = model_source.rstrip("/") + "/sparkml"

print("Spark ML path:", spark_model_path)

structure_accessible = False
top_level_entries = []

try:
    top_level_entries = dbutils.fs.ls(spark_model_path)

    structure_accessible = True

    print("Spark ML directory accessible: YES")
    print("Top-level entries:")

    for item in top_level_entries:
        print(" -", item.name)

except Exception as e:
    print("Spark ML directory accessible: NO")
    print("Reason:", str(e)[:300])

# --------------------------------------------------
# 3. Inspect internal pipeline stages
# --------------------------------------------------

print("\n3. INTERNAL PIPELINE STAGES")

internal_stages = []

if structure_accessible:

    try:
        stage_path = spark_model_path + "/stages"

        stage_entries = dbutils.fs.ls(stage_path)

        internal_stages = sorted(
            item.name.rstrip("/")
            for item in stage_entries
        )

        print("Internal stage count:", len(internal_stages))

        for name in internal_stages:
            print(" -", name)

    except Exception as e:
        print("Internal stage inspection: NOT VERIFIED")
        print("Reason:", str(e)[:300])

# --------------------------------------------------
# 4. Preliminary assessment
# --------------------------------------------------

print("\n4. MODEL STRUCTURE ASSESSMENT")

print("Internal stages found:", len(internal_stages))

if internal_stages:

    print("Stage directory listing: PASS")

    print(
        "Contains preprocessing-like stages:",
        any(
            name.split("_", 1)[-1].startswith((
                "Imputer",
                "StringIndexer",
                "OneHotEncoder",
                "StandardScaler",
                "VectorAssembler"
            ))
            for name in internal_stages
        )
    )

    print(
        "Contains LogisticRegression stage:",
        any(
            "LogisticRegression" in name
            for name in internal_stages
        )
    )

else:
    print("Stage directory listing: NOT VERIFIED")

print("Deployment input contract: PENDING")
print("Double preprocessing risk: PENDING")

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Model loaded: False")
print("Artifact files downloaded: False")
print("Data transformed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.16 COMPLETE")


# COMMAND ----------


# STEP 17.17 — EXISTING INFERENCE NOTEBOOK INSPECTION
# READ-ONLY: Retrieve notebook source only
# No notebook execution, inference, writes, or job submission

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ExportFormat
import base64

print("=" * 70)
print("STEP 17.17 — INFERENCE NOTEBOOK SOURCE INSPECTION")
print("=" * 70)

notebook_path = (
    "/Telco_Customer_Churn_ML/"
    "15C_Telco_Guarded_Batch_Job_Candidate"
)

client = WorkspaceClient()

# --------------------------------------------------
# 1. Inspect notebook identity
# --------------------------------------------------

print("\n1. NOTEBOOK IDENTITY")

try:
    notebook_info = client.workspace.get_status(
        path=notebook_path
    )

    print("Notebook path:", notebook_path)
    print("Object type:", notebook_info.object_type)
    print("Language:", notebook_info.language)

except Exception as e:
    print("Notebook metadata: NOT VERIFIED")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 2. Retrieve notebook source
# --------------------------------------------------

print("\n2. NOTEBOOK SOURCE ACCESS")

notebook_source = None

try:
    exported = client.workspace.export(
        path=notebook_path,
        format=ExportFormat.SOURCE
    )

    if exported.content:
        notebook_source = base64.b64decode(
            exported.content
        ).decode("utf-8")

        print("Notebook source accessible: YES")
        print("Source characters:", len(notebook_source))
    else:
        print("Notebook source accessible: NO")

except Exception as e:
    print("Notebook source accessible: NO")
    print("Reason:", str(e)[:300])


# --------------------------------------------------
# 3. Identify relevant source lines
# --------------------------------------------------

print("\n3. MODEL AND PREPROCESSING REFERENCES")

keywords = [
    "PipelineModel.load",
    "mlflow.spark.load_model",
    "mlflow.pyfunc.load_model",
    "spark_udf",
    "preprocessing_pipeline",
    "models:/",
    "transform(",
    "vector_to_array",
    "features"
]

matching_lines = []

if notebook_source:

    for line_number, line in enumerate(
        notebook_source.splitlines(),
        start=1
    ):
        if any(
            keyword.lower() in line.lower()
            for keyword in keywords
        ):
            matching_lines.append(
                (line_number, line.strip())
            )

    print("Matching source lines:", len(matching_lines))

    for line_number, line in matching_lines[:40]:
        print(f"Line {line_number}: {line[:220]}")

else:
    print("Source inspection unavailable")


# --------------------------------------------------
# 4. Assessment
# --------------------------------------------------

print("\n4. DEPLOYMENT CONTRACT ASSESSMENT")

print("Notebook executed: False")
print("Model loaded: False")
print("Preprocessing inclusion: NOT YET VERIFIED")
print("Double preprocessing risk: NOT YET RULED OUT")
print("Deployment input contract: PENDING")


# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")
print("Artifact files downloaded: False")

print("\nSTEP 17.17 COMPLETE")


# COMMAND ----------


# STEP 17.18 — VERIFY INFERENCE EXECUTION SEQUENCE
# READ-ONLY: Inspect previously retrieved notebook source
# No notebook execution or model loading

print("=" * 70)
print("STEP 17.18 — INFERENCE SEQUENCE VERIFICATION")
print("=" * 70)

if not notebook_source:
    raise ValueError(
        "Notebook source unavailable. "
        "Complete Step 17.17 first."
    )

source_lines = notebook_source.splitlines()

# --------------------------------------------------
# 1. Inspect model-loading section
# --------------------------------------------------

print("\n1. MODEL AND PREPROCESSING LOADING")

for line_number in range(40, 106):

    if line_number <= len(source_lines):
        print(
            f"{line_number:04d}: "
            f"{source_lines[line_number - 1]}"
        )

# --------------------------------------------------
# 2. Inspect transformation sequence
# --------------------------------------------------

print("\n2. PREPROCESSING AND PREDICTION SEQUENCE")

for line_number in range(315, 381):

    if line_number <= len(source_lines):
        print(
            f"{line_number:04d}: "
            f"{source_lines[line_number - 1]}"
        )

# --------------------------------------------------
# 3. Inspect final inference invocation
# --------------------------------------------------

print("\n3. FINAL INFERENCE INVOCATION")

for line_number in range(1335, 1371):

    if line_number <= len(source_lines):
        print(
            f"{line_number:04d}: "
            f"{source_lines[line_number - 1]}"
        )

# --------------------------------------------------
# 4. Safety
# --------------------------------------------------

print("\n4. SAFETY")

print("Notebook executed: False")
print("Pipeline loaded: False")
print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.18 COMPLETE")


# COMMAND ----------


# STEP 17.19 — PROTECTED INFERENCE SAFETY REVIEW
# READ-ONLY: Inspect notebook source in memory

print("=" * 70)
print("STEP 17.19 — PROTECTED INFERENCE SAFETY REVIEW")
print("=" * 70)

if not notebook_source:
    raise ValueError(
        "Notebook source unavailable. Run Step 17.17 first."
    )

source_lines = notebook_source.splitlines()

# --------------------------------------------------
# 1. Locate protected execution function
# --------------------------------------------------

print("\n1. PROTECTED EXECUTION FUNCTION")

function_name = "execute_snapshot_protected_batch_job"

function_line = None

for index, line in enumerate(source_lines, start=1):
    if line.lstrip().startswith(
        f"def {function_name}("
    ):
        function_line = index
        break

print("Function found:", function_line is not None)
print("Definition line:", function_line)

# --------------------------------------------------
# 2. Inspect function implementation
# --------------------------------------------------

print("\n2. FUNCTION IMPLEMENTATION")

if function_line is not None:

    # Print the function until the next top-level
    # function/class definition or notebook command.
    for index in range(
        function_line - 1,
        len(source_lines)
    ):

        line = source_lines[index]

        if index > function_line - 1 and (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("# COMMAND ----------")
        ):
            break

        print(f"{index + 1:04d}: {line}")

else:
    print("Function implementation unavailable")


# --------------------------------------------------
# 3. Inspect safety-related references
# --------------------------------------------------

print("\n3. SAFETY CONTROL REFERENCES")

keywords = [
    "source_version",
    "source_fingerprint",
    "batch_id",
    "registry_table",
    "destination_table",
    "duplicate",
    "overwrite",
    "append",
    "execution_status"
]

for keyword in keywords:

    matches = sum(
        keyword.lower() in line.lower()
        for line in source_lines
    )

    print(f"{keyword}: {matches} references")

# --------------------------------------------------
# 4. Safety
# --------------------------------------------------

print("\n4. SAFETY")

print("Notebook executed: False")
print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.19 COMPLETE")


# COMMAND ----------


# STEP 17.20 — BATCH REGISTRATION AND PUBLICATION REVIEW
# READ-ONLY: Source inspection only

print("=" * 70)
print("STEP 17.20 — REGISTRATION AND PUBLICATION REVIEW")
print("=" * 70)

if not notebook_source:
    raise ValueError("Run Step 17.17 first.")

source_lines = notebook_source.splitlines()

function_names = [
    "register_batch_identity",
    "publish_quality_approved_batch"
]

for function_name in function_names:

    print("\n" + "-" * 70)
    print("FUNCTION:", function_name)
    print("-" * 70)

    start_index = None

    for index, line in enumerate(source_lines):
        if line.startswith(f"def {function_name}("):
            start_index = index
            break

    if start_index is None:
        print("FUNCTION NOT FOUND")
        continue

    print("Definition line:", start_index + 1)

    for index in range(start_index, len(source_lines)):

        line = source_lines[index]

        if index > start_index and (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("# COMMAND ----------")
        ):
            break

        print(f"{index + 1:04d}: {line}")

print("\nSAFETY CHECK")
print("Notebook executed: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.20 COMPLETE")


# COMMAND ----------


# STEP 17.21 — PREDICTION QUALITY GATE REVIEW
# READ-ONLY: Inspect existing notebook source in memory

print("=" * 70)
print("STEP 17.21 — PREDICTION QUALITY GATE REVIEW")
print("=" * 70)

if not notebook_source:
    raise ValueError("Run Step 17.17 first.")

source_lines = notebook_source.splitlines()
function_name = "validate_batch_predictions"

start_index = None

for index, line in enumerate(source_lines):
    if line.startswith(f"def {function_name}("):
        start_index = index
        break

print("\n1. QUALITY VALIDATION FUNCTION")
print("Function found:", start_index is not None)

if start_index is not None:
    print("Definition line:", start_index + 1)

    print("\n2. QUALITY VALIDATION IMPLEMENTATION")

    for index in range(start_index, len(source_lines)):
        line = source_lines[index]

        if index > start_index and (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("# COMMAND ----------")
        ):
            break

        print(f"{index + 1:04d}: {line}")

else:
    print("Function implementation unavailable.")

print("\n3. SAFETY")
print("Notebook executed: False")
print("Model loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.21 COMPLETE")


# COMMAND ----------


# STEP 17.22 — APPROVED OUTPUT SCHEMA REVIEW
# READ-ONLY: Schema metadata only
# No inference, writes, or Job execution

print("=" * 70)
print("STEP 17.22 — APPROVED OUTPUT SCHEMA REVIEW")
print("=" * 70)

approved_table = (
    "workspace.default.telco_15_approved_predictions"
)

print("\n1. APPROVED PREDICTION TABLE")
print("Table:", approved_table)

if not spark.catalog.tableExists(approved_table):
    raise ValueError(
        "Approved prediction table is not accessible."
    )

# Retrieve schema metadata only
approved_schema = spark.table(approved_table).schema

print("\n2. COLUMN DEFINITIONS")

for field in approved_schema.fields:
    print(
        f"{field.name:30} "
        f"{field.dataType.simpleString():20} "
        f"nullable={field.nullable}"
    )

# --------------------------------------------------
# 3. Required prediction columns
# --------------------------------------------------

print("\n3. REQUIRED OUTPUT CONTRACT")

required_columns = [
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "batch_id",
    "prediction_timestamp",
    "model_name",
    "model_version",
    "classification_threshold",
    "source_table",
    "publication_timestamp"
]

actual_columns = approved_schema.fieldNames()

missing_columns = [
    name for name in required_columns
    if name not in actual_columns
]

extra_columns = [
    name for name in actual_columns
    if name not in required_columns
]

print("Expected columns:", len(required_columns))
print("Actual columns:", len(actual_columns))
print("Missing columns:", missing_columns)
print("Extra columns:", extra_columns)

schema_status = (
    "PASS"
    if not missing_columns and not extra_columns
    else "REVIEW"
)

print("Schema contract status:", schema_status)

# --------------------------------------------------
# 4. Staging isolation
# --------------------------------------------------

print("\n4. STAGING ISOLATION")

staging_table = staging_config[
    "proposed_prediction_table"
]

print("Proposed staging table:", staging_table)
print(
    "Distinct from approved table:",
    staging_table != approved_table
)

print(
    "Staging table currently exists:",
    spark.catalog.tableExists(staging_table)
)

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Prediction rows collected: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.22 COMPLETE")


# COMMAND ----------


# STEP 17.23 — STAGING OUTPUT CONTRACT
# READ-ONLY: In-memory configuration and schema review

print("=" * 70)
print("STEP 17.23 — STAGING OUTPUT CONTRACT")
print("=" * 70)

approved_table = (
    "workspace.default.telco_15_approved_predictions"
)

staging_table = staging_config[
    "proposed_prediction_table"
]

# --------------------------------------------------
# 1. Capture approved output schema
# --------------------------------------------------

approved_schema = spark.table(approved_table).schema

expected_schema = {
    field.name: field.dataType.simpleString()
    for field in approved_schema.fields
}

required_columns = [
    "batch_id",
    "model_name",
    "model_version",
    "customerID",
    "churn_probability",
    "predicted_class",
    "predicted_churn",
    "prediction_timestamp",
    "classification_threshold",
    "source_table",
    "publication_timestamp"
]

expected_types = {
    "batch_id": "string",
    "model_name": "string",
    "model_version": "string",
    "customerID": "string",
    "churn_probability": "double",
    "predicted_class": "int",
    "predicted_churn": "string",
    "prediction_timestamp": "timestamp",
    "classification_threshold": "double",
    "source_table": "string",
    "publication_timestamp": "timestamp"
}

# --------------------------------------------------
# 2. Validate exact output contract
# --------------------------------------------------

print("\n1. OUTPUT SCHEMA VALIDATION")

column_match = (
    list(expected_schema.keys()) == required_columns
)

type_match = (
    expected_schema == expected_types
)

print("Column count:", len(expected_schema))
print("Column order matches:", column_match)
print("Data types match:", type_match)

for name in required_columns:
    print(
        f"{name:30} "
        f"{expected_schema.get(name, 'MISSING')}"
    )

# --------------------------------------------------
# 3. Define staging contract
# --------------------------------------------------

staging_output_contract = {
    "environment": "STAGING",
    "source_table": staging_config["source_table"],
    "approved_reference_table": approved_table,
    "destination_table": staging_table,
    "required_columns": required_columns,
    "expected_types": expected_types,
    "expected_prediction_rows": 1409,
    "require_quality_validation": True,
    "require_unique_customer_ids": True,
    "require_source_reconciliation": True,
    "allow_inference": False,
    "allow_writes": False,
    "allow_job_submission": False
}

print("\n2. STAGING CONFIGURATION")

for key, value in staging_output_contract.items():
    if key not in ["required_columns", "expected_types"]:
        print(f"{key}: {value}")

# --------------------------------------------------
# 4. Isolation checks
# --------------------------------------------------

print("\n3. ISOLATION VALIDATION")

protected_tables = {
    "workspace.default.telco_15_approved_predictions",
    "workspace.default.telco_15_job_batch_registry",
    "workspace.default.telco_test_raw"
}

destination_isolated = (
    staging_table not in protected_tables
)

staging_exists = spark.catalog.tableExists(staging_table)

print("Destination isolated:", destination_isolated)
print("Staging table already exists:", staging_exists)

# --------------------------------------------------
# 5. Overall assessment
# --------------------------------------------------

print("\n4. CONTRACT ASSESSMENT")

checks = {
    "Column order": column_match,
    "Column types": type_match,
    "Destination isolation": destination_isolated,
    "Staging name available": not staging_exists,
    "Inference disabled": not staging_output_contract["allow_inference"],
    "Writes disabled": not staging_output_contract["allow_writes"],
    "Job submission disabled": not staging_output_contract["allow_job_submission"]
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

overall_pass = all(checks.values())

print(
    "Overall staging contract:",
    "PASS" if overall_pass else "REVIEW"
)

print("\n5. SAFETY")

print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.23 COMPLETE")


# COMMAND ----------


# STEP 17.24 — SOURCE-TO-PREDICTION RECONCILIATION
# READ-ONLY: Existing source and approved predictions

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 17.24 — CUSTOMER RECONCILIATION")
print("=" * 70)

source_table = deployment_config["source_table"]
approved_table = deployment_config["prediction_table"]

approved_batch_id = (
    "d972dd16-029d-489f-a406-3c6c6e1e6265"
)

# --------------------------------------------------
# 1. Load existing customer IDs
# --------------------------------------------------

source_ids = (
    spark.table(source_table)
    .select("customerID")
)

prediction_ids = (
    spark.table(approved_table)
    .filter(F.col("batch_id") == approved_batch_id)
    .select("customerID")
)

print("\n1. RECORD COUNTS")

source_count = source_ids.count()
prediction_count = prediction_ids.count()

print("Source records:", source_count)
print("Approved batch records:", prediction_count)

# --------------------------------------------------
# 2. Check uniqueness and null/blank IDs
# --------------------------------------------------

print("\n2. CUSTOMER ID INTEGRITY")

source_unique = source_ids.distinct().count()
prediction_unique = prediction_ids.distinct().count()

source_invalid = source_ids.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

prediction_invalid = prediction_ids.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

print("Source unique IDs:", source_unique)
print("Prediction unique IDs:", prediction_unique)
print("Source invalid IDs:", source_invalid)
print("Prediction invalid IDs:", prediction_invalid)

# --------------------------------------------------
# 3. Find missing predictions
# --------------------------------------------------

print("\n3. MISSING PREDICTIONS")

missing_predictions = (
    source_ids.distinct()
    .join(
        prediction_ids.distinct(),
        on="customerID",
        how="left_anti"
    )
    .count()
)

print("Source IDs missing predictions:", missing_predictions)

# --------------------------------------------------
# 4. Find unexpected predictions
# --------------------------------------------------

print("\n4. UNEXPECTED PREDICTIONS")

unexpected_predictions = (
    prediction_ids.distinct()
    .join(
        source_ids.distinct(),
        on="customerID",
        how="left_anti"
    )
    .count()
)

print("Prediction IDs absent from source:", unexpected_predictions)

# --------------------------------------------------
# 5. Reconciliation assessment
# --------------------------------------------------

print("\n5. RECONCILIATION ASSESSMENT")

checks = {
    "Record counts match": source_count == prediction_count,
    "Source IDs unique": source_unique == source_count,
    "Prediction IDs unique": prediction_unique == prediction_count,
    "Source IDs valid": source_invalid == 0,
    "Prediction IDs valid": prediction_invalid == 0,
    "No missing predictions": missing_predictions == 0,
    "No unexpected predictions": unexpected_predictions == 0
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print(
    "Overall reconciliation:",
    "PASS" if all(checks.values()) else "REVIEW"
)

# --------------------------------------------------
# 6. Safety
# --------------------------------------------------

print("\n6. SAFETY")
print("Models loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.24 COMPLETE")


# COMMAND ----------


# STEP 17.25 — APPROVED PREDICTION BATCH INVENTORY
# READ-ONLY: Existing table metadata and aggregations
# No inference, writes, or Job execution

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 17.25 — APPROVED BATCH INVENTORY")
print("=" * 70)

approved_table = deployment_config["prediction_table"]

print("\n1. APPROVED PREDICTION TABLE")
print("Table:", approved_table)

if not spark.catalog.tableExists(approved_table):
    raise ValueError("Approved prediction table is unavailable.")

# --------------------------------------------------
# 2. Total records and batch inventory
# --------------------------------------------------

approved_df = spark.table(approved_table)

print("\n2. TABLE SUMMARY")
print("Total records:", approved_df.count())

print("\n3. BATCH INVENTORY")

batch_inventory = (
    approved_df
    .groupBy(
        "batch_id",
        "model_name",
        "model_version",
        "source_table"
    )
    .agg(
        F.count("*").alias("record_count"),
        F.countDistinct("customerID").alias("unique_customers"),
        F.min("prediction_timestamp").alias(
            "earliest_prediction"
        ),
        F.max("prediction_timestamp").alias(
            "latest_prediction"
        )
    )
    .orderBy(F.desc("record_count"))
)

batch_inventory.show(
    20,
    truncate=False
)

# --------------------------------------------------
# 4. Check previously selected batch ID
# --------------------------------------------------

print("\n4. PREVIOUSLY SELECTED BATCH")

previous_batch_id = (
    "d972dd16-029d-489f-a406-3c6c6e1e6265"
)

matching_count = (
    approved_df
    .filter(F.col("batch_id") == previous_batch_id)
    .count()
)

print("Selected batch ID:", previous_batch_id)
print("Matching records:", matching_count)

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")
print("Models loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.25 COMPLETE")


# COMMAND ----------


# STEP 17.26 — CORRECT APPROVED BATCH RECONCILIATION
# READ-ONLY: Existing source and prediction tables

from pyspark.sql import functions as F

print("=" * 70)
print("STEP 17.26 — CORRECT BATCH RECONCILIATION")
print("=" * 70)

source_table = deployment_config["source_table"]
approved_table = deployment_config["prediction_table"]

approved_batch_id = (
    "59e85469-4fb9-4b0f-882c-761cec0d93eb"
)

# --------------------------------------------------
# 1. Read existing customer IDs
# --------------------------------------------------

source_ids = (
    spark.table(source_table)
    .select("customerID")
)

prediction_ids = (
    spark.table(approved_table)
    .filter(F.col("batch_id") == approved_batch_id)
    .select("customerID")
)

print("\n1. RECORD COUNTS")

source_count = source_ids.count()
prediction_count = prediction_ids.count()

print("Source records:", source_count)
print("Approved prediction records:", prediction_count)

# --------------------------------------------------
# 2. Customer ID integrity
# --------------------------------------------------

print("\n2. CUSTOMER ID INTEGRITY")

source_unique = source_ids.distinct().count()
prediction_unique = prediction_ids.distinct().count()

source_invalid = source_ids.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

prediction_invalid = prediction_ids.filter(
    F.col("customerID").isNull()
    | (F.trim(F.col("customerID")) == "")
).count()

print("Source unique IDs:", source_unique)
print("Prediction unique IDs:", prediction_unique)
print("Source invalid IDs:", source_invalid)
print("Prediction invalid IDs:", prediction_invalid)

# --------------------------------------------------
# 3. Reconcile customer ID sets
# --------------------------------------------------

print("\n3. CUSTOMER RECONCILIATION")

missing_predictions = (
    source_ids.distinct()
    .join(
        prediction_ids.distinct(),
        "customerID",
        "left_anti"
    )
    .count()
)

unexpected_predictions = (
    prediction_ids.distinct()
    .join(
        source_ids.distinct(),
        "customerID",
        "left_anti"
    )
    .count()
)

print("Missing prediction IDs:", missing_predictions)
print("Unexpected prediction IDs:", unexpected_predictions)

# --------------------------------------------------
# 4. Validation gates
# --------------------------------------------------

print("\n4. RECONCILIATION GATES")

checks = {
    "Source nonempty": source_count > 0,
    "Prediction batch nonempty": prediction_count > 0,
    "Record counts match": source_count == prediction_count,
    "Source IDs unique": source_unique == source_count,
    "Prediction IDs unique": prediction_unique == prediction_count,
    "Source IDs valid": source_invalid == 0,
    "Prediction IDs valid": prediction_invalid == 0,
    "No missing predictions": missing_predictions == 0,
    "No unexpected predictions": unexpected_predictions == 0
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print(
    "\nOverall reconciliation:",
    "PASS" if all(checks.values()) else "REVIEW"
)

# --------------------------------------------------
# 5. Safety
# --------------------------------------------------

print("\n5. SAFETY")

print("Models loaded: False")
print("Inference executed: False")
print("Job submitted: False")
print("Delta writes performed: False")
print("Protected assets modified: False")

print("\nSTEP 17.26 COMPLETE")
