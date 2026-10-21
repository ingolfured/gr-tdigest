# Spark consumer check

This check compiles Scala against a prebuilt JAR and uses two separate executor
JVMs. It exercises weighted queries, broadcast restoration after closing the
original, a 40-shard shuffle, on-disk caching and checkpoint restoration without
rebuilding the source. It verifies native payload identity on both executors.
The successful result prints the actual Spark/Scala versions and process IDs.

Stage Spark, the Scala compiler and the candidate JAR first. The runner downloads
nothing. For example, with Spark 3.5.9 and Scala compiler 2.12.16 already staged:

```sh
export JAVA_HOME=/staged/jdk-11
export RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 bindings/java/examples/spark/check.py \
  --spark-home /staged/spark-3.5.9-bin-hadoop3 \
  --scala-compiler-classpath '/staged/scala-compiler-2.12.16.jar:/staged/scala-library-2.12.16.jar:/staged/scala-reflect-2.12.16.jar' \
  --jar /staged/gr-tdigest-CANDIDATE.jar \
  --serializer java --output /results/java11-java
```

Repeat with JDK 17 and a fresh output directory. Repeat both runtimes with
`--serializer kryo`. The Kryo example explicitly registers a serializer that
writes TDIG bytes and restores a fresh native handle. Generic field serialization
of a process-local handle is insufficient. Registration must be part of the
application's Spark configuration. This check sets its selected serializer
explicitly, even if an input properties file selects another one.

The default `local-cluster[2,1,512]` starts a local master and two executor
processes. `local[4]` is not a substitute: the check requires distinct executor
processes outside the driver. The local checkpoint directory is temporary and
shared only on this machine. This does not qualify an executor image, a remote
filesystem or a multi-host cluster. The sample's byte-header mass inspection is
a check of the current TDIG v3 format, not an additional public Java API.

For an owner-run cluster check, supply `--master`, `--checkpoint` with a shared
URI, and optionally `--properties-file` and repeated `--conf` options. The check
uses one core and 512 MiB per executor and disables retries to expose failures;
it is a correctness exercise, not a representative capacity or memory test.
Stage all tools and artifacts through the approved internal services. Keep
credentials and trust configuration external. Cluster-specific images, minimum
ABI, deployed serializer settings and real workloads still need owner approval.

Every output directory contains the compiled check application, compiler
version, command logs and a receipt identifying the exact JAR, bundled native
payload and check source. Failed runs retain their receipts. No production
publication or application configuration is changed by this runner.
