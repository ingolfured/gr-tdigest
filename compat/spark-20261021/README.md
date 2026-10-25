# Public Spark check receipt

These four fresh runs use the same prebuilt 0.2.5-dev2026102102 JAR from
source aaf7ab3956fed4269c1eaeef915a6faee78fa305. The receipt identifies the
JAR, native payload, Scala check source, runner, commands and hashed logs.

All four cells passed: Java 11/17 x Java object serialization/explicit Kryo
state serialization. Compilation uses Scala 2.12.16; Spark 3.5.9 supplies
Scala 2.12.18 at runtime. Each run uses two separate executor JVMs.

Use [the staged-tool runner](../../bindings/java/examples/spark/README.md)
to reproduce the broadcast, shuffle and checkpoint checks. The local
checkpoint filesystem is temporary and shared by local processes only.
This is not qualification of an intranet executor image, remote filesystem,
application-specific serializer configuration or representative server load.
Those checks remain with the consuming owner; no access to private artifacts
is needed for this public exercise.
