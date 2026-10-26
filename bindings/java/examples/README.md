# Standalone JVM consumers

These examples resolve the same prebuilt `gr.tdigest:gr-tdigest` JAR used by Java.
They need no Rust compiler or Python at runtime. Use the unique candidate version
from its artifact manifest. A successful local example is not Spark cluster
qualification.

## Kotlin and Gradle

The example pins Kotlin 2.4.21 and emits Java 11 bytecode. This is a tested
compiler selection, not a claim about the minimum supported Kotlin compiler.
Incremental compilation tracks source edits and application classpath changes.
Use the repository's complete Gradle wrapper or its configured Gradle bootstrap:

```sh
bindings/java/gradlew -p bindings/java/examples/kotlin \
  -PtdigestVersion=VERSION run
```

Set `TDIGEST_MAVEN_REPOSITORY_URL` for an exclusive internal dependency source.
Optional `TDIGEST_MAVEN_USERNAME` and `TDIGEST_MAVEN_PASSWORD` are read from the
environment. Without that URL the example uses Maven Central. Publication
destinations are separate settings; running the example publishes nothing.
After staging all dependencies, add `--offline` for an offline check. Run the
same JAR with each required runtime, rather than rebuilding it for each JDK.

## Scala and sbt

The example pins Scala 2.12.16 and sbt 1.12.11. Build with the declared JDK 17;
the compiled example targets Java 11. Install or stage sbt separately, then run:

```sh
cd bindings/java/examples/scala
sbt -Dtdigest.version=VERSION run
```

`TDIGEST_MAVEN_REPOSITORY_URL` selects the sole project dependency repository.
The optional credential environment variables are the same as above; set
`TDIGEST_MAVEN_REALM` to the server's authentication realm. For an internal-only
sbt launcher and compiler bootstrap, also supply sbt's external repository and
credential configuration using `sbt.override.build.repos`,
`sbt.repository.config` and `sbt.boot.credentials`. Project resolution alone
does not configure the launcher. Keep credentials and trust stores outside the
source checkout.

After staging the launcher, compiler and application dependencies, run
`sbt -Dtdigest.version=VERSION 'set offline := true' run` with the configured
repository unavailable. `TDIGEST_RUNTIME_JAVA_HOME` selects the forked example's
runtime independently of the build JDK. Test that runtime explicitly.

Both examples exercise weighted construction, array merge-all and a byte
round-trip. They print the loaded JAR origin and native payload hash through
`TDigest.runtimeInfo()`. The coverage manifest identifies the exact artifact and
runtime checks actually completed. Spark 3.5 / Scala 2.12 remains provisional;
driver/executor serialization, shuffles, broadcasts, checkpoint settings and
executor images still require the consuming environment's checks.
