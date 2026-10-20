# Java 11 and later

Java code and tests compile with `--release 11`. The pinned Gradle8.14.5 can
itself run on JDK11. CI builds and runs the JNI tests and packaged JAR on JDK11
and17, in addition to the full project's JDK21 build.

With an installed JDK, the normal wrapper needs no Python:

```sh
JAVA_HOME=/opt/jdk-11 ./bindings/java/gradlew --no-daemon -p bindings/java clean test jar smoke verifyJava11Bytecode
```

`verifyJava11Bytecode` checks compiled class files, not just Gradle settings.
Native code is built for the host OS/architecture and packaged in the JAR.
`CARGO_TARGET_DIR` is respected without a target symlink.

## Internal downloads and Artifactory

The optional bootstrap needs Python3.10+ and its standard library. Configure
JDK/Gradle archive URLs, local paths or `file:` URLs and their expected SHA256:

```sh
export TDIGEST_JDK_URL=https://artifacts.example/jdks/jdk-11-linux-x64.tar.gz
export TDIGEST_JDK_SHA256=EXPECTED_64_CHARACTER_SHA256
export TDIGEST_GRADLE_URL=https://artifacts.example/gradle/gradle-8.14.5-bin.zip
export TDIGEST_GRADLE_SHA256=EXPECTED_64_CHARACTER_SHA256
export TDIGEST_MAVEN_REPOSITORY_URL=https://artifacts.example/artifactory/maven-virtual
make java-prepare
make build-java java-test
```

The Maven dependency source is exclusive when configured: it replaces
MavenCentral, with no public fallback. `TDIGEST_MAVEN_USERNAME` and
`TDIGEST_MAVEN_PASSWORD` supply its credentials. `TDIGEST_DOWNLOAD_USERNAME` and
`TDIGEST_DOWNLOAD_PASSWORD` supply HTTP Basic credentials for archive downloads.
Configured download failures have no alternate source. There is no implicit JDK
vendor download; choose the archive for your OS and architecture.

| Variable | Meaning |
| --- | --- |
| `JAVA_HOME` | Existing JDK; an explicit `TDIGEST_JDK_URL` selects its verified JDK instead. |
| `TDIGEST_GRADLE_HOME` | Existing Gradle installation; skips distribution download. |
| `TDIGEST_TOOLCHAIN_CACHE` | Bootstrap cache; default `~/.cache/gr-tdigest/toolchains`. |
| `TDIGEST_BOOTSTRAP_PYTHON` | Python executable used by the optional bootstrap. |
| `GRADLE_USER_HOME` | Gradle dependency cache, independent of the toolchain cache. |
| `CARGO_HOME`, `CARGO_TARGET_DIR` | Rust source cache and native output directory. |

Archives are checksum-verified before extraction. Paths and links are checked,
and complete installations enter the cache by atomic rename. Partial downloads
never become a completed installation. Bootstrap errors do not print configured
URLs or passwords. POSIX and Windows wrappers use the same environment controls;
the POSIX fake-launcher test is distinct from the portable archive tests.

## Building with no network

Prepare the tools and dependency caches through internal sources first, or
pre-stage archives and dependencies. Then use:

```sh
TDIGEST_OFFLINE=1 make build-java java-test
```

Offline mode refuses uncached network tool downloads and passes `--offline` to
both Gradle and Cargo. Local archive files remain usable. Keep the same internal
Maven URL when warming and reusing Gradle's cache. Missing dependencies fail
instead of falling back to a public source.

The native build also needs the pinned Rust toolchain and Cargo sources.
Provision Rust in advance; populate Cargo with `cargo fetch --locked` using the
organization's Cargo mirror, or supply a vendored Cargo tree. Python development
dependencies are needed only for the full project's Python targets and have
their own uv/index configuration. Maven *publishing* still uses the separate
`MAVEN_REPOSITORY_URL` settings; the dependency mirror never changes the release
destination.

Fast bootstrap checks need no external network, installed JDK or Gradle:
`make java-bootstrap-test`.

Direct Java subprocesses in the cross-language tests use the same selected JDK as Gradle. Setting the archive environment variables is sufficient for `make build test`, including on a host with no usable `java`/`javac` on `PATH`; `JAVA_HOME` is also honored when explicitly supplied.
