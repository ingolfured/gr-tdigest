# JVM validation, 20 October 2026

Java 11 and 17 each passed 13 JNI tests, JAR creation, the packaged smoke test and the class-file check (major version55). Each JAR also ran from a temporary directory with an empty native-library search path, loading its embedded native library.

A local HTTP mirror with Basic authentication supplied actual JDK11/JDK17 archives, Gradle8.14.5 and all JUnit dependencies. Archive caches started empty. The exclusive Maven mirror was then stopped; clean Java builds and tests passed on both JDKs with `TDIGEST_OFFLINE=1`. Rust sources and its compiler were already cached. No public fallback is configured when an internal source is selected.

`make build`, `make test` and `make lint` passed under JDK11. The original source/wrapper checksum gate and all91 original Python tests passed. Eight bootstrap contracts cover local/cache/offline behavior, authentication, checksums, archive containment, argument handling and concurrent cache completion.

The first full test attempt caught a fixture that ignored `CARGO_TARGET_DIR`; the fixture was corrected before the passing rerun. No numerical source was changed. [Machine-readable receipt](validation.json) binds the changed build/test files and packaged artifacts. These checks cover Linux x86_64; they do not establish Windows or macOS runtime support.
