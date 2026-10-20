# Build and qualify the combined consumer candidate

This is a development candidate. Do not publish it as a production release.
The manifest records the exact source revision and artifact bytes. The source
and every built artifact must use one unique development version. Test/import
the supplied wheel and JAR; rebuilding under that version is not the same
artifact qualification.

## Tools and internal sources

Use the pinned Rust/Cargo 1.95.0 toolchain (lock format 4), a declared JDK and
Gradle 8.14.5. `scripts/check_build_tools.py` checks selection without installing
tools. The complete Gradle wrapper is included in checkout and source archive.
Build JDK and tested runtime JDKs are separate manifest fields.

For JDK/Gradle use `TDIGEST_JDK_URL` / `TDIGEST_JDK_SHA256` and
`TDIGEST_GRADLE_URL` / `TDIGEST_GRADLE_SHA256`. Local archives and authenticated
internal HTTP sources are supported. `TDIGEST_TOOLCHAIN_CACHE` controls staging;
`TDIGEST_GRADLE_HOME` selects a preinstalled Gradle. Download credentials use
`TDIGEST_DOWNLOAD_USERNAME` / `TDIGEST_DOWNLOAD_PASSWORD` and remain external.

`TDIGEST_MAVEN_REPOSITORY_URL` exclusively replaces Maven Central for build
dependencies. Use `TDIGEST_MAVEN_USERNAME` / `TDIGEST_MAVEN_PASSWORD` as needed.
Publication settings `MAVEN_REPOSITORY_URL` / credentials are separate and are
not used by ordinary dependency resolution. Configured-source failure must not
fall back to a public repository. Real internal service and trust tests are
owned by the user; controlled local authenticated fixtures are external checks.

Stage Rust with standard `RUSTUP_DIST_SERVER` / `RUSTUP_UPDATE_ROOT` pointing at
approved internal sources. Configure Cargo source replacement in an external
`CARGO_HOME/config.toml`, or vendor exact dependencies. Registry credentials and
trust configuration stay outside source control. Use `UV_DEFAULT_INDEX` for an
exclusive internal Python index and a staged Python/tool environment.

After staging all tools and dependencies, set `TDIGEST_OFFLINE=1` and
`CARGO_NET_OFFLINE=true`. Gradle and its native Cargo task use offline mode.
Use the documented staged Python environment with `uv run --no-sync`; bootstrap
HTTP cache misses are refused. A clean source archive must not depend on a
developer checkout, private data or a target cache from another source revision.

## Component and combined checks

```sh
python3 scripts/check_build_tools.py
make build
make test
make lint
make quality-extended      # fails if accuracy fails; do not conceal the result
make quality-boundaries    # separate exact numerical boundary controls
```

Ordinary Rust, rustdoc, Python/NumPy, Polars, original compatibility, Java/JNI,
Scala/Kotlin compilation, installed-artifact and distributed checks are distinct
coverage rows. A unit-test count cannot replace an unsuccessful rustdoc/build
step. Known-outcome quality regression is not numerical acceptance.

## Release build and install

Build Python from `bindings/python`, whose pyproject selects the actual package:

```sh
cd bindings/python
maturin sdist --out /chosen/artifacts
maturin build --release --locked --zig --compatibility manylinux_2_28 --out /chosen/artifacts
```

The Python floor is glibc 2.28. JNI has a separate glibc 2.17 requirement. JNI
uses the system allocator when built with the `java` feature. Cross-build the
native library for the JNI floor, then set `TDIGEST_NATIVE_FILE` to that exact
file and `TDIGEST_SOURCE_SHA` to the frozen source SHA when packaging with
`bindings/java/gradlew -p bindings/java jar`. The supplied JAR must contain that
payload; inspect symbols and actual execution separately. A tag change is not
an ABI fix. Build tools, targets and any unperformed floor checks belong in the
manifest, not in an inferred compatibility claim.

Install exact candidate wheel bytes into a fresh environment, pin the optional
Polars extra and selected NumPy/PyArrow versions, run `pip check`, and import
both `gr_tdigest` (modern) and `tdigest_rs` (original) outside the checkout.
The `gr-tdigest` distribution owns both module paths. Remove conflicting older
distributions such as `gr-tdigest-rs`, `gr-polars-tdigest-tmp` or `tdigest-rs`
before installing this owner; do not leave two owners of the same files.

For an application upgrade, save its exact old wheel/hash and environment lock,
install the uniquely pinned candidate, then run that application's regression
gates. Roll back using those saved bytes and lock. The supplied internal
0.1.9/0.1.11 hashes identify baseline binaries, whose native provenance is
unrecovered; a public 0.2.4 wheel is not a substitute for those private tests.

## JVM loading and ownership

Run the JAR outside the checkout with Java 11/17/21. No Rust/Python installation
or developer library directory is required at runtime. `TDigest.runtimeInfo()`
reports resolved JAR version/origin and actual loaded native SHA-256.

`-Dgr.tdigest.extractDir=/existing/writable/path` selects a directory on a mount
that permits native loading. `-Dgr.tdigest.native=relative/or/absolute/library`
selects an exact override, including executor distributed-file paths. A failed
explicit override has no fallback. An incompatible bundled resource reports
its origin/path and does not silently select another payload.

Every digest owns one native handle. Close it or use try-with-resources. Queries
are read-only; independent digests may run concurrently. Sharing a mutable
handle between threads is not supported. Bytes/object/stream restoration owns
fresh handles. Catchable Rust panics become Java exceptions; allocator aborts,
invalid process pointers and all possible out-of-memory failures cannot be
promised catchable.

Actual Spark driver/executor shuffles, broadcasts, checkpoint restoration,
executor images, private data, representative servers and internal publication
remain user-arranged acceptance steps. Spark 3.5 / Scala 2.12 is provisional;
the precise configuration must be confirmed before cluster sign-off.

Modern Rust builders and seed restoration are fallible and validate state before exposing it. Handle their `Result`; malformed configuration/state is an error. Requested capacity is retained when oversized seeds are compressed. Public codecs and modern binding reconstruction use checked state. The original `tdigest_core` API remains separately pinned.

Current version003 review bytes and exact-source qualification are in [the artifact report](accuracy-investigation/ARTIFACTS.md) and [coverage](accuracy-investigation/coverage-v3.json). The original manifests beside this guide describe the historical version001 artifact source; they do not replace current receipts.
