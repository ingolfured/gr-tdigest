package gr.tdigest;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * Java wrapper for the native Rust TDigest.
 *
 * Queries:
 *   - cdf(double[] x) -> double[]
 *   - quantile(double q) -> double                 // scalar JNI
 *   - quantile(double[] qs) -> double[]            // vector via scalar loop
 *   - median() -> double
 *
 * Owns a native handle; frees it via Cleaner or explicit close().
 */
public final class TDigest implements AutoCloseable {

  /* ======================= Type-safe enums for builder ======================= */

  public enum Scale {
    QUAD, K1, K2, K3;
    String asWire() { return name().toLowerCase(Locale.ROOT); }
  }

  public enum SingletonPolicy {
    OFF, USE, USE_WITH_PROTECTED_EDGES;

    int toNativeCode() {
      switch (this) {
        case OFF:  return 0;
        case USE:  return 1;
        case USE_WITH_PROTECTED_EDGES: return 2;
        default:   return 1;
      }
    }
  }

  public enum Precision {
    F64, F32
  }

  /* ======================= Backing native ======================= */

  private static String norm(String s) {
    if (s == null) throw new NullPointerException();
    return s.trim().toLowerCase(Locale.ROOT);
  }

  private static String parseScale(String s) {
    if (s == null) return "k2";
    String v = norm(s);
    switch (v) {
      case "quad":
      case "k1":
      case "k2":
      case "k3":
        return v;
      default:
        throw new IllegalArgumentException(
            "invalid scale: " + s + " (expected 'quad', 'k1', 'k2', or 'k3')");
    }
  }

  private static int parsePolicyCode(String kind) {
    if (kind == null) return 1; // default "use"
    String v = norm(kind);
    switch (v) {
      case "off":   return 0;
      case "use":   return 1;
      case "edges":
      case "usewithprotectededges":
        return 2;
      default:
        throw new IllegalArgumentException(
            "invalid singleton_policy: " + kind + " (expected 'off', 'use', or 'edges')");
    }
  }

  private static int coalesceEdges(Integer edges) {
    return edges == null ? 3 : Math.max(0, edges);
  }

  private static int policyCode(SingletonPolicy p) {
    switch (p) {
      case OFF:  return 0;
      case USE:  return 1;
      case USE_WITH_PROTECTED_EDGES:return 2;
      default:   return 1;
    }
  }

  /* ======================= Cleaner & state ======================= */

  private static final Cleaner CLEANER = Cleaner.create();

  private static final class State implements Runnable {
    private long handle;
    State(long handle) { this.handle = handle; }

    @Override
    public void run() {
      if (handle != 0) {
        try {
          TDigestNative.free(handle);
        } finally {
          handle = 0;
        }
      }
    }
  }

  private State state;
  private final Cleaner.Cleanable cleanable;

  private TDigest(long handle) {
    if (handle == 0) {
      throw new IllegalStateException("Failed to create TDigest (null native handle)");
    }
    this.state = new State(handle);
    this.cleanable = CLEANER.register(this, state);
  }

  /* ======================= Builder (fluent) ======================= */

  public static final class Builder {
    private int maxSize = 1000;
    private Scale scale = Scale.K2;
    private SingletonPolicy policy = SingletonPolicy.USE;
    private int edgesPerSide = 3;
    private Precision precision = Precision.F64;

    public Builder maxSize(int n) {
      if (n <= 0) throw new IllegalArgumentException("maxSize must be > 0");
      this.maxSize = n;
      return this;
    }

    public Builder scale(Scale s) {
      this.scale = Objects.requireNonNull(s, "scale");
      return this;
    }

    public Builder singletonPolicy(SingletonPolicy p) {
      this.policy = Objects.requireNonNull(p, "policy");
      return this;
    }

    public Builder precision(Precision p) {
      this.precision = Objects.requireNonNull(p, "precision");
      return this;
    }

    public Builder keep(int k) { this.edgesPerSide = Math.max(0, k); return this; }
    public Builder edgesPerSide(int k) { this.edgesPerSide = Math.max(0, k); return this; }

    public TDigest build(double[] values) {
      Objects.requireNonNull(values, "values");
      final boolean f32 = (precision == Precision.F32);
      final int edges = (policy == SingletonPolicy.USE_WITH_PROTECTED_EDGES) ? edgesPerSide : 0;
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policy.toNativeCode(), edges, f32);
      return new TDigest(h);
    }

    public TDigest build(float[] values) {
      Objects.requireNonNull(values, "values");
      final boolean f32 = (precision == Precision.F32);
      final int edges = (policy == SingletonPolicy.USE_WITH_PROTECTED_EDGES) ? edgesPerSide : 0;
      long h = TDigestNative.fromArray(
          values, maxSize, scale.asWire(), policy.toNativeCode(), edges, f32);
      return new TDigest(h);
    }
  }

  public static Builder builder() { return new Builder(); }

  /* ======================= Construction (stringly-typed) ======================= */

  public static TDigest fromArray(
      double[] values, Integer max_size, String scale, Boolean f32_mode, String singleton_policy, Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");
    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;
    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  public static TDigest fromArray(
      float[] values, Integer max_size, String scale, Boolean f32_mode, String singleton_policy, Integer edges
  ) {
    Objects.requireNonNull(values, "values");
    final int maxSize = (max_size == null) ? 1000 : max_size;
    if (maxSize <= 0) throw new IllegalArgumentException("max_size must be > 0");
    final String sc = parseScale(scale);
    final int policyCode = parsePolicyCode(singleton_policy);
    final int keep = coalesceEdges(edges);
    final boolean f32 = (f32_mode != null) && f32_mode;
    long h = TDigestNative.fromArray(values, maxSize, sc, policyCode, keep, f32);
    return new TDigest(h);
  }

  public static TDigest fromBytes(byte[] bytes) {
    Objects.requireNonNull(bytes, "bytes");
    long h = TDigestNative.fromBytes(bytes);
    return new TDigest(h);
  }

  public static TDigest create(int maxSize, String scale) {
    return TDigest.fromArray(new double[0], maxSize, scale, false, "use", 3);
  }

  public static TDigest fromValues(double[] values, int maxSize, String scale) {
    return TDigest.fromArray(values, maxSize, scale, false, "use", 3);
  }

  /* ======================= Queries ======================= */

  /**
   * CDF evaluated at array-like x (returns double[]).
   * Fast path: pass input array directly when all finite.
   * For NaN/±inf probes:
   *   - NaN   → NaN
   *   - -inf  → 0.0
   *   - +inf  → 1.0
   */
  public double[] cdf(double[] values) {
    ensureOpen();
    Objects.requireNonNull(values, "values");

    boolean allFinite = true;
    for (double v : values) {
      if (!Double.isFinite(v)) { allFinite = false; break; }
    }
    if (allFinite) {
      return TDigestNative.cdf(state.handle, values);
    }

    // Clean non-finite to some finite placeholder for native, then post-fix outputs.
    double[] temp = Arrays.copyOf(values, values.length);
    for (int i = 0; i < temp.length; i++) {
      if (!Double.isFinite(temp[i])) temp[i] = 0.0;
    }
    double[] out = TDigestNative.cdf(state.handle, temp);
    if (out == null) return null;
    for (int i = 0; i < values.length && i < out.length; i++) {
      double v = values[i];
      if (Double.isNaN(v)) {
        out[i] = Double.NaN;
      } else if (Double.isInfinite(v)) {
        out[i] = v < 0.0 ? 0.0 : 1.0;
      }
    }
    return out;
  }

  /** Vector quantile: implemented via scalar JNI. NaN/±inf → NaN. */
  public double[] quantile(double[] qs) {
    ensureOpen();
    Objects.requireNonNull(qs, "qs");
    double[] out = new double[qs.length];
    for (int i = 0; i < qs.length; i++) {
      double q = qs[i];
      if (Double.isNaN(q) || Double.isInfinite(q)) {
        out[i] = Double.NaN;
      } else {
        out[i] = TDigestNative.quantile(state.handle, q);
      }
    }
    return out;
  }

  /** Quantile for q in [0,1].  NaN/±inf → NaN. */
  public double quantile(double q) {
    ensureOpen();
    if (Double.isNaN(q) || Double.isInfinite(q)) {
      return Double.NaN;
    }
    return TDigestNative.quantile(state.handle, q);
  }

  public double median() {
    ensureOpen();
    return TDigestNative.median(state.handle);
  }

  /* ======================= Serialization & lifecycle ======================= */

  public byte[] toBytes() {
    ensureOpen();
    return TDigestNative.toBytes(state.handle);
  }

  @Override
  public void close() {
    if (state != null && state.handle != 0) {
      cleanable.clean();
      state = null;
    }
  }

  public boolean isClosed() {
    return state == null || state.handle == 0;
  }

  private void ensureOpen() {
    if (isClosed()) {
      throw new IllegalStateException("TDigest is closed");
    }
  }

  @Override
  public String toString() {
    return "TDigest(handle=" + (state == null ? 0 : state.handle) + ")";
  }
}
