package gr.tdigest_rs;

import java.lang.ref.Cleaner;
import java.util.Objects;

/**
 * Java wrapper for the native Rust TDigest.
 * Owns a native handle; frees it via Cleaner or explicit close().
 */
public final class TDigest implements AutoCloseable {

  /** Single global cleaner used to free native memory when forgotten. */
  private static final Cleaner CLEANER = Cleaner.create();

  /** Internal holder for the native pointer. */
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

  private State state;                    // native handle state
  private final Cleaner.Cleanable cleanable;

  private TDigest(long handle) {
    if (handle == 0) {
      throw new IllegalStateException("Failed to create TDigest (null native handle)");
    }
    this.state = new State(handle);
    this.cleanable = CLEANER.register(this, state);
  }

  /** Create an empty digest. */
  public static TDigest create(int maxSize, String scale) {
    long h = TDigestNative.create(maxSize, normalizeScale(scale));
    return new TDigest(h);
  }

  /** Build a digest from an array of values (sorting/merging occurs in Rust). */
  public static TDigest fromValues(double[] values, int maxSize, String scale) {
    Objects.requireNonNull(values, "values");
    long h = TDigestNative.createFromValues(values, maxSize, normalizeScale(scale));
    return new TDigest(h);
  }

  /** Estimate the cumulative distribution function at the given probe points. */
  public double[] cdf(double[] xs) {
    ensureOpen();
    Objects.requireNonNull(xs, "xs");
    return TDigestNative.estimateCdf(state.handle, xs);
  }

  /** Estimate a quantile for q in [0,1]. */
  public double quantile(double q) {
    ensureOpen();
    return TDigestNative.estimateQuantile(state.handle, q);
  }

  /** Explicitly release the underlying native memory. */
  @Override
  public void close() {
    if (state != null && state.handle != 0) {
      cleanable.clean();   // runs State.run() exactly once
      state = null;
    }
  }

  /** True if already closed (or never opened). */
  public boolean isClosed() {
    return state == null || state.handle == 0;
  }

  private void ensureOpen() {
    if (isClosed()) {
      throw new IllegalStateException("TDigest is closed");
    }
  }

  private static String normalizeScale(String s) {
    if (s == null) throw new NullPointerException("scale");
    String v = s.trim().toLowerCase();
    switch (v) {
      case "quad":
      case "k1":
      case "k2":
      case "k3":
        return v;
      default:
        throw new IllegalArgumentException("scale must be one of: quad, k1, k2, k3");
    }
  }

  @Override
  public String toString() {
    return "TDigest(handle=" + (state == null ? 0 : state.handle) + ")";
  }
}
