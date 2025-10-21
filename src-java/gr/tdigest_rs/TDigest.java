package gr.tdigest_rs;

import java.lang.ref.Cleaner;
import java.util.Objects;

public final class TDigest implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();

  private static final class State implements Runnable {
    private long handle;
    State(long handle) { this.handle = handle; }
    @Override public void run() {
      if (handle != 0) {
        TDigestNative.free(handle);
        handle = 0;
      }
    }
  }

  private State state;               // holds the native handle
  private final Cleaner.Cleanable cleanable;

  private TDigest(long handle) {
    if (handle == 0) throw new IllegalStateException("Failed to create TDigest (null native handle)");
    this.state = new State(handle);
    this.cleanable = CLEANER.register(this, state);
  }

  /** Build an empty digest. */
  public static TDigest create(int maxSize, String scale) {
    long h = TDigestNative.create(maxSize, requireScale(scale));
    return new TDigest(h);
  }

  /** Build a digest from values (will sort/merge on the Rust side as implemented). */
  public static TDigest fromValues(double[] values, int maxSize, String scale) {
    Objects.requireNonNull(values, "values");
    long h = TDigestNative.createFromValues(values, maxSize, requireScale(scale));
    return new TDigest(h);
  }

  /** Estimate CDF at the provided x's. */
  public double[] cdf(double[] xs) {
    ensureOpen();
    Objects.requireNonNull(xs, "xs");
    return TDigestNative.estimateCdf(state.handle, xs);
  }

  /** Estimate the q-quantile for q in [0,1]. */
  public double quantile(double q) {
    ensureOpen();
    return TDigestNative.estimateQuantile(state.handle, q);
  }

  /** Explicitly release native memory. Safe to call multiple times. */
  @Override public void close() {
    // idempotent: calling close() twice is a no-op
    if (state != null && state.handle != 0) {
      cleanable.clean();   // runs State.run() exactly once
    }
  }

  /** True if already closed (or never successfully opened). */
  public boolean isClosed() {
    return state == null || state.handle == 0;
  }

  private void ensureOpen() {
    if (isClosed()) throw new IllegalStateException("TDigest is closed");
  }

  private static String requireScale(String s) {
    if (s == null) throw new NullPointerException("scale");
    // accept common variants; normalize to what Rust expects (lowercase)
    String v = s.trim().toLowerCase();
    switch (v) {
      case "quad": case "k1": case "k2": case "k3": return v;
      default: throw new IllegalArgumentException("scale must be one of: quad, k1, k2, k3");
    }
  }
}
