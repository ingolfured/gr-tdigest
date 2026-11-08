package gr.tdigest;

import java.util.Objects;

/** Batched operations on a Linux HIP device. Each call transfers a canonical digest snapshot. */
public final class Gpu {
  static { Natives.load(); }
  private Gpu() {}

  public static native boolean compiled();
  private static native double[] query(int device, byte[] digest, double[] values, boolean quantiles);
  private static native byte[] train(int device, byte[] seed, double[] values, double[] weights);
  private static native byte[] merge(int device, byte[][] children);

  private static void requireHip() {
    if (!compiled()) throw new UnsupportedOperationException("Build the native library with java,hip");
  }
  public static double[] cdf(TDigest digest, double[] probes, int device) {
    requireHip();
    return query(device, digest.toBytes(), Objects.requireNonNull(probes), false);
  }
  public static double[] quantile(TDigest digest, double[] probabilities, int device) {
    requireHip();
    return query(device, digest.toBytes(), Objects.requireNonNull(probabilities), true);
  }
  public static double median(TDigest digest, int device) {
    return quantile(digest, new double[]{0.5}, device)[0];
  }
  /** An empty seed chooses the construction settings. A populated seed appends a batch. */
  public static TDigest add(TDigest seed, double[] values, double[] weights, int device) {
    requireHip();
    return TDigest.fromBytes(train(device, seed.toBytes(), Objects.requireNonNull(values), weights));
  }
  public static TDigest fromArray(double[] values, int device) {
    try (TDigest seed = TDigest.create(1000, "k2")) {
      return add(seed, values, null, device);
    }
  }
  public static TDigest mergeAll(TDigest[] digests, int device) {
    requireHip();
    byte[][] bytes = new byte[digests.length][];
    for (int i = 0; i < bytes.length; i++) bytes[i] = digests[i].toBytes();
    return TDigest.fromBytes(merge(device, bytes));
  }
}
