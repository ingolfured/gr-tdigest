package gr.tdigest_rs;

final class TDigestNative {
  static {
    Natives.load();
  }

  // Handle lifecycle
  public static native long  create(int maxSize, String scale);
  public static native long  createFromValues(double[] values, int maxSize, String scale);
  public static native void  free(long handle);

  // Queries
  public static native double[] estimateCdf(long handle, double[] xs);
  public static native double   estimateQuantile(long handle, double q);

  private TDigestNative() {}
}
