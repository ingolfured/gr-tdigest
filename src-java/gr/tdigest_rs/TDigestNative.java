package gr.tdigest_rs;

public final class TDigestNative {
  static { System.loadLibrary("polars_tdigest"); }

  public static native long create(int maxSize, String scale);
  public static native long createFromValues(double[] values, int maxSize, String scale);
  public static native void free(long handle);
  public static native double[] estimateCdf(long handle, double[] xs);
  public static native double   estimateQuantile(long handle, double q);
}
