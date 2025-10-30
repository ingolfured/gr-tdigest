package gr;
import java.util.Arrays;

import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;

public class TestRun {
  public static void main(String[] args) {
    // --- 1) Manual cleanup -----------------------------------------------
    TDigest d = TDigest
        .builder()
        .maxSize(1000)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)  // or EDGES with keep()
        .precision(Precision.F64)
        .build(new double[]{0, 1, 2, 3});     // <-- use build(double[])
    System.out.println("Manual cleanup example:");
    System.out.println(Arrays.toString(d.cdf(new double[]{0.0, 1.5, 3.0})));
    System.out.println("p50 = " + d.quantile(0.5));
    d.close();

    // --- 2) AutoCloseable (try-with-resources) ----------------------------
    System.out.println("\nAutoCloseable example:");
    try (TDigest digest = TDigest.builder()
        .maxSize(100)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.EDGES).keep(4)
        .precision(Precision.F32) // internal f32 sketch; API still uses double[] probes
        .build(new float[]{0, 1, 2, 3})) {    // <-- use build(float[])
      System.out.println(Arrays.toString(digest.cdf(new double[]{0.0, 1.5, 3.0})));
      System.out.println("p50 = " + digest.quantile(0.5));
    }
  }
}
