package gr;

import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;

/**
 * Minimal runtime smoke entrypoint for packaged JAR/native loading checks.
 */
public final class SmokeMain {
  private SmokeMain() {}

  public static void main(String[] args) {
    try (TDigest d = TDigest.builder()
        .maxSize(64)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      double q50 = d.quantile(0.5);
      double cdf2 = d.cdf(new double[] {2.0})[0];
      if (!Double.isFinite(q50) || !Double.isFinite(cdf2)) {
        throw new IllegalStateException("Smoke check failed: non-finite outputs");
      }
      System.out.println("ok," + q50 + "," + cdf2);
    }
  }
}
