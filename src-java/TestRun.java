import gr.tdigest_rs.TDigest;
import java.util.Arrays;

public class TestRun {
  public static void main(String[] args) {

    // --- 1. Manual cleanup -----------------------------------------------
    TDigest d = TDigest.fromValues(new double[]{0, 1, 2, 3}, 100, "k2");
    System.out.println("Manual cleanup example:");
    System.out.println(Arrays.toString(d.cdf(new double[]{0.0, 1.5, 3.0})));
    System.out.println("p50 = " + d.quantile(0.5));
    d.close();

    // --- 2. Automatic cleanup (try-with-resources) ------------------------
    System.out.println("\nAutoCloseable example:");
    try (TDigest digest = TDigest.fromValues(new double[]{0, 1, 2, 3}, 100, "k2")) {
      System.out.println(Arrays.toString(digest.cdf(new double[]{0.0, 1.5, 3.0})));
      System.out.println("p50 = " + digest.quantile(0.5));
    } // digest.close() is called automatically here
  }
}
