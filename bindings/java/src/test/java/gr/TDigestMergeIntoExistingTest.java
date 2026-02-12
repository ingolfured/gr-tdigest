package gr;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import gr.tdigest.TDigest;
import gr.tdigest.TDigest.Precision;
import gr.tdigest.TDigest.Scale;
import gr.tdigest.TDigest.SingletonPolicy;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class TDigestMergeIntoExistingTest {
  @Test
  void addDoubleArrayAndScalarRaisesMedianForLargerValues() {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F64)
        .build(new double[] {0.0, 1.0, 2.0, 3.0})) {
      double before = d.quantile(0.5);
      d.add(new double[] {10.0, 11.0, 12.0}).add(13.0);
      double after = d.quantile(0.5);
      assertTrue(after > before, "expected median to increase after adding larger values");
    }
  }

  @Test
  void addFloatArrayAndScalarRaisesMedianForLargerValues() {
    try (TDigest d = TDigest.builder()
        .maxSize(256)
        .scale(Scale.K2)
        .singletonPolicy(SingletonPolicy.USE)
        .precision(Precision.F32)
        .build(new float[] {0.0f, 1.0f, 2.0f, 3.0f})) {
      double before = d.quantile(0.5);
      d.add(new float[] {10.0f, 11.0f, 12.0f}).add(13.0f);
      double after = d.quantile(0.5);
      assertTrue(after > before, "expected median to increase after adding larger float values");
    }
  }

  @Test
  void mergeTwoF64DigestsMatchesCombinedDistribution() {
    try (TDigest left = TDigest.builder()
            .maxSize(256)
            .scale(Scale.K2)
            .singletonPolicy(SingletonPolicy.USE)
            .precision(Precision.F64)
            .build(new double[] {0.0, 1.0, 2.0, 3.0});
         TDigest right = TDigest.builder()
            .maxSize(256)
            .scale(Scale.K2)
            .singletonPolicy(SingletonPolicy.USE)
            .precision(Precision.F64)
            .build(new double[] {10.0, 11.0, 12.0, 13.0});
         TDigest expected = TDigest.builder()
            .maxSize(256)
            .scale(Scale.K2)
            .singletonPolicy(SingletonPolicy.USE)
            .precision(Precision.F64)
            .build(new double[] {0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0})) {
      left.merge(right);
      assertEquals(expected.quantile(0.5), left.quantile(0.5), 1e-9);
    }
  }

  @Test
  void mergeAllSupportsThreeF32Digests() {
    try (TDigest a = TDigest.builder().precision(Precision.F32).build(new float[] {0f, 1f, 2f, 3f});
         TDigest b = TDigest.builder().precision(Precision.F32).build(new float[] {10f, 11f, 12f, 13f});
         TDigest c = TDigest.builder().precision(Precision.F32).build(new float[] {20f, 21f, 22f, 23f});
         TDigest expected = TDigest.builder().precision(Precision.F32).build(
             new float[] {0f, 1f, 2f, 3f, 10f, 11f, 12f, 13f, 20f, 21f, 22f, 23f});
         TDigest merged = TDigest.mergeAll(a, b, c)) {
      assertEquals(expected.quantile(0.5), merged.quantile(0.5), 1e-6);
    }
  }

  @Test
  void mergeRejectsMixedPrecision() {
    try (TDigest f64 = TDigest.builder().precision(Precision.F64).build(new double[] {1.0, 2.0, 3.0});
         TDigest f32 = TDigest.builder().precision(Precision.F32).build(new float[] {1.0f, 2.0f, 3.0f})) {
      assertThrows(IllegalArgumentException.class, () -> f64.merge(f32));
      assertThrows(IllegalArgumentException.class, () -> TDigest.mergeAll(f64, f32));
    }
  }

  @Test
  void mergeAllIterableMatchesVarargs() {
    try (TDigest a = TDigest.builder().precision(Precision.F64).build(new double[] {0.0, 1.0, 2.0});
         TDigest b = TDigest.builder().precision(Precision.F64).build(new double[] {10.0, 11.0, 12.0});
         TDigest c = TDigest.builder().precision(Precision.F64).build(new double[] {20.0, 21.0, 22.0});
         TDigest byVarargs = TDigest.mergeAll(a, b, c);
         TDigest byIterable = TDigest.mergeAll(Arrays.asList(a, b, c))) {
      assertEquals(byVarargs.quantile(0.5), byIterable.quantile(0.5), 1e-12);
    }
  }

  @Test
  void mergeAllEmptyIterableReturnsCanonicalEmptyDigest() {
    try (TDigest empty = TDigest.mergeAll(Arrays.<TDigest>asList());
         TDigest canonical = TDigest.builder()
             .maxSize(1000)
             .scale(Scale.K2)
             .singletonPolicy(SingletonPolicy.USE)
             .precision(Precision.F64)
             .build(new double[] {})) {
      assertTrue(Double.isNaN(empty.quantile(0.5)));
      empty.merge(canonical); // must not throw config mismatch
      assertTrue(Double.isNaN(empty.quantile(0.5)));
    }
  }
}
