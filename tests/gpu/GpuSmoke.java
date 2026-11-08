import gr.tdigest.Gpu;
import gr.tdigest.TDigest;
import java.util.Arrays;

public final class GpuSmoke {
  public static void main(String[] args) {
    int device = Integer.parseInt(args[0]);
    if (!Gpu.compiled()) throw new AssertionError("HIP JNI library required");
    double[] xs = {3, -2, -2, 0, 5};
    double[] ws = {.5, 2, .25, 1, .75};
    for (TDigest.Precision precision : new TDigest.Precision[]{TDigest.Precision.F32, TDigest.Precision.F64}) {
      TDigest.Builder builder = TDigest.builder().maxSize(40).precision(precision);
      try (TDigest empty = builder.build(new double[0]);
           TDigest cpu = builder.buildWeighted(xs, ws);
           TDigest gpu = Gpu.add(empty, xs, ws, device);
           TDigest expected = TDigest.mergeAll(cpu, cpu);
           TDigest actual = Gpu.mergeAll(new TDigest[]{gpu, gpu}, device);
           TDigest ordinary = TDigest.fromValues(xs, 1000, "k2");
           TDigest built = Gpu.fromArray(xs, device)) {
        if (!Arrays.equals(cpu.toBytes(), gpu.toBytes()) || !Arrays.equals(expected.toBytes(), actual.toBytes()) ||
            !Arrays.equals(ordinary.toBytes(), built.toBytes())) throw new AssertionError("TDIG mismatch");
        double[] qs = {0, .1, .5, .75, 1};
        if (!Arrays.equals(cpu.cdf(xs), Gpu.cdf(gpu, xs, device)) ||
            !Arrays.equals(cpu.quantile(qs), Gpu.quantile(gpu, qs, device)) ||
            cpu.median() != Gpu.median(gpu, device)) throw new AssertionError("query mismatch");
        try {
          Gpu.quantile(gpu, new double[]{Double.NaN}, device);
          throw new AssertionError("invalid probability accepted");
        } catch (IllegalArgumentException expectedError) {
          if (!expectedError.getMessage().contains("quantile")) throw expectedError;
        }
      }
    }
    System.out.println("HIP Java parity passed on device " + device);
  }
}
