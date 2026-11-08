// Compile this independent of Python to exercise the device training ABI.
#include "../../src/tdigest/gpu/kernels.hip"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int device = argc == 2 ? std::atoi(argv[1]) : 0;
    const double xs[] = {2., -1., 2., 4.}, ws[] = {.25, 2., .75, 1.};
    double *values = nullptr, *weights = nullptr;
    Snapshot *snapshot = nullptr;
    if (gr_gpu_alloc(device, xs, 4, &values) || gr_gpu_alloc(device, ws, 4, &weights) ||
        gr_gpu_upload(device, nullptr, 0, 0., 0., NAN, NAN, &snapshot))
        return 1;
    bool ok = true;
    for (size_t cap : {1, 2, 3}) {
        Config config = {cap, 0, 2, 0, 0, 0.};
        Stats stats{};
        Cent rows[4];
        int err = gr_gpu_train(snapshot, {}, config, values, weights, 4, rows, 4, &stats);
        ok &= !err && stats.rows == cap && stats.sum == 4. && stats.count == 4. && stats.min == -1. &&
              stats.max == 4.;
        if (ok && cap == 1) ok = rows[0].mean == 1. && rows[0].weight == 4. && !rows[0].atomic;
        if (ok && cap == 2) ok = rows[0].mean == -1. && rows[0].atomic && rows[1].mean == 3. && !rows[1].atomic;
        if (ok && cap == 3) ok = rows[1].mean == 2. && rows[1].weight == 1. && rows[1].atomic;
    }
    gr_gpu_drop(snapshot);
    gr_gpu_free(device, values);
    gr_gpu_free(device, weights);
    std::printf("device %d compression fixture %s\n", device, ok ? "passed" : "failed");
    return ok ? 0 : 1;
}
