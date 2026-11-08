// Included by kernels.hip to share snapshot, stream and device ownership.
#include "compress.hpp"
#include <cfloat>
#include <climits>
#include <hipcub/device/device_radix_sort.hpp>

struct Stats {
    double sum, count, min, max;
    size_t rows;
};
template <class T> struct Scratch {
    T *p = nullptr;
    hipError_t alloc(size_t n) { return hipMalloc(&p, n * sizeof(T)); }
    ~Scratch() {
        if (p) (void)hipFree(p);
    }
};
__device__ static uint64_t sort_key(double x) {
    uint64_t bits = uint64_t(__double_as_longlong(x));
    return bits ^ ((bits >> 63) ? UINT64_MAX : (uint64_t(1) << 63));
}
__device__ static double key_mean(uint64_t key) {
    return __longlong_as_double(key ^ ((key >> 63) ? (uint64_t(1) << 63) : UINT64_MAX));
}
__global__ static void prepare(const double *xs, const double *ws, size_t n, bool narrow, uint64_t *keys,
                               double *weights) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        double x = narrow ? double(float(xs[i])) : xs[i];
        keys[i] = sort_key(xs[i] != 0. && x == 0. ? NAN : x);
        weights[i] = ws ? ws[i] : 1.;
    }
}
__device__ static Cent raw_row(Row r) { return {r.mean, r.raw_weight, r.atomic}; }
__device__ static bool point_underflow(Stats s) {
    return s.rows == 1 && s.min == s.max && s.min != 0. && fabs(s.sum) < DBL_MIN && s.min * s.count == s.sum;
}
__device__ static void append(Cent *items, size_t &k, Sum &run, Cent c) {
    if (k && items[k - 1].mean == c.mean) {
        run.add(c.weight);
        Cent &last = items[k - 1];
        last.weight += c.weight;
        if (!isfinite(last.weight)) last.weight = run.total();
        last.atomic &= c.atomic;
    } else {
        run = Sum{};
        run.add(c.weight);
        items[k++] = c;
    }
}
__device__ static void finish(Cent *items, size_t k, Cent *work, Cent *out, Config cfg, Stats stats, Stats *result,
                              bool cache_trials = false) {
    if (!valid_stream(items, k)) return;
    stats.rows = compress(items, k, work, out, cfg, cache_trials);
    if (stats.rows && valid_stream(out, stats.rows)) *result = stats;
}
__global__ static void train(const Row *old, Stats prior, Config cfg, const uint64_t *keys, const double *weights,
                             size_t n, bool weighted, Cent *items, Cent *work, Cent *out, Stats *result) {
    result->rows = 0; // A nonempty batch always yields at least one row when valid.
    Sum moment, mass;
    if (!weighted) moment.add(prior.sum);
    bool underflow = false;
    for (size_t i = 0; i < n; ++i) {
        double x = key_mean(keys[i]), w = weights[i], product = x * w;
        if (!isfinite(x) || !(w > 0) || !isfinite(w) || !isfinite(product)) return;
        underflow |= x != 0. && product == 0.;
        mass.add(w);
        moment.add(product);
    }
    double lo = key_mean(keys[0]), hi = key_mean(keys[n - 1]);
    double count = prior.count + mass.total(), sum = moment.total();
    if (prior.rows) {
        // Rust keeps the existing extrema on ties.
        lo = lo < prior.min ? lo : prior.min;
        hi = hi > prior.max ? hi : prior.max;
    }
    if (underflow && lo != hi) return;
    if (weighted) sum += prior.sum;
    if (weighted && (underflow || (point_underflow(prior) && old[0].atomic)) && lo == hi) sum = lo * count;
    if (!isfinite(count) || !isfinite(sum)) return;

    size_t i = 0, j = 0, k = 0;
    Sum run;
    while (i < prior.rows || j < n) {
        Cent c;
        if (j == n || (i < prior.rows && old[i].mean <= key_mean(keys[j])))
            c = raw_row(old[i++]);
        else {
            c = {key_mean(keys[j]), weights[j], 1};
            ++j;
        }
        append(items, k, run, c);
    }
    finish(items, k, work, out, cfg, {sum, count, lo, hi, 0}, result);
}
static hipError_t run_compressor(hipStream_t stream, const void *kernel, void **args, Stats *result, Cent *out,
                                 Cent *host, size_t limit, Stats *stats) {
    hipError_t err;
    if ((err = hipLaunchKernel(kernel, dim3(1), dim3(1), args, 0, stream)) ||
        (err = hipStreamSynchronize(stream)) ||
        (err = hipMemcpy(stats, result, sizeof(Stats), hipMemcpyDeviceToHost)))
        return err;
    if (stats->rows > limit) return hipErrorInvalidValue;
    return hipMemcpy(host, out, stats->rows * sizeof(Cent), hipMemcpyDeviceToHost);
}
extern "C" int gr_gpu_train(Snapshot *s, Stats prior, Config cfg, const double *xs, const double *ws, size_t n,
                            Cent *output, size_t limit, Stats *stats) {
    if (n > INT_MAX || s->count > size_t(INT_MAX) - n) return hipErrorInvalidValue;
    return on_device(s->device, [&] {
        Scratch<uint64_t> keys, sorted_keys;
        Scratch<double> weights, sorted_weights;
        Scratch<Cent> items, work, out;
        Scratch<Stats> result;
        Scratch<char> radix;
        auto stream = s->stream;
        const size_t total = s->count + n;
        hipError_t err;
        if ((err = keys.alloc(n)) || (err = sorted_keys.alloc(n)) || (err = weights.alloc(n)) ||
            (err = sorted_weights.alloc(n)) || (err = items.alloc(total)) || (err = work.alloc(total)) ||
            (err = out.alloc(total)) || (err = result.alloc(1)))
            return err;
        size_t temp = 0;
        auto sort = [&](void *buf) {
            return hipcub::DeviceRadixSort::SortPairs(buf, temp, keys.p, sorted_keys.p, weights.p,
                                                      sorted_weights.p, int(n), 0, 64, stream);
        };
        if ((err = sort(nullptr)) || (err = radix.alloc(std::max<size_t>(1, temp)))) return err;
        bool weighted = ws != nullptr, narrow = cfg.narrow;
        void *args[] = {&xs, &ws, &n, &narrow, &keys.p, &weights.p};
        unsigned blocks = std::min<size_t>((n + 255) / 256, 65535);
        if ((err = hipLaunchKernel(reinterpret_cast<const void *>(prepare), dim3(blocks), dim3(256), args, 0,
                                   stream)) ||
            (err = sort(radix.p)))
            return err;
        void *training[] = {&s->rows,  &prior,   &cfg,    &sorted_keys.p, &sorted_weights.p, &n,
                            &weighted, &items.p, &work.p, &out.p,         &result.p};
        return run_compressor(stream, reinterpret_cast<const void *>(train), training, result.p, out.p, output,
                              limit, stats);
    });
}
