__device__ static Cent raw_row(Cent c) { return c; }
template <class Left>
__device__ static void merge_pair(const Left *left, Stats a, const Row *right, Stats b, Config cfg, Cent *items,
                                  Cent *work, Cent *out, Stats *result, bool cache_trials = false) {
    result->rows = 0;
    Sum mass, moment, run;
    mass.add(a.count);
    mass.add(b.count);
    moment.add(a.sum);
    moment.add(b.sum);
    Stats stats = {moment.total(), mass.total(), b.min < a.min ? b.min : a.min, a.max > b.max ? a.max : b.max, 0};
    if (!isfinite(stats.sum) || !isfinite(stats.count)) return;
    size_t i = 0, j = 0, k = 0;
    while (i < a.rows || j < b.rows) {
        Cent row = j == b.rows || (i < a.rows && left[i].mean <= right[j].mean) ? raw_row(left[i++])
                                                                                : raw_row(right[j++]);
        append(items, k, run, row);
    }
    if (k == 1 && items[0].atomic && stats.min == stats.max && (point_underflow(a) || point_underflow(b)))
        stats.sum = items[0].mean * stats.count;
    finish(items, k, work, out, cfg, stats, result, cache_trials);
}
__global__ static void merge_sorted(const Row *left, Stats a, const Row *right, Stats b, Config cfg, Cent *items,
                                    Cent *work, Cent *out, Stats *result) {
    merge_pair(left, a, right, b, cfg, items, work, out, result);
}
extern "C" int gr_gpu_merge(Snapshot *left, Stats a, Snapshot *right, Stats b, Config cfg, Cent *output,
                            size_t limit, Stats *stats) {
    return on_device(left->device, [&] {
        size_t n = a.rows + b.rows;
        Scratch<Cent> items, work, out;
        Scratch<Stats> result;
        hipError_t err;
        if ((err = items.alloc(n)) || (err = work.alloc(n)) || (err = out.alloc(n)) || (err = result.alloc(1)))
            return err;
        void *args[] = {&left->rows, &a, &right->rows, &b, &cfg, &items.p, &work.p, &out.p, &result.p};
        return run_compressor(left->stream, reinterpret_cast<const void *>(merge_sorted), args, result.p, out.p,
                              output, limit, stats);
    });
}
