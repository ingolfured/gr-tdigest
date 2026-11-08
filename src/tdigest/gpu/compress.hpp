// The ordered scan mirrors compressor.rs. Trial passes only count groups.
struct Cent {
    double mean, weight;
    uint64_t atomic;
};
struct Config {
    size_t cap, pins;
    int family, policy, narrow;
    double delta;
};
struct Sum {
    double main = 0, correction = 0, scale = 1;
    __device__ void add(double value) {
        value *= scale;
        while (isfinite(main) && isfinite(value) && !isfinite(main + value)) {
            main *= 0.25;
            correction *= 0.25;
            scale *= 0.25;
            value *= 0.25;
        }
        double next = main + value;
        if (isfinite(next))
            correction += fabs(main) >= fabs(value) ? (main - next) + value : (value - next) + main;
        main = next;
    }
    __device__ double total() const { return (main + correction) / scale; }
};
struct Cluster {
    Cent first{};
    size_t count = 0;
    double w = 0, mw = 0, last = 0;
    Sum moment;
    __device__ void add(Cent c) {
        if (!count++) first = c;
        w += c.weight;
        last = c.mean;
        double product = c.mean * c.weight;
        mw += product;
        moment.add(product);
    }
    __device__ Cent finish(bool narrow) const {
        if (count == 1) return first;
        double m = ((first.mean < 0 && last > 0) || !isfinite(mw) ? moment.total() : mw) / w;
        if (isfinite(w) && isfinite(mw) && isfinite(m)) m = fmin(last, fmax(first.mean, m));
        return {narrow ? double(float(m)) : m, w, 0};
    }
};
__device__ static double scale_k(double q, double d, int family, double z) {
    q = fmin(1. - 1e-15, fmax(1e-15, q));
    switch (family) {
    case 0:
        return d * (q < 0.5 ? sqrt(q * 0.5) : 1. - sqrt((1. - q) * 0.5));
    case 1:
        return (d / (2. * 3.141592653589793)) * asin(2. * q - 1.);
    case 2:
        return (d / (4. * 0.6931471805599453)) * log(q / (1. - q));
    case 3:
        return (d / z) * log(q / (1. - q));
    default:
        return (d / 4.) * log(fmax(1e-15, log(1. / (1. - q)) / log(1. / q)));
    }
}
__device__ static double mass(const Cent *cs, size_t n) {
    double total = 0;
    for (size_t i = 0; i < n; ++i) total += cs[i].weight;
    return total;
}
__device__ static bool valid_stream(const Cent *cs, size_t n) {
    if (!isfinite(mass(cs, n))) return false;
    for (size_t i = 0; i < n; ++i)
        if (!isfinite(cs[i].mean) || !(cs[i].weight > 0)) return false;
    return true;
}
__device__ static size_t klimit(const Cent *cs, size_t n, Cent *out, Config cfg, double d) {
    if (!n) return 0;
    double z = cfg.family == 3 ? fma(log(double(n) / d), 4., 24.) : 1.;
    if (z <= 0) {
        if (out)
            for (size_t i = 0; i < n; ++i) out[i] = cs[i];
        return n;
    }
    double total = mass(cs, n), completed = 0, weight = 0, left = scale_k(0, d, cfg.family, z);
    size_t written = 0;
    Cluster cluster;
    for (size_t i = 0; i < n; ++i) {
        double right = scale_k((completed + weight + cs[i].weight) / total, d, cfg.family, z);
        if (!(right - left <= 1. + (cfg.delta ? 0. : 1e-12))) {
            if (weight > 0.) {
                if (out) out[written] = cluster.finish(cfg.narrow);
                ++written;
                completed += weight;
                weight = 0.;
                cluster = Cluster{};
            }
            left = scale_k(completed / total, d, cfg.family, z);
        }
        weight += cs[i].weight;
        if (out) cluster.add(cs[i]);
    }
    if (out) out[written] = cluster.finish(cfg.narrow);
    return written + 1;
}
__device__ static size_t bucketize(const Cent *cs, size_t n, Cent *out, Config cfg, size_t cap) {
    double target = mass(cs, n) / double(cap);
    Cluster acc;
    size_t written = 0;
    for (size_t i = 0; i < n; ++i) {
        acc.add(cs[i]);
        if (acc.w >= target && written < cap - 1) {
            out[written++] = acc.finish(cfg.narrow);
            acc = Cluster{};
        }
    }
    if (acc.count) out[written++] = acc.finish(cfg.narrow);
    return written;
}
__device__ static size_t cap_core(const Cent *core, size_t n, Cent *out, Config cfg, size_t cap) {
    double lo = 1e-6, hi = fmax(lo, double(cfg.cap)), best = lo;
    for (int trial = 0; trial < 28; ++trial) {
        double mid = 0.5 * (lo + hi);
        if (klimit(core, n, nullptr, cfg, mid) <= cap)
            lo = best = mid;
        else
            hi = mid;
    }
    for (int trial = 0; trial <= 12; ++trial) {
        size_t count = klimit(core, n, out, cfg, best);
        if (count <= cap) return count;
        best *= 0.95;
    }
    return bucketize(core, n, out, cfg, cap);
}
// 'items' is already coalesced. 'work' and 'out' each have room for n centroids.
__device__ static size_t compress(const Cent *items, size_t n, Cent *work, Cent *out, Config cfg) {
    if (!cfg.delta && cfg.policy == 0 && n <= cfg.cap) {
        for (size_t i = 0; i < n; ++i) out[i] = items[i];
        return n;
    }
    if (!cfg.delta && cfg.policy != 2 && cfg.cap <= 2) return bucketize(items, n, out, cfg, cfg.cap);
    size_t l = 0, r = n, cap = cfg.cap;
    if (cfg.policy == 1) {
        l = n ? 1 : 0;
        r = n > 1 ? n - 1 : l;
        cap -= l + n - r;
    } else if (cfg.policy == 2) {
        while (l < n && l < cfg.pins && items[l].atomic) ++l;
        while (r > l && n - r < cfg.pins && items[r - 1].atomic) --r;
    }
    for (size_t i = 0; i < l; ++i) out[i] = items[i];
    size_t count = klimit(items + l, r - l, work, cfg, cfg.delta ? cfg.delta : double(cfg.cap));
    if (!valid_stream(work, count)) return 0;
    if (!cfg.delta && count > cap)
        count = cap_core(work, count, out + l, cfg, cap);
    else
        for (size_t i = 0; i < count; ++i) out[l + i] = work[i];
    for (size_t i = r; i < n; ++i) out[l + count + i - r] = items[i];
    return l + count + n - r;
}
