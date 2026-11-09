// One lane per independent group. Each round reads the next child and leaves
// the compact accumulator in device memory for the following round.
struct Child {
    const Row *rows;
    Stats stats;
};
struct Group {
    size_t first, count, scratch, stride, output, limit;
    Config cfg;
};
static constexpr unsigned group_threads = 64;
extern "C" const Row *gr_gpu_rows(const Snapshot *s) { return s->rows; }

// Intermediate states do not visit Rust's validate_state. Apply its numeric
// bounds here so an invalid intermediate cannot be hidden by a later merge.
__device__ static bool valid_intermediate(const Cent *cs, Stats s) {
    if (!s.rows || !isfinite(s.sum) || !(s.count > 0.) || !isfinite(s.count)) return false;
    const double mag = fmax(fabs(s.min), fabs(s.max)), avg = s.sum / s.count;
    const double quantum = mag >= 0x1p-149 ? 0x1p-149 : 0x1p-1074;
    const double tol = 4. * FLT_EPSILON * mag + quantum;
    const bool point = cs[0].atomic && cs[0].mean == s.min && point_underflow(s);
    if (!point && (!isfinite(avg) || avg < s.min - tol || avg > s.max + tol)) return false;
    double total = 0., prev = -INFINITY;
    Sum moment;
    bool atomic = true;
    for (size_t i = 0; i < s.rows; ++i) {
        Cent c = cs[i];
        if (!(c.mean > prev && c.mean >= s.min && c.mean <= s.max)) return false;
        prev = c.mean;
        total += c.weight;
        atomic &= c.atomic != 0;
        if (mag) moment.add((c.mean / mag) * (c.weight / s.count));
    }
    const double terms = double(s.rows) + 2.;
    if (!isfinite(total) || fabs(total - s.count) > DBL_EPSILON * terms * fmax(total, s.count)) return false;
    if (atomic && (s.min != cs[0].mean || s.max != prev)) return false;
    if (!mag) return s.sum == 0.;
    return point || fabs(moment.total() - avg / mag) <= 32. * FLT_EPSILON + 8. * DBL_EPSILON * terms +
                                                            quantum / mag + (0x1p-1074 / s.count) / mag * terms;
}
__global__ static void merge_round(const Group *groups, size_t n, const Child *children, size_t step, Cent *arena,
                                   Stats *states, Cent *results) {
    const size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
    Group g = groups[id];
    if (step >= g.count) return;
    Cent *items = arena + g.scratch, *work = items + g.stride, *acc = work + g.stride;
    Stats prior = step ? states[id] : Stats{};
    if (prior.rows == SIZE_MAX) return;
    Child next = children[g.first + step];
    if (!prior.rows) {
        prior = next.stats;
        for (size_t i = 0; i < prior.rows; ++i) acc[i] = raw_row(next.rows[i]);
    } else if (next.stats.rows) {
        merge_pair(acc, prior, next.rows, next.stats, g.cfg, items, work, acc, &prior, true);
        if (prior.rows > g.limit || !valid_intermediate(acc, prior)) prior.rows = SIZE_MAX;
    }
    states[id] = prior;
    if (step + 1 == g.count && prior.rows != SIZE_MAX)
        for (size_t i = 0; i < prior.rows; ++i) results[g.output + i] = acc[i];
}
// One group per thread. Size a wave of work for this kernel on this device.
extern "C" int gr_gpu_group_limits(int device, size_t *resident, size_t *free_bytes) {
    return on_device(device, [&] {
        if (!*resident) {
            int units, blocks;
            hipError_t err;
            if ((err = hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device)) ||
                (err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                     &blocks, reinterpret_cast<const void *>(merge_round), group_threads, 0)))
                return err;
            *resident = std::max<size_t>(1, size_t(units) * blocks * group_threads);
        }
        size_t total;
        return hipMemGetInfo(free_bytes, &total);
    });
}
extern "C" int gr_gpu_groups(int device, const Group *gs, size_t n, const Child *cs, size_t count, size_t rounds,
                             size_t scratch, size_t slots, Stats *stats, Cent *rows, bool *allocation_oom) {
    return on_device(device, [&] {
        Scratch<Group> groups;
        Scratch<Child> children;
        Scratch<Cent> arena, out;
        Scratch<Stats> states;
        hipError_t err;
        if ((err = groups.alloc(n)) || (err = children.alloc(count)) || (err = arena.alloc(scratch)) ||
            (err = out.alloc(slots)) || (err = states.alloc(n))) {
            if (err == hipErrorOutOfMemory) {
                *allocation_oom = true;
                (void)hipGetLastError();
            }
            return err;
        }
        if ((err = hipMemcpy(groups.p, gs, n * sizeof(Group), hipMemcpyHostToDevice)) ||
            (err = hipMemcpy(children.p, cs, count * sizeof(Child), hipMemcpyHostToDevice)) ||
            (err = hipMemsetAsync(states.p, 0, n * sizeof(Stats), hipStreamPerThread)) ||
            (err = hipMemsetAsync(out.p, 0, slots * sizeof(Cent), hipStreamPerThread)))
            return err;
        for (size_t step = 0; step < rounds; ++step) {
            void *args[] = {&groups.p, &n, &children.p, &step, &arena.p, &states.p, &out.p};
            err = hipLaunchKernel(reinterpret_cast<const void *>(merge_round),
                                  dim3((n + group_threads - 1) / group_threads), dim3(group_threads), args, 0,
                                  hipStreamPerThread);
            if (err != hipSuccess) return err;
        }
        if ((err = hipStreamSynchronize(hipStreamPerThread)) ||
            (err = hipMemcpy(stats, states.p, n * sizeof(Stats), hipMemcpyDeviceToHost)))
            return err;
        return hipMemcpy(rows, out.p, slots * sizeof(Cent), hipMemcpyDeviceToHost);
    });
}
