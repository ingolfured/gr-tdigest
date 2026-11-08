#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstring>
#include <new>

struct Row { double mean, weight, prefix; uint64_t atomic; };
struct Snapshot {
    int device;
    Row *rows = nullptr;
    double *host = nullptr, *mapped = nullptr;
    hipStream_t stream = nullptr;
    size_t count, capacity = 0;
    double total, min, max;
    Snapshot(int id, size_t n, double mass, double lo, double hi)
        : device(id), count(n), total(mass), min(lo), max(hi) {}
    ~Snapshot() {
        if (stream) (void)hipStreamDestroy(stream);
        if (host) (void)hipHostFree(host);
        if (rows) (void)hipFree(rows);
    }
};

template<class F> static hipError_t on_device(int device, F operation) {
    int previous;
    auto err = hipGetDevice(&previous);
    if (err != hipSuccess) return err;
    if (previous != device && (err = hipSetDevice(device)) != hipSuccess) return err;
    err = operation();
    if (previous != device) {
        auto restore = hipSetDevice(previous);
        if (err == hipSuccess) err = restore;
    }
    return err;
}

extern "C" int gr_gpu_upload(int device, const Row *rows, size_t n, double total, double lo,
                             double hi, Snapshot **out) {
    return on_device(device, [&] {
        std::unique_ptr<Snapshot> s(new(std::nothrow) Snapshot(device,n,total,lo,hi));
        if (!s) return hipErrorOutOfMemory;
        auto stream_err = hipStreamCreateWithFlags(&s->stream, hipStreamNonBlocking);
        if (stream_err != hipSuccess) return stream_err;
        if (n) {
            auto err = hipMalloc(&s->rows, n*sizeof(Row));
            if (err != hipSuccess) return err;
            err = hipMemcpy(s->rows, rows, n*sizeof(Row), hipMemcpyHostToDevice);
            if (err != hipSuccess) return err;
        }
        *out = s.release();
        return hipSuccess;
    });
}

extern "C" void gr_gpu_drop(Snapshot *s) {
    (void)on_device(s->device, [&] { delete s; return hipSuccess; });
}
extern "C" const char *gr_gpu_error(int code) { return hipGetErrorString(static_cast<hipError_t>(code)); }

__device__ static double fraction(double x, double a, double b) {
    return isinf(b-a) ? (0.5*x-0.5*a)/(0.5*b-0.5*a) : (x-a)/(b-a);
}
__device__ static double cdf(const Row *rows, size_t n, double total, double min, double max, double x) {
    if (!n || isnan(x)) return NAN;
    if (x < min) return 0.;
    if (x > max) return 1.;
    size_t lo=0, hi=n;
    while (lo<hi) {
        size_t mid=lo+(hi-lo)/2;
        if (rows[mid].mean < x) lo=mid+1; else hi=mid;
    }
    if (lo<n && rows[lo].mean==x) return (rows[lo].prefix+0.5*rows[lo].weight)/total;
    if (lo==0) return rows[0].mean>min ? fraction(x,min,rows[0].mean)*(0.5*(rows[0].weight/total)) : 0.;
    if (lo==n) return max>rows[n-1].mean ? 1.-fraction(x,rows[n-1].mean,max)*(0.5*(rows[n-1].weight/total)) : 1.;
    const Row l=rows[lo-1], r=rows[lo];
    const double le=l.atomic ? l.weight*0.5 : 0., re=r.atomic ? r.weight*0.5 : 0.;
    const double span=0.5*(l.weight+r.weight)-le-re;
    return (l.prefix+l.weight*0.5+le+span*fraction(x,l.mean,r.mean))/total;
}
__global__ static void cdf_batch(const Row *rows, size_t n, double total, double min, double max,
                                 double *xs, size_t size) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<size; i+=size_t(gridDim.x)*blockDim.x)
        xs[i]=cdf(rows,n,total,min,max,xs[i]);
}

extern "C" int gr_gpu_cdf(Snapshot *s, const double *xs, const double **out, size_t n) {
    if (!n) return hipSuccess;
    return on_device(s->device, [&] {
        if (n>s->capacity) {
            double *next, *mapped;
            auto err=hipHostMalloc(&next,n*sizeof(double),hipHostMallocMapped);
            if (err!=hipSuccess) return err;
            err=hipHostGetDevicePointer(reinterpret_cast<void**>(&mapped),next,0);
            if (err!=hipSuccess) { (void)hipHostFree(next); return err; }
            if (s->host) (void)hipHostFree(s->host);
            s->host=next; s->mapped=mapped; s->capacity=n;
        }
        std::memcpy(s->host,xs,n*sizeof(double));
        const unsigned blocks=static_cast<unsigned>(std::min<size_t>((n+255)/256,65535));
        void *args[]={&s->rows,&s->count,&s->total,&s->min,&s->max,&s->mapped,&n};
        auto err=hipLaunchKernel(reinterpret_cast<const void*>(cdf_batch),dim3(blocks),dim3(256),args,0,s->stream);
        if (err!=hipSuccess) return err;
        err=hipStreamSynchronize(s->stream);
        if (err==hipSuccess) *out=s->host;
        return err;
    });
}
