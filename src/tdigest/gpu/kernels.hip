#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstring>
#include <new>

struct Row { double mean, weight, prefix; uint64_t atomic; double center, boundary, raw_weight; };
struct Snapshot {
    int device;
    Row *rows = nullptr;
    double *host = nullptr, *mapped = nullptr;
    // Calls synchronize before returning. Share a per-thread stream across snapshots.
    hipStream_t stream = hipStreamPerThread;
    int *invalid = nullptr;
    size_t count, capacity = 0;
    double total, count_mass, min, max;
    Snapshot(int id, size_t n, double mass, double count_mass, double lo, double hi)
        : device(id), count(n), total(mass), count_mass(count_mass), min(lo), max(hi) {}
    ~Snapshot() {
        if (host) (void)hipHostFree(host);
        if (rows) (void)hipFree(rows);
        if (invalid) (void)hipFree(invalid);
    }
};

template<class F> static hipError_t on_device(int device, F operation) {
    int previous;
    auto err = hipGetDevice(&previous);
    if (err != hipSuccess) return err;
    if (previous != device && (err = hipSetDevice(device)) != hipSuccess) {
        // hipCUB reads the thread-local launch status internally. Consume the
        // ordinal error from this call before it reaches an unrelated sort.
        (void)hipGetLastError();
        return err;
    }
    err = operation();
    if (previous != device) {
        auto restore = hipSetDevice(previous);
        if (err == hipSuccess) err = restore;
    }
    return err;
}

extern "C" int gr_gpu_upload(int device, const Row *rows, size_t n, double total, double count_mass, double lo,
                             double hi, Snapshot **out) {
    return on_device(device, [&] {
        std::unique_ptr<Snapshot> s(new(std::nothrow) Snapshot(device,n,total,count_mass,lo,hi));
        if (!s) return hipErrorOutOfMemory;
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
__device__ static double interpolate(double a, double b, double t) {
    if (t<=0) return a;
    if (t>=1) return b;
    return signbit(a)==signbit(b) ? a+(b-a)*t : a*(1.-t)+b*t;
}
__device__ static double quantile(const Row *rows, size_t n, double mass, double min, double max, double q) {
    if (!n) return NAN;
    if (q==0) return min;
    if (q==1) return max;
    const Row first=rows[0], last=rows[n-1];
    if (n==1) {
        if (first.atomic) return first.mean;
        return q<=0.5 ? interpolate(min,first.mean,2*q) : interpolate(first.mean,max,2*q-1);
    }
    double target=q*mass;
    if (target<=first.center) return first.atomic ? first.mean : interpolate(min,first.mean,target/first.center);
    if (target>=mass-last.weight/2) return last.atomic ? last.mean : interpolate(last.mean,max,1-(mass-target)/(last.weight/2));
    size_t lo=1,hi=n;
    while (lo<hi) {
        size_t mid=lo+(hi-lo)/2;
        if (rows[mid].center<=target) lo=mid+1; else hi=mid;
    }
    const size_t right=lo==n ? n-1 : lo;
    const Row l=rows[right-1],r=rows[right];
    const double span=(l.weight+r.weight)/2;
    const double start=lo==n ? mass-last.weight/2-span : l.center;
    const double end=start+span;
    if (l.atomic && r.atomic)
        return target<l.boundary ? l.mean : target>l.boundary ? r.mean : interpolate(l.mean,r.mean,0.5);
    if (l.atomic && start-l.weight/2<target && target<start+l.weight/2) return l.mean;
    if (r.atomic && end-r.weight/2<target && target<end+r.weight/2) return r.mean;
    const double w1=target-start-(l.atomic ? l.weight/2 : 0.);
    const double w2=end-target-(r.atomic ? r.weight/2 : 0.);
    return interpolate(l.mean,r.mean,w1+w2<=0 ? 0.5 : w1/(w1+w2));
}
__global__ static void query_batch(const Row *rows,size_t n,double total,double count,double min,double max,
                                   bool quantiles,const double *xs,double *out,size_t size,int *invalid) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<size; i+=size_t(gridDim.x)*blockDim.x) {
        const double x=xs[i];
        if (invalid && !(x>=0. && x<=1.)) { atomicExch(invalid,1); continue; }
        out[i]=quantiles ? quantile(rows,n,count,min,max,x) : cdf(rows,n,total,min,max,x);
    }
}

static hipError_t launch(Snapshot *s, bool quantiles, const double *src, double *dst, size_t n,int *invalid) {
    const unsigned blocks=static_cast<unsigned>(std::min<size_t>((n+255)/256,65535));
    void *args[]={&s->rows,&s->count,&s->total,&s->count_mass,&s->min,&s->max,&quantiles,&src,&dst,&n,&invalid};
    auto err=hipLaunchKernel(reinterpret_cast<const void*>(query_batch),dim3(blocks),dim3(256),args,0,s->stream);
    return err==hipSuccess ? hipStreamSynchronize(s->stream) : err;
}

extern "C" int gr_gpu_query(Snapshot *s, bool quantiles, const double *xs, const double **out, size_t n) {
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
        auto err=launch(s,quantiles,s->mapped,s->mapped,n,nullptr);
        if (err==hipSuccess) *out=s->host;
        return err;
    });
}

extern "C" int gr_gpu_alloc(int device, const double *xs, size_t n, double **out) {
    return on_device(device,[&] {
        if (!n) return hipSuccess;
        auto err=hipMalloc(out,n*sizeof(double));
        if (err==hipSuccess && xs) err=hipMemcpy(*out,xs,n*sizeof(double),hipMemcpyHostToDevice);
        return err;
    });
}
extern "C" void gr_gpu_free(int device, double *ptr) {
    if (ptr) (void)on_device(device,[&] { return hipFree(ptr); });
}
extern "C" int gr_gpu_download(int device, const double *xs, double *out, size_t n) {
    if (!n) return hipSuccess;
    return on_device(device,[&] { return hipMemcpy(out,xs,n*sizeof(double),hipMemcpyDeviceToHost); });
}
extern "C" int gr_gpu_resident_query(Snapshot *s,bool quantiles,const double *xs,double *out,size_t n) {
    if (!n) return hipSuccess;
    int invalid=0;
    int status=on_device(s->device,[&] {
        if (!quantiles) return launch(s,false,xs,out,n,nullptr);
        if (!s->invalid) {
            auto err=hipMalloc(&s->invalid,sizeof(int));
            if (err!=hipSuccess) return err;
        }
        auto err=hipMemsetAsync(s->invalid,0,sizeof(int),s->stream);
        if (err!=hipSuccess) return err;
        err=launch(s,true,xs,out,n,s->invalid);
        if (err!=hipSuccess) return err;
        return hipMemcpy(&invalid,s->invalid,sizeof(int),hipMemcpyDeviceToHost);
    });
    return status==hipSuccess && invalid ? -1 : status;
}

#include "training.hpp"
#include "merge.hpp"
#include "groups.hpp"
