# CLIC v1.6

## On CPU

Tested on Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz, single thread.

```
batch_size=20 bin_size=256 num_features=17 use_gpu=False num_threads=1
Nelem=256 mean_time=171.83 ms stddev_time=0.71 ms mem_used=193 MB
Nelem=512 mean_time=343.16 ms stddev_time=3.18 ms mem_used=282 MB
Nelem=2560 mean_time=1689.36 ms stddev_time=6.76 ms mem_used=1056 MB
Nelem=5120 mean_time=3339.05 ms stddev_time=6.48 ms mem_used=2038 MB
Nelem=10240 mean_time=6707.42 ms stddev_time=5.38 ms mem_used=3997 MB
```

On 12 threads
```
batch_size=20 bin_size=256 num_features=17 use_gpu=False num_threads=12
Nelem=256 mean_time=42.46 ms stddev_time=1.02 ms mem_used=169 MB
Nelem=512 mean_time=78.22 ms stddev_time=0.65 ms mem_used=213 MB
Nelem=2560 mean_time=377.50 ms stddev_time=4.07 ms mem_used=612 MB
Nelem=5120 mean_time=740.40 ms stddev_time=4.85 ms mem_used=1181 MB
Nelem=10240 mean_time=1458.50 ms stddev_time=11.97 ms mem_used=2319 MB
```

## On GPU

Tested on RTX2070S 8192MiB.
```
batch_size=20 bin_size=256 num_features=17 use_gpu=True num_threads=1
Nelem=256 mean_time=1.96 ms stddev_time=0.52 ms mem_used=782 MB
Nelem=512 mean_time=3.21 ms stddev_time=0.15 ms mem_used=916 MB
Nelem=2560 mean_time=21.48 ms stddev_time=0.14 ms mem_used=1721 MB
Nelem=5120 mean_time=50.09 ms stddev_time=0.19 ms mem_used=2795 MB
Nelem=10240 mean_time=106.15 ms stddev_time=0.63 ms mem_used=4943 MB
```
