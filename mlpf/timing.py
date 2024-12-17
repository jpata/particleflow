import time

import numpy as np
import onnxruntime as rt
import resource
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-size", type=int, default=256)
    parser.add_argument("--num-features", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--input-dtype", type=str, default="float32")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--model", type=str, default="test.onnx")
    parser.add_argument(
        "--execution-provider",
        type=str,
        default="CPUExecutionProvider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider"],
    )
    args = parser.parse_args()
    return args


def get_mem_cpu_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000


def get_mem_gpu_mb():
    import pynvml

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem.used / 1000 / 1000


def get_mem_mb(use_gpu):
    if use_gpu:
        return get_mem_gpu_mb()
    else:
        return get_mem_cpu_mb()


if __name__ == "__main__":
    # for GPU testing, you need to
    # pip install only onnxruntime_gpu, not onnxruntime!
    args = parse_args()

    bin_size = args.bin_size
    num_features = args.num_features
    use_gpu = args.execution_provider == "CUDAExecutionProvider"
    batch_size = args.batch_size
    num_threads = args.num_threads

    if use_gpu:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    print("batch_size={} bin_size={} num_features={} use_gpu={} num_threads={}".format(batch_size, bin_size, num_features, use_gpu, num_threads))

    EP_list = [args.execution_provider]

    time.sleep(5)

    mem_initial = get_mem_mb(use_gpu)
    print("mem_initial", mem_initial)

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

    onnx_sess = rt.InferenceSession(args.model, sess_options, providers=EP_list)
    # warmup

    mem_onnx = get_mem_mb(use_gpu)
    print("mem_onnx", mem_onnx)

    X = np.array(np.random.randn(batch_size, bin_size, num_features), getattr(np, args.input_dtype))
    for i in range(10):
        onnx_sess.run(None, {"Xfeat_normed": X, "mask": (X[..., 0] != 0).astype(np.float32)})

    for bin_mul in [
        10,
        20,
        40,
    ]:
        num_elems = bin_size * bin_mul
        times = []
        mem_used = []

        # average over 100 events
        for i in range(100):

            # allocate array in system memory
            X = np.array(np.random.randn(batch_size, num_elems, num_features), getattr(np, args.input_dtype))

            # transfer data to GPU, run model, transfer data back
            t0 = time.time()
            # pred_onx = onnx_sess.run(None, {"Xfeat_normed": X, "l_mask_": X[..., 0]==0})
            pred_onx = onnx_sess.run(None, {"Xfeat_normed": X, "mask": (X[..., 0] != 0).astype(np.float32)})
            t1 = time.time()
            dt = (t1 - t0) / batch_size
            times.append(dt)

            mem_used.append(get_mem_mb(use_gpu))

        print(
            "Nelem={} mean_time={:.2f} ms stddev_time={:.2f} ms mem_used={:.0f} MB".format(
                num_elems,
                1000.0 * np.mean(times),
                1000.0 * np.std(times),
                np.max(mem_used),
            )
        )
        time.sleep(5)
