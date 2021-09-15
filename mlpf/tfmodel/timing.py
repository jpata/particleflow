import numpy as np
import time
import subprocess
import shlex

#pip install only onnxruntime_gpu, not onnxruntime!
import onnxruntime

if __name__ == "__main__":
    EP_list = ['CUDAExecutionProvider']

    nvidia_smi_call = "nvidia-smi --query-gpu=timestamp,name,pci.bus_id,pstate,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 -f nvidia_smi_log.csv"
    p = subprocess.Popen(shlex.split(nvidia_smi_call))

    time.sleep(5)
    onnx_sess = onnxruntime.InferenceSession("model.onnx", providers=EP_list)

    for num_elems in [3200, 6400, 12800, 25600, 12800, 6400, 3200]:
        times = []
        for i in range(250):

            #allocate array in system RAM
            X = np.array(np.random.randn(1, num_elems, 15), np.float32)
            
            #transfer data to GPU, run model, transfer data back
            t0 = time.time()
            pred_onx = onnx_sess.run(None, {"x:0": X})
            t1 = time.time()
            dt = t1 - t0
            times.append(dt)

        print("Nelem={} mean_time={:.2f}ms stddev_time={:.2f} ms".format(num_elems, 1000.0*np.mean(times), 1000.0*np.std(times)))
        time.sleep(5)

    p.terminate()