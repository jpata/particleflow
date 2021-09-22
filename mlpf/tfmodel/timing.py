import numpy as np
import time
import pynvml

#pip install only onnxruntime_gpu, not onnxruntime!
import onnxruntime

if __name__ == "__main__":
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    EP_list = ['CUDAExecutionProvider']

    time.sleep(5)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_initial = mem.used/1000/1000
    print("mem_initial", mem_initial)
    
    onnx_sess = onnxruntime.InferenceSession("model.onnx", providers=EP_list)
    time.sleep(5)
    
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_onnx = mem.used/1000/1000
    print("mem_onnx", mem_initial)

    for num_elems in range(1600, 25600, 320):
        times = []
        mem_used = []
        
        #average over 100 events
        for i in range(100):

            #allocate array in system RAM
            X = np.array(np.random.randn(1, num_elems, 18), np.float32)
            
            #transfer data to GPU, run model, transfer data back
            t0 = time.time()
            pred_onx = onnx_sess.run(None, {"x:0": X})
            t1 = time.time()
            dt = t1 - t0
            times.append(dt)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used.append(mem.used/1000/1000)

        print("Nelem={} mean_time={:.2f} ms stddev_time={:.2f} ms mem_used={:.0f} MB".format(num_elems, 1000.0*np.mean(times), 1000.0*np.std(times), np.max(mem_used)))
        time.sleep(5)
