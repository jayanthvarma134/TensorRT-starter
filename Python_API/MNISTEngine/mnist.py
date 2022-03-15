import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
from mnist_test_loader import MNIST_Test

parser = argparse.ArgumentParser(description="Build TensorRT Engine from scratch using the Python API")
parser.add_argument("-e", "--engine", required=True, default="models/sample/sample.engine", help="path to the engine file. If not present, creates one at this path")
parser.add_argument("-pt", "--pytorch", required=False, default="models/sample/sample.pt", help="path to the pytorch file if TRT engine is not present.")
parser.add_argument("-rn", "--render", required=False, default=False, help="render the image that you randomly sampled for testing.")
args = parser.parse_args()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
WORKSPACE_SIZE = 1 << 30 
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def populate_network(network, weights):
    """ 
        Network Architecture :
        Net(
            (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
            (conv2): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
            (fc1): Linear(in_features=800, out_features=500, bias=True)
            (fc2): Linear(in_features=500, out_features=10, bias=True)
        )
    """
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    conv1_w = weights['conv1.weight'].numpy()
    conv1_b = weights['conv1.bias'].numpy()
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv2_w = weights['conv2.weight'].numpy()
    conv2_b = weights['conv2.bias'].numpy()
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights['fc1.weight'].numpy()
    fc1_b = weights['fc1.bias'].numpy()
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights['fc2.weight'].numpy()
    fc2_b = weights['fc2.bias'].numpy()
    fc2 = network.add_fully_connected(relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=fc2.get_output(0))

def build_engine(weights):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    config.max_workspace_size = WORKSPACE_SIZE
    populate_network(network, weights)
    engine = builder.build_engine(network, config)
    serialized_engine = engine.serialize()
    save_engine(args.engine, serialized_engine)
    return runtime.deserialize_cuda_engine(serialized_engine)

def save_engine(engine_file, serialized_engine):
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

    
def load_random_test_case(mnist_test, pagelocked_buffer):
    img, expected_output = mnist_test.get_random_testcase()
    np.copyto(pagelocked_buffer, img)
    return img, expected_output

def run():
    engine_found = os.path.exists(args.engine)
    pytorch_found = os.path.exists(args.pytorch)
    engine = None
    mnist_test = MNIST_Test()

    if engine_found:
        print(f"[ENGINE FOUND]")
        with open(args.engine, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
    else :
        if pytorch_found:
            print(f"[ENGINE NOT FOUND] : \nPopulating tensorrt.INetworkDefinition with weights from pytorch model: {args.pytorch}")
            weights = torch.load(args.pytorch)
            engine = build_engine(weights)
        else:
            print(f"[ARGUMENT ERROR] : Engine not found and pytorch file not found at path. Please check your input arguments.")
            return

    if engine is not None:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        context = engine.create_execution_context()

        case_img, case_num = load_random_test_case(mnist_test, pagelocked_buffer=inputs[0].host)
        [output] = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = np.argmax(output)
        print(f"[Original Value ] : {case_num}")
        print(f"[Prediction] : {pred}")

        # if args.render:
        #     mnist_test.show_test_case(case_img)

if __name__ == "__main__":
    run()