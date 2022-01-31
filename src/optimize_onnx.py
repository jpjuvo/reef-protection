import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType
import argparse

def main(onnx_pth):

    opt_onnx = onnx_pth.replace('.onnx', '_opt.onnx')
    quant_onnx = onnx_pth.replace('.onnx', '_quant.onnx')

    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = opt_onnx

    # https://onnxruntime.ai/docs/performance/tune-performance.html#convolution-heavy-models-and-the-cuda-ep
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]

    session = rt.InferenceSession(onnx_pth, sess_options, providers=providers)

    # quantization
    quantized_model = quantize_dynamic(
        opt_onnx, 
        quant_onnx, 
        weight_type=QuantType.QUInt8
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[])
    
    parser.add_argument('--onnx_pth', type=str)

    params = parser.parse_args()
    onnx_pth = params.onnx_pth
    
    main(onnx_pth)