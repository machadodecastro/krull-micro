import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="artifacts/krull_micro.onnx")
    p.add_argument("--out", default="artifacts/krull_micro_int8.onnx")
    args = p.parse_args()
    quantize_dynamic(args.onnx, args.out, weight_type=QuantType.QInt8)
    print(f"Saved INT8 ONNX to {args.out}")

if __name__ == "__main__":
    main()
