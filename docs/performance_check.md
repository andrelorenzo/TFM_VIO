# Performance comparisions between libraries

## GPU
model-small.onnx => [OpenCV] frames=400 | avg=13.337 ms | min=10.876 ms | max=18.559 ms | FPS=74.98
model-small.onnx => [ORT] frames=400 | avg=11.242 ms | min=10.984 ms | max=15.099 ms | FPS=88.95

model-f6b98070.onnx => [ORT] frames=400 | avg=112.439 ms | min=96.194 ms | max=126.326 ms | FPS=8.89
model-f6b98070.onnx => [OpenCV] frames=400 | avg=137.412 ms | min=115.193 ms | max=156.157 ms | FPS=7.28

Depth-Anything-V2.onnx => [OpenCV] frames=400 | avg=116.233 ms | min=109.870 ms | max=133.522 ms | FPS=8.60
Depth-Anything-V2.onnx => [ORT] frames=400 | avg=116.149 ms | min=101.537 ms | max=130.958 ms | FPS=8.61

depth_anything_v2_vits.onnx => [ORT] frames=400 | avg=95.016 ms | min=86.731 ms | max=100.242 ms | FPS=10.52
depth_anything_vits14.quant.onnx  => [ORT] frames=400 | avg=312.255 ms | min=261.587 ms | max=539.990 ms | FPS=3.20
DA3METRIC-LARGE.onnx => [ORT] frames=400 | avg=303.545 ms | min=215.126 ms | max=396.304 ms | FPS=3.29

model-hybrid-fp16.onnx => [ORT] frames=400 | avg=113.172 ms | min=81.993 ms | max=372.221 ms | FPS=8.84
model-hybrid.onnx => [ORT] frames=400 | avg=179.939 ms | min=126.693 ms | max=281.930 ms | FPS=5.56
model_fp16.onnx => [ORT] frames=400 | avg=76.741 ms | min=64.493 ms | max=104.587 ms | FPS=13.03

(WINNER) depth_anything/DA3METRIC-SMALL.onnx => [ORT-DA3] frames=400 | avg=56.765 ms | min=43.850 ms | max=91.179 ms | FPS=17.62 
(WINNER | NO VIDEO) depth_anything/DA3METRIC-SMALL.onnx => [ORT-DA3] frames=400 | avg=51.283 ms | min=42.439 ms | max=64.330 ms | FPS=19.50

## CPU
model-small.onnx => [OpenCV] frames=400 | avg=85.849 ms | min=67.114 ms | max=143.330 ms | FPS=11.65
model-small.onnx => [ORT] frames=400 | avg=20.011 ms | min=16.870 ms | max=28.432 ms | FPS=49.97
