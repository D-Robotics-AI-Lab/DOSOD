## The Usage for building model on X5 ai toolchain

### 0. Run on AI toolchain docker image at first

Get the support from D-Robotics FAE.

### 1. Generate calibration data

```shell
cd /root/DOSOD/ai_toolchain/x5

wget https://huggingface.co/D-Robotics/DOSOD/resolve/main/coco_val_100.tar
# or 'wget https://modelscope.cn/models/D-Robotics/DOSOD/resolve/master/coco_val_100.tar'
tar -xvf coco_val_100.tar

python3 gen_calibration_data.py
```

### 2. Prepare onnx model

Option 1: Get the onnx model from [4.1 Train and export model](https://github.com/D-Robotics-AI-Lab/DOSOD)

Option 2: Download the onnx model from [DOSOD](https://huggingface.co/D-Robotics/DOSOD)

```shell
cd /root/DOSOD/ai_toolchain/x5

wget https://huggingface.co/D-Robotics/DOSOD/resolve/main/dosod_mlp3x_l_rep.onnx
# or 'wget https://modelscope.cn/models/D-Robotics/DOSOD/resolve/master/dosod_mlp3x_l_rep.onnx'
```

### 3. Quantization

```shell
cd /root/DOSOD/ai_toolchain/x5
hb_mapper makertbin -c con_DOSOD_L.yaml --model-type onnx
```

### 4. Verification

```shell
cd /root/DOSOD/ai_toolchain/x5
python3 infer_quant.py --image_path ./000000162415.jpg --onnx_quant_path ./model_output_l/dosod_mlp3x_l_rep-int16_quantized_model.onnx
```

### 5. Run on board

Follow [hobot_dosod](https://github.com/D-Robotics/hobot_dosod) or [DOSOD Usage](https://developer.d-robotics.cc/rdk_doc/Robot_development/boxs/detection/hobot_dosod). Run on RDK X5 board.

```shell
source /opt/tros/humble/setup.bash
ros2 launch hobot_dosod dosod.launch.py dosod_model_file_name:="config/dosod_mlp3x_l_rep-int16.bin"
```