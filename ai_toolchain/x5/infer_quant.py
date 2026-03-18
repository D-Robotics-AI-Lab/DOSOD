import os
import cv2
import numpy as np
import os.path as osp
from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.data.transformer import RGB2YUV444Transformer, BGR2NV12Transformer, NV12ToYUV444Transformer, RGB2NV12Transformer
import argparse

from preprocess import preprocess_custom


classes_coco_str = """
    person
    bicycle
    car
    motorbike
    aeroplane
    bus
    train
    truck
    boat
    traffic light
    fire hydrant
    stop sign
    parking meter
    bench
    bird
    cat
    dog
    horse
    sheep
    cow
    elephant
    bear
    zebra
    giraffe
    backpack
    umbrella
    handbag
    tie
    suitcase
    frisbee
    skis
    snowboard
    sports ball
    kite
    baseball bat
    baseball glove
    skateboard
    surfboard
    tennis racket
    bottle
    wine glass
    cup
    fork
    knife
    spoon
    bowl
    banana
    apple
    sandwich
    orange
    broccoli
    carrot
    hot dog
    pizza
    donut
    cake
    chair
    sofa
    pottedplant
    bed
    diningtable
    toilet
    tvmonitor
    laptop
    mouse
    remote
    keyboard
    cell phone
    microwave
    oven
    toaster
    sink
    refrigerator
    book
    clock
    vase
    scissors
    teddy bear
    hair drier
    toothbrush
"""
classes_coco = [c.strip() for c in classes_coco_str.strip().split("\n")]


def infer_quant_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height:int=512, width:int = 1024, lossy=False):
    # model
    sess = HB_ONNXRuntime(model_file=onnx_model_path)
    input_names = [input.name for input in sess.get_inputs()]
    output_names = [output.name for output in sess.get_outputs()]

    # image
    image = preprocess_custom(
        image_path, 
        height=height, 
        width=width,
    )
    # 因为量化后的onnx会在onnx的开始插入nv12转rgb的操作，而我们输入的数据是rgb，所以这里需要转换下
    image = image * 255
    image = np.expand_dims(image, axis=0)
    image_show = image.astype(np.uint8)
    if not lossy:
        fun_t = RGB2YUV444Transformer(data_format="CHW")  # 这个是无损的和板端有区别, 替换成下面完全模拟板端的
        input_data = fun_t.run_transform(image[0])
    else:
        fun_t1 = RGB2NV12Transformer(data_format="CHW")
        fun_t2 = NV12ToYUV444Transformer((height, width), yuv444_output_layout="CHW")
        input_data = fun_t1.run_transform(image[0])
        input_data = fun_t2.run_transform(input_data)
    input_data = input_data[np.newaxis, ...]
    input_data -= 128
    input_data = input_data.astype(np.int8)
    input_data = input_data.transpose(0, 2, 3, 1)

    # infer
    feed_dict = {
        input_names[0]: input_data,
    }
    outputs = sess.run(output_names, feed_dict)

    # NMS
    scores, bboxes = outputs
    bboxes = bboxes.squeeze(0)
    scores = scores.squeeze(0)
    argmax_idx = np.argmax(scores, axis=1).astype(np.int8)
    argmax_scores = scores[np.arange(scores.shape[0]), argmax_idx]
    indexs = cv2.dnn.NMSBoxes(bboxes, argmax_scores, 0.1, 0.5)

    # 画图
    image_show = image_show.transpose(0, 2, 3, 1)
    image_show = cv2.cvtColor(image_show[0], cv2.COLOR_RGB2BGR)
    for idx in indexs:
        cv2.rectangle(image_show, 
                    (int(bboxes[idx][0]), int(bboxes[idx][1])), 
                    (int(bboxes[idx][2]), int(bboxes[idx][3])),
                    (0, 255, 0), 
                    2)
        cv2.putText(image_show, str(argmax_scores[idx]), (int(bboxes[idx][0]), int(bboxes[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image_show, str(classes_coco[argmax_idx[idx]]), (int(bboxes[idx][0]), int(bboxes[idx][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_quant.png")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, image_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer_onnx")
    parser.add_argument("--onnx_quant_path", type=str, default= "./model_output_l/dosod_mlp3x_l_rep-int16_quantized_model.onnx", help="onnx path")
    parser.add_argument("--image_path", type=str, default="./000000162415.jpg", help="image path")
    parser.add_argument("--result_dir", type=str, default="./", help="result dir")
    parser.add_argument("--height", type=int, default=640, help="height")
    parser.add_argument("--width", type=int, default=640, help="width")
    parser.add_argument('--lossy', action='store_true', help='lossy')
    args = parser.parse_args()

    infer_quant_onnx(
        onnx_model_path=args.onnx_quant_path,
        image_path=args.image_path,
        result_dir=args.result_dir,
        height=args.height,
        width=args.width,
        lossy=args.lossy
    )