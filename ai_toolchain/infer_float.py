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


def infer_onnx(onnx_model_path: str, image_path: str, result_dir: str = "./", height=640, width=640):
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
    image = np.expand_dims(image, axis=0)
    
    # infer
    feed_dict = {
        input_names[0]: image,
    }
    outputs = sess.run(output_names, feed_dict)
    
    # 后处理
    image_show = (image * 255).astype(np.uint8)

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
    dst_path = os.path.join(result_dir, os.path.basename(image_path)[:-4]+"_result_float.png")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, image_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="infer onnx")
    parser.add_argument("--onnx_float_path", type=str, default="./dosod_mlp3x_l_rep.onnx", help="onnx path")
    parser.add_argument("--image_path", type=str, default="./000000162415.jpg", help="image path")
    parser.add_argument("--result_dir", type=str, default="./", help="result dir")
    parser.add_argument("--height", type=int, default=640, help="height")
    parser.add_argument("--width", type=int, default=640, help="width")
    args = parser.parse_args()

    # 使用原始onnx推理查看下onnx是否正确
    infer_onnx(
        onnx_model_path=args.onnx_float_path,
        image_path=args.image_path,
        result_dir=args.result_dir,
        height=args.height,
        width=args.width,
    )
    