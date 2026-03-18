import os
import numpy as np
import argparse
from tqdm import tqdm

from preprocess import preprocess_custom


def gen_calibration_data_nv12(data_dir: str, save_dir: str, height: int = 640, width: int = 640):
    all_images = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p.endswith("jpg") or p.endswith("JPG")]
    os.makedirs(save_dir, exist_ok=True)
    for image_path in tqdm(all_images, desc="Generate Calibration Data"):
        image = preprocess_custom(
            image_path, 
            height=height, 
            width=width,
        )
        image = image[np.newaxis, ...]
        # S100 需要归一化后的数据作为校准数据
        # image = image * 255 # 因为减均值除方差的过程放到校准过程中, 放到onnx里面了, 所以这里乘以255
        assert image.shape == (1, 3, height, width)
        np.save(os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".npy"), image.astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Calibration Data")
    parser.add_argument("--data_dir", type=str, default= "./coco_val_100", help="The directory of calibration images")
    parser.add_argument("--save_dir", type=str, default="./coco_val_100_calib_s100", help="The directory to save calibration data")
    parser.add_argument("--height", type=int, default=640, help="height")
    parser.add_argument("--width", type=int, default=640, help="width")
    args = parser.parse_args()

    gen_calibration_data_nv12(args.data_dir, args.save_dir, height=args.height, width=args.width)