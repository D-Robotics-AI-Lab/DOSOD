import json
import os
import argparse
import numpy as np
import torch
from mmdet.apis import init_detector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--text',
                        type=str,
                        default='data/texts/coco_class_texts.json',
                        help='Path to text file''')
    parser.add_argument('--out-dir', type=str, help='The dir to save text embeddings npy')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='Device used for run')

    args = parser.parse_args()

    device = args.device

    with open(args.text) as f:
        data = json.load(f)
    texts = [x[0] for x in data]

    # generate text embeddings
    print('init model......')
    model = init_detector(args.config, args.checkpoint, device=device)
    model.eval()

    print('start to generate text embeddings......')
    with torch.no_grad():
        text_embeddings = model.backbone_text([texts], enable_assertion=False)
        text_embeddings = model.bbox_head.head_module.forward_text(text_embeddings)
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-1])

    print('start to save text embeddings......')
    os.makedirs(args.out_dir, exist_ok=True)
    text_embeddings = text_embeddings.cpu().data.numpy()
    np.save(os.path.join(args.out_dir,
                         os.path.splitext(os.path.basename(args.text))[0] + '_' + os.path.splitext(os.path.basename(args.checkpoint))[0]) + ".npy",
            text_embeddings)
