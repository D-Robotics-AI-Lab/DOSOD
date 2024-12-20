import json
import argparse
import numpy as np
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='/horizon-bucket/AIoT-data-bucket/yonghao01.he/pretrain_models/clip-vit-base-patch32')
    parser.add_argument('--text',
                        type=str,
                        default='/home/users/yonghao01.he/projects/YOLO-World-Workspace/yolo-world-reparameterize-show/open_word.json')
    parser.add_argument('--out', type=str, default='/home/users/yonghao01.he/projects/YOLO-World-Workspace/yolo-world-reparameterize-show/open_word.npy')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = CLIPTextModelWithProjection.from_pretrained(args.model)

    with open(args.text) as f:
        data = json.load(f)
    texts = [x[0] for x in data]
    device = 'cuda:0'
    model.to(device)
    texts = tokenizer(text=texts, return_tensors='pt', padding=True)
    texts = texts.to(device)
    text_outputs = model(**texts)
    txt_feats = text_outputs.text_embeds
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    txt_feats = txt_feats.reshape(-1, txt_feats.shape[-1])

    np.save(args.out, txt_feats.cpu().data.numpy())
