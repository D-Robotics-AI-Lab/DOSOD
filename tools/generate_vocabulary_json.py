import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Vocabulary Text to Json File")
    parser.add_argument("--text", type=str, default="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush", help='Texts')
    parser.add_argument("--output", type=str, default='offline_vocabulary.json', help='Output path')

    args = parser.parse_args()

    # 指定输出的 JSON 文件名
    text = args.text
    output_file = args.output

    # 将文本按逗号分割并去除多余的空格
    items = [item.strip() for item in text.split(",")]

    # 将每个项目转换为单独的列表
    nested_items = [[item] for item in items]

    print("len items:", len(nested_items), nested_items)

    # 将嵌套列表保存为 JSON 文件
    with open(output_file, "w", encoding="utf-8") as file:
        # indent=4
        json.dump(nested_items, file, ensure_ascii=False)

    print(f"Finshed. Save vocabulary file: {output_file}")

