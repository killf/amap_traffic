import json
import shutil
import cv2
import os

DATA_DIR = "/home/killf/dataset/amap_traffic/amap"


def main():
    tagging = dict()
    for line in open("tagging.csv").readlines():
        idx, status = line.strip().split(',')
        tagging[idx] = int(status) if status else -1

    file_path = os.path.join(DATA_DIR, "amap_traffic_annotations_test.tag.json")
    if not os.path.exists(file_path):
        shutil.copy(os.path.join(DATA_DIR, "amap_traffic_annotations_test.json"), file_path)

    meta = json.load(open(file_path))
    for item in meta["annotations"]:
        item["status"] = tagging[item['id']]

    json.dump(meta, open(file_path, "w"), indent=4)


if __name__ == '__main__':
    main()
