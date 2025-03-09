import os
import cv2
import torch
import argparse
from pathlib import Path
from doclayout_yolo import YOLOv10
from doclayout_yolo.engine.results import Results
from doclayout_yolo.engine.results import Boxes
from huggingface_hub import hf_hub_download
from copy import deepcopy
import numpy as np
import torch


def merge_boxes(results: Results) -> Results:
    if results.boxes is None:
        return results
    boxes_tensor = results.boxes.data.cpu() if isinstance(results.boxes.data, torch.Tensor) else results.boxes.data
    boxes_np = boxes_tensor.numpy() if isinstance(boxes_tensor, torch.Tensor) else np.copy(boxes_tensor)
    num_cols = boxes_np.shape[1]
    has_track = (num_cols == 7)
    det_list = []
    for row in boxes_np:
        if has_track:
            x1, y1, x2, y2, track, conf, cls = row
        else:
            x1, y1, x2, y2, conf, cls = row
        cls = int(cls)
        # 映射到大类
        if cls in [0, 1, 8, 9]:
            group = "Text"
        elif cls in [3, 4]:
            group = "Figure"
        elif cls in [5, 6, 7]:
            group = "Table"
        elif cls == 2:
            group = "abandon"
        else:
            group = None

        det_list.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": conf,
            "cls": cls,   
            "group": group,
            "used": False, 
        })
        
    abandon_list = [d for d in det_list if d["group"] == "abandon"]
    non_abandon = [d for d in det_list if d["group"] != "abandon"]

    for d in non_abandon:
        d["cx"] = (d["x1"] + d["x2"]) / 2

    page_width = results.orig_shape[1]
    col_thresh = page_width * 0.1  
    non_abandon_sorted = sorted(non_abandon, key=lambda d: d["cx"])
    columns = []
    for d in non_abandon_sorted:
        assigned = False
        for col in columns:
            avg_cx = np.mean([item["cx"] for item in col])
            if abs(d["cx"] - avg_cx) < col_thresh:
                col.append(d)
                assigned = True
                break
        if not assigned:
            columns.append([d])
    for col in columns:
        col.sort(key=lambda d: d["y1"])

    gap_thresh = results.orig_img.shape[0] * 0.05  # 2%页面高度作为垂直间隔阈值

    def merge_vertical(boxes_in_group, merge_type):
        merged_list = []
        i = 0
        while i < len(boxes_in_group):
            # 从当前框开始构造一个合并候选组
            group_candidates = [boxes_in_group[i]]
            i += 1
            while i < len(boxes_in_group):
                prev = group_candidates[-1]
                curr = boxes_in_group[i]
                # 计算前一框的下边界与当前框上边界的间隔
                vertical_gap = curr["y1"] - prev["y2"]
                # 判断水平重叠率：计算两个框在x方向的重叠长度，除以较窄框的宽度
                overlap = max(0, min(prev["x2"], curr["x2"]) - max(prev["x1"], curr["x1"]))
                width_min = min(prev["x2"] - prev["x1"], curr["x2"] - curr["x1"])
                overlap_ratio = overlap / width_min if width_min > 0 else 0
                if vertical_gap <= gap_thresh and overlap_ratio >= 0.5:
                    group_candidates.append(curr)
                    i += 1
                else:
                    break
            # 合并候选组处理：如果组内只有1个框，不进行合并，否则检查特定条件
            if len(group_candidates) == 1:
                merged_list.append(group_candidates[0])
            else:
                # 对于Figure，要求组内至少有一个原始类别为3（figure）
                if merge_type == "Figure":
                    if not any(d["cls"] == 3 for d in group_candidates):
                        # 不满足合并条件，则单独处理每个框
                        merged_list.extend(group_candidates)
                        continue
                # 对于Table，要求组内仅有一个原始 table（类别5）
                if merge_type == "Table":
                    table_count = sum(1 for d in group_candidates if d["cls"] == 5)
                    if table_count != 1:
                        merged_list.extend(group_candidates)
                        continue
                # 计算合并后的包围矩形：取所有框的最小x1、最小y1、最大x2、最大y2
                merged_box = {
                    "x1": min(d["x1"] for d in group_candidates),
                    "y1": min(d["y1"] for d in group_candidates),
                    "x2": max(d["x2"] for d in group_candidates),
                    "y2": max(d["y2"] for d in group_candidates),
                    "conf": max(d["conf"] for d in group_candidates),
                    "cls": merge_type  # 合并后的类别直接记录大类字符串
                }
                merged_list.append(merged_box)
        return merged_list

    merged_results = []  # 用于存放所有合并结果（不含 abandon)

    for col in columns:
        # 对于每个大类分别取出同一列内的框（注意：同一列内可能存在多个大类）
        for merge_type in ["Text", "Figure", "Table"]:
            group_boxes = [d for d in col if d["group"] == merge_type]
            if group_boxes:
                merged = merge_vertical(group_boxes, merge_type)
                merged_results.extend(merged)

    merged_results.extend(abandon_list)

    merge_class_map = {"Text": 0, "Figure": 1, "Table": 2, "abandon": 3}
    new_boxes_list = []
    for d in merged_results:
        x1, y1, x2, y2, conf = d["x1"], d["y1"], d["x2"], d["y2"], d["conf"]
        # 如果d["cls"]为字符串，则用映射，否则说明未参与合并（依然是原始类别），按原规则转大类
        if isinstance(d["cls"], str):
            cls_new = merge_class_map[d["cls"]]
        else:
            # 对于未合并的情况（单个目标），依照原始类别映射到大类：
            if d["cls"] in [0, 1, 8, 9]:
                cls_new = merge_class_map["Text"]
            elif d["cls"] in [3, 4]:
                cls_new = merge_class_map["Figure"]
            elif d["cls"] in [5, 6, 7]:
                cls_new = merge_class_map["Table"]
            elif d["cls"] == 2:
                cls_new = merge_class_map["abandon"]
            else:
                cls_new = -1  # 未知
        new_boxes_list.append([x1, y1, x2, y2, conf, cls_new])
    new_boxes_array = np.array(new_boxes_list)
    new_boxes_tensor = torch.tensor(new_boxes_array, dtype=torch.float32)
    new_results = results.new()
    new_results.boxes = Boxes(new_boxes_tensor, results.orig_shape)
    new_results.names = {0: "Text", 1: "Figure", 2: "Table", 3: "abandon"}
    return new_results
def crop_and_save_images(results, save_dir):
    save_path = Path(save_dir)
    save_path = Path(os.path.join(save_path, "cropped_images"))
    save_path.mkdir(parents=True, exist_ok=True)
    
    orig_image = results.orig_img
    orig_height, orig_width = orig_image.shape[:2]

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(orig_width, int(x2)), min(orig_height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cropped = orig_image[y1:y2, x1:x2]
            filename = save_path / f'crop_{i}_{x1}_{y1}_{x2}_{y2}.jpg'
            cv2.imwrite(str(filename), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))    
        print(f"成功保存 {len(boxes)} 个切割图片到 {save_path}")
    else:
        print("未检测到任何目标")
        
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/model/doclayout_yolo_docstructbench_imgsz1024.pt", required=False, type=str)
    parser.add_argument('--image-path', default="/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/assets/example", required=False, type=str)
    parser.add_argument('--res-path', default='/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/outputs', required=False, type=str)
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    return parser.parse_args()

if __name__ == "__main__":

    args = create_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    images = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if f.endswith('.jpg')]
    
    model = YOLOv10(args.model)
    det_res = model.predict(
        images,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
    )
    det_res = [merge_boxes(det)for det in det_res]
    # print(det_res)
    for det in det_res:
        origin_image_path = det.path
        annotated_frame = det.plot(pil=True, line_width=args.line_width, font_size=args.font_size)
        if not os.path.exists(args.res_path):
            os.makedirs(args.res_path)
        output_path = os.path.join(args.res_path,origin_image_path.split("/")[-1].replace(".jpg", "_res.jpg"))
        # crop_and_save_images(det_res[0], save_dir=args.res_path)
        cv2.imwrite(output_path, annotated_frame)
        print(f"Result saved to {output_path}")