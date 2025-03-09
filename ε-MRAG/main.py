import argparse



def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_model', default="/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/model/doclayout_yolo_docstructbench_imgsz1024.pt", required=False, type=str)
    parser.add_argument('--pdf_path', default="/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/assets/example", required=False, type=str)
    parser.add_argument('--output_path', default='/data1/zhh/baselines/mm/icrr/DocLayout-YOLO/outputs', required=False, type=str)
    
    
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    return parser.parse_args()



def main(args):
    



if __name__ == "__main__":
    args = create_args()
    main(args)