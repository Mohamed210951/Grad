import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_and_evaluate(gt_path, det_path):
    # Load ground truth and detections
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(det_path)

    # COCO Evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

# Assuming you have a file path for ground truth
gt_path = r'C:\Graduation_Project\Train\Faster_rcnn\final_dataset_coco\valid\_annotations.coco.json'  # Adjust path as needed
det_path = r'C:\Graduation_Project\detections.json'  # Adjust path as needed

# Run evaluation
evaluation_results = load_and_evaluate(gt_path, det_path)
print("COCO Evaluation Results:", evaluation_results)