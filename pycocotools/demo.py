from src.coco import COCO
from src.cocoeval import COCOeval

import sys
import pickle
import numpy as np

# json file path of annotations
ANNOTATION_PATH = sys.argv[1]

# json file path of bbox and/or segm results
RESULT_PATH = sys.argv[2]

# Annotation type is segm or bbox
ANN_TYPE = sys.argv[3]

DATASET_TYPE = sys.argv[4]
if DATASET_TYPE != "train" and DATASET_TYPE != "val":
  print("Unknown dataset type")
  sys.exit(1)

cocoGt = COCO(ANNOTATION_PATH)

# load detection file
cocoDt = cocoGt.loadRes(RESULT_PATH)

imgIds = sorted(cocoGt.getImgIds())

# If you want to have LRP components for different sizes (e.g. small, 
# medium and large for object detection), then set this variable 
# to True.
print_lrp_components_over_size = True

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, ANN_TYPE, print_lrp_components_over_size)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize(f"{DATASET_TYPE}_result_summary.txt")

data_keys = ["lrp_values", "dt_scores", "tps", "fps", "lrp_opt_thr"]
results = cocoEval.get_results()
for key in data_keys:
  values = results[key]
  file_name = f"{DATASET_TYPE}_{key}"
  with open(file_name, 'wb') as out_file:
    pickle.dump(values, out_file)

if DATASET_TYPE == "train":
  lrp_opt_thrs = results["lrp_opt_thr"]
  file_name = f"{DATASET_TYPE}_shift.py"
  with open(file_name, 'w') as out_file:
    out_file.write("SHIFT=[")
    for lrp_opt_thr in lrp_opt_thrs:
      val = 0 if np.isnan(lrp_opt_thr) else lrp_opt_thr
      val = 0.5-val
      out_file.write(f"{round(val,3)},")
    out_file.write("0]\n\n") # last 0 is for background class
