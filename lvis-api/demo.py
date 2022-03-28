from lvis import LVIS, LVISResults, LVISEval
import sys
import pickle
import numpy as np

# json file path of annotations
ANNOTATION_PATH = sys.argv[1]
# json file path of bbox and/or segm results
RESULT_PATH = sys.argv[2]

# Annotation type is segm or bbox
ANN_TYPE = sys.argv[3]

# Number of detections to be collected from each image
# LVIS uses 300 by default. 
# If you want to use all detections in the detection 
# file, then you can set it to -1.
MAX_DETS = 300

gt = LVIS(ANNOTATION_PATH)
results = LVISResults(gt, RESULT_PATH, max_dets=MAX_DETS)
lvis_eval = LVISEval(gt, results, iou_type=ANN_TYPE)
params = lvis_eval.params
params.max_dets = MAX_DETS  # No limit on detections per image.

data_keys = ["lrp_values", "dt_scores", "tps", "fps", "lrp_opt_thr"]
dataset = sys.argv[4]
if dataset != "train" or dataset != "val":
  print("Unknown dataset type")
  sys.exit(1)

lvis_eval.run()
lvis_eval.print_results(f"{dataset}_result_summary.txt")

results = lvis_eval.get_results()
for key in data_keys:
  values = results[key]
  file_name = f"{dataset}_{key}"
  with open(file_name, 'wb') as out_file:
    pickle.dump(values, out_file)

if dataset == "train":
  lrp_opt_thrs = results["lrp_opt_thr"]
  file_name = f"{dataset}_shift.py"
  with open(file_name, 'w') as out_file:
    out_file.write("SHIFT=[")
    for lrp_opt_thr in lrp_opt_thrs:
      val = 0 if np.isnan(lrp_opt_thr) else lrp_opt_thr
      val = 0.5-val
      out_file.write(f"{round(val,3)},")
    out_file.write("0]\n\n") # last 0 is for background class
