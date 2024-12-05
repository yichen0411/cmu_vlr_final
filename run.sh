#!/bin/bash

python3 "./inference.py" \
  -c "../Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py" \
  -p "./checkpoint0008.pth" \
  -i "./banana.png" \
  -t "car" \
  -o "./pred_images"


# -p: path to checkpoint file (replace it with model updated model weight: https://drive.google.com/drive/folders/1C4XZZ60DPiUzIQX1qYNzFCSEutpe_RDq?usp=drive_link)
# -i: path to image file 
# -t: text prompt
# -o: output directory


