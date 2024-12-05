## Inference

To run inference using the script, use the following command:

```bash
python "../Open-GroundingDino/tools/inference_on_a_image.py" \
  -c "../Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py" \
  -p "../output/checkpoint0008.pth" \
  -i "../bdd100k/images/100k/val/b1c9c847-3bda4659.jpg" \
  -t "car" \
  -o "../pred_images"


-p: path to checkpoint file (replace it with model updated model weight: https://drive.google.com/drive/folders/1C4XZZ60DPiUzIQX1qYNzFCSEutpe_RDq?usp=drive_link)
-i: path to image file 
-t: text prompt
-o: output directory
```

## Inference in CARLA
Start CARLA, then:

```bash
$ cd cmu_vlr_final
$ python3 carla_example.py
```
