# Expert data preparation

# 1. Prepare expert training data for Medical-Diff-VQA 

We assume you followed the data preparation steps detailed [here](../../data_prepare/...)

Now, we can take the existing training data and expert model cards and triggers to the training conversation examples to train M3.

```commandline
export PYTHONPATH=${PWD}/..
IN_DATAPATH=/Users/hroth/Data/Childrens/VisionAndLanguage/all_images_json/llava_med_instruct_mimicvqa_train.json
PRED_ROOT=/Users/hroth/Data/VLM/cxr/MIMIC_VQA/images
OUT_DATAPATH=/Users/hroth/Code/monai_vlm/experts/mimic_vqa/llava_med_instruct_mimicvqa_expert_binary_variations_all_train.json
python expert_train_data_cxr.py --in_datapath ${IN_DATAPATH} --pred_root ${PRED_ROOT} --out_datapath ${OUT_DATAPATH}
```

