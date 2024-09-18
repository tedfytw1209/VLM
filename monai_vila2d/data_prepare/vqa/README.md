Please refer the `instruction_files` folder for the pre-generated instruction data files for each dataset. 

The below commands are for reproducing the instruction data files and are not necessary to run.
### RadVQA
Example command to run generate the instruction training data json file for RadVQA dataset:

```
python radvqa_instruct_data_generate.py \
    --input_json /path/to/VQA_RAD_Dataset_Public.json \
    --output_json /path/to/output/radvqa_instruct.json \
    --data_type train
```

### SLAKE
Example commands to run generate the instruction training data json file for Slake dataset:

Training Instruction Data:
```
python slake_instruct_data_generate.py \
    --input_paths /path/to/slake_dataset/train.json /path/to/slake_dataset/validate.json \
    --output_path /path/to/output/slake_train_val_instruct.json
```
Testing Instruction Data:
```
python slake_instruct_data_generate.py \
    --input_paths /path/to/slake_dataset/test.json \
    --output_path /path/to/output/slake_test_instruct.json
```

### PathVQA
Example command to run generate the instruction training data json file for PathVQA dataset:

```
python pathvqa_instruct_data_generate.py \
    --train_pkl /path/to/train_vqa.pkl \
    --val_pkl /path/to/val_vqa.pkl \
    --test_pkl /path/to/test_vqa.pkl \
    --output_json /path/to/output/merged_pathvqa_instruct.json
```

### MIMIC-VQA
