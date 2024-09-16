# #!/usr/bin/env python

import os
import base64
import pickle


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_string


def save_dataset(dataset_type, dataset_name, save_path, data):
    save_filename = f"{dataset_type}_{dataset_name}.pkl"
    save_pathname = os.path.join(save_path, save_filename)
    with open(save_pathname, "wb") as f:
        pickle.dump(data, f)


image_dir = "/mnt/drive1/mimic-cxr-512_v2/images"
output_dir = "/localhome/local-dongy/projects/vlm/gt/gt_train_v3_training_1.1"

list_filepath = "/localhome/local-dongy/projects/vlm/lists/mimic-cxr-2.0.0-train_v3_training_1.1_text.txt"
text_dir = "/localhome/local-dongy/projects/vlm/text_gt/training_1.1"

with open(list_filepath, "r") as file:
    filepaths = file.readlines()
filepaths = [_item.strip() for _item in filepaths]
num_cases = len(filepaths)

data_dict = []
for _i in range(num_cases):
    print(f"{_i + 1}/{num_cases}")
    text_filepath = os.path.join(text_dir, filepaths[_i])
    image_filepath = os.path.join(image_dir, filepaths[_i])
    image_filepath = image_filepath.replace(".txt", "")
    print(image_filepath, text_filepath)

    if not os.path.exists(image_filepath):
        continue

    image_base64_str = encode_image_to_base64(image_filepath)
    with open(text_filepath, "r") as file:
        reference_report = file.read()
    # reference_report = reference_report.strip()
    # reference_report = reference_report.replace("\n", " ").replace("  ", " ")
    print(reference_report)

    _dict = {
        "question": "Describe the image in detail.",
        "image": [image_base64_str],
        "answer": reference_report,
    }
    data_dict.append(_dict)
    print(f"data_dict: {len(data_dict)}")

    # input()

save_dataset("captioning", "mimic_v2_train_v3_training_1.1", output_dir, data_dict)
