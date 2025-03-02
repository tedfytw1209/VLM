# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
import warnings
from glob import glob

import numpy as np
from monai.bundle.config_parser import ConfigParser
from monai.utils import set_determinism

set_determinism(20240723)

chars = "ABCDEFGHIJKLMN"
c = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged cardiomediastinum",
    "Fracture",
    "Lung lesion",
    "Lung opacity",
    "Pleural effusion",
    "Pleural other",
    "Pneumonia",
    "Pneumothorax",
    "Support devices",
    "None of the above",
)
classes = dict(zip(chars, c))
prompt = (
    "<image>\nThe following is a multiple-choice question about findings in chest X-ray in the frontal view. "
    "Please reply with the corresponding answer choice letter(s).\n"
    "Question: What are the potential abnormalities according to the provided X-ray image?\n"
)
prompt += "\n".join([f"({item}) {classes[item]}" for item in classes])
prompt += "\n"

binary_prompt = (
    "<image>\nThe following is a question about findings in chest X-ray in the frontal view. "
    "Please reply with yes or no.\n"
    "Question: is there <placeholder> according to the provided X-ray image?\n"
)
prompts = [binary_prompt.replace("<placeholder>", classes[item].lower()) for item in classes if item != "N"]
prompts = [prompt] + prompts  # all prompts
# prompts = [prompt]  # multiple-choice only
binary_prompt_only = False

split_csv = "mimic-cxr-2.0.0-split.csv"
orig_csv = "mimic-cxr-2.0.0-merged-chexpert.csv"
image_base = "/red/chenaokun1990/VLM_dataset/ReportGeneration/MIMIC-CXR_JPG/files/"
expert_base = "/red/chenaokun1990/tienyu/torchxrayvision"
expert_models = ["ensemble"]
out_json = "mimic-cxr-2.0.0-train-expert-balanced-multi-simple.json"
freqs = [
    47629.0,
    46373.0,
    11231.0,
    28339.0,
    7456.0,
    4675.0,
    6436.0,
    53254.0,
    56118.0,
    2005.0,
    16757.0,
    11046.0,
    71447.0,
    80000.0,
]
freqs = 5000.0 / np.asarray(freqs)
balanced = True


def get_item(fname, p_in, p_out, expert=None):
    """Get a dictionary item to form the struct input to VLM."""
    if not expert:
        return {
            "id": f"{uuid.uuid4()}",
            "image": fname,
            "conversations": [{"from": "human", "value": f"{p_in}"}, {"from": "gpt", "value": f"{p_out}"}],
        }
    return {
        "id": f"{uuid.uuid4()}",
        "image": fname,
        "conversations": [{"from": "human", "value": f"{p_in}{expert}"}, {"from": "gpt", "value": f"{p_out}"}],
    }


with open(orig_csv, "r") as f:
    lines = f.readlines()
label_dict = {x.split(",", 1)[1].split(",", 1)[0]: x.strip() for x in lines[1:]}
path_dict = {os.path.basename(fname):fname for fname in glob(os.path.join(image_base, "*.jpg"))}
headers = lines[0]

with open(split_csv, "r") as f:
    split_lines = f.readlines()
split_lines = split_lines[1:]

sum_cls, sum_others, sum_neg = 0, 0, 0
out_dicts = []
for idx, line in enumerate(split_lines):
    fname, *_, phase = line.strip().split(",")
    if phase != "train":
        continue
    if fname not in label_dict:
        warnings.warn(f"image not found in merged csv {idx}: {fname}", stacklevel=2)
        continue
    print(idx, fname, phase, label_dict[fname])
    label_line = label_dict[fname]
    if not fname.endswith("jpg"):
        fname += ".jpg"
    fname_path = path_dict.get(fname, None)
    full_name = os.path.join(image_base, fname_path)
    if not os.path.isfile(full_name):
        warnings.warn(f"image not found {idx}: {full_name}", stacklevel=2)
        continue
    flags = [float(x.lower()) == 1.0 for x in label_line.split(",")[-14:]]
    flags.pop(8)  # no finding in the mimic-cxr-2.0.0-merged-chexpert.csv
    if balanced:
        prob = np.random.rand()
        if (not any(flags)) and (freqs[-1] < prob):
            continue
        elif any(flags) and all(freqs[:-1][flags] < prob):
            continue
    expert_results = ""
    if expert_models:
        expert_results = (
            "When answering the question, please analyze the image and "
            "incorporate the additional results generated by an expert classification model:\n"
        )
        ens = {}
        for expert_res in expert_models:
            json_path = os.path.join(expert_base, expert_res, fname.replace("jpg", "jpg.json"))
            res = ConfigParser.load_config_files(json_path)
            for cls_name in res:
                if cls_name not in ens:
                    ens[cls_name] = [float(res[cls_name])]
                else:
                    ens[cls_name] += [float(res[cls_name])]
        for cls_name in ens:
            prob_cls = sum(ens[cls_name]) / len(ens[cls_name])
            expert_results += f" {cls_name.lower().replace('_', ' ')}: {'yes' if prob_cls > 0.5 else 'no'}\n"
    for p_id, p_in in enumerate(prompts):
        if p_id == 0 and binary_prompt_only:
            continue
        if p_id == 0:
            for v in [1, 2]:  # generate multiple types of answers as augmentation
                if v == 1:
                    p_out = "(N)" if not any(flags) else ",".join([f"({x})" for x, y in zip(chars, flags) if y])
                elif v == 2:
                    p_out = (
                        "(N) None of the above"
                        if not any(flags)
                        else ",".join([f"({x}) {classes[x]}" for x, y in zip(chars, flags) if y])
                    )
                out_dicts.append(get_item(fname_path, p_in, p_out, expert_results))
        else:
            p_out = "Yes" if flags[p_id - 1] else "No"
            out_dicts.append(get_item(fname_path, p_in, p_out, expert_results))
    sum_cls += np.asarray(flags)
    sum_others += 1
    sum_neg += float(not any(flags))
print(sum_cls)
print(idx, sum_others, sum_neg)
ConfigParser.export_config_file(out_dicts, out_json, indent=4)
