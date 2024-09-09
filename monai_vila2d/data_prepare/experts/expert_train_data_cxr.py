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

import os.path

from expert_utils import model_list
from ...data_utils import read_json, write_json
from tqdm import tqdm
import random
random.seed(0)

assert isinstance(model_list, str)

pred_root = "/Users/hroth/Data/VLM/cxr/MIMIC_VQA/images"
root_dir = "./"

n = 100_000
test_frac = 0.5


def get_predictions(root, image_name):
    predictions_file = os.path.join(root, image_name + ".json")
    pred = read_json(predictions_file)

    pred_str = []
    for k, v in pred.items():
        pred_str.append(f"{k}: {v:.2f}")
    return ", ".join(pred_str)


def main():
    in_data = read_json("/Users/hroth/Data/Childrens/VisionAndLanguage/all_images_json/llava_med_instruct_mimicvqa_train.json")

    assert n < len(in_data)

    out_fileprefix = "../experts/mimic_vqa/llava_med_instruct_mimicvqa_expert"

    in_data = random.sample(in_data, k=n)

    count = 0
    all_conversations = []
    for entry in tqdm(in_data, desc="creating train data..."):
        conv = entry["conversations"]
        # append data
        # Keep first question
        assert conv[0]["from"] == "human"
        first_prompt = conv[0]["value"]
        predictions = get_predictions(pred_root, entry["image"])
        new_conv = list()
        new_conv.append({"from": "human", "value": model_list + f" This is a CXR image.\n" + first_prompt})
        new_conv.append({"from": "gpt", "value": "This looks like an chest x-ray. Let me trigger <CXR()>."})
        first_prompt = first_prompt.replace("\n<image>", "")
        first_prompt = first_prompt.replace("<image>", "")
        new_conv.append({"from": "human", "value": f"The resulting predictions are: {predictions}. Take these likelihoods into account when responding to this prompt:\n{first_prompt}"})
        new_conv.extend(conv[1::])

        entry["conversations"] = new_conv

        all_conversations.append(entry)
        count += 1

    print(f"Converted {len(all_conversations)} conversations")

    out_train_file = out_fileprefix + "_train.json"
    out_test_file = out_fileprefix + "_test.json"

    split_idx = int(test_frac*len(all_conversations))

    random.shuffle(all_conversations)
    test_conversations = all_conversations[0:split_idx]
    train_conversations = all_conversations[split_idx::]

    write_json(train_conversations, out_train_file)
    write_json(test_conversations, out_test_file)


if __name__ == "__main__":
    main()
