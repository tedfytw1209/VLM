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

import argparse
import random

from data_utils import read_json, write_json
from expert_utils import add_expert_conversation, assert_image_placeholder, get_predictions, model_list
from tqdm import tqdm

random.seed(0)

assert isinstance(model_list, str)


def main(args):
    """Prepare expert training data for CXR."""
    in_data = read_json(args.in_datapath)

    assert len(in_data) > 0, f"No data read from {args.in_datapath}"

    count = 0
    all_conversations = []
    for entry in tqdm(in_data, desc="creating train data"):
        conv = entry["conversations"]

        # add expert instructions
        predictions = get_predictions(args.pred_root, entry["image"])
        new_conv = add_expert_conversation(conv, predictions)

        assert_image_placeholder(new_conv)
        entry["conversations"] = new_conv

        all_conversations.append(entry)
        count += 1

    print(f"Converted {len(all_conversations)} conversations")

    write_json(all_conversations, args.out_datapath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_datapath", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--out_datapath", type=str, required=True)
    args = parser.parse_args()

    main(args)
