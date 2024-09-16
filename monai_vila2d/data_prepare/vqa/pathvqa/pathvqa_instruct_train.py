import os
import json
import pickle
import random

def main():

    pkl_train_f_path = os.path.normpath('/home/vnath/datasets_2024/1613713/qas/train_vqa.pkl')
    pkl_val_f_path = os.path.normpath('/home/vnath/datasets_2024/1613713/qas/val_vqa.pkl')    
    
    # Step 1: Load the pickle file
    with open(pkl_train_f_path, 'rb') as t_file:
        train_data = pickle.load(t_file)
    t_file.close()

    with open(pkl_val_f_path, 'rb') as v_file:
        val_data = pickle.load(v_file)
    v_file.close()
    
    # Transform the data
    transformed_data = []
    total_questions = 0

    # Process data from data1
    for item in train_data:
        if random.choice([True, False]):
            human_value = str(f"<image>\n{item['sent']}")
        else:
            human_value = str(f"{item['sent']}\n<image>")

        # Ensure all values are strings
        new_item = {
            "id": str(item['question_id']),
            "image": str(f"train/{item['img_id']}.jpg"),
            "conversations": [
                {
                    "from": str("human"),
                    "value": human_value
                },
                {
                    "from": str("gpt"),
                    "value": str(list(item['label'].keys())[0])
                }
            ]
        }
        transformed_data.append(new_item)
        total_questions += 1
        print(f"Total questions recorded so far: {total_questions}")

    # Process data from data2
    for item in val_data:
        if random.choice([True, False]):
            human_value = str(f"<image>\n{item['sent']}")
        else:
            human_value = str(f"{item['sent']}\n<image>")

        # Ensure all values are strings
        new_item = {
            "id": str(item['question_id']),
            "image": str(f"val/{item['img_id']}.jpg"),
            "conversations": [
                {
                    "from": str("human"),
                    "value": human_value
                },
                {
                    "from": str("gpt"),
                    "value": str(list(item['label'].keys())[0])
                }
            ]
        }
        transformed_data.append(new_item)
        total_questions += 1
        print(f"Total questions recorded so far: {total_questions}")

    # Step 3: Write the new JSON file
    with open('/home/vnath/Code/vlfm_2024/pathvqa/pathvqa_instruct_train.json', 'w') as json_file:
        json.dump(transformed_data, json_file, indent=4)

if __name__=="__main__":
    main()