import os
import json
import pickle
import random

def main():

    pkl_train_f_path = os.path.normpath('/home/vnath/datasets_2024/1613713/qas/test_vqa.pkl')
    
    # Step 1: Load the pickle file
    with open(pkl_train_f_path, 'rb') as t_file:
        test_data = pickle.load(t_file)
    t_file.close()

    data = test_data
    # Step 2: Transform the data
    transformed_data = []
    counter = 0
    for item in data:
        if random.choice([True, False]):
            human_value = f"<image>\n{item['sent']}"
        else:
            human_value = f"{item['sent']}\n<image>"
        new_item = {
            "id": str(item['question_id']),
            "image": str(os.path.join('test', f"{item['img_id']}.jpg")),
            "conversations": [
                {
                    "from": "human",
                    "value": human_value
                },
                {
                    "from": "gpt",
                    "value": str(list(item['label'].keys())[0])
                }
            ]
        }
        transformed_data.append(new_item)

        counter = counter + 1
        print('Questions recorded so far: {}'.format(counter))

    # Step 3: Write the new JSON file
    with open('/home/vnath/Code/vlfm_2024/pathvqa/pathvqa_instruct_test.json', 'w') as json_file:
        json.dump(transformed_data, json_file, indent=4)

if __name__=="__main__":
    main()