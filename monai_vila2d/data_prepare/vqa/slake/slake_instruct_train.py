import json
import random

# Function to randomly place "\n<image>" either before or after the question
def add_image_tag(question):
    if random.choice([True, False]):
        return f"{question}\n<image>"
    else:
        return f"<image>\n{question}"

# Function to transform input data
def transform_data(input_data):
    output_data = []
    
    for idx, item in enumerate(input_data):
        if item['q_lang'] == 'en':
            transformed_item = {
                "id": str(idx + 1),
                "image": str(item["img_name"]),
                "conversations": [
                    {
                        "from": str("human"),
                        "value": add_image_tag(str(item['question']))
                    },
                    {
                        "from": str("gpt"),
                        "value": str(item["answer"])
                    }
                ]
            }
            output_data.append(transformed_item)
    
    return output_data

def main():

    input_path_1 = '/home/vnath/Downloads/Slake/Slake1/train.json'
    input_path_2 = '/home/vnath/Downloads/Slake/Slake1/validate.json'
    output_path = '/home/vnath/Downloads/Slake/Slake1/slake_train_val_instruct.json'
    # Read input JSON files
    with open(input_path_1, 'r') as input_file1, open(input_path_2, 'r') as input_file2:
        input_data1 = json.load(input_file1)
        input_data2 = json.load(input_file2)

    # Combine the input data from both files
    combined_input_data = input_data1 + input_data2

    # Transform the combined input data
    output_data = transform_data(combined_input_data)

    # Write to output JSON file
    with open(output_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=4)

    # Output to check
    print(json.dumps(output_data, indent=4))
    print(len(output_data))
if __name__=="__main__":
    main()