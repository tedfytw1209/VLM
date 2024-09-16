import os
import random
import json

def main():
    input_json = os.path.normpath('/home/vnath/Downloads/osfstorage-archive/VQA_RAD Dataset Public.json')
    output_json = os.path.normpath('/home/vnath/Downloads/osfstorage-archive/radvqa_train_instruct_v3.json')
    with open(input_json, 'r') as j_file:
        j_data = json.load(j_file)
    j_file.close()

    # Organ list of questions
    organ_q_list = [
        "What is the visible organ?",
        "What organ is it?",
        "Is any organ visible?",
        "Which can organ can be seen?",
        "Any visible organs?"
    ]

    train_instruct_data = []
    counter = 1
    for idx, each_data in enumerate(j_data):
        print('Status: {} / {}'.format(idx, len(j_data)))

        if each_data["phrase_type"] != "test_para" or each_data["phrase_type"] != "test_freeform":
            t_dict = {}
            t_dict["id"] = str(counter)
            counter = counter + 1
            print('Instruct Row Count: {}'.format(counter))
            t_dict["image"] = each_data["image_name"]
            conv_list = [
                {
                    "from": "human",
                    "value": str(each_data["question"] + "\n<image>")
                },
                {
                    "from": "gpt",
                    "value": str(each_data["answer"])
                }
            ]
            t_dict["conversations"] = conv_list
            train_instruct_data.append(t_dict)

            # Adding an organ question for training purposes
            random_choice = random.randint(0, 4)
            t_dict = {}
            t_dict["id"] = str(counter)
            counter = counter + 1
            print('Instruct Row Count: {}'.format(counter))
            t_dict["image"] = each_data["image_name"]
            conv_list = [
                {
                    "from": "human",
                    "value": str(organ_q_list[random_choice] + "\n<image>")
                },
                {
                    "from": "gpt",
                    "value": str(each_data["image_organ"])
                }
            ]
            t_dict["conversations"] = conv_list
            train_instruct_data.append(t_dict)

            if "question_rephrase" in each_data and each_data["question_rephrase"] != "NULL":
                t_dict = {}
                t_dict["id"] = str(counter)
                counter = counter + 1
                print('Instruct Row Count: {}'.format(counter))
                t_dict["image"] = each_data["image_name"]
                conv_list = [
                    {
                        "from": "human",
                        "value": str(each_data["question_rephrase"] + "\n<image>")
                    },
                    {
                        "from": "gpt",
                        "value": str(each_data["answer"])
                    }
                ]
                t_dict["conversations"] = conv_list
                train_instruct_data.append(t_dict)

    with open(output_json, 'w') as j_file:
        json.dump(train_instruct_data, j_file, indent=4)
    j_file.close()

    print('All set to fly ...')

if __name__=="__main__":
    main()