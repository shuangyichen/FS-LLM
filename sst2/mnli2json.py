from datasets import load_dataset
import json

def process_mnli_to_json(json_file_path):
    # Load the SST-2 dataset
    dataset = load_dataset("glue", "mnli")

    # Initialize an empty list to store data
    data = []
    # data2 = []
    # print(dataset['train'][0][0])
    # print(len(dataset['train']))

    # Process the new training set (first 66,675 items of the original training set)
    for item in dataset['train']:
        # print(item)
        # label = 'positive' if item['label'] == 1 else 'negative'
        if len(data)<=388774:
            data.append({
                'instruction': None,  # or any default instruction
                'input': item['premise'],
                'output': item['label'],
                'category': item['hypothesis']
            })
        else:
            data.append({
                'instruction': None,  # or any default instruction
                'input': item['premise'],
                'output': item['label'],
                'category': item['hypothesis']
            })


    # Process the new test set (original validation set)
    for item in dataset['validation_matched']:
        # label = 'positive' if item['label'] == 1 else 'negative'
        data.append({
            'instruction': None,  # or any default instruction
            'input': item['premise'],
            'output': item['label'],
            'category': item['hypothesis']
        })

    # Write to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

json_file_path = 'mnli.json'  # Path to save the JSON file
process_mnli_to_json(json_file_path)
