# from datasets import load_dataset
# import json

# def process_sst2_to_json(json_file_path):
#     # Load the SST-2 dataset
#     dataset = load_dataset("glue", "sst2")

#     # Initialize an empty list to store data
#     data = []

#     # Process each subset (train, validation, test)
#     for subset in ['train', 'validation', 'test']:
#         # Check if the subset exists in the dataset
#         if subset in dataset:
#             for item in dataset[subset]:
#                 # Convert label to string ('positive' or 'negative')
#                 label = 'positive' if item['label'] == 1 else 'negative'
#                 # Append the item to the data list
#                 data.append({
#                     'instruction': None,  # or any default instruction
#                     'input': item['sentence'],
#                     'output': item['label'],
#                     'category': subset  # Using subset name as category
#                 })

#     # Write to a JSON file
#     with open(json_file_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# # Example usage
# json_file_path = 'sst2.json'  # Path to save the JSON file
# process_sst2_to_json(json_file_path)

# Return the path to the saved file
# json_file_path

from datasets import load_dataset
import json

def process_sst2_to_json(json_file_path):
    # Load the SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # Initialize an empty list to store data
    data = []
    # data2 = []
    # print(dataset['train'][0][0])
    # print(len(dataset['train']))

    # Process the new training set (first 66,675 items of the original training set)
    for item in dataset['train']:
        # print(item)
        # label = 'positive' if item['label'] == 1 else 'negative'
        if len(data)<=66675:
            data.append({
                'instruction': None,  # or any default instruction
                'input': item['sentence'],
                'output': item['label'],
                'category': 'train'
            })
        else:
            data.append({
                'instruction': None,  # or any default instruction
                'input': item['sentence'],
                'output': item['label'],
                'category': 'validation'
            })


    # Process the new test set (original validation set)
    for item in dataset['validation']:
        # label = 'positive' if item['label'] == 1 else 'negative'
        data.append({
            'instruction': None,  # or any default instruction
            'input': item['sentence'],
            'output': item['label'],
            'category': 'test'
        })

    # Write to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

json_file_path = 'sst2.json'  # Path to save the JSON file
process_sst2_to_json(json_file_path)
