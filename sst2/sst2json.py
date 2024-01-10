from datasets import load_dataset
import json

def process_sst2_to_json(json_file_path):
    # Load the SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # Initialize an empty list to store data
    data = []

    # Process each subset (train, validation, test)
    for subset in ['train', 'validation', 'test']:
        # Check if the subset exists in the dataset
        if subset in dataset:
            for item in dataset[subset]:
                # Convert label to string ('positive' or 'negative')
                label = 'positive' if item['label'] == 1 else 'negative'
                # Append the item to the data list
                data.append({
                    'instruction': None,  # or any default instruction
                    'input': item['sentence'],
                    'output': item['label'],
                    'category': subset  # Using subset name as category
                })

    # Write to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage
json_file_path = 'sst2.json'  # Path to save the JSON file
process_sst2_to_json(json_file_path)

# Return the path to the saved file
# json_file_path

