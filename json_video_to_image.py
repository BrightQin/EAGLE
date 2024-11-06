import json

# Load the JSON data from the file
with open('dataset/Video/train/videochatgpt_tune/videochatgpt_llavaimage_tune_filtered.json', 'r') as file:
    data = json.load(file)

# Iterate over each item in the data
for item in data:
    # Replace 'video' key with 'image'
    if 'video' in item:
        item['image'] = item.pop('video')
    # Replace '<video>' with '<image>' in conversations
    for conversation in item.get('conversations', []):
        conversation['value'] = conversation['value'].replace('<video>', '<image>')

# Save the modified data back to a file
with open('dataset/Video/train/videochatgpt_tune/videochatgpt_llavaimage_tune_filtered_image.json', 'w') as file:
    json.dump(data, file, ensure_ascii=False)