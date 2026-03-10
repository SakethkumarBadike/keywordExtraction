import json
import os

def prepare_ner_dataset(source_folder, output_filename="ner_ready_data.json"):
    ner_dataset = []
    
    # Get the path to the current folder where the script is running
    current_dir = os.getcwd()
    target_path = os.path.join(current_dir, output_filename)

    # Ensure source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' not found.")
        return

    for filename in os.listdir(source_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(source_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    text = data.get("text", "")
                    raw_annotations = data.get("annotations", [])
                    
                    cleaned_entities = []
                    for ann in raw_annotations:
                        # ann format: [start, end, "LABEL: Word"]
                        start, end, label_str = ann[0], ann[1], ann[2]
                        
                        # Split by ':' and take only the first part (the Label)
                        label_only = label_str.split(':')[0].strip()
                        
                        cleaned_entities.append([start, end, label_only])
                    
                    ner_dataset.append({
                        "text": text,
                        "entities": cleaned_entities
                    })
                    
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

    # Save the file to the current directory
    with open(target_path, 'w', encoding='utf-8') as out_f:
        json.dump(ner_dataset, out_f, indent=4)
    
    print(f"✔ Success! Merged {len(ner_dataset)} files.")
    print(f"✔ Saved to: {target_path}")

# --- EXECUTION ---
# Replace 'json_files_folder' with the name of your folder containing the jsons
prepare_ner_dataset('C:\\Users\\Saketh\\Desktop\\lasttry\\ResumesJsonAnnotated')