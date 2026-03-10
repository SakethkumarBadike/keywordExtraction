from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# 1. Path to your saved folder (Change if your folder name is different)
model_path = "./my_ner_model/final_ner_model" 

# 2. Load the model and tokenizer from the local directory
model = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. Create the NER pipeline
# aggregation_strategy="simple" merges B-SKILL and I-SKILL into one "SKILL" group
nlp_ner = pipeline(
    "ner", 
    model=model, 
    tokenizer=tokenizer, 
    aggregation_strategy="simple" 
)

# 4. Your test text
text = """
Experienced Data Scientist with a focus on Deep Learning and Natural Language Processing. 
Proficient in Python, PyTorch, and developing Computer Vision algorithms. 
Strong understanding of SQL and cloud platforms like AWS.
"""

# 5. Run Inference
results = nlp_ner(text)

# 6. Display results with a confidence threshold
print(f"{'Entity':<20} | {'Label':<10} | {'Score':<8}")
print("-" * 45)

for entity in results:
    if entity['score'] > 0.90:  # You can adjust this threshold
        print(f"{entity['word']:<20} | {entity['entity_group']:<10} | {entity['score']:.4f}")