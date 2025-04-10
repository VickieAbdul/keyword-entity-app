from flask import Flask, render_template, request, send_file, jsonify
import spacy
import pandas as pd
from collections import defaultdict
import re
import subprocess
import tempfile
import os
from spacy.tokens import Span
import json

app = Flask(__name__)

# Ensure spaCy model is loaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load spaCy model and configure it with custom rules
def load_nlp_model_with_custom_rules():
    nlp = spacy.load("en_core_web_sm")
    
    # Add entity ruler with custom rules
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Add patterns for AI companies and technologies
    patterns = [
        {"label": "ORG", "pattern": "OpenAI"},
        {"label": "ORG", "pattern": "OpenAI Inc"},
        {"label": "ORG", "pattern": "OpenAI Inc."},
        {"label": "ORG", "pattern": "OpenAI Corporation"},
        {"label": "PRODUCT", "pattern": "GPT"},
        {"label": "PRODUCT", "pattern": "GPT-3"},
        {"label": "PRODUCT", "pattern": "GPT-4"},
        {"label": "PRODUCT", "pattern": "ChatGPT"},
        {"label": "ORG", "pattern": "Google DeepMind"},
        {"label": "ORG", "pattern": "DeepMind"},
        {"label": "ORG", "pattern": "Anthropic"},
        {"label": "PRODUCT", "pattern": "Claude"},
        {"label": "ORG", "pattern": "Meta AI"},
        {"label": "ORG", "pattern": "Facebook AI Research"},
        {"label": "ORG", "pattern": "FAIR"},
    ]
    ruler.add_patterns(patterns)
    
    # Custom component to fix "AI" and other tech terms being classified as GPE
    @spacy.Language.component("fix_tech_entities")
    def fix_tech_entities(doc):
        ai_terms = ["AI", "Artificial Intelligence", "Machine Learning", "ML", "NLP"]
        tech_orgs = ["AI Lab", "AI Research", "AI Alliance"]
        
        new_ents = []
        for ent in doc.ents:
            # Fix AI as standalone term
            if ent.text in ai_terms and ent.label_ == "GPE":
                new_ents.append(Span(doc, ent.start, ent.end, label="PRODUCT"))
            # Fix AI as part of organization name
            elif any(term in ent.text for term in tech_orgs) and ent.label_ == "GPE":
                new_ents.append(Span(doc, ent.start, ent.end, label="ORG"))
            else:
                new_ents.append(ent)
        
        doc.ents = new_ents
        return doc
    
    # Add custom component to pipeline
    nlp.add_pipe("fix_tech_entities", after="ner")
    
    return nlp

nlp = load_nlp_model_with_custom_rules()

# Define colors for entity labels
ENTITY_COLORS = {
    "PERSON": "#ffadad",
    "ORG": "#ffd6a5",
    "GPE": "#caffbf",
    "LOC": "#ffc6ff",
    "FAC": "#bdb2ff",
    "PRODUCT": "#a0c4ff",
    "EVENT": "#fdffb6",
    "WORK_OF_ART": "#9bf6ff",
    "DATE": "#d8f8ff",
    "TIME": "#a0c4ff",
    "MONEY": "#bdb2ff",
    "QUANTITY": "#ffc6ff",
    "PERCENT": "#caffbf",
    "CARDINAL": "#fffffc",
    "ORDINAL": "#fffffc",
    "LANGUAGE": "#ffd6a5",
    "NORP": "#fdffb6",
}

ENTITY_DESCRIPTIONS = {
    "PERSON": "People, including fictional",
    "ORG": "Organizations, companies, agencies",
    "GPE": "Countries, cities, states (Geopolitical Entities)",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water",
    "FAC": "Buildings, airports, highways, bridges",
    "PRODUCT": "Objects, vehicles, foods, tech products",
    "EVENT": "Named events like wars, sports events, hurricanes",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements, as of weight or distance",
    "PERCENT": "Percentage",
    "CARDINAL": "Numerals that do not fall under another type",
    "ORDINAL": "Ordinal numbers like 'first', 'second'",
    "LANGUAGE": "Any named language",
    "NORP": "Nationalities, religious or political groups",
}

@app.route('/')
def index():
    example_text = """
    Microsoft announced a $10 billion investment in OpenAI on March 24, 2023. 
    The deal was signed by Satya Nadella, CEO of Microsoft. 
    OpenAI, based in San Francisco, California, is known for developing ChatGPT.
    The company plans to use these funds to enhance their AI research capabilities.
    Google DeepMind and Anthropic are also major players in the AI industry.
    """
    return render_template('index.html', example_text=example_text, 
                          entity_descriptions=ENTITY_DESCRIPTIONS)

@app.route('/process', methods=['POST'])
def process_text():
    text = request.form.get('text', '')
    highlight_word = request.form.get('highlight_word', '')
    highlight_entities = 'highlight_entities' in request.form
    show_entity_count = 'show_entity_count' in request.form
    
    # Process the text
    doc = nlp(text)
    
    # Collect word highlight spans
    highlights = []
    if highlight_word:
        pattern = re.compile(r'\b' + re.escape(highlight_word) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            highlights.append((match.start(), match.end(), "#ffff00"))  # yellow

    # Add entity highlight spans if enabled
    entities = []
    entity_counts = defaultdict(int)
    
    if highlight_entities:
        for ent in doc.ents:
            color = ENTITY_COLORS.get(ent.label_, "#e0e0e0")  # default light gray
            highlights.append((ent.start_char, ent.end_char, color))
            entities.append({"text": ent.text, "label": ent.label_})
            entity_counts[ent.label_] += 1

    # Sort and build highlighted HTML
    highlights.sort(key=lambda x: x[0])  # sort by start index
    highlighted_text = ""
    last_idx = 0

    for start, end, color in highlights:
        # Handle overlapping spans by taking the later one
        if start < last_idx:
            continue
        highlighted_text += text[last_idx:start]
        highlighted_text += f"<span style='background-color: {color}; padding: 1px 3px; border-radius: 3px;'>{text[start:end]}</span>"
        last_idx = end

    highlighted_text += text[last_idx:]  # Add remaining text
    
    # Format entity counts for chart
    entity_count_data = [{"label": label, "count": count} 
                          for label, count in entity_counts.items()]
    
    # Return results
    result = {
        "highlighted_text": highlighted_text,
        "entities": entities,
        "entity_counts": dict(entity_counts),
        "entity_count_data": entity_count_data,
        "entity_colors": ENTITY_COLORS
    }
    
    return render_template('results.html', 
                          text=text,
                          highlighted_text=highlighted_text,
                          entities=entities,
                          entity_counts=sorted(entity_counts.items(), key=lambda x: x[1], reverse=True),
                          entity_colors=ENTITY_COLORS,
                          entity_descriptions=ENTITY_DESCRIPTIONS,
                          highlight_word=highlight_word,
                          highlight_entities=highlight_entities,
                          show_entity_count=show_entity_count,
                          chart_data=json.dumps(entity_count_data))

@app.route('/download-csv')
def download_csv():
    entities_json = request.args.get('entities', '[]')
    entities = json.loads(entities_json)
    
    # Create DataFrame and save to temporary CSV file
    df = pd.DataFrame(entities)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    
    @after_this_request
    def remove_file(response):
        try:
            os.remove(temp_file.name)
        except Exception as error:
            app.logger.error(f"Error removing temporary file: {error}")
        return response
        
    return send_file(temp_file.name, 
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='extracted_entities.csv')

def after_this_request(func):
    if not hasattr(request, 'after_request_callbacks'):
        request.after_request_callbacks = []
    request.after_request_callbacks.append(func)
    return func

@app.after_request
def call_after_request_callbacks(response):
    for callback in getattr(request, 'after_request_callbacks', []):
        response = callback(response)
    return response

if __name__ == '__main__':
    app.run(debug=True)