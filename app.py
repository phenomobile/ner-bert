import streamlit as st
import numpy as np
from transformers import pipeline
from transformers import AutoModelForTokenClassification, BertTokenizerFast
import torch
import spacy
from spacy import displacy

@st.cache(allow_output_mutation=True)
def get_model(): 
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = AutoModelForTokenClassification.from_pretrained('phenomobile/ner-bert')
    return tokenizer, model

tokenizer, model = get_model()

user_input = st.text_area('Enter text')
button = st.button("Analyze")

if user_input and button:
    ner_pipeline = pipeline(task = "ner", model = model, tokenizer = tokenizer)
    output = ner_pipeline(user_input)
    
    ner_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

    entities = []

    for i in range(len(output)):
        if output[i]['entity'] != 0:
            if output[i]['entity'][0] == 'B':
                j = i + 1
                while j < len(output) and output[j]['entity'][0] == 'I':
                    j += 1
                entities.append((output[i]['entity'].split('-')[1], output[i]['start'], output[j - 1]['end']))
                
    nlp = spacy.blank("en")
    doc = nlp(user_input)

    ents = []

    for ee in entities:
        ents.append(doc.char_span(ee[1], ee[2], label = ee[0], alignment_mode = 'expand'))

    ents = list(set(ents))

    doc.ents = ents
    options = {"ents": ["PER", "ORG", "LOC", "MISC"],
               "colors": {"PER": "lightblue", "ORG": "lightcoral", "LOC": "lightgreen", "MISC": "green"}}
    
    displacy_html = displacy.render(doc, style="ent", options=options, )

    st.markdown(displacy_html, unsafe_allow_html=True)