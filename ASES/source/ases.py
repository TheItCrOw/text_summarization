# see: https://huggingface.co/Einmalumdiewelt/T5-Base_GNAD?
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, MarianMTModel
from transformers import pipeline
import requests
import spacy
import pytextrank
import re
import json
import sys
import traceback


# Text Rank looks at all sentences and assigns a rank to them
# the most highest ranked sentences are then chosen to be the summary
# We use spacy to implement the textrank algorithm
# First we need a spacy pipeline. The model we use is en_core_web_lg
nlp = spacy.load("de_core_news_sm")


def getText():
    f = open("input.txt", "r", encoding='utf-8')
    return f.read()


def summarize_local(text):
    '''Summarizes with the local model'''
    # prepared_text = " ".join(text.split())
    # prepared_text = prepared_text.split('.')
    # prepared_text = [line.strip() for line in prepared_text]
    # prepared_text = tuple(prepared_text)[:-1]
    # is_split_into_words=True
    tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")

    model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")

    inputs = tokenizer(text, max_length=1024, truncation=True,
                       return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, 
                                 min_length=0,
                                 max_length=2048)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
    return summary


def summarize_via_api(text):
    '''Summarizes with the huggingface interference api of this model'''
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/Einmalumdiewelt/T5-Base_GNAD"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text
    })
    print("Output:")
    print(output)
    print("End Output ====================")
    return output[0]['summary_text']


def generate_abstract_summary(text):
    # https://huggingface.co/philschmid/bart-large-cnn-samsum
    summarizer = pipeline("summarization",
                          model="philschmid/bart-large-cnn-samsum",
                          truncation=True)
    return summarizer(text)[0]['summary_text']
    # return summarize_local(text)


def generate_extractive_summary(text):
    # Add the textrank algorithm to the spacy pipeline
    nlp.add_pipe("textrank")

    # The doc now contains the summary. That is it!
    doc = nlp(text)

    # max_sentences = len(list(doc.sents))//2
    max_sentences = 4

    result = ""
    # Print out the results
    for sentence in doc._.textrank.summary(limit_sentences=max_sentences):
        result += " " + str(sentence)
    return result


def clean_text(text):
    # We want to cleanup the text a bit.    
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(' +', ' ', text)
    return text


def translate_german_to_english(text):
    # https://huggingface.co/Helsinki-NLP/opus-mt-de-en
    src = "de"  # source language
    trg = "en"  # target language

    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # We tokenize the text and translate each sentence in the
    # opus-mt model.
    tokens = nlp(text)
    result = ""
    for sentence in tokens.sents:
        sentence = str(sentence)
        batch = tokenizer(sentence, return_tensors="pt")

        generated_ids = model.generate(**batch)
        result += tokenizer.batch_decode(generated_ids,
                                         skip_special_tokens=True)[0]
        result += " "
    return result


def translate_english_to_german(text):
    # https://huggingface.co/Helsinki-NLP/opus-mt-de-en
    src = "en"  # source language
    trg = "de"  # target language

    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    batch = tokenizer(text, return_tensors="pt")

    generated_ids = model.generate(**batch)
    return tokenizer.batch_decode(generated_ids,
                                  skip_special_tokens=True)[0]


# Start point
if __name__ == '__main__':
    try:
        text = getText()
        text = clean_text(text)
        # print("The original text: ==========================================================")
        # print(text)

        # print("As english text: ============================================================")
        english_text = translate_german_to_english(text)
        # print(english_text)

        extractive_summary = generate_extractive_summary(text)
        # print("Extractive summary ==========================================================")
        # print(extractive_summary)

        abstract_summary = generate_abstract_summary(english_text)
        # print("Abstract Summary in English: ==========================================================")
        # print(abstract_summary)

        in_german = translate_english_to_german(abstract_summary)
        # print("Abstract Summary in German: ==========================================================")
        # print(in_german)

        # We json this result doct and write it as a result
        result = {
            "english_translation": english_text,
            "abstract_summary": in_german,
            "extractive_summary": extractive_summary
        }
        # print("The results: ==========================================================")
        # print(result)

        # Write the result as json
        with open("output.json", "w", encoding="utf-8") as outfile:
            json.dump(result, outfile)
        print('GOOD')
    except Exception as ex:
        # print(str(ex))
        # print(traceback.format_exc())
        bad = {
            'ex': str(ex),
            'traceback': traceback.format_exc()
        }
        with open("output.json", "w", encoding="utf-8") as outfile:
            json.dump(bad, outfile)
        print('BAD')
    sys.stdout.flush()
