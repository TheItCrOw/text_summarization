# see: https://huggingface.co
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, MarianMTModel
from transformers import pipeline
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
import requests
import spacy
import pytextrank
import re
import json
import sys
import traceback


nlp = spacy.load("de_core_news_sm")
pegasus_model_name = "transformersbook/pegasus-samsum"


def getText():
    f = open("pegasus_input.txt", "r", encoding='utf-8')
    return f.read()


def generate_abstract_summary(text):

    pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)
    pipe = pipeline(
        "summarization",
        model=pegasus_model_name,
        tokenizer=pegasus_tokenizer,
        truncation=True,
        framework="pt"
    )
    summarizer = pipe
    summary = summarizer(text, min_length=0,
                         max_length=512)
    return summary[0]['summary_text']


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

        abstract_summary = generate_abstract_summary(text)
        # print("Abstract Summary in English: ==========================================================")
        # print(abstract_summary)

        in_german = translate_english_to_german(abstract_summary)
        # print("Abstract Summary in German: ==========================================================")
        # print(in_german)

        # We json this result doct and write it as a result
        result = {
            "abstract_summary": in_german,
        }
        # print("The results: ==========================================================")
        # print(result)

        # Write the result as json
        with open("pegasus_output.json", "w", encoding="utf-8") as outfile:
            json.dump(result, outfile)
        print('GOOD')
    except Exception as ex:
        # print(str(ex))
        # print(traceback.format_exc())
        bad = {
            'ex': str(ex),
            'traceback': traceback.format_exc()
        }
        with open("pegasus_output.json", "w", encoding="utf-8") as outfile:
            json.dump(bad, outfile)
        print('BAD')
    sys.stdout.flush()
