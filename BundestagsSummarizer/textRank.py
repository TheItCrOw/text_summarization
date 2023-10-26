# see: https://huggingface.co/Einmalumdiewelt/T5-Base_GNAD?
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
    f = open("textRank_input.txt", "r", encoding='utf-8')
    return f.read()


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


# Start point
if __name__ == '__main__':
    try:
        text = getText()
        text = clean_text(text)
        # print("The original text: ==========================================================")
        # print(text)

        extractive_summary = generate_extractive_summary(text)
        # print("Extractive summary ==========================================================")
        # print(extractive_summary)

        # We json this result doct and write it as a result
        result = {
            "extractive_summary": extractive_summary
        }

        # Write the result as json
        with open("textRank_output.json", "w", encoding="utf-8") as outfile:
            json.dump(result, outfile)
        print('GOOD')
    except Exception as ex:
        # print(str(ex))
        # print(traceback.format_exc())
        bad = {
            'ex': str(ex),
            'traceback': traceback.format_exc()
        }
        with open("textRank_output.json", "w", encoding="utf-8") as outfile:
            json.dump(bad, outfile)
        print('BAD')
    sys.stdout.flush()
