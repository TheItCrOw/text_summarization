from tqe import TQE
import re
import json
import sys
import traceback


def similarity(german, english):
    # lang_1 = ["Why are you here?", "What is your name?"]
    # lang_2 = ["Das Wetter ist sehr angenehm.", "Was ist dein Name?"]
    model = TQE('LaBSE')
    cos_sim_values = model.fit([german], [english])
    return cos_sim_values[0]


def getText():
    f = open("input.txt", "r", encoding='utf-8')
    return f.read()


def clean_text(text):
    # We want to cleanup the text a bit.    
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(' +', ' ', text)
    text = re.sub('(DE)', '', text)
    return text


# Start point
if __name__ == '__main__':
    try:
        text = getText()
        text = json.loads(text, strict=False)
        german_text = clean_text(text['german'])
        english_text = clean_text(text['english'])
        sim = similarity(german_text, english_text)

        # We json this result doct and write it as a result
        result = {
            "similarity": sim,
        }

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
