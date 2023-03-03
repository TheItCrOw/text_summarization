# see: https://huggingface.co/Einmalumdiewelt/T5-Base_GNAD?
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import requests

tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")

model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")

text = """Man kann in den baupolitischen Debatten nicht immer sagen: „13 Milliarden Euro für Sanierung und nur 1 Milliarde für Neubau“, und hier jetzt sagen, für die Sanierung stünde nichts zur Verfügung – es stand nie mehr zur Verfügung!
Er ist jetzt obligatorisch ohne Förderung, weil es nur Mitnahmeeffekte gegeben hat und weil Herr Altmaier – das muss man sagen – und Herr Seehofer das Geld mit der Schubkarre herausgefahren haben, ohne dass die Förderung eine ökonomische Wirkung, eine ökologische Wirkung, eine soziale Wirkung gehabt hat.
Frau Skudelny hat völlig recht mit dem Hinweis auf das Building Information Modeling und die Rolle, die BIM für die Recyclingwirtschaft hat.
denn das Qualitätssiegel Nachhaltiges Gebäude, das QNG-Siegel, ist schon da.
Beim Thema Industrie stehen wir wirklich mit den Betroffenen in Verbindung.1"""


def summarize_local():
    '''Summarizes with the local model'''
    prepared_text = " ".join(text.split())
    prepared_text = prepared_text.split('.')
    prepared_text = [line.strip() for line in prepared_text]
    prepared_text = tuple(prepared_text)[:-1]

    inputs = tokenizer([prepared_text], max_length=1024, truncation=True,
                       return_tensors="pt", is_split_into_words=True)

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0,
                                 max_length=1024)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
    print(summary)


def summarize_via_api():
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


# Start point
if __name__ == '__main__':
    # summarize_local()
    print(summarize_via_api())
