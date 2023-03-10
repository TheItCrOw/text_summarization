# Summaries texts with spacy and some text scoring. See
# https://www.w3schools.com/python/python_tuples_access.asp

import spacy
from spacy.lang.de.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest


def getText():
    '''Get the text we want to summarize'''
    return " ".join("""Frau Präsidentin! Meine Damen und Herren! Herr Kuhle, haben Sie das bei der Atlantik-Brücke gelernt, im Young-Leaders-Programm?
Ich muss Ihnen wirklich sagen: Die heutige Debatte ist viel zu ernsthaft, als dass man auf so eine Art und Weise hier Polemik betreibt. Ich muss Ihnen
                    wirklich sagen: Das geht so nicht.
Meine Damen und Herren, die Lage ist sehr ernst. Im September wurde Nord Stream in die Luft gesprengt. Jetzt, fast fünf Monate später, steht ein
                    furchtbarer Verdacht im Raum: Nord Stream sollte von unseren eigenen Verbündeten angegriffen worden sein. Es steht der Verdacht im Raum,
dass die Vereinigten Staaten von Amerika und das Königreich Norwegen einen Sprengstoffanschlag auf Nord Stream geplant und ausgeführt haben. Erhoben
                    wurde dieser Vorwurf von einem Staatsbürger der USA, dem mehrfach ausgezeichneten Journalisten Seymour Hersh. Hersh beruft sich in einem am Mittwoch
                    veröffentlichten Artikel auf eine Quelle,
die anscheinend direkte Kenntnis von der Planung der Operation hatte. Hersh behauptet, die Vereinigten Staaten und Norwegen hätten im vergangenen
                    Sommer dafür gesorgt, dass während eines Marinemanövers der NATO in der Ostsee Sprengstoff an beiden Nord-Stream-Pipelines platziert und einige Monate später
                    durch ein Signal gezündet wurde.
Das Weiße Haus und die CIA bestreiten diesen Vorwurf. Doch wie sehr man amerikanischen Geheimdiensten vertrauen kann, das wissen wir ja.
Ich erinnere nur an die NSA-Affäre, als herauskam, dass Frau Merkel von den Amerikanern abgehört wurde.
Ein Sprengstoffangriff – das muss man aber in aller Deutlichkeit sagen – würde eine rote Linie überschreiten.
In Artikel 5 des Nordatlantikvertrages heißt es:
Die Parteien vereinbaren, daß ein bewaffneter Angriff gegen eine oder mehrere von ihnen in Europa oder Nordamerika als ein Angriff gegen sie alle
                    angesehen werden wird …
Ein Angriff auf Nord Stream ist ein Angriff auf Deutschland und damit auf das gesamte Bündnis. Aber wenn es ein Bundesgenosse wäre, der unsere
                    kritische Infrastruktur angegriffen hätte, dann wäre das Vertrauen, das Grundlage für jedes Bündnis ist, zerstört.
Meine Damen und Herren, ich möchte eines sehr klar sagen, ehe ich bewusst missverstanden werde: Ich kann nicht ausschließen, dass es die USA und
                    Norwegen waren, die Nord Stream in die Luft gesprengt haben. Ich kann nicht ausschließen, dass Russland Nord Stream in die Luft gesprengt hat;
wahrscheinlich würden wir dann aber darüber lesen. Um genau zu sein, kann ich gar nichts ausschließen. Und warum kann ich nichts ausschließen?
Warum weiß ich als Abgeordneter, 137 Tage nachdem Nord Stream in die Luft gesprengt wurde, immer noch nicht, wer den Anschlag verübt hat?
Weil in diesen 137 Tagen die Regierung von Scholz, Baerbock und Habeck nichts, null, keinen Nanometer zur Aufklärung beigetragen hat.
Die Regierung der viertgrößten Volkswirtschaft der Erde weiß nach fast fünf Monaten immer noch nicht, wer uns vor der eigenen Haustür angegriffen hat.
                    Die freundlichste Deutung dieses Totalversagens ist, dass unsere Regierung aus inkompetenten Gauklern besteht, die man weder in Moskau noch in Washington noch
                    in Oslo ernst nimmt.
Im schlimmsten Fall aber heißt es etwas ganz anderes: dass diese Regierung kein Interesse an Aufklärung hat,
dass diese Regierung die Wahrheit unterdrückt, dass diese Regierung nicht im deutschen Interesse, sondern im Interesse des Auslands handelt.
Anzeichen gibt es leider genug.
Im Februar 2022 erklärte US-Präsident Biden während einer Pressekonferenz mit Olaf Scholz, dass man Nord Stream ein Ende setzen würde. Widerspruch von
                    Scholz? – Fehlanzeige. Und erst diese Woche hat „Die Zeit“ aufgedeckt, dass Außenministerin Baerbock systematisch mit ausländischen Regierungen zusammengewirkt
                    hat, um den Bundeskanzler zu Leopard-Lieferungen zu zwingen. Eine solche Außenministerin, die kennt keine deutschen Interessen. Eine solche Außenministerin,
die kennt auch kein Vaterland. Eine solche Außenministerin, meine Damen und Herren, der ist alles zuzutrauen.
In dieser schwierigen Stunde ist es an uns, den gewählten Vertretern des deutschen Volkes, das zu tun, woran diese Regierung offenkundig scheitert.
                    Wir müssen für Aufklärung sorgen.
Wir müssen Wahrheit ans Licht bringen. Die im Raum stehenden Vorwürfe müssen vollständig und restlos aufgeklärt werden.
Es muss aufgeklärt werden, wer die Drahtzieher und wer hier die Mitwisser dieses hinterhältigen Angriffs waren. Das deutsche Volk, lieber Herr Kuhle,
                    hat ein Recht darauf, zu erfahren, wer uns angegriffen hat.
""".split())


def get_doc():
    # Creates a nlp pipeline and a doc for the given text
    # Create a new pipeline
    nlp = spacy.load('en_core_web_lg')

    # Create a document out of the text. A doc is a sequence of tokens
    # Its a distinct text
    doc = nlp(getText())
    print("\n=======================================")
    print("Doc: ")
    print(doc)
    print("With number of sentences: " + str(len(list(doc.sents))))
    print("=======================================\n")
    return doc


def filter_token(doc):
    # Here we want to filter the tokens. We only want ADJ, NOUN, VERB
    # because these hold the most relevant information in each sentence
    keywords = []
    # The .?! to seperate into sentencens
    stopwords = list(STOP_WORDS)
    # These part of speeches we want to check
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']

    for token in doc:
        if(token.text in stopwords or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            keywords.append(token.text)
    print("\n=======================================")
    print("Keywords: ")
    print(keywords)
    print("=======================================\n")
    # return the keywords
    return keywords


def count_keywords(keywords, take):
    # Counts each keywords in the doc
    frequence = Counter(keywords)
    print("\n=======================================")
    print("Top " + str(take) + " Keywords with their frequence: ")
    print(frequence.most_common(take))
    print("=======================================\n")
    return frequence.most_common(take)


def normalize(keywords, frequence, take):
    # Normalize the occurences. So like ('deep', 6) to ('deep', 0.33)
    max_frequence = Counter(keywords).most_common(1)[0][1]
    normalized_frequences = []
    for i in range(0, len(frequence)):
        word_and_count = frequence[i]
        word = word_and_count[0]
        count = word_and_count[1]
        normalize = (count/max_frequence)
        normalized_frequences.append((word, normalize))
    print("\n=======================================")
    print("Normalized keywords with their frequence: ")
    print(dict(normalized_frequences))
    print("=======================================\n")
    return dict(normalized_frequences)


def weigh_sentences(doc, frequence):
    # Now we need to calculate what sentences we take for the summary.
    # We do that by the word frequency
    sentence_strength = {}
    for sentence in doc.sents:
        for word in sentence:
            if(word.text in frequence.keys()):
                if(sentence in sentence_strength.keys()):
                    sentence_strength[sentence] += frequence[word.text]
                else:
                    sentence_strength[sentence] = frequence[word.text]
    print("\n=======================================")
    print("Sentences with their strength: ")
    print(sentence_strength)
    print("=======================================\n")
    return sentence_strength


def main():
    doc = get_doc()
    keywords = filter_token(doc)
    frequence = count_keywords(keywords, 10)
    frequence = normalize(keywords, frequence, 10)
    sentence_strength = weigh_sentences(doc, frequence)

    # and now print the top ranked sentences as the summary
    summary = " ".join([str(line) for line in nlargest(4, sentence_strength, key=sentence_strength.get)])
    print("The given text: ")
    print(getText())
    print("\n==========================Summary============================\n")
    print(summary)
    # Print out the results


# Start point
if __name__ == '__main__':
    print("Simple SpaCy Text Summarization")
    main()
