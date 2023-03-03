# Summaries texts with Text Rank. See
# https://www.youtube.com/watch?v=SNimr_nOC7w
# and https://www.edlitera.com/blog/posts/text-summarization-nlp-how-to
# Math behind textrank ranking:
# https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

import spacy
import pytextrank
import re


def getText():
    '''Get the text we want to summarize'''
    return """Frau Präsidentin! Liebe Kolleginnen und Kollegen! Ich will zunächst, lieber Michael Kießling, sagen: Man kann in den baupolitischen Debatten nicht
                    immer sagen: „13 Milliarden Euro für Sanierung und nur 1 Milliarde für Neubau“, und hier jetzt sagen, für die Sanierung stünde nichts zur Verfügung – es stand
                    nie mehr zur Verfügung! Im Rahmen der Sanierung wird fast alles gefördert, was man sich nur vorstellen kann. Erste Bemerkung.
Zweite Bemerkung. Frau Skudelny hat völlig recht mit dem Hinweis auf das Building Information Modeling und die Rolle, die BIM für die
                    Recyclingwirtschaft hat. Wir haben uns das Circular Economy House in Berlin-Neukölln angeguckt. Dafür ist BIM unmittelbar erforderlich. Guckt euch das einfach
                    mal an! Dann werdet ihr sehen, Frau Skudelny liegt mit ihrem Hinweis gar nicht falsch. Ich mache das auch, ich betrachte auch mal das Ganze.
Wenn man in den Antrag schaut, den die Union geschrieben hat, stellt man fest, es geht nicht nur um die Frage der mineralischen Rohstoffe, sondern
                    beispielsweise auch um den Neubau. Es wird kritisiert, dass es immer höhere energetische Anforderungen gibt, die Fokussierung auf die verbrauchte Energie am
                    Gebäude usw.
Jetzt nenne ich Ihnen mal zwei Zahlen: 1990 hatten wir in der Bundesrepublik Deutschland 210 Millionen Tonnen CO2-Belastung aus dem Gebäudesektor.
                    2019 waren es nicht mehr 210 Millionen Tonnen, sondern 120 Millionen Tonnen. Wir haben dreißig Jahre gebraucht, um das annähernd zu halbieren. Wir haben uns
                    jetzt vorgenommen, das von 2022 bis 2030, in acht Jahren, erneut zu halbieren, auf 67 Millionen Tonnen.
Was Ihre Kritik zum Gebäudestandard KfW 55 angeht: Er ist jetzt obligatorisch ohne Förderung, weil es nur Mitnahmeeffekte gegeben hat und weil Herr
                    Altmaier – das muss man sagen – und Herr Seehofer das Geld mit der Schubkarre herausgefahren haben, ohne dass die Förderung eine ökonomische Wirkung, eine
                    ökologische Wirkung, eine soziale Wirkung gehabt hat. Das war falsch; das kann sich keiner erlauben. Mit 5 Milliarden Euro haben wir angefangen, bei
                    16 Milliarden Euro haben wir aufgehört, weil sich im Bereich der Gebäudeemissionen im Grunde genommen fast nichts geändert hat.
Jetzt ändert sich etwas, weil wir hier – das ist das Gute an Ihrem Antrag; aber wir sind auch schon selber darauf gekommen – tatsächlich über
                    Kreislaufwirtschaft reden, weil das die Ökobilanz verbessert. Ehrlich gesagt, die Antwort auf die Frage „Mit welchen Maßnahmen wollen Sie eigentlich
                    Klimaneutralität bis 2045 erreichen?“ bleiben Sie permanent schuldig.
Mosern, meckern und mehr Geld fordern. Ja, mehr Geld möchte ich auch, immer; das ist aber, ehrlich gesagt, nicht die pfiffigste Geschichte. Zweiter
                    Punkt.
Dritter Punkt. Was in Ihrem Antrag steht, haben wir zu einem Teil schon erledigt; denn das Qualitätssiegel Nachhaltiges Gebäude, das QNG-Siegel, ist
                    schon da. Das achtet auf den Lebenszyklus, auf den ökologischen Fußabdruck.
Wir konzentrieren uns nicht nur auf Gebäude, sondern bringen jetzt die kommunale Wärmeplanung auf den Weg. Das ist unmittelbar wichtig.
Ich komme zu Ihrer Überschrift zurück und sage Ihnen: Ja, wir führen den Gebäuderessourcenpass ein, wie im Koalitionsvertrag vereinbart. Das
                    grundlegende Prinzip: In dem Gebäuderessourcenpass sollen für jedes Gebäude die wesentlichen Informationen rund um den Ressourcenverbrauch, die Klimawirkung und
                    die Kreislauffähigkeit transparent angegeben werden. Langfristig schafft der Pass also die Grundlage für konsistente Kreislaufwirtschaft im Bausektor. Das ist
                    vernünftig, ausgesprochen vernünftig.
Jetzt will ich noch einen Punkt ansprechen: Das ist eine gesamtgesellschaftliche Aufgabe. Nur mit dem Finger aufs Gesetz zu zeigen, reicht nicht aus.
                    Erstens. Sie fordern mehr Mittel für Baustoffforschung. Ja, die wurde sträflich vernachlässigt. Fragen wir mal, von wem. Jetzt passiert was: 5 Millionen Euro
                    sind im Haushalt eingestellt, 10 Millionen Euro in Form von Verpflichtungsermächtigungen vorgesehen. Bei Ihnen: null, überhaupt nichts.
Zweiter Punkt: die Eigentümer; dazu habe ich schon etwas gesagt.
Dritter Punkt: die Industrie. Beim Thema Industrie stehen wir wirklich mit den Betroffenen in Verbindung. Normung, technische Zulassung, all das sind
                    Aufgaben, die erledigt werden müssen, und das ist auf dem Weg.
Aber trotzdem herzlichen Dank, dass Sie mit Ihrem Antrag mitgedacht haben.
Schönen Dank.
"""
    #return """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""


def main():
    # Text Rank looks at all sentences and assigns a rank to them
    # the most highest ranked sentences are then chosen to be the summary
    # We use spacy to implement the textrank algorithm
    # First we need a spacy pipeline. The model we use is en_core_web_lg
    nlp = spacy.load("de_core_news_sm")

    # Add the textrank algorithm to the spacy pipeline
    nlp.add_pipe("textrank")

    # We want to cleanup the text a bit.
    text = getText()
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(' +', ' ', text)
    # The doc now contains the summary. That is it!
    doc = nlp(text)

    print("The given text: ")
    print(text)
    print("==========================Summary============================")
    # resulting_sentences = len(list(doc.sents))//8
    resulting_sentences = 5
    print(resulting_sentences)
    # Print out the results
    for sentence in doc._.textrank.summary(limit_sentences=resulting_sentences):
        print(sentence)


# Start point
if __name__ == '__main__':
    print("Extractive Text Summarization with the Text Rank")
    main()
