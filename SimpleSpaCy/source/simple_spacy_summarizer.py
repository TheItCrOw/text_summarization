# Summaries texts with spacy and some text scoring. See
# https://www.w3schools.com/python/python_tuples_access.asp


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest


def getText():
    '''Get the text we want to summarize'''
    return """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""


def get_doc():
    # Creates a nlp pipeline and a doc for the given text
    # Create a new pipeline
    nlp = spacy.load('en_core_web_lg')

    # Create a document out of the text. A doc is a sequence of tokens
    doc = nlp(getText())
    print("Doc: ")
    print(doc)
    print("With number of sentences: " + str(len(list(doc.sents))))
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
    print("Keywords: ")
    print(keywords)
    # return the keywords
    return keywords


def count_keywords(keywords, take):
    # Counts each keywords in the doc
    frequence = Counter(keywords)
    print("Top " + str(take) + " Keywords with their frequence: ")
    print(frequence.most_common(take))
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
    print("Normalized keywords with their frequence: ")
    print(dict(normalized_frequences))
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
    print("Sentences with their strength: ")
    print(sentence_strength)
    return sentence_strength


def main():
    doc = get_doc()
    keywords = filter_token(doc)
    frequence = count_keywords(keywords, 5)
    frequence = normalize(keywords, frequence, 5)
    sentence_strength = weigh_sentences(doc, frequence)

    # and now print the top ranked sentences as the summary
    summary = nlargest(3, sentence_strength, key=sentence_strength.get)
    print("The given text: ")
    print(getText())
    print("==========================Summary============================")
    print(summary)
    # Print out the results


# Start point
if __name__ == '__main__':
    print("Simple SpaCy Text Summarization")
    main()
