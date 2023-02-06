# Summaries texts with Text Rank. See
# https://www.youtube.com/watch?v=SNimr_nOC7w
# and https://www.edlitera.com/blog/posts/text-summarization-nlp-how-to
# Math behind textrank ranking:
# https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

import spacy
import pytextrank


def getText():
    '''Get the text we want to summarize'''
    return """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""


def main():
    # Text Rank looks at all sentences and assigns a rank to them
    # the most highest ranked sentences are then chosen to be the summary
    # We use spacy to implement the textrank algorithm
    # First we need a spacy pipeline. The model we use is en_core_web_lg
    nlp = spacy.load("en_core_web_lg")

    # Add the textrank algorithm to the spacy pipeline
    nlp.add_pipe("textrank")

    # The doc now contains the summary. That is it!
    doc = nlp(getText())

    print("The given text: ")
    print(getText())
    print("==========================Summary============================")
    # Print out the results
    for sentence in doc._.textrank.summary(limit_sentences=2):
        print(sentence)


# Start point
if __name__ == '__main__':
    print("Extractive Text Summarization with the Text Rank")
    main()
