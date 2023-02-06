# Lets import the pegasus dependencies provided by the HuggingFace
# community. (See: https://huggingface.co/) (Contains pretrained models)
# Blog on how to use the pegasus like we do here:
# https://www.edlitera.com/blog/posts/text-summarization-nlp-how-to
# or in video form: https://www.youtube.com/watch?v=SNimr_nOC7w

from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline


# Lets define a pretrained model we want to work with. This
# model comes from the huggingface community.
# There are other datasets like CCN/DailyMail, NEWSROOM, Gigaword
# arXiv, BIGPATENT, WikiHow, Reddit TIFU
model_name = "google/pegasus-xsum"


def getText():
    '''Get the text we want to summarize'''
    return """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance. Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems. ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, whence the structured part."""


def get_tokenizer():
    # We are using the pretrained model for PEGASUS tokenizer
    # We need a tokenizer, since models, transformers and other
    # deep learning models cant take in text as input data, but
    # only numbers. The tokenizer creates numerical represantations
    # for our words in the text. These numbers are used to operate
    # with our model
    return PegasusTokenizer.from_pretrained(model_name)


def get_model():
    # define the model
    return PegasusForConditionalGeneration.from_pretrained(model_name)


def get_pipeline():
    # Creates a summary pipeline and returns it.
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=get_tokenizer(),
        framework="pt"
    )


def create_summary_pipeline(min, max):
    # min_length: min_length of summary (Number of tokens)
    # max_length: max_length of summary (Number of tokens)
    # Summarizes the text like create_summary, but with a
    # pipeline that gives us more options.
    summarizer = get_pipeline()
    summary = summarizer(getText(), min_length=min,
                         max_length=max)
    return summary


def create_summary():
    # Creates a summary without a pipeline. These steps are
    # the ones we do by "hand" what the pipeline version does
    # in the background
    # ==================================
    # We need the tokenizer
    tokenizer = get_tokenizer()

    # We need the model
    model = get_model()

    # we got the tokenizer, we got the model. Lets tokenize the data then
    # truncation=True => If the text is too long, chunk it
    # padding="longest" => If the text is too short, padd the text
    # so that each inputs are of the same length
    # return_tensors="pt" => Cause we are using pytorch
    tokens = tokenizer(getText(), truncation=True, padding="longest",
                       return_tensors="pt")
    print("Extracted tensors and tokens:")
    # Tensors are the inputs, outputs and transformations within
    # neural networks: https://deeplizard.com/learn/video/Csa5R12jYRg
    print(tokens)

    # We got our tokens now. Lets summarize
    # The ** operator:
    # mydict = {'x':1,'y':2,'z':3}
    # foo(**mydict)
    # x=1, y=2, z=3
    # Summary is encoded because we got tokens again. Models only work
    # with tokens
    encoded_summary = model.generate(**tokens)

    # Decode the token summary
    decoded_summary = tokenizer.decode(encoded_summary[0],
                                       skip_special_tokens=True)
    return decoded_summary


def main():
    # summary = create_summary()
    summary = create_summary_pipeline(30, 150)

    print("The given text: ")
    print(getText())
    print("==========================Summary============================")
    print(summary)


# Start point
if __name__ == '__main__':
    print("Abstractive Text Summarization with the PEGASUS Model")
    main()
