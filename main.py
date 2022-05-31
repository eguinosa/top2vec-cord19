# Gelin Eguinosa Rosique

from sys import argv
from pprint import pprint
from top2vec import Top2Vec

from papers_analyzer import PapersAnalyzer
from time_keeper import TimeKeeper
from extra_funcs import big_number


if __name__ == '__main__':
    # To record the runtime of the program
    stopwatch = TimeKeeper()

    # Separate the CORD-19 papers by their size in 3 categories
    # (small - 1 paragraph, medium - 1 page, big - more than 1 page)
    print("\nSorting and separating the CORD-19 papers by size in 3 categories...")
    sorted_papers = PapersAnalyzer(show_progress=True)
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get random documents from the 'big' category.
    number = 3_000
    print(f"\nExtracting {big_number(number)} Big Papers randomly from CORD-19...")
    # Top2Vec requires a list of strings (weird)
    papers_text = list(sorted_papers.big_papers_content(number, show_progress=True))
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Train Top2Vec Model.
    print("\nTraining Top2Vec Model...")
    # # Universal Sentence Encoder to train the model.
    # top2vec_model = Top2Vec(papers_text, embedding_model='universal-sentence-encoder')
    # BERT Sentence Transformer to train the model.
    top2vec_model = Top2Vec(papers_text, embedding_model='distiluse-base-multilingual-cased')
    print("Done.")
    print(f"[{stopwatch.formatted_runtime()}]")

    # Get the number of Topics found.
    num_topics = top2vec_model.get_num_topics()
    print(f"\nTop2Vec found {num_topics} topics.")

    # Get Topic sizes & Topic number.
    print("\nTopic Sizes:")
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
    for topic_size, topic_id in zip(topic_sizes, topic_nums):
        print(f"  Topic #{topic_id}: {topic_size}")

    # Get Topic's Words & Word Scores.
    topics_words, words_scores, topic_ids = top2vec_model.get_topics()
    for topic_words, word_scores, topic_id in zip(topics_words, words_scores, topic_ids):
        print(f"\nTopic #{topic_id} Words:")
        words_data = list(zip(word_scores, topic_words))
        pprint(words_data[:20])    


# # Import Libraries
# import numpy as np
# import pandas as pd
# from top2vec import Top2Vec
# from sklearn.datasets import fetch_20newsgroups

# # Reading the Data
# newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# # print(f"\nThe type of the newsgroup is {type(newsgroups)}.")

# # print("\nFirst article of the newsgroup:")
# # print(f"\n{newsgroups.data[0]}\n")

# # Training Top2Vec Model
# model = Top2Vec(newsgroups.data, embedding_model='universal-sentence-encoder')

# # Viewing the Number of Topics
# num_topics = model.get_num_topics()
# print(f"\nThe number of topics is: {num_topics}.\n")


