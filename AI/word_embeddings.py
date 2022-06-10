#***********************************
# SPDX-FileCopyrightText: 2009-2020 Vtenext S.r.l. <info@vtenext.com> and KLONDIKE S.r.l. <info@klondike.ai>
# SPDX-License-Identifier: AGPL-3.0-only
#***********************************

from gensim.models import word2vec
from utils import clean_text, convert_data_to_index
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_word_embedding(text):

    text_clean = clean_text(text)
    train_tokens = []

    for sentence in text_clean:
        train_tokens.append(sentence.split())

    embedding_lenght = 300
    max_sequence_len = 1000

    # USING GENSIM
    model = word2vec.Word2Vec(train_tokens, iter=10, min_count=10, size=embedding_lenght, workers=4)
    indexes = convert_data_to_index(train_tokens, model.wv)
    sequence = pad_sequences(indexes, maxlen=max_sequence_len, padding="pre", truncating="post")  # e padding = post?

    return sequence