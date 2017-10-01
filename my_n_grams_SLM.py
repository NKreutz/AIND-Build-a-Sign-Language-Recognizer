from collections import Counter

class N_Gram_SLM(object):
    """Statistical Language Model Selector"""
    def sentence_to_n_grams(tokenized_sentence, N:int):
        sentence_n_grams = []
        for i in range(len(sentence_tokens)-(N-1)):
            count = i
            n_grams = []
            for num in range(N):
                word = sentence_tokens[count]
                n_grams.append(word)
                count += 1
            sentence_n_grams.append(tuple(n_grams))
        return sentence_n_grams

    def get_tokenized_sentences(test_set):
        tokenized_sentences = []
        for video_num in test_set.sentences_index:
            sentence_tokens = ['<s>'] + [test_set.wordlist[i] for i in test_set.sentences_index[video_num]] + ['</s>']
            tokenized_sentences.append(sentence_tokens)
        return tokenized_sentences

    def n_gram_SLM_select(self, test_set, N:int, probabilities):
        all_n_grams = []
        all_tokens = []
        for tokenized_sentence in get_tokenized_sentences(test_set):
            for token in tokenized_sentence:
                all_tokens.append(token)
            sentence_n_grams = sentence_to_n_grams(tokenized_sentence, N)
            for n_gram in sentence_n_grams:
                all_n_grams.append(n_gram)
        token_raw_counts = Counter(all_tokens)
        n_gram_raw_counts = Counter(all_n_grams)


        return
