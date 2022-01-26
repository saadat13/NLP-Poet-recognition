from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter

WORD_COUNT_THRESHOLD = 5
ESKALEI = 0.0001
L3, L2, L1, EPSILON = 0.80, 0.10, 0.10, 0.05


class LanguageModel:
    backoff_model = None

    def __init__(self, filename, lambda3=L3, lambda2=L2, lambda1=L1, epsilon=EPSILON):
        with open(filename, 'r') as f:
            sentences = f.readlines()

        unigram = {}
        words = []
        for sent in sentences:
            for word in sent.split():
                words.append(word)
        temp = Counter(words)
        count_dict = {}
        for k, v in temp.items():
            if v > WORD_COUNT_THRESHOLD:
                count_dict[k] = v

        total_count = sum(count_dict.values())
        for word, count in temp.items():
            unigram[word] = count / total_count

        # bigram
        couple_words = []
        bigram = {}
        for sent in sentences:
            token = word_tokenize(sent)
            for w1, w2 in ngrams(token, 2):
                couple_words.append((w1, w2))
        bi_count_dict = Counter(couple_words)
        for w1_w2, count in bi_count_dict.items():
            w1, w2 = w1_w2
            try:
                bigram[(w2, w1)] = bi_count_dict[(w1, w2)] / count_dict[w1]
            except KeyError:
                bigram[(w2, w1)] = ESKALEI

        bigram_backoff_model = {}

        for w2_w1, p in bigram.items():
            w2, w1 = w2_w1
            try:
                unigram_value = unigram[w2]
            except KeyError:
                unigram_value = 0
            bigram_backoff_model[(w2, w1)] = lambda3 * bigram[(w2, w1)] + lambda2 * unigram_value + lambda1 * epsilon

        self.backoff_model = bigram_backoff_model

    def calculate_probability(self, mesra):
        prob = 1
        token = word_tokenize(mesra)
        for w1, w2 in ngrams(token, 2):
            try:
                prob *= self.backoff_model[(w2, w1)]
            except KeyError:
                prob *= ESKALEI
        return prob


def main():
    hafez = LanguageModel('../train_set/hafez_train.txt')
    ferdowsi = LanguageModel('../train_set/ferdowsi_train.txt')
    molavi = LanguageModel('../train_set/molavi_train.txt')

    lines = open('../test_set/test_file.txt', 'r').readlines()
    true_count = 0
    total_count = len(lines)

    for line in lines:
        ans, mesra = line.split('\t')
        ans = int(ans)
        p_hafez = hafez.calculate_probability(mesra)
        p_ferdowsi = ferdowsi.calculate_probability(mesra)
        p_molavi = molavi.calculate_probability(mesra)
        m = max(p_hafez, p_ferdowsi, p_molavi)
        if m == p_ferdowsi and ans == 1:
            print("ok it's from ferdowsi")
            true_count += 1
        elif m == p_hafez and ans == 2:
            print("ok it's from hafez")
            true_count += 1
        elif m == p_molavi and ans == 3:
            print("ok it's from molavi")
            true_count += 1
        else:
            print("not ok")

    print('precision is :', 100 * true_count / total_count)


if __name__ == '__main__':
    main()
