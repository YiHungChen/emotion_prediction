from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from autocorrect import Speller
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
spell = Speller()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmas_sentence(words_list):
    output = []
    for index, sentence in enumerate(words_list):
        tokens = word_tokenize(sentence)
        tokens = [spell(x) for x in tokens]
        tagged_sent = pos_tag(tokens)
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 詞形還原
        output.append(lemmas_sent)
    return output


def lemmas_words(words_list):
    output = []
    tokens = word_tokenize(words_list)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 詞形還原

    return lemmas_sent


if __name__ == '__main__':
    sentence = 'football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.'
    tokens = word_tokenize(sentence)  # 分詞
    tagged = pos_tag(tokens)  # 獲取單詞詞性
    lemmas(tokens)


