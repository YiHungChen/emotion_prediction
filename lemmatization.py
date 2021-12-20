from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


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


def lemmas(words_list):
    tagged_sent = pos_tag(words_list)
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


