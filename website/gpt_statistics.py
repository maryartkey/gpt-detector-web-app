import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import statistics as st
from nltk.tokenize import sent_tokenize
from collections import Counter
import plotly.express as px
import gpt_statistics



stop_words = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]

class TextGroup:
    def __init__(self, texts):
        self._texts = texts  

    def get_mean_word_length(self):
        texts = self._texts
        mean_word_lengths = []

        for text in texts['text']:
            words_length = 0
            number_of_words = 0
            for word in text.split():
                words_length += len(word)
                number_of_words += 1
            if number_of_words != 0:
                mean_word_lengths.append(words_length/number_of_words)
            else:
                mean_word_lengths.append(1)
        return mean_word_lengths
    
    def get_stdev_word_lengths(self):
        texts = self._texts
        stdevs = []

        def stdev_text(text):
            lengths = list(map(len, text.split()))
            values = 0
            if lengths != []:
                values = st.stdev(lengths)            
            return values
        for text in texts['text']:
            if text != []:
                stdevs.append(stdev_text(text))    
            else:
                stdevs.append(0)             
        result_values = []
        for value in stdevs:
            if value == 0:
                value = min(stdevs)   
            result_values.append(value)
        return result_values

    def get_mean_sentence_length(self):      
        texts = self._texts
        mean_sentence_lengths = []

        for text in texts['text']:
            sentences_length = 0
            number_of_sentences = 0
            for sentence in sent_tokenize(text):                  
                sentences_length += len(sentence.split())
                number_of_sentences += 1
            if number_of_sentences == 0:
                number_of_sentences = 1
            if sentences_length == 0:
                sentences_length = 1
            mean_sentence_lengths.append(sentences_length/number_of_sentences)
        return mean_sentence_lengths
    
    def get_stdev_sentence_lengths(self):
        texts = self._texts
        stdevs = []

        def stdev_text(text):
            lengths = list(map(len, sent_tokenize(text)))
            values = 0
            if len(lengths) > 1:
                values = st.stdev(lengths)
            return values
        for text in texts['text']:
            if text != []:
                stdevs.append(stdev_text(text))
            else:
                stdevs.append(0)
        result_values = []
        for value in stdevs:
            if value == 0:
                value = min(stdevs)   
            result_values.append(value)
        return result_values
    
    def get_dots_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == '.':
                    marks_in_text += 1
                elif word == ',' or word == '!' or word == '?' or word == ':' or word == ';':
                    other_marks_in_text += 1
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values

    def get_commas_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == ',':
                    marks_in_text += 1
                elif word == '.' or word == '!' or word == '?' or word == ':' or word == ';':
                    other_marks_in_text += 1
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values
    
    def get_excpoints_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == '!':
                    marks_in_text += 1
                elif word == '.' or word == ',' or word == '?' or word == ':' or word == ';':
                    other_marks_in_text += 1
            
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values
    
    def get_questmarks_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == '?':
                    marks_in_text += 1
                elif word == '.' or word == ',' or word == '!' or word == ':' or word == ';':
                    other_marks_in_text += 1
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values
    
    def get_colons_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == ':':
                    marks_in_text += 1
                elif word == '.' or word == ',' or word == '!' or word == '?' or word == ';':
                    other_marks_in_text += 1
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values

    def get_semicolons_count(self):
        list_values = []
        texts = self._texts
        
        for text in texts['text']:
            marks_in_text = 0
            other_marks_in_text = 0
            for word in text.split():
                if word == ';':
                    marks_in_text += 1
                elif word == '.' or word == ',' or word == '!' or word == '?' or word == ':':
                    other_marks_in_text += 1
            total_marks = marks_in_text + other_marks_in_text
            if total_marks == 0:
                total_marks = 1
            list_values.append(marks_in_text/total_marks)
        return list_values
    
    def get_homogeneity(self):
        list_values = []        
        texts = self._texts
        
        # Процесс уникальных слов слева направо
        def forward(text):
            text_forward = text[:]
            unic_words = []
            unic_words_count = [1]
            for index in range (1, len(text_forward)):
                word = text_forward[index]
                if word not in unic_words:
                    unic_words_count.append(unic_words_count[index - 1] + 1)
                    unic_words.append(text_forward[index])
                else:
                    unic_words_count.append(unic_words_count[index - 1])
                    
            return unic_words_count[:]
        
        # Процесс уникальных слов справа налево
        def reverse(text):
            text_reversed = text[::-1]
            unic_words = []
            unic_words_count = [1]
            for index in range (1, len(text_reversed)):
                word = text_reversed[index]
                if word not in unic_words:
                    unic_words_count.append(unic_words_count[index - 1] + 1)
                    unic_words.append(text_reversed[index])
                else:
                    unic_words_count.append(unic_words_count[index - 1])
                    
            return unic_words_count[:]
        
        # Подсчёт численных статистик
        area = 0
        sup = 0
        area_abs = 0
        for text in texts['text']:
            area = 0
            text_forward = forward(text)
            text_reverse = reverse(text)
            for i in range(len(text_forward)):
                diff = text_forward[i] - text_reverse[i]
                area += diff
                if sup < abs(diff):
                    sup = abs(diff)
                area_abs = abs(area)
            list_values.append(area_abs)

        return list_values

    def get_average_rank(self):
        texts = self._texts
        average_ranks = []

        def build_library_for_rank():
            library = self._texts

            frequency = Counter()
            for text in library['text']:
                frequency.update(text.split())
            
            rank_library = dict(sorted(frequency.items(), key=lambda elem: elem[1], reverse=True))                     

            return rank_library

        def rank_for_text(library, text):
            
            rank_library = library
            
            most_frequent_count = rank_library.keys()

            rank_for_lib = {}

            for i, word in enumerate(most_frequent_count, start=1):
                rank_for_lib[word] = i

            frequency_in_text = {}
            all_words = 0
            mapped_words = {}
            average_r = {}
            average_rank = 0

            for word in text.split():
                if not word in stop_words:
                    count = frequency_in_text.get(word, 0)
                    frequency_in_text[word] = count + 1

            frequencies_in_text = dict(sorted(frequency_in_text.items(), key=lambda elem: elem[1], reverse=True))                

            all_words = len(text.split())

            for word in frequencies_in_text.keys():                
                average_rank += int(frequencies_in_text[word] * rank_for_lib[word]) / all_words
            return average_rank

        library = build_library_for_rank()

        for text in texts['text']:
            average_ranks.append(rank_for_text(library, text))

        return average_ranks

    def plot_hist(self, statistic, values):
        dict_for_dataframe = {str(statistic) : values}
        mean_word_lengths = pd.DataFrame(dict_for_dataframe)
        # mean_word_lengths.hist()
        _ = mean_word_lengths.hist(figsize=(14, 9), bins=25) ## или в виде гистограмм
        plt.tight_layout()
        plt.show()

    def get_pos_count(self):
        print("Enter pos count.")
        noun_list = []
        verb_list = []
        adj_list = []
        adv_list = []

        texts = self._texts
        for text in texts['text']:
            noun_count = 0
            verb_count = 0
            adj_count = 0
            adv_count = 0

            total_count = 0
            doc = self.nlp(text)
            for token in doc:
                if token.pos_ == 'NOUN':
                    noun_count += 1
                elif token.pos_ == 'VERB':
                    verb_count += 1
                elif token.pos_ == 'ADJ':
                    adj_count += 1
                elif token.pos_ == 'ADV':
                    adv_count += 1
                total_count += 1
            if total_count == 0:
                total_count == 1
            noun_list.append(noun_count/total_count)
            verb_list.append(verb_count/total_count)
            adj_list.append(adj_count/total_count)
            adv_list.append(adv_count/total_count)

        return noun_list, verb_list, adj_list, adv_list
    
    
    def plot_double_hist(self, statistic_1, values_1, statistic_2, values_2):
        nat = values_1
        art = values_2
        sns.histplot(nat, color='blue', legend='natural')
        sns.histplot(art, color='orange', legend='artificial')
        plt.show()

def main():
    print("Enter main.")

    # text_nat_semi = pd.read_excel('../all_essays_natural_semi.xlsx')
    # text_art_semi = pd.read_excel('../all_essays_artificial_semi.xlsx')
    # text_nat_lemma = pd.read_excel('../all_essays_natural_lemma.xlsx')
    # text_art_lemma = pd.read_excel('../all_essays_artificial_lemma.xlsx')

    text_nat_semi = pd.read_excel('text_nat_semi_w_values_new.xlsx')
    text_art_semi = pd.read_excel('text_art_semi_w_values_new.xlsx')
    text_group_art_semi = TextGroup(text_art_semi)
    text_group_art_semi.plot_double_hist('qmark_count', text_nat_semi["qmark_count"], 'qmark_count', text_art_semi["qmark_count"])
    text_group_art_semi.plot_double_hist('colon_count', text_nat_semi["colon_count"], 'colon_count', text_art_semi["colon_count"])
    text_group_art_semi.plot_double_hist('semicolon_count', text_nat_semi["semicolon_count"], 'semicolon_count', text_art_semi["semicolon_count"])

    # text_nat_lemma = pd.read_excel('../all_essays_natural_lemma.xlsx')
    # text_art_lemma = pd.read_excel('../all_essays_artificial_lemma.xlsx')

    # text_group_nat_lemma = TextGroup(text_nat_lemma)   
    # text_group_art_lemma = TextGroup(text_art_lemma)    

    # values_nat_rank = text_group_nat_lemma.get_average_rank()
    # values_art_rank = text_group_art_lemma.get_average_rank()

    # text_nat_lemma['rank'] = values_nat_rank
    # text_art_lemma['rank'] = values_art_rank

    # text_group_art_lemma.plot_double_hist('text_nat_lemma', values_nat_rank, 'text_art_lemma', values_art_rank)


    # text_group_nat_semi = TextGroup(text_nat_semi)
    # text_group_nat_lemma = TextGroup(text_nat_lemma)    
    
    # values_nat_mean_w_len = text_group_nat_semi.get_mean_word_length()
    # values_nat_stdev_w_len = text_group_nat_semi.get_stdev_word_lengths()
    # values_nat_mean_sent_len = text_group_nat_semi.get_mean_sentence_length()
    # values_nat_stdev_sent_len = text_group_nat_semi.get_stdev_sentence_lengths()
    # values_nat_dots = text_group_nat_semi.get_dots_count()
    # values_nat_commas = text_group_nat_semi.get_commas_count()
    # values_nat_excpts = text_group_nat_semi.get_excpoints_count()
    # values_nat_qmarks = text_group_nat_semi.get_questmarks_count()
    # values_nat_semicolons = text_group_nat_semi.get_semicolons_count()
    # values_nat_colons = text_group_nat_semi.get_colons_count()
    # values_nat_rank = text_group_nat_lemma.get_average_rank()
    # values_nat_homogeneity = text_group_nat_lemma.get_homogeneity()
    # values_nat_nouns, values_nat_verbs, values_nat_adj, values_nat_adv = text_group_nat_lemma.get_pos_count()


    # text_nat_semi['mean_word_len'] = values_nat_mean_w_len
    # text_nat_semi['stdev_word_len'] = values_nat_stdev_w_len
    # text_nat_semi['mean_sent_len'] = values_nat_mean_sent_len
    # text_nat_semi['stdev_sent_len'] = values_nat_stdev_sent_len
    # text_nat_semi['dot_count'] = values_nat_dots
    # text_nat_semi['comma_count'] = values_nat_commas
    # text_nat_semi['excpoint_count'] = values_nat_excpts
    # text_nat_semi['qmark_count'] = values_nat_qmarks
    # text_nat_semi['semicolon_count'] = values_nat_semicolons
    # text_nat_semi['colon_count'] = values_nat_colons
    # text_nat_semi['rank'] = values_nat_rank
    # text_nat_semi['homogeneity'] = values_nat_homogeneity
    # text_nat_semi['noun'] = values_nat_nouns
    # text_nat_semi['verb'] = values_nat_verbs
    # text_nat_semi['adj'] = values_nat_adj
    # text_nat_semi['adv'] = values_nat_adv

    # text_nat_semi.to_excel("text_nat_semi_w_values_new.xlsx")

    # text_group_art_semi = TextGroup(text_art_semi)
    # text_group_art_lemma = TextGroup(text_art_lemma)

    # values_art_mean_w_len = text_group_art_semi.get_mean_word_length()
    # values_art_stdev_w_len = text_group_art_semi.get_stdev_word_lengths()
    # values_art_mean_sent_len = text_group_art_semi.get_mean_sentence_length()
    # values_art_stdev_sent_len = text_group_art_semi.get_stdev_sentence_lengths()
    # values_art_dots = text_group_art_semi.get_dots_count()
    # values_art_commas = text_group_art_semi.get_commas_count()
    # values_art_excpts = text_group_art_semi.get_excpoints_count()
    # values_art_qmarks = text_group_art_semi.get_questmarks_count()
    # values_art_semicolons = text_group_art_semi.get_semicolons_count()
    # values_art_colons = text_group_art_semi.get_colons_count()
    # values_art_rank = text_group_art_lemma.get_average_rank()
    # values_art_homogeneity = text_group_art_lemma.get_homogeneity()
    # values_art_nouns, values_art_verbs, values_art_adj, values_art_adv = text_group_art_lemma.get_pos_count()

    # text_art_semi['mean_word_len'] = values_art_mean_w_len
    # text_art_semi['stdev_word_len'] = values_art_stdev_w_len
    # text_art_semi['mean_sent_len'] = values_art_mean_sent_len
    # text_art_semi['stdev_sent_len'] = values_art_stdev_sent_len
    # text_art_semi['dot_count'] = values_art_dots
    # text_art_semi['comma_count'] = values_art_commas
    # text_art_semi['excpoint_count'] = values_art_excpts
    # text_art_semi['qmark_count'] = values_art_qmarks
    # text_art_semi['semicolon_count'] = values_art_semicolons
    # text_art_semi['colon_count'] = values_art_colons
    # text_art_semi['rank'] = values_art_rank
    # text_art_semi['homogeneity'] = values_art_homogeneity
    # text_art_semi['noun'] = values_art_nouns
    # text_art_semi['verb'] = values_art_verbs
    # text_art_semi['adj'] = values_art_adj
    # text_art_semi['adv'] = values_art_adv

    # text_art_semi.to_excel("text_art_semi_w_values_new.xlsx")

    # datasets = [text_nat_semi, text_art_semi]
    # dataset_for_training_art_nat_all_values = pd.concat(datasets)
    # dataset_for_training_art_nat_all_values.to_excel("dataset_for_training_art_nat_all_values_new.xlsx")
    
    # text_group_nat_semi.plot_double_hist('mean_word_length', values_nat, 'mean_word_length', values_art)
      
if __name__ == "__main__":
    main()

