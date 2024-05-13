from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Text
from . import db
import json
import numpy as np
import pandas as pd
import spacy
import statistics as st
from nltk.tokenize import sent_tokenize
from collections import Counter
import pickle
import joblib

with open('model.pkl', 'rb') as f:
    clf2 = joblib.load(f)

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

views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_data = request.form.get('textarea')
        print(len(text_data))
        new_text = Text(data=text_data, user_id=current_user.id)  #providing the schema for the note 
        db.session.add(new_text) #adding the note to the database 
        db.session.commit()
        avg = average_word_len(text_data)
        tmp = TextGroup(text_data)

        return render_template("text.html", user=current_user, text_data=text_data, tmp=tmp)
    
    return render_template("home.html", user=current_user)

def prediction_main(text):
    text_list = text.split()
    count = 0
    for word in text_list:
        count += len(word)
    return count/len(text)

@views.route('/faq')
def faq():
    return render_template("faq.html")

@views.route('/about')
def about():
    return render_template("about.html")

@views.route('/research')
def research():
    return render_template("research.html")

@views.route('/api')
def api():
    return render_template("api.html")