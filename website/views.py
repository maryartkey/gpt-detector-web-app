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

with open('model_nohomo.pkl', 'rb') as f:
    regr = pickle.load(f)

class TextGroup:
    def __init__(self, texts):
        self._texts = texts  
        self.nlp = spacy.load("en_core_web_sm")

    def get_mean_word_length(self):
        texts = self._texts
        mean_word_length = 0
        number_of_words = 0
        words_length = 0
        for word in texts.split():
            words_length += len(word)
            number_of_words += 1
        if number_of_words != 0:
            mean_word_length = (words_length/number_of_words)
        else:
            mean_word_length = 1
        return mean_word_length
    
    def get_stdev_word_lengths(self):
        texts = self._texts
        mean_word_length = 0
        number_of_words = 0
        words_length = 0
        for word in texts.split():
            words_length += len(word)
            number_of_words += 1
        if number_of_words != 0:
            mean_word_length = (words_length/number_of_words)
        else:
            mean_word_length = 1
        return mean_word_length
        

    def get_mean_sentence_length(self):      
        sentences_length = 0
        number_of_sentences = 0
        texts = self._texts
        for sentence in sent_tokenize(texts):                  
            sentences_length += len(sentence.split())
            number_of_sentences += 1
        if number_of_sentences == 0:
            number_of_sentences = 1
        if sentences_length == 0:
            sentences_length = 1
        mean_sentence_lengths = (sentences_length/number_of_sentences)
        return mean_sentence_lengths
    
    def get_stdev_sentence_lengths(self):
        texts = self._texts
        lengths = list(map(len, sent_tokenize(texts)))
        values = 0
        if len(lengths) > 1:
            values = st.stdev(lengths)
        return values
    
    def get_dots_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == '.':
                marks_in_text += 1
            elif word == ',' or word == '!' or word == '?' or word == ':' or word == ';':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values

    def get_commas_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == ',':
                marks_in_text += 1
            elif word == '.' or word == '!' or word == '?' or word == ':' or word == ';':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values
    
    def get_excpoints_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == '!':
                marks_in_text += 1
            elif word == ',' or word == '.' or word == '?' or word == ':' or word == ';':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values
    
    def get_questmarks_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == '?':
                marks_in_text += 1
            elif word == ',' or word == '.' or word == '!' or word == ':' or word == ';':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values
    
    def get_colons_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == ':':
                marks_in_text += 1
            elif word == ',' or word == '.' or word == '!' or word == '?' or word == ';':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values

    def get_semicolons_count(self):
        texts = self._texts
        marks_in_text = 0
        other_marks_in_text = 0
        for word in texts.split():
            if word == ';':
                marks_in_text += 1
            elif word == ',' or word == '.' or word == '!' or word == '?' or word == ':':
                other_marks_in_text += 1
        total_marks = marks_in_text + other_marks_in_text
        if total_marks == 0:
            total_marks = 1
        list_values = (marks_in_text/total_marks)
        return list_values   
   

    def get_pos_count(self):
        text = self._texts
        nouns = 0
        verbs = 0
        adjs = 0
        advs = 0
        total_count = 0
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ == 'NOUN':
                nouns += 1
            elif token.pos_ == 'VERB':
                verbs += 1
            elif token.pos_ == 'ADJ':
                adjs += 1
            elif token.pos_ == 'ADV':
                advs += 1
            
            total_count += 1
            if total_count == 0:
                total_count == 1
            nouns = (nouns/total_count)
            verbs = (verbs/total_count)
            adjs = (adjs/total_count)
            advs = (advs/total_count)
        return nouns, verbs, adjs, advs


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_data = request.form.get('textarea')
        print(len(text_data))
        # new_text = Text(data=text_data, user_id=current_user.id)  #providing the schema for the note 
        # db.session.add(new_text) #adding the note to the database 
        # db.session.commit()
        text = text_data
        data = TextGroup(text)        

        values_mean_w_len = data.get_mean_word_length()
        values_stdev_w_len = data.get_stdev_word_lengths()
        values_mean_sent_len = data.get_mean_sentence_length()
        values_stdev_sent_len = data.get_stdev_sentence_lengths()
        values_dots = data.get_dots_count()
        values_commas = data.get_commas_count()
        values_excpts = data.get_excpoints_count()
        values_qmarks = data.get_questmarks_count()
        values_semicolons = data.get_semicolons_count()
        values_colons = data.get_colons_count()
        values_nouns, values_verbs, values_adj, values_adv = data.get_pos_count()

        values_dict = {'mean_word_len':  values_mean_w_len, 'stdev_word_len' : values_stdev_w_len, 'mean_sent_len': values_mean_sent_len, 'stdev_sent_len' : values_stdev_sent_len,
       'dot_count': values_dots, 'comma_count' : values_commas, 'excpoint_count':values_excpts, 'qmark_count':values_qmarks,
       'semicolon_count':values_semicolons, 'colon_count':values_colons, 'noun':values_nouns, 'verb':values_verbs, 'adj':values_adj, 'adv':values_adv}
        input = pd.DataFrame(values_dict, index=[0])
        print(values_dict)
        prediction = regr.predict(input[0:1])
        if prediction == ['human']:
            output_text = 'Текст, скорее всего, написан человеком.'
            generated = 0
        else:
            output_text = 'Похоже, что это сгенерированный текст.'
            generated = 1

        print(output_text)
        print(prediction)


        return render_template("text.html", user=current_user, text_data=text_data, output_text=output_text, generated=generated)
    
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