from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Text
from . import db
import json

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
        return render_template("text.html", user=current_user, text_data=text_data, avg=avg)
    
    return render_template("home.html", user=current_user)

def average_word_len(text):
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