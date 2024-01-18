import streamlit as st
import requests, time, math, os, json
from util.util import generate_mdd_for_app, get_phoneme_ipa_form
from util.map_color import map_color

import ast

url = "http://127.0.0.1:2103"
current_folder = os.path.dirname(os.path.realpath(__file__))
img_folder = os.path.join(current_folder, 'img')
audio_folder = os.path.join(current_folder, 'audio')
if not os.path.exists(audio_folder):
    os.mkdir(audio_folder) 

input = st.session_state['input']
phonetics = st.session_state['phonetics']
phoneme_result = st.session_state['phoneme_result']
correct_rate = st.session_state['correct_rate']
correct_rate = float(correct_rate) * 100

st.markdown(f"""<p style="font-size: 21px;">{input}</p>""", unsafe_allow_html=True)
text = "/" + phonetics + "/"

result_list = ast.literal_eval(phoneme_result)
color = map_color(result_list)
text = '<span style="font-size: 21px;">/</span>' + "".join(f'<span style="color: {c}; font-size: 21px;">{gt}</span>' for gt, pred, x, c in color) + '<span style="font-size: 21px;">/</span>'
st.markdown(text, unsafe_allow_html=True)

if correct_rate < 20:
    text = '<p style="color:#CB4335; font-size: 24px;">Poor</p>'
    st.markdown(text, unsafe_allow_html=True)
elif correct_rate < 40:
    text = '<p style="color:#CB4335; font-size: 24px;">Try Better</p>'
    st.markdown(text, unsafe_allow_html=True)
elif correct_rate < 60:
    text = '<p style="color:#D4AC0D; font-size: 24px;">Fair</p>'
    st.markdown(text, unsafe_allow_html=True)
elif correct_rate < 80:
    text = '<p style="color:#D4AC0D; font-size: 24px;">Good</p>'
    st.markdown(text, unsafe_allow_html=True)
elif correct_rate < 90:
    text = '<p style="color:#28B463; font-size: 24px;">Very Good</p>'
    st.markdown(text, unsafe_allow_html=True)
elif correct_rate < 100:
    text = '<p style="color:#28B463; font-size: 24px;">Excellent!</p>'
    st.markdown(text, unsafe_allow_html=True)
else:
    text = '<p style="color:#28B463; font-size: 24px;">Perfect!</p>'
    st.markdown(text, unsafe_allow_html=True) 

audio = st.session_state['audio']
st.audio(audio, format='audio/wav')

text = 'Your speaking was ' + str(correct_rate) + '% correct'
st.markdown(f"""<p style="font-size: 21px;">{text}</p>""", unsafe_allow_html=True)


try_again = st.button("Try again")
try_other_word = st.button("Try other words")
main_menu = st.button("Main menu")

if try_again:
    st.switch_page('pages/record.py')

if try_other_word:
    st.switch_page('pages/input.py')

if main_menu:
    st.switch_page('choice.py')