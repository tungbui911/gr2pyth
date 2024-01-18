import streamlit as st
import requests, time, math, os, json
from util.util import generate_mdd_for_app, get_phoneme_ipa_form


main_menu = st.button("Main menu")

if main_menu:
    st.switch_page("choice.py")

url = "http://127.0.0.1:2103"
current_folder = os.path.dirname(os.path.realpath(__file__))
img_folder = os.path.join(current_folder, 'img')
audio_folder = os.path.join(current_folder, 'audio')
if not os.path.exists(audio_folder):
    os.mkdir(audio_folder) 

user_input = st.text_input("Enter some text:")

submit_input = st.button("Submit")

if submit_input:
    st.session_state["input"] = user_input
    st.switch_page("pages/record.py")


