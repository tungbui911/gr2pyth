import streamlit as st
import requests, time, math, os, json
from util.util import generate_mdd_for_app, get_phoneme_ipa_form

url = "http://127.0.0.1:2103"
current_folder = os.path.dirname(os.path.realpath(__file__))
img_folder = os.path.join(current_folder, 'img')
audio_folder = os.path.join(current_folder, 'audio')
if not os.path.exists(audio_folder):
    os.mkdir(audio_folder) 

input = st.session_state['input']

st.markdown(f"""<p style="font-size: 21px;">{input}</p>""", unsafe_allow_html=True)
result = requests.post(url=f'{url}/phonemes', data={'text':input}).text
result = json.loads(result)
if not result:
    result = get_phoneme_ipa_form(input)

st.session_state['phonetics'] = result.get('phonetics')
text = "/" + result.get('phonetics') + "/"

st.markdown(f"""<p style="font-size: 21px;">{text}</p>""", unsafe_allow_html=True)


from st_audiorec import st_audiorec
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    result = requests.post(url=f'{url}/predict', data={'text':input}, files={'audio': wav_audio_data}).text
    result = json.loads(result)

#    if not result:
#        result = generate_mdd_for_app(log_proba, canonical, word_phoneme_in)

    st.session_state['correct_rate'] = result.get('correct_rate')
    st.session_state['phoneme_result'] = result.get('phoneme_result')

    result = st.button("Result")
    if result: 
        st.session_state['audio'] = wav_audio_data
        st.switch_page('pages/result.py')

go_back = st.button("Go back")
main_menu = st.button("Main menu")

if go_back:
    st.switch_page("pages/input.py")

if main_menu:
    st.switch_page("choice.py")