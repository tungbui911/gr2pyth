import streamlit as st

text = 'Welcome to the English Learning Website! Which course would you like to choose?'
st.markdown(f"""<p style="font-size: 36px;">{text}</p>""", unsafe_allow_html=True)

vocab_input = st.button("Vocabulary Test")
grammar_input = st.button("Grammar Test")
reading_input = st.button("Reading Test")
speak_input = st.button("Speaking Test")

if vocab_input:
    st.switch_page("pages/vocab.py")
if grammar_input:
    st.switch_page("pages/grammar.py")
if reading_input:
    st.switch_page("pages/reading.py")
if speak_input:
    st.switch_page("pages/input.py")