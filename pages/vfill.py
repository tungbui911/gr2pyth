import streamlit as st
import streamlit_book as stb

go_back = st.button("Go back")

if go_back:
    st.switch_page("pages/vocab.py")

st.title("Fill in the Blank")

stb.single_choice("Nam is trying to break the _____ of staying up too late.", 
                  ["sound", "habit", "option", "race"], 
                  1)

stb.single_choice("The journalist is talking about a new _____ published in the newspaper next week.", 
                  ["editor", "documentary", "cartoon", "article"], 
                  3)

stb.single_choice("Binh is 18 years old. Linh is 16 years old. Binh is _____ than Linh.", 
                  ["taller", "older", "shorter", "younger"], 
                  1)

stb.single_choice("My uncle _____ to save up for a new house.", 
                  ["arrives", "connects", "follows", "plans"], 
                  3)

