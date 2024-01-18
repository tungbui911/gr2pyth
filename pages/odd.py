import streamlit as st
import streamlit_book as stb


go_back = st.button("Go back")

if go_back:
    st.switch_page("pages/vocab.py")

#st.title("Odd one out")

stb.single_choice("Odd one out", 
                  ["dog", "cat", "tree", "pig"], 
                  2)

stb.single_choice("Odd one out", 
                  ["soccer", "volleyball", "basketball", "housework"], 
                  3)

stb.single_choice("Odd one out", 
                  ["my", "she", "her", "his"], 
                  1)

stb.single_choice("Odd one out", 
                  ["math", "literature", "history", "book"], 
                  3)
