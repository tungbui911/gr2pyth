import streamlit as st
import streamlit_book as stb

go_back = st.button("Go back")

if go_back:
    st.switch_page("pages/vocab.py")

st.title("Find the best synonym")

stb.single_choice("meaningful", 
                  ["big", "useful", "bad", "smart"], 
                  1)

stb.single_choice("satisfy", 
                  ["warn", "accept", "please", "sit"], 
                  2)

stb.single_choice("tolerate", 
                  ["spoil", "appreciate", "ignore", "endure"], 
                  3)

stb.single_choice("profession", 
                  ["truth", "decision", "job", "play"], 
                  2)

