import streamlit as st

go_back = st.button("Go back")
odd = st.button("Odd one out")
synonym = st.button("Find the synonym")
vfill = st.button("Fill in the blank")

if go_back:
    st.switch_page("choice.py")
if odd:
    st.switch_page("pages/odd.py")
if synonym:
    st.switch_page("pages/synonym.py")
if vfill:
    st.switch_page("pages/vfill.py")
