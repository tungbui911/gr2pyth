import streamlit as st
import streamlit_book as stb


go_back = st.button("Go back")

if go_back:
    st.switch_page("choice.py")

st.title("Fill in the Blank")

stb.single_choice("The man _____ home when his car broke down.", 
                  ["drives", "is driving", "was driving", "has driven"], 
                  2)

stb.single_choice("She promised _____ to my birthday party, but she didn't.", 
                  ["to come", "coming", "come", "to coming"], 
                  0)

stb.single_choice("Her parents are working on the farm, _____ they?", 
                  ["aren't", "are", "don't", "do"], 
                  0)

stb.single_choice("Mary lives _____ the countryside.", 
                  ["in", "to", "about", "with"], 
                  0)

stb.single_choice("The more talkative she was, _____ uncomfortable we felt.", 
                  ["the more", "the more than", "more than", "the better than"], 
                  0)

stb.single_choice("He bought this _____ ring.", 
                  ["nice small Japanese", "small nice Japanese", "Japanese nice small", "nice Japanese small"], 
                  0)

stb.single_choice("She failed the driving test _____ she practiced a lot.", 
                  ["in spite of", "because of", "although", "despite"], 
                  2)

stb.single_choice("A new supermarket _____ last week.", 
                  ["opens", "open", "have opened", "was opened"], 
                  3)
