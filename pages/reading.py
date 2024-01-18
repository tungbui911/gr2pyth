import streamlit as st
import streamlit_book as stb


go_back = st.button("Go back")

if go_back:
    st.switch_page("choice.py")

prg1 = "Device-centred communication has become almost universal over the past twenty years. More than three quarters of people in the world now own a mobile device, and more than half communicate via social networking."
st.markdown(f"""<p style="font-size: 21px;">{prg1}</p>""", unsafe_allow_html=True)
prg2 = "It is now hard to imagine a world without mobile devices consisting of such things as mobile phones, laptops and tablets. They allow us to stay in touch with a large network of friends, no matter where they are. But many experts say that communicating with a device is nothing like talking with someone in person. “Body language, eye contact and tone of voice can tell us so much,” psychologist Mary Peters says. “And none of those exist on a device. Even video chat removes many subtle clues.”"
st.markdown(f"""<p style="font-size: 21px;">{prg2}</p>""", unsafe_allow_html=True)
prg3 = "We don’t know to what extent these technologies will permanently change the way people interact. People will always want to meet up with others in small and large groups. Indeed, it is fair to say that social media makes it easier than ever before for people to organise social events. However, there is still a danger that device-centred communication may have a negative long-term impact on the way people interact with each other on a day-to-day basis."
st.markdown(f"""<p style="font-size: 21px;">{prg3}</p>""", unsafe_allow_html=True)
prg4 = "We must not, therefore, lose sight of the need to focus on the actual people around us, and remember thatthey deserve our real – not virtual – attention. The idea of a culture where people always have a screen between them sounds a bit funny, because deep understanding comes when we see the reactions on other people’s faces."
st.markdown(f"""<p style="font-size: 21px;">{prg4}</p>""", unsafe_allow_html=True)

stb.single_choice("The passage is mainly about ______.", 
                  ["the development of device-centred communication", "the impact of device-centred communication", "the definition of device-centred communication", "the misunderstanding of device-centred communication"], 
                  1)

stb.single_choice("In paragraph 2, in her statement about the advantages of communicating in person, Mary Peters mentioned all of the following EXCEPT ______.", 
                  ["body language", "eye contact", "handshake", "tone of voice"], 
                  2)

stb.single_choice("The word 'meet up' in paragraph 3 is closest in meaning to ______.", 
                  ["come down", "get together", "get away", "come away"], 
                  1)

stb.single_choice("According to paragraph 4, deep understanding appears when ______.", 
                  ["we communicate through social networking", "we interact with modern technology", "we care about our virtual friends", "we see the reactions on the faces of other people"], 
                  3)
