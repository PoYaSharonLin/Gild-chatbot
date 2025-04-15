import streamlit as st
from openai import OpenAI
import time
import re

placeholderstr = "Please input your command"
user_name = "On-Boarding Mentor"
user_image = "https://www.flaticon.com/free-sticker/job-seeker_9263242?term=hr&page=1&position=72&origin=search&related_id=9263242"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def generate_response(prompt):
        # Normalize prompt for easier matching (lowercase, strip whitespace)
        prompt = prompt.strip().lower()
        
        # Patterns for negative self-statements
        negative_pattern = r'\b(i(\'?m| am| feel| think i(\'?)?m)?\s*(so\s+)?(stupid|ugly|dumb|idiot|worthless|loser|useless))\b'
        
        # Patterns for onboarding-related queries
        help_pattern = r'\b(how|what|help|start|use|do|guide|onboard)\b'
        question_pattern = r'\b(why|when|where|who|what)\b'
        
        # Handle negative self-statements
        if re.search(negative_pattern, prompt, re.IGNORECASE):
            return "I'm sorry you feel that way! You're here to learn and grow, and I'm here to help you every step of the way. Want to explore how this app can support you?"
        
        # Handle onboarding-related questions or requests
        elif re.search(help_pattern, prompt, re.IGNORECASE) or re.search(question_pattern, prompt, re.IGNORECASE):
            return "I'd be happy to help! Could you share a bit more about what you're curious about? For example, are you wondering how to set up your profile or explore app features?"
        
        # Default response for generic input
        else:
            return "Thanks for sharing! I'm here to guide you through onboarding. Try asking something like 'How do I start?' or tell me what you're thinking about!"

    # Chat function section (timing included inside function)
    def chat(prompt: str):
        st_c_chat.chat_message("user",avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        # response = f"You type: {prompt}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()
