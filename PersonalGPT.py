from openai import OpenAI
import openai
import streamlit as st
import tiktoken
import json
import datetime
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import re

def get_url(input):
    urls = search(input, num_results=2)
    urls = list(set(url for url in urls))
    return urls

def scrape_website(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        page_content = response.content
        soup = BeautifulSoup(page_content, 'html.parser')
        paragraphs = soup.find_all('p')
        scraped_data = [p.get_text() for p in paragraphs]
        formatted_data = '\n'.join(scraped_data)
        return [url, formatted_data]
    
    else:
        return [url]

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, but keep letters and numbers
    text = re.sub(r'[^a-z0-9\s,.\n]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def scrape_gpt(user_input):
    encoding = tiktoken.get_encoding('cl100k_base')
    urls = get_url(user_input)
    
    full_web_content = ""
    for i, url in enumerate(urls):
        try:
            web_content = scrape_website(url)[1]
            clean_web_content = clean_text(web_content)
            encode = encoding.encode(clean_web_content)
            clean_web_content = encoding.decode(encode[:500])

            full_web_content += f'WEB {i+1}: ' + clean_web_content + f'\nURL: {url}\n\n'
        except:
            continue
    
    
    return full_web_content

tools = [
    {
        "type": "function",
        "function": {
            "name": "scrape_gpt",
            "description": "Use this function to search answers from the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The search queries that you want to browse. Make sure to use optimized search queries.",
                    },
                },
                "required": ["user_input"],
            },
        }
    }
]

def get_completion(use_tools=False, stream=True):
    if not use_tools:
        completion = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        temperature=temperature,
                        stream=stream,
                    )
    else:
        
        completion = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        temperature=temperature,
                        tools=tools,
                        stream=stream,
                    )
    return completion

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def reset_state():
    for key in st.session_state.keys():
        if key != 'api_key':
            del st.session_state[key]

def create_chat_title():
    system_title_content = """You are a chatbot to create a chat title from user input, for example, if user input "what is clustering", you give a title "Clustering Explanation". Make sure to use a proper word for the title. Keep it simple and don't make the title too long. If user input an opening statement/question, like "hi", "how are you", etc, answer with "Personal AI Assistant"."""
    system_title_prompt = {"role": 'system', "content": system_title_content}

    title_prompt = st.session_state.messages[-2]
    title_messages = [system_title_prompt, {'role': 'user', 'content': title_prompt['content']}]

    title = ''
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=title_messages,
        temperature=1,
        stream=True
    )
    for resp in response:
        title += (resp.choices[0].delta.content or "")
        title_placeholder.markdown(f"<h5 style='text-align: center;'>{title + '|'}</h5>", unsafe_allow_html=True)
    
    title_placeholder.markdown(f"<h5 style='text-align: center;'>{title}</h5>", unsafe_allow_html=True)
    st.session_state.title.append(title)

# Define the options for the dropdown menu
model_options = ["GPT-3.5", "GPT-4"]
audio_options = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

st.markdown("<h1 style='text-align: center;'>PersonalGPT</h1>", unsafe_allow_html=True)

if 'api_key' in st.session_state:
    with st.sidebar:
        st.success('API key provided!', icon='✅')
else:
    api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (api_key.startswith('sk-') and len(api_key)==51):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
        st.session_state.api_key = {'OPENAI_API_KEY': api_key}

with st.sidebar:
    if st.button('New Chat'):
        reset_state()

    select_model = st.toggle('GPT-4')
    if not select_model:
        model = 'gpt-3.5-turbo'
        st.write("You are using **GPT-3.5**")
    else:
        model = 'gpt-4-1106-preview'
        st.write("You are using **GPT-4**")
        
    st.session_state["openai_model"] = model

    temperature = st.slider('Model Temperature', 0.0, 2.0, 1.0, 0.01, label_visibility='collapsed')
    browsing = st.toggle('Browsing')

    selected_audio = st.selectbox("TTS sound:", audio_options)
    st.markdown('*Write "TTS: (text)" to use text-to-speech*')

try:
    openai.api_key = st.session_state.api_key['OPENAI_API_KEY']
    client = OpenAI(api_key=st.session_state.api_key['OPENAI_API_KEY'])

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        system_prompt = """You are a helpful and respectful AI assistant"""
        st.session_state.messages.append({"role": "system", "content": system_prompt, 'audio': None})

    if len(st.session_state.messages) < 3:
        title_placeholder = st.empty()
    else:
        if type(st.session_state.title) == list:
            chat_title = st.session_state.title[0]
        else:
            chat_title = st.session_state.title

        st.markdown(f"<h5 style='text-align: center;'>{chat_title}</h5>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message['role'] != 'system' and message['content'][:14] != '```WEB CONTEXT':
            if message['audio'] == None:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.audio(message["audio"], format='audio/mp3')

    prompt = st.chat_input("Say something...")
    if prompt:
        if browsing and len(st.session_state.messages) == 1:
            st.session_state.messages.append({'role': 'system', 'content': 'You have ability to browse the internet using function. If you are unsure about your answer, or the user ask for real time factual data, use your ability to search the internet. Provide a good and optimized search query when searching the internet. If the user have some texts inside triple backticks starting with WEB CONTEXT  (```WEB CONTEXT ...```) after their question, that string inside the triple backticks is the answer from the internet, so you don\'t need to call the browsing function. All of the answers from the internet is updated, so don\'t assume your knowledge is true because it isn\'t updated. If user ask you for follow up question that are cannot be answered from the previous web searching, then search it again through the internet.', 'audio': None})

        st.session_state.messages.append({"role": "user", "content": prompt, 'audio': None})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if prompt[:4] == 'TTS:':
            with st.chat_message('assistant'):
                message_placeholder = st.empty()
                response = openai.audio.speech.create(
                    model="tts-1",
                    voice=selected_audio,
                    input=prompt[4:]
                )

                audio = b""
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    audio += chunk

                message_placeholder.audio(audio, format="audio/mp3")
            st.session_state.messages.append({"role": "assistant", "content": prompt[4:], 'audio': audio})

        else:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                if not browsing:
                    completion = get_completion()

                    for response in completion:
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.markdown(full_response + "▌")
                else:
                    completion = get_completion(use_tools=True)

                    search_query = ""
                    for response in completion:
                        if response.choices[0].delta.content != None:
                            full_response += (response.choices[0].delta.content or "")
                            message_placeholder.markdown(full_response + "▌")
                        elif response.choices[0].delta.content == None and full_response == "":
                            word_query = response.choices[0].delta.tool_calls
                            if word_query:
                                search_query += word_query[0].function.arguments
                                continue
                    
                    if len(search_query) != 0:
                        user_input = json.loads(search_query)['user_input']
                        message_placeholder.markdown(f'*Browsing Google... ({user_input})*')
                        web_content = scrape_gpt(user_input)
                        message_placeholder.markdown(f'*Finish Browsing. Total web content tokens: {num_tokens_from_string(web_content, "cl100k_base")}*')

                        st.session_state.messages.append({'role': 'user', 'content': f"```WEB CONTEXT {web_content}```", 'audio': None})

                        completion_browse = get_completion(use_tools=False)
                        for response_browse in completion_browse:
                            full_response += (response_browse.choices[0].delta.content or "")
                            message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response, 'audio': None})


    if 2 < len(st.session_state.messages) <= 5:
        try:
            if type(st.session_state.title) == list:
                st.session_state.title = st.session_state.title[0]
            else:
                pass
        except:
            st.session_state.title = []
            create_chat_title()

    full_message = ""
    for message in st.session_state.messages:
        full_message += message['content'] + " "

    num_tokens = num_tokens_from_string(full_message, "cl100k_base")

    if (not browsing and num_tokens > 9) or (browsing and num_tokens > 159):
        with st.sidebar: 
            st.write(f'Total tokens: {num_tokens}')
            
            if st.button('Save Chat'):
                json_temp = {}
                for i, msg in enumerate(st.session_state.messages):
                    json_temp[i] = {'role': msg['role'], 'content': msg['content'], 'type': 'text' if msg['audio'] is None else 'audio'}

                now = datetime.datetime.now()
                date_now = now.strftime("%Y%m%d-%H%M%S.%f")
                filename = f"history/chat_{date_now}"
                with open(f'{filename}.json', 'w') as f:
                    json.dump(json_temp, f)

except:
    pass
