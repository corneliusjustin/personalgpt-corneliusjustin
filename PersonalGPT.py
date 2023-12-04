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

def get_completion(use_browsing=False, stream=True):
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
    
    tool_choice = 'auto' if use_browsing else 'none'
    
    completion = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream
    )
    
    return completion

def get_stream_full_response(completion):

    available_functions = {
        'scrape_gpt': scrape_gpt
    }

    full_response = ""
    tool_calls_args = []
    tool_calls_arg = ""
    function_names = []
    tool_call_ids = []
    idx_temp = 0

    for response in completion:
        content = response.choices[0].delta.content
        finish = response.choices[0].finish_reason
        if content != None or full_response != "":
            full_response += (response.choices[0].delta.content or "")
            
            if finish:
                assistant_response = {'role': 'assistant', 'content': full_response, 'audio': None}
                st.session_state.messages.append(assistant_response)
                
                return full_response
            else:
                message_placeholder.markdown(full_response + "▌")
        
        elif content == None and full_response == "":
            tool_calls = response.choices[0].delta.tool_calls
            
            if tool_calls:
                function_name = tool_calls[0].function.name
                tool_call_id = tool_calls[0].id
                if function_name:
                    function_names.append(function_name)
                    tool_call_ids.append(tool_call_id)

                tool_index = tool_calls[0].index
                if tool_index > idx_temp:
                    tool_calls_args.append(tool_calls_arg)
                    tool_calls_arg = ""
                    idx_temp += 1

                tool_calls_arg += tool_calls[0].function.arguments
            
            elif finish:
                tool_calls_args.append(tool_calls_arg)
    
    if len(tool_calls_args) != 0:
        for i, function_name in enumerate(function_names):
            function_to_call = available_functions[function_name]
            function_arg = json.loads(tool_calls_args[i])

            with st.spinner(f'*Browsing Google... ({list(function_arg.values())[0]})*'):
                function_response = function_to_call(**function_arg)
            
            message_placeholder.markdown(f'*Finish retreiving knowledge. Total context tokens: {num_tokens_from_string(function_response, "cl100k_base")}*')

            st.session_state.messages.append(
                {
                    'role': 'system',
                    "content": f"QUERY:\n{function_arg}\n\nTOOL RESPONSE:\n{function_response}",
                    'audio': None
                }
            )

        completion_func = get_completion(use_browsing=True)

        return get_stream_full_response(completion_func)

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
    system_title_content = """You are an expert to create a chat title from user input for chatbot app, for example, if user input "what is clustering", you give a title "Clustering Explanation". Another example, user input "tell me about latest AI news", you give "Latest AI News Update" or something like that. Make sure to use a proper word for the title. Keep it simple and don't make the title too long. If user input an opening statement/question, like "hi", "how are you", etc, answer with "Personal AI Assistant". You are not a question answer bot, don't answer user's question, instead create a title from user question."""
    system_title_prompt = {"role": 'system', "content": system_title_content}

    for messages in st.session_state.messages[:5]:
        if messages['role'] == 'user':
            title_prompt = messages['content']
            break
            
    title_messages = [system_title_prompt, {'role': 'user', 'content': title_prompt}]

    title = ''
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=title_messages,
        temperature=0.3,
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
        model = 'gpt-3.5-turbo-1106'
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
        
        system_prompt = """You are a helpful and respectful AI assistant. You don't have ability to browse the internet, unless there's a system prompt after this telling you that you are able"""
        st.session_state.messages.append({"role": "system", "content": system_prompt, 'audio': None})
    
    if 'title' not in st.session_state:
        title_placeholder = st.empty()
    else:
        st.markdown(f"<h5 style='text-align: center;'>{st.session_state.title[0]}</h5>", unsafe_allow_html=True)

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
            st.session_state.messages.append({'role': 'system', 'content': 'You have ability to browse the internet using function. If you are unsure about your answer, or the user ask for real time factual data, use your ability to search the internet. Provide a good and optimized search query when searching the internet. If system with "QUERY" and "TOOL RESPONSE", that\'s the response from the tool that you call, so you don\'t need to call the browsing function. All of the answers from the internet is updated, so don\'t assume your knowledge is true because it isn\'t updated. If user ask you for follow up question that are cannot be answered from the previous web searching, then search it again through the internet.', 'audio': None})

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

                if not browsing:
                    completion = get_completion()

                else:
                    completion = get_completion(use_browsing=True)

                full_response = get_stream_full_response(completion)
                message_placeholder.markdown(full_response)


    if 2 < len(st.session_state.messages) <= 5:
        if 'title' not in st.session_state: 
            st.session_state.title = []
            create_chat_title()

    full_message = ""
    for message in st.session_state.messages:
        full_message += message['content'] + " "

    num_tokens = num_tokens_from_string(full_message, "cl100k_base")

    if (not browsing and num_tokens > 34) or (browsing and num_tokens > 184):
        with st.sidebar: 
            st.write(f'Total tokens: {num_tokens}')
            
            # if st.button('Save Chat'):
            #     json_temp = {}
            #     for i, msg in enumerate(st.session_state.messages):
            #         json_temp[i] = {'role': msg['role'], 'content': msg['content'], 'type': 'text' if msg['audio'] is None else 'audio'}

            #     now = datetime.datetime.now()
            #     date_now = now.strftime("%Y%m%d-%H%M%S.%f")
            #     filename = f"history/chat_{date_now}"
            #     with open(f'{filename}.json', 'w') as f:
            #         json.dump(json_temp, f)

except:
    pass