import random
import time
import io

import streamlit as st
import json
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import numpy as np
max_questions = 5

af3_model_id = "nvidia/audio-flamingo-3-hf"
target_device = "cuda:0" 

@st.cache_resource
def load_model():
    print("Loading model...")
    try:
        processor = AutoProcessor.from_pretrained(af3_model_id)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(af3_model_id, device_map=target_device)
        print(f"Model loaded successfully to {target_device}.")
        return processor, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

af3_processor, af3_model = load_model()

def infer_af3(question, options, audio_file):
    if af3_model is None or af3_processor is None:
        return "Model not loaded properly."

    options_text = "\n".join([f"- {opt}" for opt in options])
    prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        "Instruction: You must answer strictly by copying the exact text of the correct option. "
        "Do NOT answer with letters like A, B, C, or D. Output absolutely nothing else but the exact option text."
    )

    conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": audio_file},
        ],
    }
    ]
    inputs = af3_processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(af3_model.device)

    outputs = af3_model.generate(**inputs, max_new_tokens=100)
    decoded_outputs = af3_processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded_outputs[0]


@st.cache_data
def load_json_dataset(max_questions):
    print("Loading local data.json...")
    try:
        with open("audio.json", "r") as f:
            all_data = json.load(f)
        
        sampled_data = random.sample(all_data, min(max_questions, len(all_data)))
        return sampled_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        st.error(f"Failed to load dataset: {e}")
        return []


data = load_json_dataset(max_questions)


col1, col2 = st.columns([3, 1])
with col1:
    st.title("ALM vs Human", anchor=False)
with col2:
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "Audio Flamingo 3"
    
    selected_model = st.selectbox(
        "Select local LLM:",
        ["Audio Flamingo 3", "Qwen-Audio", "SALMONN"],
        index=["Audio Flamingo 3", "Qwen-Audio", "SALMONN"].index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model

st.divider()


def reset():
    st.session_state.page = 0
    st.session_state.correct = 0
    st.session_state.llm_correct = 0


if "page" not in st.session_state:
    reset()


def set_page(name):

    st.session_state.page = name


def next_question(q_name, selected, answer):
    if selected == answer:
        st.session_state.correct += 1

    with st.spinner(f"Waiting for {st.session_state.selected_model}..."):
        try:
            current_data = data[q_name - 1]
            instruction_text = current_data.get("question", current_data.get("instruction", "Listen to the audio and choose the correct option."))
            choices = current_data.get("choices", [])
            
            audio_data = current_data.get("audio_id", current_data.get("context")) 
            audio_file = audio_data.get("path") if isinstance(audio_data, dict) else audio_data

            llm_response = infer_af3(instruction_text, choices, audio_file)
            print(f"AF3 Response: {llm_response}")
            
            if answer.lower() in llm_response.lower():
                st.session_state.llm_correct += 1
        except Exception as e:
            print(f"Error during AF3 inference: {e}")
            st.error(f"Error during inference: {e}")
            
    st.session_state.page = q_name + 1


if st.session_state.page == 0:

    st.space("xlarge")

    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        st.subheader("Start the game", text_alignment="center")
        st.button("Start game", on_click=set_page(1), type="primary")

elif (st.session_state.page >= 1) and (st.session_state.page <= max_questions):

    with st.container(horizontal_alignment="center", vertical_alignment="center"):

        current_data = data[st.session_state.page - 1]
        
        instruction_text = current_data.get("question", current_data.get("instruction", "Listen to the audio and choose the correct option."))
        choices = current_data.get("choices", [])
        answer = current_data.get("answer", "")
        
        st.subheader(f"Question {st.session_state.page}", anchor=False)
        st.write(f"**{instruction_text}**")
        
        audio_data = current_data.get("audio_id", current_data.get("context"))
        
        try:
            st.audio(audio_data, autoplay=True)
        except Exception as e:
            st.error(f"Failed to play audio path `{audio_data}`: {e}")

        st.space("medium")

        with st.container():
            c1, c2 = st.columns(2)

            if len(choices) > 0:
                c1.button(choices[0], on_click=next_question, args=(st.session_state.page, choices[0], answer), use_container_width=True, key=choices[0])
            if len(choices) > 1:
                c2.button(choices[1], on_click=next_question, args=(st.session_state.page, choices[1], answer), use_container_width=True, key=choices[1])

            c3, c4 = st.columns(2)
            if len(choices) > 2:
                c3.button(choices[2], on_click=next_question, args=(st.session_state.page, choices[2], answer), use_container_width=True, key=choices[2])
            if len(choices) > 3:
                c4.button(choices[3], on_click=next_question, args=(st.session_state.page, choices[3], answer), use_container_width=True, key=choices[3])

        st.space("medium")
        with st.container(horizontal_alignment="right"):

            st.button(
                "Restart",
                on_click=reset,
                type="primary",
            )


elif st.session_state.page == max_questions + 1:
    st.space("xlarge")
    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        st.header("Game over!", text_alignment="center")

        st.subheader(
            f"You scored {st.session_state.correct} / {max_questions}",
            text_alignment="center",
        )

        st.subheader(
            f"LLM scored {st.session_state.llm_correct} / {max_questions}",
            text_alignment="center",
        )

        if st.session_state.correct > st.session_state.llm_correct:
            st.subheader("You won!!", text_alignment="center")
        elif st.session_state.correct < st.session_state.llm_correct:
            st.subheader("LLM won!!", text_alignment="center")
        else:
            st.subheader("Both won!", text_alignment="center")

        st.space("medium")

        st.button("New game", on_click=reset, type="primary")

else:
    reset()
