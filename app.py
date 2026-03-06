import random
import time

import streamlit as st
import json
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

max_questions = 5

af3_model_id = "nvidia/audio-flamingo-3-hf"
print("Loading model...")

target_device = "cuda:0" 

try:
    af3_processor = AutoProcessor.from_pretrained(af3_model_id)
    af3_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(af3_model_id, device_map=target_device)
    print(f"Model loaded successfully to {target_device}.")
except Exception as e:
    print(f"Failed to load model: {e}")
    af3_processor = None
    af3_model = None


def infer_af3(question, options, audio_file):
    if af3_model is None:
        return "Model not loaded properly."

    options_text = "\n".join([f"- {opt}" for opt in options])
    prompt = f"{question}\nOptions:\n{options_text}\nAnswer accurately using the text from the correct option."

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
def load_json(max_questions):
    with open("/Users/hrishikeshhpillai/Hrishi/ALMvsHuman/audio.json", "r") as f:
        data = json.load(f)
    # print(list(data.values()))

    data = random.sample(list(data.values()), max_questions)

    sample = [data[i] for i in range(len(data))]

    return sample


data = load_json(max_questions)


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
    # print("Clicked", q_name, st.session_state.clicked)
    st.session_state.page = q_name + 1

    if selected == answer:
        st.session_state.correct += 1

    with st.spinner(f"Waiting for {st.session_state.selected_model}..."):
        try:
            current_data = data[q_name - 1]
            audio_file = current_data.get("audio_id", current_data.get("file", ""))
            question_text = current_data.get("question", "Listen to the audio and choose the correct option.")
            options = current_data.get("options", [])

            llm_response = infer_af3(question_text, options, audio_file)
            # llm_response = "testing"
            print(f"LLM Response: {llm_response}")
            
            if answer.lower() in llm_response.lower():
                st.session_state.llm_correct += 1
        except Exception as e:
            print(f"Error during LLM inference: {e}")


if st.session_state.page == 0:

    st.space("xlarge")

    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        st.subheader("Start the game", text_alignment="center")
        st.button("Start game", on_click=set_page(1), type="primary")

elif (st.session_state.page >= 1) and (st.session_state.page <= max_questions):

    with st.container(horizontal_alignment="center", vertical_alignment="center"):

        current_data = data[st.session_state.page - 1]
        audio_file = current_data.get("audio_id", current_data.get("file", ""))
        question_text = current_data.get("question", "Listen to the audio and choose the correct option.")
        options = current_data.get("options", [])
        answer = current_data.get("answer", "")

        st.subheader(f"Question {st.session_state.page}", anchor=False)
        st.write(f"**{question_text}**")
        st.audio(audio_file, autoplay=True)

        st.space("medium")

        with st.container():
            c1, c2 = st.columns(2)

            c1.button(
                options[0],
                on_click=next_question,
                args=(st.session_state.page, options[0], answer),
                use_container_width=True,
                key=options[0],
            )
            c2.button(
                options[1],
                on_click=next_question,
                args=(st.session_state.page, options[1], answer),
                use_container_width=True,
                key=options[1],
            )

            c3, c4 = st.columns(2)
            c3.button(
                options[2],
                on_click=next_question,
                args=(st.session_state.page, options[2], answer),
                use_container_width=True,
                key=options[2],
            )
            c4.button(
                options[3],
                on_click=next_question,
                args=(st.session_state.page, options[3], answer),
                use_container_width=True,
                key=options[3],
            )

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
