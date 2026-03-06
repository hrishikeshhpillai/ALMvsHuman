import random
import time

import streamlit as st
import json

max_questions = 10
audio_folder = "./audio_files"


@st.cache_data
def load_json(max_questions):
    with open("audio.json", "r") as f:
        data = json.load(f)

    print(list(data.values()))

    data = random.sample(list(data.values()), max_questions)

    sample = [data[i] for i in range(len(data))]

    return sample


data = load_json(max_questions)


st.title("ALLM vs Human", anchor=False)
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

    with spinner_container:
        with st.spinner():
            time.sleep(2)  # TODO: Replace with LLM call and calculate llm output score

            # TODO: Add score if llm output is correct
            st.session_state.llm_correct = random.choice([0, 1])


if st.session_state.page == 0:

    st.space("xlarge")

    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        st.subheader("Start the game", text_alignment="center")
        st.button("Start game", on_click=set_page(1), type="primary")

elif (st.session_state.page >= 1) and (st.session_state.page <= max_questions):

    spinner_container = st.container()

    with st.container(horizontal_alignment="center", vertical_alignment="center"):

        question = data[st.session_state.page - 1]
        audio_file, options, answer = question.values()

        st.subheader(f"Question {st.session_state.page}", anchor=False)
        st.audio(f"{audio_folder}/{audio_file}", autoplay=True)

        st.space("medium")
        st.write("Choose correct option.")

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
            st.balloons()
            st.subheader("You won!!", text_alignment="center")
        elif st.session_state.correct < st.session_state.llm_correct:
            st.subheader("LLM won!!", text_alignment="center")
        else:
            st.balloons()
            st.subheader("Both won!", text_alignment="center")

        st.space("medium")

        st.button("New game", on_click=reset, type="primary")

else:
    reset()
