import random
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

# st.write(data)
# data

st.title("AI vs Human")


def reset():
    st.session_state.page = 0
    st.session_state.correct = 0


if "page" not in st.session_state:
    reset()


def set_page(name):

    st.session_state.page = name


def next_question(q_name, selected, answer):
    # print("Clicked", q_name, st.session_state.clicked)
    st.session_state.page = q_name + 1

    if selected == answer:
        st.session_state.correct += 1


if st.session_state.page == 0:

    for _ in range(10):
        st.write(" ")

    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        # st.markdown('<div class="full-height">', unsafe_allow_html=True)
        st.subheader("Start the game", text_alignment="center")
        st.button("Start game", on_click=set_page(1), type="primary")

        # st.markdown('</div>', unsafe_allow_html=True)

elif (st.session_state.page >= 1) and (st.session_state.page <= max_questions):
    # for _ in range(10):
    #     st.write(" ")

    # q_id = st.session_state.page - 1

    with st.container(horizontal_alignment="center", vertical_alignment="center"):

        question = data[st.session_state.page - 1]
        audio_file, options, answer = question.values()

        st.subheader(f"Question {st.session_state.page}")
        st.audio(f"{audio_folder}/{audio_file}", autoplay=True)

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

        # st.write(answer)
        # print(st.session_state.page)

        # st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == max_questions + 1:
    for _ in range(10):
        st.write(" ")
    with st.container(horizontal_alignment="center", vertical_alignment="center"):
        st.header("Game over!", text_alignment="center")

        st.subheader(
            f"You scored {st.session_state.correct} / {max_questions}",
            text_alignment="center",
        )

else:
    reset()
