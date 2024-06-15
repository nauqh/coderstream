import pandas as pd
import streamlit as st

from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI
from streamlit_chat import message

st.set_page_config(
    page_title="Coderstream",
    page_icon="🧑‍💻",
)

if "datasets" not in st.session_state:
    datasets = {}
    datasets["Titanic"] = pd.read_csv("titanic.csv")

    st.session_state["datasets"] = datasets
else:
    datasets = st.session_state["datasets"]


# NOTE: SIDEBAR
with st.sidebar:
    st.info("**🚧Status**: `BETA`")
    openai_key = st.text_input(
        label="🔑 OpenAI Key:",
        help="Required for OpenAI's models",
        type="password")
    container = st.empty()

    # NOTE: FILE UPLOAD
    file = st.file_uploader("💻 Load a CSV file:", type="csv")
    index_no = 0
    if file:
        file_name = file.name[:-4].capitalize()
        datasets[file_name] = pd.read_csv(file)
        index_no = len(datasets)-1

    chosen_dataset = container.radio(
        "📑 Choose your data:",
        datasets.keys(),
        index=index_no
    )
    st.markdown("##")
    st.markdown("Designed and built by [**Nauqh**](https://github.com/nauqh).")

# NOTE: MAIN
st.markdown("""<h1 style='
                font-family: Inconsolata; font-weight: 400;
                font-size: 3rem'>🧑‍💻Coderstream</h1>""",
            unsafe_allow_html=True)
st.markdown("""<h3 style='
                font-family: Inconsolata; font-weight: 400;
                font-size: 1.5rem'>Our sophisticated A.I. addresses your awful inquiry</h3>""",
            unsafe_allow_html=True)

st.markdown("##")
df = datasets[chosen_dataset]
empty = st.empty()
st.data_editor(df)


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}]

# Load message history
for msg in st.session_state.messages:

    message(msg["content"], is_user=True if msg["role"] == "user" else False)

# Main conversation
if prompt := st.chat_input(placeholder="What is this data about?"):
    if not openai_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    llm = ChatOpenAI(temperature=0,
                     model="gpt-3.5-turbo-0613",
                     openai_api_key=openai_key,
                     streaming=True)

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    try:
        st_cb = StreamlitCallbackHandler(
            st.empty(),
            collapse_completed_thoughts=True
        )

        response = agent.run(
            {
                "input": st.session_state.messages[-1]
            },
            callbacks=[st_cb]
        )
        st.session_state.messages.append({"role": "assistant",
                                          "content": response})
        message(response)
    except Exception:
        st.session_state["messages"] = [{"role": "assistant",
                                         "content": "How can I help you?"}]
        st.rerun()
