#
# Project that deploys a RAG based chatbot on the M1 Macbook Pro.
# We can load the index from a local file, or from Chroma
# Load initially by setting the doc_path file to the directory you want to use as a source
# Vector data based path is determined by vdb_path = "./docs/examples/chroma/..."
#
# Then set the doc_path to empty, so the system loads from the vector data base.
#
# Project was based initially off of the Streamlit Chat Pack
# https://www.linkedin.com/posts/llamaindex_you-can-now-spin-up-a-rag-streamlit-app-activity-7137126999172550656-AXhz?utm_source=share&utm_medium=member_desktop
# I replaced the reference to OpenAI GPT-3.5 Turbo with a local LLM
# Also replaced the default in memory vector db with Chroma.
#
# 20240220 Upgraded with advanced RAG based on this article
# https://www.linkedin.com/posts/804250ab_advanced-retrieval-augmented-generation-activity-7165375010663055360-PN_v/?utm_source=share&utm_medium=member_ios
# Also upgraded to llama_index 0.10.7 which necessitated a bunch of code changes for migration.
#
# TODOs:
# 1. make it more flexible to start a new session and load from different sources       
# 2. Can switch between vector databases
# 3. Detect if vector db data already available and use it instead of reading documents

from typing import Dict, Any

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate,download_loader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.llms.openai import OpenAI
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

import chromadb
from chromadb.utils import embedding_functions

from importlib.metadata import version
import time

# What version of llama_index are we using?
print(f"LlamaIndex version: {version('llama_index')}")

class StreamlitChatPack(BaseLlamaPack):
    """Streamlit chatbot pack."""

    def __init__(
        self,
        wikipedia_page: str = "Summarization assistant",
        run_from_main: bool = False,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if not run_from_main:
            raise ValueError(
                "Please run this llama-pack directly with "
                "`streamlit run [download_dir]/streamlit_chatbot/base.py`"
            )

        self.wikipedia_page = wikipedia_page

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""
        import streamlit as st
        from streamlit_pills import pills

        st.set_page_config(
            page_title=f"Chat with {self.wikipedia_page} today",
            page_icon="🦙",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )

        if "messages" not in st.session_state:  # Initialize the chat messages history
            st.session_state["messages"] = [
                {"role": "assistant", "content": "What do you want to do today!"},
            ]

        st.title(
            f"Chat with {self.wikipedia_page}"
        )

        # st.info(
        #    "This example is powered by the **[Llama Hub Wikipedia Loader](https://llamahub.ai/l/wikipedia)**. Use any of [Llama Hub's many loaders](https://llamahub.ai/) to retrieve and chat with your data via a Streamlit app.",
        #    icon="ℹ️",
        # )

        def add_to_message_history(role, content):
            message = {"role": role, "content": str(content)}
            st.session_state["messages"].append(
                message
            )  # Add response to message history

        # Use this if you want to use the OpenAI LLM, which we're not
        def get_openai_llm():
            llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5)
            return llm

        # Load the model
        def get_llama2_llm():
            llm = LlamaCPP(
                # You can pass in the URL to a GGML model to download it automatically
                model_url=None,
                # Or you can pass in the path to a local model
                model_path="../llama2/llama.cpp/models/7Bchat/speechless-llama2-hermes-orca-platypus-wizardlm-13b.Q5_K_S.gguf",
                temperature=0.1,
                max_new_tokens=1024,
                # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                context_window=3900,
                # kwargs to pass to __call__()
                generate_kwargs={},
                # kwargs to pass to __init__()
                # set to at least 1 to use GPU
                model_kwargs={"n_gpu_layers": 1},
                # transform inputs into Llama2 format
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                verbose=True,
            )
            return llm

        @st.cache_resource
        def load_chroma_data():

            hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")

            start_time = time.time()

            doc_path = "./docs/examples/data/scrumofscrums/"
            # doc_path = "./docs/examples/data/empty/"

            # load some documents
            print("Loading documents: " + doc_path)
            documents = SimpleDirectoryReader(doc_path).load_data()

            checkpoint_time = time.time()
            print(f"Documents loaded in {checkpoint_time - start_time:.2f} seconds")
            start_time = checkpoint_time

            # initialize client, setting path to save vector db data
            vdb_path = "./docs/examples/chroma/scrumofscrums"

            # initialize client, setting path to save data
            db = chromadb.PersistentClient(path=vdb_path, )

            checkpoint_time = time.time()
            print("Creating db store in " + vdb_path)
            print(f"DB Client created in {checkpoint_time - start_time:.2f} seconds")
            start_time = checkpoint_time

            default_ef = embedding_functions.DefaultEmbeddingFunction()

            # create collection
            chroma_collection = db.get_or_create_collection("quickstart")

            checkpoint_time = time.time()
            print(f"Collection created in {checkpoint_time - start_time:.2f} seconds")
            start_time = checkpoint_time

            # assign chroma as the vector_store to the context
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            checkpoint_time = time.time()
            print(f"Storage context created in {checkpoint_time - start_time:.2f} seconds")
            start_time = checkpoint_time

            # create service context
            # llm_loaded = get_openai_llm()
            llm_loaded = get_llama2_llm()

            service_context = ServiceContext.from_defaults(
                # llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5)
                llm = llm_loaded,
                chunk_size=256,
                # embed_model="local:BAAI/bge-large-en"
                # embed_model = default_ef
                embed_model=hf_embed_model
            )

            # create your index
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context,
                service_context=service_context
            )
            return index

        # This uses the in memory vector db.  Note the hard coded doc path
        @st.cache_resource
        def load_index_data_2():
            hf_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
            llm_loaded = get_llama2_llm()

            doc_path = "./docs/examples/data/b2b_simplif_readout2/"
            # doc_path = "./docs/examples/data/scrumofscrums/"
            documents = SimpleDirectoryReader(doc_path).load_data()

            service_context = ServiceContext.from_defaults(
                llm = llm_loaded,
                chunk_size=256,
                chunk_overlap=64,
                context_window=4096,
                embed_model=hf_embed_model
            )

            index = VectorStoreIndex.from_documents(
                documents, service_context=service_context
            )
            return index

        # index = load_index_data_2()
        index = load_chroma_data()

        selected = pills(
            "Choose a question to get started or write your own below.",
            [
                "Summarize the meeting",
                "What were the main concepts in the meeting",
                "Who were the main speakers",
                "What are the action items from the meeting",
            ],
            clearable=True,
            index=None,
        )

        print("What is my index: " + str(index))

        from llama_index.core.postprocessor import MetadataReplacementPostProcessor
        from llama_index.core.postprocessor import SentenceTransformerRerank

        # The target key defaults to `window` to match the node_parser's default
        postproc = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )

        # Advanced RAG - reranker
        rerank = SentenceTransformerRerank(
            top_n=2,
            model="BAAI/bge-reranker-base"
        )

        # Modified for Advanced RAG
        # More similarity_top_k
        # Set alpha to 0.5
        if "chat_engine" not in st.session_state:  # Initialize the query engine
            st.session_state["chat_engine"] = index.as_chat_engine(
                chat_mode="context", verbose=True,
                similarity_top_k=6,
#                vector_store_query_mode="hybrid",
                alpha=0.5,
                node_postprocessors=[postproc, rerank],
            )
            print("Initializing chat engine")

        for message in st.session_state["messages"]:  # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
                print("Writing messages: ", message["content"])

        if selected:
            with st.chat_message("user"):
                st.write(selected)
            with st.chat_message("assistant"):
                response = st.session_state["chat_engine"].stream_chat(selected)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                add_to_message_history("user", selected)
                add_to_message_history("assistant", response)
                print("\n IS SELECTED\n")

        if prompt := st.chat_input(
            "Your question"
        ):  # Prompt for user input and save to chat history
            with st.chat_message("user"):
                st.markdown(prompt)
            add_to_message_history("user", prompt)
            print("***Added to message history **user**" + str(prompt))
            # print(" **user**" + str(prompt))

        # If last message is not from assistant, generate a new response
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = st.session_state["chat_engine"].stream_chat(prompt)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)

                # st.write(response.response)
                add_to_message_history("assistant", response.response)

                print("***Response container: " + str(response_str))
                # print("***Added to message history **assistant**" + str(response.response))

if __name__ == "__main__":
    StreamlitChatPack(run_from_main=True).run()
