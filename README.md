# Local RAG

Almost all the components needed to run RAG on your local M1 Macbook Pro. 

# Overview

Creates a chatbot that reads a document, populates a vector database with embeddings and answers questions.

- Based off of the streamlit chatbot pack
- Replaced OpenAI with local LLM ( llama2 ) running locally
- Replaced OpenAI embeddings with Hugging Face
- Replaced in memory vector db with Chroma

# Execution

Probably missing some pre requisites, but run it using 
% streamlit run ./streamlit_chatbot_pack/base3.py

