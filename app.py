import os
from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from groq import Groq
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone_vector

app = Flask(__name__)

# Initialize Pinecone index
pinecone_index = pinecone_vector.initialize_pinecone_index()

# Initialize Groq client
def initialize_groq_client():
    groq_api_key = os.getenv('GROQ_API_KEY')
    return Groq(api_key=groq_api_key)

groq_client = initialize_groq_client()

# Initialize Pinecone vector store
def initialize_pinecone_vector_store():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "youtube-transcripts"
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

pinecone_vector_store = initialize_pinecone_vector_store()

def get_relevant_excerpts(user_question, vector_store):
    """Retrieve the most relevant excerpts from Pinecone based on the user's question."""
    relevant_docs = vector_store.similarity_search(user_question)
    relevant_excerpts = '\n\n---\n\n'.join([doc.page_content for doc in relevant_docs[:3]])
    return relevant_excerpts

def generate_response_with_context(client, model_name, user_question, relevant_excerpts):
    """Generate a response to the user's question using Groq and relevant excerpts."""
    system_prompt = '''
    You are a Q&A bot. Given the user's question and relevant excerpts from the content, answer the question by including direct quotes from the excerpts. If the information cannot be found, respond with "I don't know."
    '''
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ]
    )

    conversation_chain = LLMChain(
        llm=client,
        prompt=prompt,
        verbose=False
    )

    combined_query = f"User Question: {user_question}\n\nRelevant Excerpt(s):\n\n{relevant_excerpts}"
    
    response = conversation_chain.predict(human_input=combined_query)
    return response

@app.route('/vectorize', methods=['POST'])
def vectorize_video():
    """Endpoint to vectorize a video URL and store it in Pinecone."""
    request_data = request.json
    video_url = request_data['video_url']
    
    video_id, tokenized_chunks = next(pinecone_vector.process_videos_from_file('video_store.json'))

    if tokenized_chunks:
        pinecone_vector.upsert_to_pinecone(pinecone_index, tokenized_chunks, video_id)
        pinecone_vector.update_video_status('video_store.json', video_id)
        return jsonify({"message": f"Video {video_url} vectorized and stored in Pinecone."})
    else:
        return jsonify({"message": f"Failed to vectorize video {video_url}."}), 400

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Endpoint to answer a question using Groq and Pinecone."""
    request_data = request.json
    user_question = request_data['question']

    # Get relevant excerpts from Pinecone
    relevant_excerpts = get_relevant_excerpts(user_question, pinecone_vector_store)

    # Generate a response using Groq
    model_name = 'llama3-8b-8192'
    response = generate_response_with_context(groq_client, model_name, user_question, relevant_excerpts)

    return jsonify({"response": response, "context": relevant_excerpts})

if __name__ == "__main__":
    app.run()
