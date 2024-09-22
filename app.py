import os
import json
from flask import Flask, request, jsonify
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.messages import SystemMessage
from groq import Groq
from langchain_pinecone import PineconeVectorStore
import pinecone_vector
import transcription
from azure_embeddings import generate_embeddings

app = Flask(__name__)

# Initialize Pinecone index
pinecone_index = pinecone_vector.initialize_pinecone_index()

# Initialize Groq client
def initialize_groq_client():
    groq_api_key = os.getenv('GROQ_API_KEY')
    return Groq(api_key=groq_api_key)

groq_client = initialize_groq_client()

system_prompt = '''
    You are a Q&A bot. Given the user's question and relevant excerpts from the content, answer the question by including direct quotes from the excerpts. If the information cannot be found, respond with "I don't know."
'''

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ]
)

# Custom embedding class for Azure OpenAI
class AzureEmbeddingFunction:
    def embed_query(self, text):
        """Generate embeddings for a single query."""
        return generate_embeddings(text)

    def embed(self, texts):
        """Generate embeddings for a list of texts."""
        return [self.embed_query(text) for text in texts if text]  # Filter out empty strings

# Initialize Pinecone vector store
def initialize_pinecone_vector_store():
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = "youtube-transcripts"
    
    embedding_function = AzureEmbeddingFunction()
    
    return PineconeVectorStore(index_name=pinecone_index_name, embedding=embedding_function)

# Initialize the vector store
pinecone_vector_store = initialize_pinecone_vector_store()

def get_relevant_excerpts(user_question, vector_store):
    """Retrieve the most relevant excerpts from Pinecone based on the user's question."""
    if not user_question:  # Check for empty question
        print("No question provided.")
        return ""  # Return empty string if no question is provided
    
    # Perform similarity search with the embedded question
    relevant_docs = vector_store.similarity_search(user_question)

    # Extract the page_content from all relevant documents
    relevant_excerpts = [doc.page_content for doc in relevant_docs]  # No limit on the number of documents

    return '\n\n---\n\n'.join(relevant_excerpts)


def generate_response_with_groq(client, model_name, combined_query):
    """Generate a response using the Groq API."""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": combined_query}
        ],
        model=model_name
    )
    
    # Extract the response from the Groq API
    response = chat_completion.choices[0].message.content
    return response

@app.route("/vectorize", methods=['POST', 'GET'])
def vectorize_video():
    """Endpoint to vectorize a video URL and store it in Pinecone."""
    
    video_url = None
    if request.method == 'POST':
        if request.content_type == 'application/json':
            request_data = request.json
            video_url = request_data.get('video_url')
        else:
            return jsonify({"error": "Content-Type must be 'application/json' for POST requests"}), 400
    
    elif request.method == 'GET':
        video_url = request.args.get('video_url')

    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    with open('video_store.json', 'r') as f:
        video_data = json.load(f)

    existing_video = next((video for video in video_data if video['video_url'] == video_url), None)
    
    if existing_video:
        if existing_video['vectorized']:
            return jsonify({"message": f"Video {video_url} is already vectorized."}), 200
        else:
            try:
                video_id = video_url.split('v=')[-1]
                transcript = transcription.get_transcript(video_id)
                
                if transcript:
                    tokenized_chunks = transcription.tokenize_text(transcript, video_url=video_url)
                    pinecone_vector.upsert_to_pinecone(pinecone_index, tokenized_chunks, video_id)

                    existing_video['vectorized'] = True
                    with open('video_store.json', 'w') as f:
                        json.dump(video_data, f, indent=4)

                    return jsonify({"message": f"Video {video_url} has been vectorized and updated."}), 200
                else:
                    return jsonify({"error": "Transcript not available for video."}), 400
            except StopIteration:
                return jsonify({"error": "Failed to vectorize video."}), 400
    else:
        try:
            video_id = video_url.split('v=')[-1]
            transcript = transcription.get_transcript(video_id)

            if transcript:
                tokenized_chunks = transcription.tokenize_text(transcript, video_url=video_url)

                pinecone_vector.upsert_to_pinecone(pinecone_index, tokenized_chunks, video_id)

                new_video = {"video_url": video_url, "vectorized": True}
                video_data.append(new_video)

                with open('video_store.json', 'w') as f:
                    json.dump(video_data, f, indent=4)

                return jsonify({"message": f"Video {video_url} vectorized and added to the store."}), 200
            else:
                return jsonify({"error": "Transcript not available for video."}), 400
        except StopIteration:
            return jsonify({"error": "Failed to vectorize new video."}), 400

@app.route('/ask_question', methods=['POST', 'GET'])
def ask_question():
    """Endpoint to answer a question using Groq and Pinecone."""
    user_question = None
    if request.method == 'POST':
        request_data = request.json
        user_question = request_data.get('question')
    elif request.method == 'GET':
        user_question = request.args.get('question')

    relevant_excerpts = get_relevant_excerpts(user_question, pinecone_vector_store)
    
    if not relevant_excerpts:  # Check if there are no relevant excerpts
        return jsonify({"response": "No relevant data found."}), 404  # Suitable message with a 404 status

    model_name = 'llama3-8b-8192'
    combined_query = f"User Question: {user_question}\n\nRelevant Excerpt(s):\n\n{relevant_excerpts}"
    
    # Generate a response using Groq
    response = generate_response_with_groq(groq_client, model_name, combined_query)

    return jsonify({"response": response, "context": relevant_excerpts})

if __name__ == "__main__":
    app.run()
