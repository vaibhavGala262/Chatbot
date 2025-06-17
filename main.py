import chromadb.proto
import streamlit as st 
import pdfplumber
import chromadb 
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

st.title("Pdf Reader") 
query = st.chat_input(placeholder="Write here")
chroma_client = chromadb.PersistentClient(path="chroma_db/")

collection = chroma_client.get_or_create_collection(name="text_collection")



pdf_file = st.file_uploader(label= "Upload you file" , type='pdf')



if pdf_file is not None:
      # get data in byrte
    
    
    
    text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text.append( page.extract_text() )  # extract text from file 
    
    # collection.add(
    # documents=text[0],
    # ids ="id1"
    # )
    
    doc_embedding = embedding.embed_documents(text)
    collection.add(
        embeddings=doc_embedding,
        ids = [f"doc_{i}" for i in range(len(doc_embedding))]
    )  # store pdf embeding in chroma dbb
    if query :
        query_embedding = embedding.embed_query(query)
    
        results = collection.query(
        query_embeddings=query_embedding, # Chroma will embed this for you
        n_results=1 # how many results to return
         )
#store query embeding in chroma db
        similarities = cosine_similarity([query_embedding], doc_embedding)
        most_representative_index = np.argmax(similarities)
        result = text[most_representative_index]
        st.write(result)

    
   
# So far: an application where we can uppload pdf and ask lines and related text will be printed from pdf
# Goal : 