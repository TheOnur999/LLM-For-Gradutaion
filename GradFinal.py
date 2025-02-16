import json
import os
import re
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI
from bs4 import SoupStrainer
from dotenv import load_dotenv
load_dotenv("C:/Users/Onur/Desktop/Stuff/Creo Projects/RAG/.env")
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# Environment değişkenlerini yükleme
load_dotenv("C:/Users/Onur/Desktop/Stuff/Creo Projects/RAG/.env")

# JSON dosyasının konumunu belirleme. Varsayılan olarak kod dosyasının bulunduğu klasöre kaydedilir
JSON_FILE_PATH = "C:/Users/Onur/Desktop/Stuff/CV/Grup4/candidates.json"
JSON_FILE_LLM_ANSWER = "C:/Users/Onur/Desktop/Stuff/CV/Grup4/llmanswer.json"

CHROMA_PATH = r"chroma_db"

# LLM versiyonu belirleme
llm = ChatOpenAI(model="gpt-4o")

def clean_json_response(response_text):
    # Remove markdown code block delimiters if present
    if response_text.startswith("```json"):
        # Remove the first line containing ```json
        response_text = response_text.split("\n", 1)[-1]
    if response_text.endswith("```"):
        # Remove the last line containing ```
        response_text = response_text.rsplit("\n", 1)[0]
    return response_text.strip()

def process_pdf_with_gpt_RAG(file_path):
    # ChromeDB kütüphaneleri entegrasyonu
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="candidates")
    
    # PDF yükleyici
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        print(f" No content extracted from {file_path}")
        return None

    pdf_content = "\n\n".join(doc.page_content for doc in documents)
    
    # Verilenleri parçalara ayırma
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        print(f" No chunks created for {file_path}")
        return None

    # Verilerin ChromeDB'ye entegresi
    docs_for_db = [chunk.page_content for chunk in chunks]
    ids = [f"{os.path.basename(file_path)}_ID{i}" for i in range(len(chunks))]
    metadata = [chunk.metadata for chunk in chunks]

    collection.upsert(
        documents=docs_for_db,
        metadatas=metadata,
        ids=ids
    )

    # RAG Entegrasyonu
    query_result = collection.query(query_texts=[pdf_content], n_results=3, include=["documents", "distances"])
    
    if query_result and "documents" in query_result and query_result["documents"]:
        docs = query_result["documents"]
        distances = query_result.get("distances", None) 
        print("Query result keys:", list(query_result.keys()))

        # Her aşamadaki Chunk'ta similarity search için distance bakılır
        retrieved_chunks = ""
        if distances:
            for doc_set, dist_set in zip(docs, distances):
                for doc, score in zip(doc_set, dist_set):
                    retrieved_chunks += f"Document: {doc}\nSimilarity Score: {score:.4f}\n\n"
        else:
            # Fallback if similarity scores are not provided
            retrieved_chunks = "\n".join(sum(docs, []))
    else:
        retrieved_chunks = pdf_content

    # Print the retrieved content along with similarity scores to the console for debugging.
    print("Retrieved content with similarity scores:\n", retrieved_chunks)


    # Kullanılacak baz prompting değerleri
    prompt = (
        "You are an Industrial Engineer HR professional tasked with analyzing resumes. "
        "Extract the following details from the resume provided below:\n\n"
        "1. Name\n"
        "2. GPA\n"
        "3. Projects\n"
        "4. Certificates\n"
        "5. Skills\n"
        "6. Work Experience\n\n"
        f"Retrieved content (with similarity scores):\n{retrieved_chunks}\n\n"
        "Provide the details in JSON format."
    )

    # LLM'den cevap alımı
    response = llm(prompt)
    response_text = response.content

    # Fonksiyon dışı kullanım için
    return response_text

# The rest of your functions remain unchanged
def load_candidates_from_json():
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r") as file:
            return json.load(file)
    return []  

def save_candidates_to_json(candidates):
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(candidates, file, indent=4)

def save_answer_to_json(rec_answer):
    with open(JSON_FILE_LLM_ANSWER, "w") as file:
        json.dump(rec_answer, file, indent=4)

def process_pdf_with_gpt(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pdf_content = "\n\n".join(doc.page_content for doc in documents)

    prompt = (
        f"You are an HR professional tasked with analyzing resumes. "
        f"Extract the following details from the resume provided below:\n\n"
        f"1. Name\n"
        f"2. GPA\n"
        f"3. Projects\n"
        f"4. Certificates\n"
        f"5. Skills\n"
        f"6. Work Experience\n\n"
        f"Resume content:\n{pdf_content}\n\n"
        f"Provide the details in JSON format."
    )
    
    response = llm(prompt)
    response_text = response.content  
    cleaned_text = clean_json_response(response_text)
    
    boolvalue = True
    while boolvalue:
        try:
            extracted_data = json.loads(cleaned_text)
            boolvalue = False  
            return extracted_data
        except json.JSONDecodeError:
            print(" Error: Could not decode GPT's response into JSON.")
            print(" GPT's response was:", response_text)
            return None
    return extracted_data

if __name__ == "__main__":
    print(" Welcome to the Candidate Analyzer!")
    
    while True:
        a = input("Would you like to read from JSON file? (y/n) ").strip().lower()
        if a in ("yes","y"):
            candidates = load_candidates_from_json()
            
            user_choice = input(" Would you like to add new PDF files? (yes/no): ").strip().lower()

            if user_choice in ("yes", "y"):
                pdf_dir = input(" Enter the directory containing the PDF files: ").strip()
                
                if os.path.isdir(pdf_dir):
                    pdf_files = [
                        os.path.join(pdf_dir, f)
                        for f in os.listdir(pdf_dir)
                        if f.lower().endswith(".pdf")  
                    ]

                    if pdf_files:
                        print(f" Found {len(pdf_files)} PDF(s). Processing...")

                        for file_path in pdf_files:
                            print(f" Processing: {file_path}")
                            extracted_data = process_pdf_with_gpt(file_path)

                            if extracted_data:
                                candidates.append(extracted_data)
                                print(f" Extracted data: {json.dumps(extracted_data, indent=4)}")
                            else:
                                print(f" Failed to process {file_path}")

                        save_candidates_to_json(candidates)
                        print(" New data has been added to the database.")
                        print(f" Candidates data saved at: {os.path.abspath(JSON_FILE_PATH)}")
                    else:
                        print(" No PDF files found in the specified directory.")
                else:
                    print(" The specified directory does not exist. Exiting...")

            if not candidates:
                print(" No candidate data found. Please add PDFs first.")
            else:
                print("\n Loaded candidates:")
                for candidate in candidates:
                    print(json.dumps(candidate, indent=4))
                
        elif a in ("no","n"):
            print("Utilizing RAG...")
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            candidates = chroma_client.get_or_create_collection(name="candidates")
            candidates = []
            user_choice2 = input("Would like to provide new pdf files for RAG?: ").strip().lower()
            if user_choice2 in ("yes","y"):
                pdf_dir = input(" Enter the directory containing the PDF files: ").strip()
                if os.path.isdir(pdf_dir):
                    pdf_files = [
                        os.path.join(pdf_dir, f)
                        for f in os.listdir(pdf_dir)
                        if f.lower().endswith(".pdf")
                    ]
                    if pdf_files:
                        print(f" Found {len(pdf_files)} PDF(s). Processing...")
                        for file_path in pdf_files:
                            print(f"Processing: {file_path}")
                            extracted_data = process_pdf_with_gpt_RAG(file_path)
                            if extracted_data:
                                candidates.append(extracted_data)
                            else:
                                print(f" Failed to process {file_path}")
                    else:
                        print("No files were found!")
                else:
                    print("No such directory was found!")        
            
        else:
            print("Please specify your answer.")

        print("\n Generating recommendations...")
        if a in ("yes","y"):
            prompt = (
                   "Now, consider the following candidate data:\n\n"
                   f"{json.dumps(candidates, indent=4)}\n\n"
                   "Who would you recommend for an Industrial Engineering position? Provide your reasons. "
                   "Write your thoughts about every candidate too. "
                   "Write your answer in a way that's easy to save in a JSON."
            )

            recommendations = llm(prompt)
            firstanswer = clean_json_response(recommendations.content)
            rec_answer = []
            rec_answer.append(json.loads(firstanswer))
            save_answer_to_json(rec_answer)
            print("\n GPT's Recommendation:")
            print(recommendations.content)
        elif a in ("no","n"):
            prompt = (
                f"Based on the following candidate data:\n\n"
                f"{candidates}\n\n"
                f"Who would you recommend for a Industrial Engineering position? Provide reasons."
            )
            recommendations = llm(prompt)
            print("\n GPT's Recommendation:")
            print(recommendations.content)
        
        print("\n You can now ask additional questions about the candidates. Type 'exit' to finish.")

        while True:
            question = input(" Your question: ").strip()
            if question.lower() == "exit":
                print("Goodbye!")
                break
            if a in ("yes","y"):
                followup_prompt = f"Candidates: {json.dumps(candidates, indent=4)}\n\nQuestion: {question}"
                followup_response = llm(followup_prompt)
                print("\n GPT's Response:")
                print(followup_response.content)
            if a in ("no","n"):
                followup_prompt = f"Candidates: {candidates}\n\nQuestion: {question}"
                followup_response = llm(followup_prompt)
                print("\n GPT's Response:")
                print(followup_response.content)
        break
