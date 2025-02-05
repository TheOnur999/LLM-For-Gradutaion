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

#JSON dosyasının konumunu belirleme. Varsayılan olarak kod dosyasının bulunduğu klasöre kaydedilir
JSON_FILE_PATH = "candidates.json"


CHROMA_PATH = r"chroma_db"

# LLM versiyonu belirleme
llm = ChatOpenAI(model="gpt-4o")

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

    #RAG Entegrasyonu
    query_result = collection.query(query_texts=[pdf_content], n_results=3)
    if query_result and "documents" in query_result and query_result["documents"]:
        
        retrieved_chunks = "\n".join(sum(query_result["documents"], []))
    else:
        
        retrieved_chunks = pdf_content

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
        f"Retrieved content:\n{retrieved_chunks}\n\n"
        "Provide the details in JSON format."
    )

    # LLM'den cevap alımı
    response = llm(prompt)
    response_text = response.content

    # Fonksiyon dışı kullanım için
    return response_text






#def utilize_rag(DATA_PATH):
# setting the environment

    #chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    #collection = chroma_client.get_or_create_collection(name="candidates")
    
# loading the document

    #loader = PyPDFDirectoryLoader(DATA_PATH)

    #raw_documents = loader.load()

# splitting the document

    #text_splitter = RecursiveCharacterTextSplitter(
        #chunk_size=300,
        #chunk_overlap=100,
        #length_function=len,
        #is_separator_regex=False,
    #)

    #chunks = text_splitter.split_documents(raw_documents)

# preparing to be added in chromadb

    #documents = []
    #metadata = []
    #ids = []

    i = 0

    #for chunk in chunks:
        #documents.append(chunk.page_content)
        #ids.append("ID"+str(i))
        #metadata.append(chunk.metadata)

        #i += 1

# adding to chromadb


    #collection.upsert(
        #documents=documents,
        #metadatas=metadata,
        #ids=ids)


def load_candidates_from_json():
    
    if os.path.exists(JSON_FILE_PATH):
        with open(JSON_FILE_PATH, "r") as file:
            return json.load(file)
    return []  


def save_candidates_to_json(candidates):
    
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(candidates, file, indent=4)


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

    try:
        extracted_data = json.loads(response_text)  
        return extracted_data
    except json.JSONDecodeError:
        print(" Error: Could not decode GPT's response into JSON.")
        print(" GPT's response was:", response_text)
        return None


if __name__ == "__main__":
    print(" Welcome to the Candidate Analyzer!")

    
    while True:
        
        a = input("Would you like to read from JSON file? (y/n)")
        a = a.lower()
        if a in ("yes","y"):

            candidates = load_candidates_from_json()
            
            user_choice = input(" Would you like to add new PDF files? (yes/no): ").strip().lower()

            if user_choice.lower() in ("yes", "y"):
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
            candidates= chroma_client.get_or_create_collection(name="candidates")
            candidates=[]
            user_choice2 = input("Would like to provide new pdf files for RAG?: ")
            if user_choice2.lower() in ("yes","y"):
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


            
            #DATA_PATH = input("Please provide directory for PDF files: ")
            #utilize_rag(DATA_PATH)
            
            
            
            
        else:
            print("Please specify your answer.")

    

        
        print("\n Generating recommendations...")
        if a in ("yes","y"):
            
            prompt = (
                f"Based on the following candidate data:\n\n"
                f"{json.dumps(candidates, indent=4)}\n\n"
                f"Who would you recommend for a Industrial Engineering position? Provide reasons."
            )

            recommendations = llm(prompt)
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