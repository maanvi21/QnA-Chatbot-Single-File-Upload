# vector_store.py (with flan-t5-base)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Step 1: Embedding model (unchanged)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Chunk text
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# Step 3: Vector store
def create_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding_model)

# Step 4: Retrieve top-k chunks
def get_top_chunks(vector_store, query, k=3):
    return vector_store.similarity_search(query, k=k)

# Step 5: Run local FLAN-T5 model
def ask_question_with_context(chunks, query):
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Build simple prompt
    context = "\n".join([doc.page_content for doc in chunks])
    prompt = PromptTemplate(
        template="Answer the question using the following context:\n\n{context}\n\nQuestion: {question}",
        input_variables=["context", "question"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"context": context, "question": query})