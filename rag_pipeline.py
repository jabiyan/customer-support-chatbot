from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def build_rag_pipeline(api_key: str, faq_path: str = "data/ecommerce_faq.txt"):

    # 1. Load documents
    loader = TextLoader(faq_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings (free, runs locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Store in ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    # 5. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 6. Custom prompt
    prompt_template = PromptTemplate.from_template("""You are a helpful and friendly customer support assistant for an e-commerce store.
Use the following information to answer the customer's question accurately and concisely.
If the answer is not in the provided information, politely say you don't have that information
and suggest they contact support directly.

Context:
{context}

Customer Question: {question}

Answer:""")

    # 7. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    # 8. Build modern LCEL chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def get_answer(rag_chain, question: str) -> dict:
    """Get answer from RAG pipeline."""
    chain, retriever = rag_chain
    answer  = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer":  answer,
        "sources": sources
    }
