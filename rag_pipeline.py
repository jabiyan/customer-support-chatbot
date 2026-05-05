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
    # 1. Load documents — embedded directly for cloud deployment
import os
from langchain_core.documents import Document

if os.path.exists(faq_path):
    loader = TextLoader(faq_path, encoding="utf-8")
    documents = loader.load()
else:
    # Fallback — embedded FAQ for Streamlit Cloud
    faq_text = """
SHIPPING & DELIVERY
Q: How long does standard shipping take?
A: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.

Q: How much does shipping cost?
A: Standard shipping is free for orders over $50. For orders under $50, standard shipping costs $4.99.

Q: How do I track my order?
A: Once your order ships, you will receive a tracking number via email. Track it under My Orders.

Q: Do you ship internationally?
A: Yes, we ship to over 50 countries. International shipping takes 10-21 business days.

RETURNS & REFUNDS
Q: What is your return policy?
A: We offer a 30-day return policy. Items must be unused and in original packaging. Sale items are final.

Q: How do I return an item?
A: Log into your account, go to My Orders, select the order and click Return Item. Print the prepaid label.

Q: How long does a refund take?
A: We process returns within 2-3 business days. Refunds appear within 5-10 business days on your statement.

Q: I received a damaged item. What do I do?
A: Take photos and contact support within 48 hours. We will send a replacement or issue a full refund.

ORDERS & PAYMENTS
Q: What payment methods do you accept?
A: We accept Visa, Mastercard, Amex, PayPal, Apple Pay, Google Pay, and Shop Pay.

Q: Can I cancel or modify my order?
A: Orders can be cancelled within 1 hour of placing them. After that they enter processing.

Q: Do you offer buy now pay later?
A: Yes, we offer Klarna and Afterpay — split into 4 interest-free payments at checkout.

Q: Is my payment information secure?
A: Yes, we use 256-bit SSL encryption and are PCI DSS compliant.

PRODUCTS & INVENTORY
Q: How do I know if an item is in stock?
A: Product availability is shown on each product page. Click Notify Me for out of stock items.

Q: Do you offer product warranties?
A: Electronics come with a 1-year manufacturer warranty. Other products have a 90-day warranty.

ACCOUNT & LOYALTY
Q: How do I reset my password?
A: Click Forgot Password on the login page and enter your email. A reset link is valid for 24 hours.

Q: Do you have a loyalty rewards program?
A: Yes — earn 1 point per $1 spent. 100 points = $1 off. Double points on your birthday month.

Q: How do I apply a discount code?
A: Enter your code in the Promo Code field at checkout. One code per order only.

CONTACT & SUPPORT
Q: How do I contact customer support?
A: Live chat 24/7, email support@shop.com (24hr response), or call 1-800-123-4567 Mon-Fri 9AM-6PM EST.
"""
    documents = [Document(page_content=faq_text, metadata={"source": "embedded_faq"})]

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
