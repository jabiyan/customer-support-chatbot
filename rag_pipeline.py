import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

FAQ_TEXT = """
SHIPPING & DELIVERY
Q: How long does standard shipping take?
A: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.

Q: How much does shipping cost?
A: Standard shipping is free for orders over $50. For orders under $50 it costs $4.99. Express costs $9.99.

Q: Do you ship internationally?
A: Yes, we ship to over 50 countries. International shipping takes 10-21 business days starting at $14.99.

Q: How do I track my order?
A: Once your order ships you will receive a tracking number via email. Track it under My Orders on our website.

Q: My order has not arrived. What should I do?
A: Check your tracking number first. If tracking shows delivered but you have not received it contact support within 7 days.

RETURNS & REFUNDS
Q: What is your return policy?
A: We offer a 30-day return policy. Items must be unused and in original packaging. Sale items are final.

Q: How do I return an item?
A: Log into your account go to My Orders select the order and click Return Item. Print the prepaid label and drop at any post office.

Q: How long does a refund take?
A: We process returns within 2-3 business days. Refunds appear within 5-10 business days depending on your bank.

Q: Can I exchange an item?
A: Yes exchanges are available for items of equal or lesser value. Select Exchange instead of Refund during the return process.

Q: I received a damaged item. What do I do?
A: Take photos and contact support within 48 hours. We will send a replacement or issue a full refund.

ORDERS & PAYMENTS
Q: What payment methods do you accept?
A: We accept Visa Mastercard American Express PayPal Apple Pay Google Pay and Shop Pay.

Q: Can I cancel or modify my order?
A: Orders can be cancelled or modified within 1 hour of placing them. After that the order enters processing.

Q: Do you offer buy now pay later?
A: Yes we offer Klarna and Afterpay. Split your purchase into 4 interest-free payments at checkout.

Q: Is my payment information secure?
A: Yes we use 256-bit SSL encryption and are PCI DSS compliant. We never store your full card details.

PRODUCTS & INVENTORY
Q: How do I know if an item is in stock?
A: Product availability is shown on each product page. Click Notify Me for out of stock items.

Q: Do you offer product warranties?
A: Electronics come with a 1-year manufacturer warranty. Other products carry a 90-day warranty against defects.

Q: Are your products authentic?
A: Yes all products are 100% authentic. We work directly with manufacturers and authorized distributors only.

ACCOUNT & LOYALTY
Q: How do I reset my password?
A: Click Forgot Password on the login page. Enter your email and we will send a reset link valid for 24 hours.

Q: Do you have a loyalty rewards program?
A: Yes earn 1 point for every $1 spent. 100 points equals $1 off your next order. Double points on your birthday month.

Q: How do I apply a discount code?
A: Enter your code in the Promo Code field at checkout and click Apply. One code per order only.

CONTACT & SUPPORT
Q: How do I contact customer support?
A: Live chat 24/7 on our website. Email support@shop.com for response within 24 hours. Call 1-800-123-4567 Mon-Fri 9AM-6PM EST.

Q: Do you have a physical store?
A: We are online only but have pickup points in New York Los Angeles Chicago and Miami.
"""


def build_rag_pipeline(api_key: str, faq_path: str = "data/ecommerce_faq.txt"):

    if os.path.exists(faq_path):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(faq_path, encoding="utf-8")
        documents = loader.load()
    else:
        documents = [Document(page_content=FAQ_TEXT, metadata={"source": "embedded_faq"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt_template = PromptTemplate.from_template(
        """You are a helpful and friendly customer support assistant for an e-commerce store.
Use the following information to answer the customer question accurately and concisely.
If the answer is not in the provided information politely say you do not have that information
and suggest they contact support directly.

Context:
{context}

Customer Question: {question}

Answer:"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2
    )

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
    chain, retriever = rag_chain
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {"answer": answer, "sources": sources}
