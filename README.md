# 🤖 Smart E-Commerce Customer Support Chatbot (RAG)

An AI-powered customer support chatbot built using **Retrieval Augmented Generation (RAG)** — the same technology behind enterprise chatbots at companies like Shopify, Zendesk, and Intercom. Ask any question about orders, shipping, returns, or payments and get instant, accurate answers grounded in real knowledge base data.

🔗 **[Live Demo → Click Here to Try It](https://customer-support-chatbot-nnwappm2k7zt6sxibwivukk.streamlit.app/)**

---

## 📌 Problem Statement

Traditional chatbots rely on rigid decision trees and often give wrong or generic answers. Large Language Models (LLMs) hallucinate and make things up. **RAG solves both problems** — it retrieves the exact relevant information first, then uses an LLM to generate a natural, accurate answer based only on that data.

This means:
- ✅ Answers are always grounded in real business data
- ✅ No hallucinations or made-up policies
- ✅ Easy to update — just change the knowledge base document

---

## 🧠 How RAG Works

```
User Question
     ↓
Search Knowledge Base (ChromaDB Vector Store)
     ↓
Retrieve Top 3 Most Relevant Chunks
     ↓
Send Question + Context to Gemini LLM
     ↓
Generate Accurate, Grounded Answer
```

---

## 🎯 Features

- 💬 Natural conversational interface with chat history
- 🔍 Semantic search — finds relevant info even if wording differs
- 🎯 7 sample question buttons for instant demo
- 🧹 Clear chat button to reset conversation
- 🔐 Secure API key input via sidebar
- 📚 Knowledge base covers: Shipping, Returns, Payments, Orders, Account & Loyalty

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LangChain (LCEL) | RAG pipeline framework |
| Google Gemini 1.5 Flash | LLM for answer generation |
| ChromaDB | Vector database for document storage |
| HuggingFace Embeddings | Free local text embeddings (all-MiniLM-L6-v2) |
| Streamlit | Web app & deployment |

---

## 🗂️ Knowledge Base Coverage

| Topic | Questions Covered |
|-------|------------------|
| 🚚 Shipping & Delivery | Timelines, costs, international, tracking |
| 🔄 Returns & Refunds | Policy, process, timelines, exchanges, damage |
| 💳 Orders & Payments | Payment methods, cancellation, security |
| 📦 Products | Stock, warranties, authenticity |
| 👤 Account & Loyalty | Sign up, password reset, rewards, promo codes |
| 📞 Support | Contact hours, channels, physical locations |

---

## 📁 Repository Structure

```
customer-support-chatbot/
│
├── app.py                    # Streamlit web app
├── rag_pipeline.py           # RAG pipeline (LCEL)
├── data/
│   └── ecommerce_faq.txt     # Knowledge base
├── requirements.txt          # Dependencies
└── README.md
```

---

## 🚀 Run Locally

1. Clone the repository:
```bash
git clone https://github.com/jabiyan/customer-support-chatbot.git
cd customer-support-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

4. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) and paste it in the sidebar.

---

## 💼 Business Value

This chatbot can be adapted for any business to:

- **Reduce support ticket volume** by 40-60% through instant self-service
- **Provide 24/7 support** without hiring additional agents
- **Scale instantly** — handles thousands of simultaneous conversations
- **Update easily** — change the FAQ text file, redeploy, done
- **Integrate anywhere** — e-commerce, SaaS, healthcare, finance

---

## 🔧 How to Customize for Your Business

1. Replace `data/ecommerce_faq.txt` with your own FAQ or policy documents
2. Update the system prompt in `rag_pipeline.py` to match your brand voice
3. Redeploy — the chatbot now answers questions about YOUR business

---

## 👤 Author

**Mustajab Hussain** — Data Scientist & ML Engineer

- 🔗 [LinkedIn](https://www.linkedin.com/in/mustajab-hussain-312475283/)
- 💻 [GitHub](https://github.com/jabiyan)
- 📧 [mustajabh015@gmail.com](mailto:mustajabh015@gmail.com)
- 💼 [Hire me on Upwork](https://www.upwork.com/freelancers/~011cf146030b8908fd)
