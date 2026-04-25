### 🎤 StudyMind AI – Intelligent Document-Based Learning Assistant 

An AI-powered study assistant built using 

- **Streamlit** 
- **LangChain** 
- **Groq LLM**

that helps students interact with PDF and PPTX documents.


---
## Table of Contents
- <a href="#Project Overview">Project Overview</a>
- <a href="#Features">Features</a>
- <a href="#Technologies Used">Technologies Used</a>
- <a href="#Project Structure">Project Structure</a>
- <a href="#Installation">Installation</a>
- <a href="#How its works">How its works</a>
- <a href="#How to Run">How to Run</a>
- <a href="#Future Improvements">Future Improvements</a>
- <a href="#Author">Author</a>

---
<h2><a class="anchor" id="Project Overview"></a>Project Overview</h2>

StudyMind AI is an AI-powered study assistant for smart learning

- Allows users to upload PDF and PPTX files.
- Enables users to chat with documents and get accurate answers
- Uses LangChain + embeddings + vector database for context-based retrieval
- Powered by Groq LLM (LLaMA 3) for fast and intelligent responses
- Automatically generates study notes from uploaded content
- Creates MCQ quizzes for self-assessment
- Generates flashcards for quick revision
- Provides document summaries for better understanding
- Built with Streamlit for a clean and interactive UI
- Supports downloading of notes, flashcards, and chat history

---
<h2><a class="anchor" id="Features"></a>Features</h2>

- Upload PDF & PPTX files
- Chat with your documents (context-aware answers)
- Auto-generate quizzes (MCQs)
- Create flashcards for revision
- Generate detailed study notes
- Get document summaries
- Download notes, flashcards, and chat history


---
<h2><a class="anchor" id="Technologies Used"></a>Technologies Used</h2>

- Frontend/UI: Streamlit
- LLM: Groq (LLaMA 3.3 70B)
- Framework: LangChain
- Embeddings: HuggingFace (MiniLM)
- Vector DB: ChromaDB

--- 
<h2><a class="anchor" id="Project Structure"></a>Project Structure</h2>

├── app.py                
├── requirements.txt    
└── README.md   

---
<h2><a class="anchor" id="Installation"></a>Installation</h2>

- Clone the repository
    git clone https://github.com/your-username/studymind-ai.git
    cd studymind-ai

- Install dependencies
   pip install -r requirements.txt

- Set API Key
   export GROQ_API_KEY="your_api_key_here"

- (Windows)
    set GROQ_API_KEY=your_api_key_here

--- 
<h2><a class="anchor" id="How its works"></a>How its works</h2>

- Upload PDF or PPTX files
- Files are split into chunks
- Embeddings are created using HuggingFace
- Stored in Chroma vector database
- Queries are answered using Groq LLM with retrieval

---

--- 
<h2><a class="anchor" id="How to Run"></a>How to Run</h2>

- streamlit run app.py


--- 
<h2><a class="anchor" id="Future Improvements"></a>Future Improvements</h2>

- Add multi-language support
- Improve UI animations
- Add voice interaction
- Export quiz results as PDF

----
## Author

Ram Krishna
- Email: ramkrishna000888@gmail.com
- Linkeddin: https://www.linkedin.com/in/ramkrishna000/

