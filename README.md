# dropdown-matching (Langchain + Ollama + FAISS)

This project helps match figma-style dropdown headers (LHS) to the most relevant field names in a JSON dataset (RHS) using local embeddings from Ollama's "llama3.2" model.

---

## Features
- match any LHS field to the top 3 RHS keys
- powered by Langchain, FAISS, and Ollama
- Flask web UI and REST API
- api/map endpoint for flexible inputs
- runs locally

## Setup Instructions:

  1. Clone the Repository
  ''' bash
git clone https://github.com/rohitkrishna20/dropdown-matching.git
cd dropdown-matching
- create a virtual environment: 

  python3 -m venv venv
  source venv/bin/activate

  2. Install python dependencies
     - pip install -r requirements.txt
       - flask
       - langchain
       - langchain-community
       - langchain-ollama
       - FAISS CPU

  3. Install Ollama Locally and Run
     - ollama pull llama3.2
     - ollama run llama3.2
    
  4. Start the App
     - python app.py and access at http://localhost:5000
    
## Using Postman to Test

- match all default headers: http://127.0.0.1:5000/api/match-all (no body required)
- match one header: https//127.0.0.1:5000/api/match   (body):
  
  {
    "header": "Account"
  }

  
