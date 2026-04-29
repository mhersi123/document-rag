# Basic Document/PDF Retrieval Augmentation Generation (RAG) Application
## Information
* This application uses Python libraries such as FastAPI, Inngest, OpenAI, LlamaIndex, Qdrant, and more to provide a user the simple capability of practicing RAG with PDF documents.

* Through a Streamlit frontend interface, users can upload PDF documents, and ask relevant questions while also specifying the number of chunks (recommended 5-10 chunks for medium sized documents)

* All functionality, monitoring, and configuration is maintained through Inngest, which provides an event-driven, durable workflow orchestration platform that allows developers to define and run background jobs, scheduled tasks, and complex, multi-step workflows in code without managing infrastructure.

## Instructions to run locally
1. Clone the repository locally
2. Create a `.env` file and variable `OPENAI_API_KEY` to hold an API key from OpenAI, accesible here: https://platform.openai.com/
3. Pre-requisites installed
    a. Python 3
    b. pip3 
    c. uv CLI
    d. Docker Desktop
* Note: To run the application, the FastAPI backend, Qdrant container, inngest server, and frontend must be running to communicate
4. Run the command to get dependencies: `uv add fastapi inngest llama-index-core llama-index-readers-file python-dotenv qdrant-client uvicorn streamlit openai langchain-text-splitters`
5. Run the FastAPI server `uv run uvicorn main:app`
6. Run the Qdrant Image (Docker must be running): `docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/vector-storage:/qdrant/storage" qdrant/qdrant`
7. To view the inngest server UI, access the URL specified (default is http://localhost:8288/)
8. To run the frontend application: `uv run streamlit run .\streamlit_app.py`