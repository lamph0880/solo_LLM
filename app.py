import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from typing import List, TypedDict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

app = Flask(__name__)
CORS(app)

# --- 설정 (Configuration) ---
PDF_FILES = [
    "국가 R&D 부적정 사례집.pdf",
    "국가연구개발사업 연구개발비 사용 기준.pdf",
    "국가연구개발혁신법.pdf"
]
DB_PATH = "./chroma_db"
MODEL_NAME = "gemma4:26b" 
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask"
OLLAMA_BASE_URL = "http://172.27.48.1:11434"

# 전역 변수
RETRIEVER = None
RAG_APP = None
INIT_DEBUG_DATA = {
    "loading": [],
    "chunking": [],
    "embedding": []
}

class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str

def load_and_index_documents():
    """문서를 로드하고 Chroma DB에 저장하거나 로드합니다."""
    global INIT_DEBUG_DATA
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs=encode_kwargs,
        model_kwargs={'device': 'cuda'} 
    )
    
    vectorstore = None
    if os.path.exists(DB_PATH):
        print(f"기존 index를 로드합니다: {DB_PATH}")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        # 시각화를 위해 기존 DB에서 샘플 추출
        try:
            sample_data = vectorstore.get(limit=5)
            for i in range(len(sample_data['documents'])):
                INIT_DEBUG_DATA["loading"].append({
                    "content": sample_data['documents'][i][:500] + "...",
                    "metadata": sample_data['metadatas'][i]
                })
                INIT_DEBUG_DATA["chunking"].append(sample_data['documents'][i][:500] + "...")
                
                # 임베딩 샘플 생성
                vec = embeddings.embed_query(sample_data['documents'][i])
                INIT_DEBUG_DATA["embedding"].append({
                    "dimension": len(vec),
                    "values": vec[:5]
                })
        except Exception as e:
            print(f"샘플 추출 중 오류: {e}")
    else:
        print("신규 index를 생성합니다...")
        documents = []
        for path in PDF_FILES:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
        
        if not documents:
            return None
            
        # Step 1: Loading samples
        for doc in documents[:5]:
            INIT_DEBUG_DATA["loading"].append({
                "content": doc.page_content[:500] + "...",
                "metadata": doc.metadata
            })
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # Step 2: Chunking samples
        for s in splits[:5]:
            INIT_DEBUG_DATA["chunking"].append(s.page_content[:500] + "...")
            
        # Step 3: Embedding samples
        for s in splits[:5]:
            vec = embeddings.embed_query(s.page_content)
            INIT_DEBUG_DATA["embedding"].append({
                "dimension": len(vec),
                "values": vec[:5]
            })
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=DB_PATH
        )
    
    return vectorstore.as_retriever()

def retrieve(state: GraphState):
    question = state["question"]
    documents = RETRIEVER.invoke(question)
    context = [doc.page_content for doc in documents]
    return {"context": context}

def generate(state: GraphState):
    question = state["question"]
    context = "\n\n".join(state["context"])
    
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    
    system_prompt = (
        "당신은 국가 연구개발(R&D) 관련 법령 및 규정 전문가입니다. "
        "제공된 문맥(Context)만을 사용하여 질문에 답하세요. "
        "전달받은 문맥에 관련 정보가 없다면 문맥에 없다고 정중히 답변하세요. "
        "답을 모른다면 모른다고 답하고, 문맥에 없는 내용은 추측하지 마세요. "
        "항상 한국어로 답변하세요."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
    ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug/init', methods=['GET'])
def get_init_debug():
    return jsonify(INIT_DEBUG_DATA)

@app.route('/ask', methods=['POST'])
def ask():
    global RAG_APP
    if RAG_APP is None:
        return jsonify({"error": "RAG 시스템이 초기화되지 않았습니다."}), 500

    data = request.json
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "질문이 없습니다."}), 400

    debug_info = []
    final_answer = ""
    
    inputs = {"question": question}
    try:
        for output in RAG_APP.stream(inputs):
            for key, value in output.items():
                if key == "retrieve":
                    debug_info.append({
                        "node": "Retrieve",
                        "status": "Success",
                        "data": value["context"]
                    })
                elif key == "generate":
                    final_answer = value["answer"]
                    debug_info.append({
                        "node": "Generate",
                        "status": "Success",
                        "data": "Answer generated successfully."
                    })
        
        return jsonify({
            "answer": final_answer,
            "debug": debug_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("시스템 초기화 중...")
    RETRIEVER = load_and_index_documents()
    RAG_APP = build_graph()
    app.run(host='0.0.0.0', port=5000, debug=False)
