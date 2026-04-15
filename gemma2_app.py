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
import easyocr
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# --- 설정 (Configuration) ---
PDF_FILES = [
    "data/국가 R&D 부적정 사례집.pdf",
    "data/국가연구개발사업 연구개발비 사용 기준.pdf",
    "data/국가연구개발혁신법.pdf"
]
DB_PATH = "./chroma_db"
MODEL_NAME = "gemma2:9b" # RTX 3070(8GB VRAM)에 최적화된 모델
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

# OCR 리더 초기화 (GPU 가속 권장)
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
except:
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

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
        
        try:
            sample_data = vectorstore.get(limit=5)
            for i in range(len(sample_data['documents'])):
                INIT_DEBUG_DATA["loading"].append({
                    "content": sample_data['documents'][i][:500] + "...",
                    "metadata": sample_data['metadatas'][i]
                })
                INIT_DEBUG_DATA["chunking"].append(sample_data['documents'][i][:500] + "...")
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
            
        for doc in documents[:5]:
            INIT_DEBUG_DATA["loading"].append({
                "content": doc.page_content[:500] + "...",
                "metadata": doc.metadata
            })
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        for s in splits[:5]:
            INIT_DEBUG_DATA["chunking"].append(s.page_content[:500] + "...")
            
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
        "제공된 문맥(Context)을 사용하여 질문에 답하세요. "
        "전달받은 문맥에 관련 정보가 있다면 그에 근거하여 답변하고, "
        "관련 규정이 없다면 규정에 없다고 답변하세요. "
        "항상 한국어로 정중하게 답변하세요."
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

@app.route('/upload', methods=['POST'])
def upload():
    global RAG_APP
    if RAG_APP is None:
        return jsonify({"error": "RAG 시스템이 초기화되지 않았습니다."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    filename = file.filename.lower()
    extracted_text = ""

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_bytes = file.read()
            results = reader.readtext(image_bytes)
            extracted_text = " ".join([res[1] for res in results])
        elif filename.endswith('.pdf'):
            temp_path = os.path.join("./", "temp_upload.pdf")
            file.save(temp_path)
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            extracted_text = " ".join([page.page_content for page in pages])
            os.remove(temp_path)
        else:
            return jsonify({"error": "지원하지 않는 파일 형식입니다."}), 400
        
        if not extracted_text.strip():
            return jsonify({"error": "텍스트 추출 실패"}), 400

        # 동적 쿼리 생성
        llm_for_query = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
        query_gen_prompt = (
            "당신은 행정 전문가입니다. 아래 텍스트를 분석하여 규정집에서 검색할 키워드 3개를 뽑아주세요.\n"
            f"텍스트: '{extracted_text}'\n"
            "출력 형식: 키워드1, 키워드2, 키워드3"
        )
        
        try:
            suggested_query = llm_for_query.invoke(query_gen_prompt).content.strip()
            retrieval_query = f"국가연구개발사업 {suggested_query}"
        except:
            retrieval_query = "국가연구개발사업 연구개발비 사용 기준"

        audit_instruction = (
            f"다음 데이터의 적정성을 분석해 주세요: '{extracted_text}'.\n"
            "검색된 규정을 근거로 [적정], [부적정], [확인필요] 판정 및 이유를 설명하세요."
        )

        debug_info = []
        debug_info.append({"node": "Query Generation", "status": "Success", "data": f"Keywords: {suggested_query}"})

        inputs = {"question": retrieval_query}
        for output in RAG_APP.stream(inputs):
            if "retrieve" in output:
                debug_info.append({"node": "Retrieve", "status": "Success", "data": output["retrieve"]["context"]})
        
        state = {"question": audit_instruction, "context": debug_info[1]["data"]}
        answer_result = generate(state)
        final_answer = answer_result["answer"]
        
        debug_info.append({"node": "Generate", "status": "Success", "data": "Analysis completed."})
        
        return jsonify({
            "extracted_text": extracted_text[:1000] + "...",
            "answer": final_answer,
            "debug": debug_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"시스템 초기화 중 (Model: {MODEL_NAME})...")
    RETRIEVER = load_and_index_documents()
    RAG_APP = build_graph()
    app.run(host='0.0.0.0', port=5000, debug=False)
