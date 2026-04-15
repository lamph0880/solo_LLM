import os
from typing import List, TypedDict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 설정 (Configuration) ---
PDF_FILES = [
    "국가 R&D 부적정 사례집.pdf",
    "국가연구개발사업 연구개발비 사용 기준.pdf",
    "국가연구개발혁신법.pdf"
]
DB_PATH = "./chroma_db"
# Ollama 모델 설정
# 사용자의 환경에 맞게 모델명을 수정하세요 (예: gemma2, gemma2:27b, gemma2:9b 등)
MODEL_NAME = "gemma4:26b" 
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask" # 한국어 임베딩 최적화 모델
OLLAMA_BASE_URL = "http://172.27.48.1:11434"

class GraphState(TypedDict):
    question: str
    context: List[str]
    answer: str

def load_and_index_documents():
    """문서를 로드하고 Chroma DB에 저장하거나 로드합니다."""
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs=encode_kwargs,
        model_kwargs={'device': 'cuda'} # GPU 사용 시 'cuda'로 변경 가능
    )
    
    if os.path.exists(DB_PATH):
        print(f"기존 index를 로드합니다: {DB_PATH}")
        vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("신규 index를 생성합니다...")
        documents = []
        for path in PDF_FILES:
            if os.path.exists(path):
                print(f"Loading {path}...")
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
            else:
                print(f"Warning: {path} 파일을 찾을 수 없습니다.")
        
        if not documents:
            raise ValueError("로드할 PDF 파일이 없습니다.")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        print("Index 생성 완료.")
    
    return vectorstore.as_retriever()

# --- LangGraph 노드 정의 ---

# 전역 retriever 변수 (또는 별도의 클래스로 관리 가능)
# LangGraph state에 retriever를 넣는 것은 권장되지 않음 (직렬화 문제 때문)
RETRIEVER = None

def retrieve(state: GraphState):
    print("---RETRIEVING---")
    question = state["question"]
    documents = RETRIEVER.invoke(question)
    context = [doc.page_content for doc in documents]
    return {"context": context}

def generate(state: GraphState):
    print("---GENERATING---")
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

# --- 그래프 구축 ---
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    try:
        RETRIEVER = load_and_index_documents()
        app = build_graph()
        
        print("\nRAG 시스템 준비 완료! 질문을 입력하세요 (종료하려면 'exit' 또는 'quit' 입력)")
        
        while True:
            user_input = input("\n[User]: ")
            if user_input.lower() in ['exit', 'quit']:
                print("시스템을 종료합니다.")
                break
            
            if not user_input.strip():
                continue
                
            inputs = {"question": user_input}
            
            try:
                # 스트리밍 출력
                for output in app.stream(inputs):
                    for key, value in output.items():
                        if key == "generate":
                            print(f"\n[Assistant]: {value['answer']}")
            except Exception as e:
                print(f"Error during execution: {e}")
                
    except Exception as e:
        print(f"Initialization Error: {e}")