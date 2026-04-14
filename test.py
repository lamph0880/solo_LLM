import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def run_local_rag():
    # 1. 파일 경로 설정 (사용자가 제공한 파일명)
    pdf_files = [
        "국가연구개발사업 연구개발비 사용 기준.pdf",
        "국가연구개발혁신법.pdf",
        "국가 R&D 부적정 사례집.pdf"
    ]

    # 파일 존재 여부 확인
    for pdf in pdf_files:
        if not os.path.exists(pdf):
            print(f"Error: 파일을 찾을 수 없습니다 -> {pdf}")
            return

    print("1. PDF 문서 로딩 중...")
    documents = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        documents.append(loader.load())
    
    # 리스트 형태의 문서를 하나로 합침
    all_docs = []
    for doc_list in documents:
        all_docs.extend(doc_list)

    # 2. 텍스트 분할 (Chunking)
    # 문서를 의미 있는 단위로 자릅니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"   - 분할 완료: {len(splits)} 개의 청크 생성됨")

    # 3. 로컬 임베딩 모델 설정 (OpenAI 대신 사용)
    # 한국어 성능이 우수한 로컬 모델을 사용합니다.
    print("2. 임베 모델 로딩 중 (최초 실행 시 다운로드 시간이 소요될 수 있습니다)...")
    model_name = "jhgan/ko-sroberta-multitask"
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
        model_kwargs={'device': 'cuda'}
    )

    # 4. 벡터 저장소 생성 (Chroma DB - 로컬 저장)
    print("3. 벡터 데이터베이스 구축 중...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"  # 데이터를 로컬에 저장하여 재사용 가능하게 함
    )
    print("   - 벡터 저장소 구축 완료")

    # 5. LLM 설정 (Ollama의 gemma4:26b 사용)
    print("4. LLM(gemma4:26b) 연결 중...")
    llm = Ollama(
        model="gemma4:26b",
        base_url="http://localhost:11434", # 기본값
        timeout=900
    )

    # 6. RAG 프롬프트 템플릿 설정
    # 모델이 문서 내용에만 기반하여 답변하도록 지시합니다.
    template = """너는 국가연구개발사업 연구비 정산 전문가야.
    반드시 아래의 문맥(Context)만을 사용하여 답변해줘. 
    만약 문맥에 답이 없다면 "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답해주고,
    어떤 이유로 부적정한 사례인지 구체적으로 설명해줘.
    답변은 한국어로 친절하게 작성해줘.

    문맥: {context}

    질문: {question}

    도움이 되는 답변:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # 7. RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # 가장 유사한 5개 청크 참조
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # 8. 질의응답 실행 루프
    print("\n" + "="*50)
    print("RAG 시스템 준비 완료! 질문을 입력하세요 (종료하려면 'exit' 입력)")
    print("="*50)

    while True:
        query = input("\n질문: ")
        if query.lower() in ['exit', 'quit', '종료']:
            break
        
        print("답변 생성 중...")
        try:
            response = qa_chain.invoke(query)
            print(f"\n[답변]:\n{response['result']}")
        except Exception as e:
            print(f"에러 발생: {e}")

if __name__ == "__main__":
    run_local_rag() 