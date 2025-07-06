from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

# Cấu hình
model_file = "model/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
vector_dp_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        temperature=0.01,
        config={'gpu_layers': 0},
        max_new_tokens=128,  
        context_length=512
    )
    return llm

# Tạo prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    return prompt

# Tạo pipeline chain (thay cho LLMChain)
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever =db.as_retriever(search_kwargs = {"k":1}),
        return_source_documents = False,
        chain_type_kwargs={'prompt':prompt}
    )
    return llm_chain

def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file = "model/all-minilm-l6-v2-q4_0.gguf") 
    db = FAISS.load_local(vector_dp_path, embedding_model,allow_dangerous_deserialization=True)
    return db

db = read_vector_db()
llm = load_llm(model_file)
# Mẫu prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

# Khởi tạo các thành phần
prompt = creat_prompt(template)
llm_chain  =create_qa_chain(prompt, llm, db)

# Chạy thử chain
question = "Khoa công nghệ thông tin thành lập năm nào ?"
response = llm_chain.invoke({"query": question})
print(response)
