import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# 1. إعداد مسار الملف
PDF_PATH = "data/notes.pdf"

# 2. تحميل نموذج التضمين متعدد اللغات (Embeddings)
print("⏳ جاري تحميل نموذج التضمين...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 3. قراءة الملف وتقسيمه وتخزين المتجهات في الذاكرة
print("📄 جاري قراءة وتقسيم ملف الـ PDF...")
loader = PyMuPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"💾 جاري بناء قاعدة البيانات المتجهة من {len(chunks)} فقرة...")
vector_db = InMemoryVectorStore.from_documents(documents=chunks, embedding=embedding_model)

# 4. إعداد النموذج اللغوي وسلسلة الاسترجاع (Retrieve & Generate)
llm = Ollama(model="qwen2.5:1.5b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3})
)

# 5. دالة الواجهة التفاعلية (Gradio UI)
def ask_assistant(user_question):
    if not user_question.strip():
        return "الرجاء كتابة سؤال أولاً."
    response = qa_chain.invoke(user_question)
    return response["result"]

interface = gr.Interface(
    fn=ask_assistant,
    inputs=gr.Textbox(lines=2, placeholder="اكتب سؤالك هنا حول محتوى الملف..."),
    outputs=gr.Textbox(lines=6, label="الإجابة"),
    title="📚 RAG-Powered Document Assistant",
    description="اسأل أي سؤال وسيتم استخراج الإجابة بدقة من ملفاتك."
)

# 6. تشغيل التطبيق
if __name__ == "__main__":
    interface.launch()