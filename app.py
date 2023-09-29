from flask import Flask, request, redirect, url_for, render_template
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import openai
import PyPDF2
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy.exc import SQLAlchemyError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI as OpenAILLM

app = Flask(__name__)
app.config['SECRET_KEY'] = '\xe7\xfa\xf9U(`\xf4X\xe25,e-\x1c\x92V' # replace this with your secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://flaskgpt3webapp:R5\'u&hg?z2hh%]ra@127.0.0.1:3306/flask_gpt_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
openai.api_key = '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'

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class File(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'))  # chatid field for file

    user = db.relationship('User', backref=db.backref('files', lazy=True))
    chat = db.relationship('Chat', backref=db.backref('files', lazy=True))

class Chat(db.Model):
    __tablename__ = 'chats'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    user = db.relationship('User', backref=db.backref('chats', lazy=True))

    def search_documents(self, query):
        # load text from all files of the user
        raw_text = ""
        for file in self.user.files:
            raw_text += extract_text_from_pdf(file.filename)

        # split the text into smaller chunks
        text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1000, chunk_overlap  = 200, length_function = len)
        texts = text_splitter.split_text(raw_text)

        # generate embeddings for the texts
        embeddings = OpenAIEmbeddings().generate_embeddings(texts)

        # create a FAISS index from the texts and embeddings
        index = FAISS(texts, embeddings)

        # search for documents similar to the query
        similar_documents = index.search_documents(query)

        # load a question-answering (QA) chain
        qa_chain = load_qa_chain(OpenAILLM())

        # get a response from the GPT-3 model by running the QA chain with the found similar documents and the query
        response = qa_chain.run_chain(similar_documents, query)

        return response

def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_number in range(reader.getNumPages()):
        page = reader.getPage(page_number)
        text += page.extract_text()
    pdf_file.close()
    return text

def get_gpt3_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

@app.route('/new_chat', methods=['GET', 'POST'])
@login_required
def new_chat():
    chat = Chat(user_id=current_user.id, text="")
    db.session.add(chat)
    db.session.commit()
    return redirect(url_for('chatbot', chat_id=chat.id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        if user is not None:
            # flash an error message
            return redirect(url_for('register'))
        user = User(username=username)
        user.set_password(request.form['password'])
        db.session.add(user)
        try:
            db.session.commit()
            return redirect(url_for('login'))
        except SQLAlchemyError:
            db.session.rollback()
            # flash an error message
    return render_template('register.html')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user is None or not user.check_password(password):
            # flash an error message
            return redirect(url_for('login'))
        login_user(user)
        return redirect(url_for('new_chat'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/chatbot/<int:chat_id>', methods=['GET', 'POST'])
@login_required
def chatbot(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return "Chat not found", 404
    if request.method == 'POST':
        query = request.form['query']
        if chat.files:
            response = chat.search_documents(query)
        else:
            response = get_gpt3_response(query)
        chat.text += "\nUser: " + query + "\nBot: " + response
        db.session.commit()
        return render_template('chatbot.html', chat=chat)
    return render_template('chatbot.html', chat=chat)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            file_path = os.path.join('uploads', file.filename)  # you need to have an uploads directory
            file.save(file_path)
            new_chat = Chat(user_id=current_user.id)
            db.session.add(new_chat)
            db.session.commit()
            new_file = File(filename=file.filename, chat_id=new_chat.id)
            db.session.add(new_file)
            db.session.commit()
            return redirect(url_for('chatbot', chat_id=new_chat.id))
    return render_template('upload.html')

@app.route('/')
def home():
    return "Server is running!"

if __name__ == "__main__":
    app.run(debug=True)
