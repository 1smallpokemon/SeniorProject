

class File(db.Model):
    __tablename__ = 'files'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'))  # chatid field for file

    user = db.relationship('User', backref=db.backref('files', lazy=True))
    chat = db.relationship('Chat', backref=db.backref('files', lazy=True))