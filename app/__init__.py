from flask import Flask
from .models import db, migrate
from .auth.routes import auth_bp
from .main.routes import main_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    migrate.init_app(app, db)

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    return app
