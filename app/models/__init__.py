# app/main/__init__.py
from flask import Blueprint

models = Blueprint('models', __name__)

from . import *
