from flask import Blueprint, render_template, redirect, url_for, request
from .forms import LoginForm, RegistrationForm # Assuming you have WTForms
from .models import User
from flask_login import login_user, logout_user, login_required

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
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

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
    
# TODO Add other authentication routes...
