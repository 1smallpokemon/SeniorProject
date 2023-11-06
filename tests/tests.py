# tests/test_user_model.py
import pytest
from app.models import User

def test_new_user():
    """
    GIVEN a User model
    WHEN a new User is created
    THEN check the email, hashed_password, authenticated, and active fields are defined correctly
    """
    user = User(username='testuser', email='testuser@example.com')
    user.set_password('FlaskIsAwesome')
    assert user.username == 'testuser'
    assert user.email == 'testuser@example.com'
    assert user.check_password('FlaskIsAwesome')
    assert not user.check_password('NotThePassword')
