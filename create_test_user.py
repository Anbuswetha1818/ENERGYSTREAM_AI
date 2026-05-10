from database import create_user
import os

if __name__ == '__main__':
    username = 'testuser'
    email = 'test@example.com'
    password = 'password123'
    if create_user(username, email, password):
        print(f"User {username} created successfully!")
    else:
        print(f"Failed to create user {username} (it might already exist).")
