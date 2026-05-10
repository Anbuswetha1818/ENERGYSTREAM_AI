import sqlite3
import os

DB_PATH = 'users.db'

def check_users():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} does not exist.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM users')
        users = cursor.fetchall()
        if not users:
            print("No users found in the database.")
        else:
            for user in users:
                print(f"ID: {user['id']}, Username: {user['username']}, Email: {user['email']}")
    except sqlite3.OperationalError as e:
        print(f"Error querying database: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    check_users()
