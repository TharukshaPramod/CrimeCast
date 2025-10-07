import sqlite3
import bcrypt
from datetime import datetime
import streamlit as st

class UserDatabase:
    def __init__(self, db_path="auth/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                phone TEXT,
                address TEXT,
                profile_picture BLOB,
                role TEXT DEFAULT 'user',
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # User sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_token TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Access logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT,
                page TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create default admin user if not exists
        admin_email = "admin@crimecast.com"
        cursor.execute("SELECT * FROM users WHERE email = ?", (admin_email,))
        if not cursor.fetchone():
            hashed_password = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
                INSERT INTO users (email, password, name, phone, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (admin_email, hashed_password, "System Admin", "+1234567890", "admin"))
        
        conn.commit()
        conn.close()
    
    def create_user(self, email, password, name, phone=None, address=None, role="user"):
        """Create a new user"""
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (email, password, name, phone, address, role)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (email, hashed_password, name, phone, address, role))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.log_action(user_id, "USER_SIGNUP", f"New user registered: {email}")
            return True, "User created successfully"
        except sqlite3.IntegrityError:
            return False, "Email already exists"
        except Exception as e:
            return False, f"Error creating user: {str(e)}"
    
    def authenticate_user(self, email, password):
        """Authenticate user with email and password"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, password, name, phone, address, role, profile_picture, 
                       created_at, last_login
                FROM users WHERE email = ? AND is_active = 1
            ''', (email,))
            
            user = cursor.fetchone()
            conn.close()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
                # Update last login
                self.update_last_login(user[0])
                self.log_action(user[0], "USER_LOGIN", "User logged in")
                return True, {
                    'id': user[0],
                    'email': user[1],
                    'name': user[3],
                    'phone': user[4],
                    'address': user[5],
                    'role': user[6],
                    'profile_picture': user[7],
                    'created_at': user[8],
                    'last_login': user[9]
                }
            else:
                return False, "Invalid email or password"
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(), user_id))
        conn.commit()
        conn.close()
    
    def log_action(self, user_id, action, details):
        """Log user actions - simplified version"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Simplified logging without page detection
            cursor.execute('''
                INSERT INTO access_logs (user_id, action, page, ip_address)
                VALUES (?, ?, ?, ?)
            ''', (user_id, action, "N/A", "127.0.0.1"))
            
            conn.commit()
            conn.close()
        except Exception as e:
            # Silently fail logging to avoid breaking the main functionality
            print(f"Logging error: {e}")
    
    def get_all_users(self):
        """Get all users (admin only)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, email, name, phone, address, role, is_active, created_at, last_login
                FROM users ORDER BY created_at DESC
            ''')
            users = cursor.fetchall()
            conn.close()
            return users
        except Exception as e:
            print(f"Error getting users: {e}")
            return []
    
    def update_user(self, user_id, name=None, phone=None, address=None, profile_picture=None):
        """Update user profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            update_fields = []
            params = []
            
            if name is not None:
                update_fields.append("name = ?")
                params.append(name)
            if phone is not None:
                update_fields.append("phone = ?")
                params.append(phone)
            if address is not None:
                update_fields.append("address = ?")
                params.append(address)
            if profile_picture is not None:
                update_fields.append("profile_picture = ?")
                params.append(profile_picture)
            
            if update_fields:
                params.append(user_id)
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, params)
                
                conn.commit()
                conn.close()
                self.log_action(user_id, "PROFILE_UPDATE", "User updated profile")
                return True, "Profile updated successfully"
            else:
                return False, "No fields to update"
        except Exception as e:
            return False, f"Error updating profile: {str(e)}"
    
    def delete_user(self, user_id, admin_id):
        """Delete user (admin only)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()
            self.log_action(admin_id, "USER_DELETED", f"Deleted user ID: {user_id}")
            return True, "User deleted successfully"
        except Exception as e:
            return False, f"Error deleting user: {str(e)}"

    def toggle_user_active_status(self, user_id, is_active):
        """Toggle user active status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET is_active = ? WHERE id = ?
            ''', (is_active, user_id))
            conn.commit()
            conn.close()
            
            status_text = "activated" if is_active else "deactivated"
            self.log_action(user_id, "USER_STATUS_CHANGE", f"User {status_text}")
            return True, f"User {status_text} successfully"
        except Exception as e:
            return False, f"Error updating user status: {str(e)}"
    
    def change_user_password(self, user_id, new_password):
        """Change user password"""
        try:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET password = ? WHERE id = ?
            ''', (hashed_password, user_id))
            conn.commit()
            conn.close()
            
            self.log_action(user_id, "PASSWORD_CHANGE", "User password changed")
            return True, "Password changed successfully"
        except Exception as e:
            return False, f"Error changing password: {str(e)}"
    
    def get_access_logs(self, limit=100):
        """Get access logs from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT al.id, al.user_id, al.action, al.page, al.timestamp, al.ip_address, u.email
                FROM access_logs al
                LEFT JOIN users u ON al.user_id = u.id
                ORDER BY al.timestamp DESC
                LIMIT ?
            ''', (limit,))
            logs = cursor.fetchall()
            conn.close()
            return logs
        except Exception as e:
            print(f"Error getting access logs: {e}")
            return []

# Global database instance
user_db = UserDatabase()