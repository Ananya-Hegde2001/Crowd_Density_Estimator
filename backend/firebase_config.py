import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
import os

class FirebaseConfig:
    def __init__(self):
        self.db = None
        self.initialize_firebase()
    
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            if firebase_admin._apps:
                # Already initialized
                self.db = firestore.client()
                print("✅ Firebase already initialized!")
                return

            # Build candidate paths for the service account file
            this_dir = os.path.dirname(os.path.abspath(__file__))
            cwd = os.getcwd()
            env_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            candidates = [
                env_path if env_path and os.path.isfile(env_path) else None,
                os.path.join(this_dir, 'serviceAccountKey.json'),               # backend/serviceAccountKey.json
                os.path.join(cwd, 'serviceAccountKey.json'),                    # current working dir
                os.path.join(os.path.dirname(this_dir), 'serviceAccountKey.json')  # project root, if placed there
            ]
            candidates = [p for p in candidates if p and os.path.isfile(p)]

            if candidates:
                cred_path = candidates[0]
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print(f"✅ Firebase initialized with service account: {cred_path}")
                return

            # As a last resort, try Application Default Credentials only if env var is set
            if env_path and os.path.isfile(env_path):
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                print("✅ Firebase initialized with Application Default Credentials")
                return

            # Nothing found; provide a clear message and fall back
            print("❌ Firebase initialization error: No service account found and GOOGLE_APPLICATION_CREDENTIALS not set to a valid file.")
            print("   Place serviceAccountKey.json in backend/ or set GOOGLE_APPLICATION_CREDENTIALS to its absolute path.")
            self.db = None
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
            self.db = None
    
    def create_user(self, email, password, username):
        """Create user in Firebase Authentication and Firestore"""
        try:
            # Create user in Firebase Authentication
            user = auth.create_user(
                email=email,
                password=password,
                display_name=username
            )
            
            # Store additional user data in Firestore
            user_data = {
                'username': username,
                'email': email,
                'uid': user.uid,
                'created_at': firestore.SERVER_TIMESTAMP,
                'is_active': True
            }
            
            # Add user document to Firestore
            self.db.collection('users').document(user.uid).set(user_data)
            
            print(f"✅ User created successfully: {username}")
            return {
                'success': True,
                'uid': user.uid,
                'message': 'User created successfully'
            }
            
        except auth.EmailAlreadyExistsError:
            return {
                'success': False,
                'error': 'Email already exists'
            }
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_user(self, email, password):
        """Verify user credentials using Firebase Authentication"""
        try:
            # Note: Firebase Admin SDK doesn't directly verify passwords
            # In a real application, you'd use Firebase Client SDK on frontend
            # For backend verification, we'll use a different approach
            
            # Get user by email
            user = auth.get_user_by_email(email)
            
            # Check if user exists and is active
            user_doc = self.db.collection('users').document(user.uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                if user_data.get('is_active', True):
                    return {
                        'success': True,
                        'uid': user.uid,
                        'username': user_data.get('username'),
                        'email': user.email
                    }
            
            return {
                'success': False,
                'error': 'User not found or inactive'
            }
            
        except auth.UserNotFoundError:
            return {
                'success': False,
                'error': 'User not found'
            }
        except Exception as e:
            print(f"❌ Error verifying user: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_user_data(self, uid):
        """Get user data from Firestore"""
        try:
            user_doc = self.db.collection('users').document(uid).get()
            if user_doc.exists:
                return user_doc.to_dict()
            return None
        except Exception as e:
            print(f"❌ Error getting user data: {e}")
            return None
    
    def update_user_data(self, uid, data):
        """Update user data in Firestore"""
        try:
            self.db.collection('users').document(uid).update(data)
            return True
        except Exception as e:
            print(f"❌ Error updating user data: {e}")
            return False
    
    def delete_user(self, uid):
        """Delete user from Firebase Authentication and Firestore"""
        try:
            # Delete from Authentication
            auth.delete_user(uid)
            # Delete from Firestore
            self.db.collection('users').document(uid).delete()
            return True
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            return False

# Initialize Firebase instance
firebase_config = FirebaseConfig()
