import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import certifi

# Dummy user credentials for demonstration
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

def rerun():
    """Rerun the Streamlit app (works across versions)"""
    if hasattr(st, "rerun"):  # Streamlit >= 1.27
        st.rerun()
    else:  # Older versions
        st.experimental_rerun()

def get_mongodb_client(uri):
    """Create and return a MongoDB client with CA file"""
    return MongoClient(uri, tlsCAFile=certifi.where())

def save_user_config(username, openai_key, mongo_uri):
    """Save user OpenAI key and Mongo URI in MongoDB for future use"""
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise_app']  # database name
        collection = db['user_configs']
        # Upsert user config by username
        collection.update_one(
            {"username": username},
            {"$set": {"openai_key": openai_key, "mongo_uri": mongo_uri}},
            upsert=True
        )
        client.close()
        return True, "Configuration saved successfully."
    except PyMongoError as e:
        return False, f"Failed to save config: {e}"

def get_user_config(username, mongo_uri):
    """Retrieve user config from MongoDB"""
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise_app']
        collection = db['user_configs']
        config = collection.find_one({"username": username}, {"_id": 0})
        client.close()
        return config
    except PyMongoError as e:
        return {"error": str(e)}

def mask_key(key: str) -> str:
    """Mask sensitive keys, keep only first 4 and last 4 chars"""
    if not key or len(key) < 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"

def login():
    st.title("Welcome to Enterprise Streamlit App")
    st.subheader("Please log in to continue")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Logged in as {username}")
            rerun()
        else:
            st.error("Invalid username or password")

def logout():
    st.write("---")
    if st.button("Logout"):
        for key in ["logged_in", "username"]:
            if key in st.session_state:
                del st.session_state[key]
        rerun()

def main_app():
    st.title(f"Hello, {st.session_state['username']}! You are logged in.")
    st.write("This will be the main app content.")
    logout()

    # Sidebar to collect OpenAI key and Mongo URI
    st.sidebar.header("Configure API Keys")

    with st.sidebar.form(key="api_config_form"):
        openai_key = st.text_input("OpenAI API Key", type="password")
        mongo_uri = st.text_input("MongoDB Atlas URI", type="password")
        submit = st.form_submit_button(label="Save Configuration")

        if submit:
            if not openai_key or not mongo_uri:
                st.sidebar.error("Both OpenAI key and Mongo URI are required.")
            else:
                success, message = save_user_config(
                    st.session_state['username'], openai_key, mongo_uri
                )
                if success:
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)

    # Show saved config from MongoDB
    st.sidebar.subheader("Your Saved Config")
    if "username" in st.session_state and st.session_state["username"] and mongo_uri:
        saved_config = get_user_config(st.session_state['username'], mongo_uri)
        if saved_config:
            # Mask OpenAI key before displaying
            if "openai_key" in saved_config:
                saved_config["openai_key"] = mask_key(saved_config["openai_key"])
            st.sidebar.json(saved_config)
        else:
            st.sidebar.info("No saved config found.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['username'] = None

    if st.session_state['logged_in']:
        main_app()
    else:
        login()

if __name__ == "__main__":
    main()
