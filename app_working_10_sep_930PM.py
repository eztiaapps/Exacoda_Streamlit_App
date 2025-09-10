import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import certifi
from datetime import datetime
from openai import OpenAI
import json
import base64
import os

# --- App Branding ---
def exacoda_branding():
    st.markdown("""
        <style>
        .exacoda-brand {
            position: fixed;
            top: 15px;
            left: 20px;
            z-index: 9999;
            font-size: 3.5rem;
            font-weight: 900;
            color: #f07e90;
            letter-spacing: 3px;
            font-family: 'Segoe UI', Arial, sans-serif;
            user-select: none;
            pointer-events: none;
            text-shadow: 0 1px 10px rgba(255, 0, 0, 0.5);
        }
        </style>
        <div class="exacoda-brand">ExaCoda</div>
    """, unsafe_allow_html=True)


# --- Utility Methods (as provided) ---
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("Please refresh the page.")

def get_mongodb_client(uri):
    return MongoClient(uri, tls=True, tlsCAFile=certifi.where())

def save_user_config(username, openai_key, mongo_uri):
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise']
        col = db['configs']
        col.update_one({"username": username}, {"$set": {"openai_key": openai_key, "mongo_uri": mongo_uri}}, upsert=True)
        client.close()
        return True, "Configuration saved successfully."
    except PyMongoError as e:
        return False, str(e)

def get_user_config(username, mongo_uri):
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise']
        col = db['configs']
        config = col.find_one({"username": username}, {"_id": 0})
        client.close()
        return config or {}
    except PyMongoError as e:
        return {"error": str(e)}

def mask_key(key):
    if not key or len(key) < 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"

def list_projects(username, mongo_uri):
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise']
        col = db['projects']
        projects = list(col.find({"username": username}))
        client.close()
        return projects
    except PyMongoError as e:
        st.error(f"Database error: {e}")
        return []

def save_project(username, mongo_uri, title, ptype):
    try:
        client = get_mongodb_client(mongo_uri)
        db = client['enterprise']
        col = db['projects']
        project = {
            "username": username,
            "title": title,
            "type": ptype,
            "created_at": datetime.utcnow(),
            "settings": {},
            "files": [],
            "prompts": {},
            "responses": {},
            "business_processes": [],
            "test_scenarios": []
        }
        res = col.insert_one(project)
        client.close()
        return res.inserted_id
    except PyMongoError as e:
        st.error(f"Database error: {e}")
        return None

def call_llm(api_key, user_prompt, system_prompt=""):
    client = OpenAI(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

def set_login_background(image_path="./assets/hq720.jpg"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background: linear-gradient(120deg, rgba(30,40,56,0.93) 0%, rgba(80,182,230,0.16) 100%), 
                            url("data:image/jpg;base64,{encoded}") no-repeat center center fixed !important;
                background-size: cover !important;
            }}
            </style>
        """, unsafe_allow_html=True)

def login():
    set_login_background("./assets/robotbg.jpg")
    col1, col2, col3 = st.columns([2, 5, 2])
    with col2:
        st.markdown('<div class="login-card-custom">', unsafe_allow_html=True)
        st.markdown('<div class="login-title-custom">Welcome</div>', unsafe_allow_html=True)
        username = st.text_input("Username", key="user", placeholder="Enter your username")
        password = st.text_input("Password", key="pass", type="password", placeholder="Enter your password")
        st.markdown(
            '<div style="font-size:0.98em; margin-bottom:1.2em; color:#aaa; text-align:right; cursor:pointer; text-decoration:underline;">Forgot password?</div>',
            unsafe_allow_html=True
        )
        login_clicked = st.button("Log in")
        st.markdown('<div class="signup-link">Sign up</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if login_clicked:
            if (username.strip() == "admin" and password == "admin123") or \
               (username.strip() == "user" and password == "user123"):
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                rerun()
            else:
                st.error("Invalid username or password")

def logout():
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        keys_to_clear = [
            'logged_in', 'username', 'temp_mongo_uri', 'selected_project_id',
            'uploaded_files', 'user_prompt', 'system_prompt',
            'llm_response', 'business_processes', 'additional_prompt'
        ]
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        rerun()

def project_card(proj, key):
    return st.button(f"{proj['title']} ({proj['type']})", key=key)


# --- Main App ---
exacoda_branding()

# Always fetch Mongo URI from secrets
if "mongo_uri" not in st.secrets:
    st.error("Missing mongo_uri in secrets!")
else:
    MONGO_URI = st.secrets["mongo_uri"]

    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        login()
    else:
        logout()
        username = st.session_state["username"]

        # --- Sidebar navigation (simplified) ---
        tabs = ["Home", "Overview", "Conflicts", "Project Dashboard", "Config"]
        st.sidebar.title("Navigation")
        tab = st.sidebar.radio("Go to", tabs, key="selected_tab")

        if tab == "Home":
            st.header("Home")
            st.write("Welcome! Use the navigation to get started.")

        elif tab == "Overview":
            st.header("Overview")
            st.write("Summary and analytics go here.")

        elif tab == "Conflicts":
            st.header("Conflicts")
            st.write("Display and resolve project or document conflicts here.")

        elif tab == "Project Dashboard":
            st.header("Project Dashboard")

            # --- Create new Project ---
            st.subheader("Create a New Project")
            with st.form("create_proj_form"):
                project_title = st.text_input("Project Title", key="nproj_title")
                project_type = st.selectbox("Type", ["App", "ML Model", "Data Pipeline", "Custom"], key="ptype")
                create_pressed = st.form_submit_button("Create")
            if create_pressed and project_title:
                result = save_project(username, MONGO_URI, project_title, project_type)
                if result:
                    st.success("Project created!")
                else:
                    st.error("Could not create project. See logs.")

            # --- List Projects Live From MongoDB ---
            st.subheader("Your Projects")
            projs = list_projects(username, MONGO_URI)
            if projs:
                for p in projs:
                    if project_card(p, key=f"proj_{p['_id']}"):
                        st.session_state.selected_project_id = str(p["_id"])
                        st.write(p)
            else:
                st.info("No projects found.")

        elif tab == "Config":
            st.header("Config")
            user_config = get_user_config(username, MONGO_URI)
            st.write("Your saved config (masked):")
            if user_config:
                st.json({k: mask_key(v) if 'key' in k or 'uri' in k else v for k, v in user_config.items()})
            else:
                st.info("No config saved yet.")

            st.subheader("Update OpenAI Key (stored per-user in DB)")
            with st.form("update_config_form"):
                openai_key = st.text_input("OpenAI Key", value=user_config.get("openai_key", ""), type="password")
                update_pressed = st.form_submit_button("Update Config")
            if update_pressed:
                success, msg = save_user_config(username, openai_key, MONGO_URI)
                if success:
                    st.success(msg)
                    rerun()
                else:
                    st.error(f"Update failed: {msg}")

            llm_choice = st.radio("Choose LLM Provider", ("OpenAI", "Google Gemini"))
            prompt = st.text_area("Prompt")
            if st.button("Call LLM"):
                user_openai_key = user_config.get("openai_key", None)
                if llm_choice == "OpenAI" and user_openai_key and prompt:
                    response = call_llm(user_openai_key, prompt)
                    st.write(response)
                elif llm_choice == "Google Gemini":
                    st.warning("Gemini logic to be added.")
                else:
                    st.warning("Missing API key or prompt.")
