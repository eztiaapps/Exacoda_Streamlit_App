import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import certifi
from datetime import datetime
from openai import OpenAI

# Dummy user credentials
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
    return MongoClient(uri, tlsCAFile=certifi.where())


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
        return config if config else {}
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
        st.error(f"DB error: {e}")
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
        st.error(f"DB error: {e}")
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


def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            rerun()
        else:
            st.error("Invalid username or password")


def logout():
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        for key in ["logged_in", "username", "temp_mongo_uri", "selected_project_id",
                    "uploaded_files", "user_prompt", "system_prompt", "llm_response", "business_processes"]:
            if key in st.session_state:
                del st.session_state[key]
        rerun()


def project_card(project, key):
    return st.button(f"{project['title']} ({project['type']})", key=key)


def main_app():
    st.title(f"Welcome, {st.session_state.username}")

    try:
        mongo_uri = st.secrets["credentials"]["mongo_uri"]
        openai_api_key = st.secrets["credentials"]["openai_api_key"]
    except Exception:
        st.error("Configure your credentials in .streamlit/secrets.toml")
        return

    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    st.sidebar.markdown("---")
    st.sidebar.write("API keys loaded securely.")

    st.sidebar.header("Create New Project")
    new_title = st.sidebar.text_input("Project Title")
    new_type = st.sidebar.selectbox("Project Type", ["web app", "gpt", "testing", "mobile app"])
    if st.sidebar.button("Create Project"):
        if new_title.strip():
            pid = save_project(st.session_state.username, mongo_uri, new_title.strip(), new_type)
            if pid:
                st.sidebar.success(f"Created project '{new_title.strip()}'")
                st.session_state.selected_project_id = pid
                rerun()
            else:
                st.sidebar.error("Failed to create project")
        else:
            st.sidebar.warning("Project title required")
    st.sidebar.markdown("---")
    logout()

    projects = list_projects(st.session_state.username, mongo_uri)
    if "selected_project_id" not in st.session_state:
        st.session_state.selected_project_id = None

    cols = st.columns(3)
    for idx, project in enumerate(projects):
        with cols[idx % 3]:
            if st.button(project['title'] + f" ({project['type']})", key=str(project["_id"])):
                st.session_state.selected_project_id = project["_id"]
                rerun()

    if st.session_state.selected_project_id:
        project = next((p for p in projects if p["_id"] == st.session_state.selected_project_id), None)
        if not project:
            st.info("Selected project not found.")
            return
        st.header(f"Project: {project['title']} ({project['type']})")

        tabs = st.tabs(["1 Upload & Prompts", "2 LLM Query", "3 View Response", "4 Edit Processes", "5 Test Cases"])

        with tabs[0]:
            uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf", "docx", "txt"])
            user_prompt = st.text_area("User Prompt", value=st.session_state.get("user_prompt", ""))
            system_prompt = st.text_area("System Prompt", value=st.session_state.get("system_prompt", ""))
            if st.button("Save Step 1"):
                st.session_state.uploaded_files = uploaded_files
                st.session_state.user_prompt = user_prompt
                st.session_state.system_prompt = system_prompt
                st.success("Saved inputs")

        with tabs[1]:
            if not st.session_state.get("uploaded_files"):
                st.warning("Upload files first")
            elif not openai_api_key:
                st.warning("OpenAI API key missing")
            else:
                if st.button("Run LLM"):
                    combined_text = ""
                    for f in st.session_state.uploaded_files:
                        try:
                            combined_text += f.read().decode() + "\n"
                        except:
                            combined_text += f"[Unreadable content from {f.name}]\n"
                    prompt = (st.session_state.user_prompt or "") + "\n\nContents:\n" + combined_text
                    # Request the LLM to respond only with JSON array
                    formatted_prompt = (
                        "You are an enterprise assistant. Extract business processes and return ONLY a JSON array "
                        "of objects with 'title' and 'description'.\n"
                        f"Prompt:\n{prompt}"
                    )
                    try:
                        response_text = call_llm(openai_api_key, formatted_prompt, st.session_state.system_prompt or "")
                        import json
                        try:
                            parsed = json.loads(response_text)
                        except Exception as e:
                            st.error(f"Failed parsing response JSON: {e}")
                            parsed = []
                        st.session_state.llm_response = {"raw": response_text, "business_processes": parsed}
                        st.success("Received LLM response.")
                    except Exception as e:
                        st.error(f"LLM call error: {e}")

        with tabs[2]:
            if st.session_state.get("llm_response"):
                st.subheader("Raw LLM Output")
                st.text_area("LLM Text", value=st.session_state.llm_response.get("raw", ""), height=200)
                st.subheader("Parsed Business Processes")
                st.json(st.session_state.llm_response.get("business_processes", []))
            else:
                st.info("Run LLM query in previous step")

        with tabs[3]:
            bps = st.session_state.get("llm_response", {}).get("business_processes", [])
            if not bps:
                st.info("No business processes available. Run earlier steps.")
            else:
                for i, bp in enumerate(bps):
                    with st.expander(f"Process {i + 1}"):
                        title = st.text_input(f"Title {i+1}", value=bp.get("title", ""), key=f"title_{i}")
                        desc = st.text_area(f"Description {i+1}", value=bp.get("description", ""), key=f"desc_{i}")
                        bp["title"] = title
                        bp["description"] = desc
                st.session_state.llm_response["business_processes"] = bps

        with tabs[4]:
            bps = st.session_state.get("llm_response", {}).get("business_processes", [])
            if not bps:
                st.info("Define business processes first")
            else:
                for i, bp in enumerate(bps):
                    st.markdown(f"### Test Scenarios for {bp.get('title', 'Untitled')}")
                    ts = bp.get("test_scenarios", [])
                    for idx, scenario in enumerate(ts):
                        updated = st.text_area(f"Scenario {idx+1}", value=scenario, key=f"ts_{i}_{idx}")
                        ts[idx] = updated
                    if st.button(f"Add scenario to {bp.get('title', '')}", key=f"add_ts_{i}"):
                        ts.append("")
                    bp["test_scenarios"] = ts
                st.session_state.llm_response["business_processes"] = bps

                if st.button("Save Project Data"):
                    save_data = {
                        "prompts": {
                            "user": st.session_state.get("user_prompt", ""),
                            "system": st.session_state.get("system_prompt", ""),
                        },
                        "files": [f.name for f in st.session_state.get("uploaded_files", [])],
                        "business_processes": bps,
                    }
                    try:
                        client = get_mongodb_client(mongo_uri)
                        db = client['enterprise']
                        col = db['projects']
                        col.update_one({"_id": st.session_state.selected_project_id}, {"$set": save_data})
                        client.close()
                        st.success("Project data saved!")
                    except Exception as e:
                        st.error(f"Error saving project data: {e}")


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.logged_in:
        main_app()
    else:
        login()


if __name__ == "__main__":
    main()
