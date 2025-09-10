import streamlit as st
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import certifi
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
import pandas as pd
import json
from bson import ObjectId
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="ExaCoda Enterprise",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .tab-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .project-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        cursor: pointer;
        transition: transform 0.3s ease;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
    }
    
    .step-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    /* Login page styling */
    .login-container {
        max-width: 400px;
        margin: 5rem auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
        color: #333;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .block-container {
        padding-top: 2rem;
        background-color: #2c3e50;
    }
    
    .nav-button {
        width: 100%;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border: none;
        border-radius: 8px;
        background: transparent;
        color: white;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover, .nav-button.active {
        background: #34495e;
        color: #f39c12;
    }
    
    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .data-table th, .data-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    
    .data-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Dummy user credentials
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

class DatabaseManager:
    """Handle all database operations"""
    
    def __init__(self, uri):
        self.uri = uri
        
    def get_client(self):
        return MongoClient(self.uri, tlsCAFile=certifi.where())
    
    def save_project(self, username, project_data):
        try:
            client = self.get_client()
            db = client['enterprise']
            collection = db['projects']
            
            project = {
                "username": username,
                "title": project_data['title'],
                "description": project_data.get('description', ''),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "steps": {
                    "1_upload": {"completed": False, "files": [], "user_prompt": "", "system_prompt": ""},
                    "2_llm_response": {"completed": False, "response_data": {}},
                    "3_business_processes": {"completed": False, "processes": []},
                    "4_test_scenarios": {"completed": False, "scenarios": []},
                    "5_test_scripts": {"completed": False, "scripts": []}
                }
            }
            
            result = collection.insert_one(project)
            client.close()
            return str(result.inserted_id)
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
    
    def get_projects(self, username):
        try:
            client = self.get_client()
            db = client['enterprise']
            collection = db['projects']
            
            projects = list(collection.find({"username": username}))
            client.close()
            
            # Convert ObjectId to string for JSON serialization
            for project in projects:
                project['_id'] = str(project['_id'])
            
            return projects
        except Exception as e:
            st.error(f"Database error: {e}")
            return []
    
    def update_project_step(self, project_id, step, data):
        try:
            client = self.get_client()
            db = client['enterprise']
            collection = db['projects']
            
            collection.update_one(
                {"_id": ObjectId(project_id)},
                {
                    "$set": {
                        f"steps.{step}": data,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            client.close()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False
    
    def get_project(self, project_id):
        try:
            client = self.get_client()
            db = client['enterprise']
            collection = db['projects']
            
            project = collection.find_one({"_id": ObjectId(project_id)})
            client.close()
            
            if project:
                project['_id'] = str(project['_id'])
            
            return project
        except Exception as e:
            st.error(f"Database error: {e}")
            return None
    
    def delete_project(self, project_id):
        try:
            client = self.get_client()
            db = client['enterprise']
            collection = db['projects']
            
            collection.delete_one({"_id": ObjectId(project_id)})
            client.close()
            return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

class LLMManager:
    """Handle LLM operations for both OpenAI and Gemini"""
    
    def __init__(self, openai_key, gemini_key):
        self.openai_client = OpenAI(api_key=openai_key) if openai_key else None
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
    
    def call_openai(self, user_prompt, system_prompt="", model="gpt-4o-mini"):
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    
    def call_gemini(self, user_prompt, system_prompt=""):
        if not self.gemini_model:
            raise Exception("Gemini model not initialized")
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = self.gemini_model.generate_content(full_prompt)
        return response.text
    
    def call_llm(self, provider, user_prompt, system_prompt=""):
        if provider == "OpenAI":
            return self.call_openai(user_prompt, system_prompt)
        elif provider == "Gemini":
            return self.call_gemini(user_prompt, system_prompt)
        else:
            raise Exception("Invalid LLM provider")

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'logged_in': False,
        'username': None,
        'current_tab': 'Home',
        'selected_project_id': None,
        'uploaded_files': [],
        'llm_provider': 'OpenAI'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_background_image():
    """Convert background image to base64 for CSS"""
    # This would be your AI robot image
    # For now, using a placeholder
    return ""

def render_login_page():
    """Render the login page with the AI robot background"""
    
    # Background styling
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }}
        
        .login-form {{
            background: rgba(255, 255, 255, 0.95);
            padding: 3rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            margin-top: 5rem;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-form">
            <h1 style="text-align: center; color: #333; margin-bottom: 2rem;">
                🤖 ExaCoda Enterprise Login
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Email")
            username = st.text_input("", placeholder="Enter your username", label_visibility="collapsed")
            
            st.markdown("### Password")
            password = st.text_input("", type="password", placeholder="Enter your password", label_visibility="collapsed")
            
            col_login, col_sso = st.columns(2)
            
            with col_login:
                login_clicked = st.form_submit_button("Login", use_container_width=True)
            
            with col_sso:
                sso_clicked = st.form_submit_button("Login with Single Sign-On", use_container_width=True)
            
            signup_clicked = st.form_submit_button("Sign up", use_container_width=True)
            
            if login_clicked:
                if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            
            if sso_clicked:
                st.info("🔄 Single Sign-On functionality would be implemented here")
            
            if signup_clicked:
                st.info("📝 Sign up functionality would be implemented here")

def render_sidebar():
    """Render the application sidebar"""
    if not st.session_state.logged_in:
        return
    
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">Exacoda QAI</h1>
        <p style="color: white; margin: 0;">{st.session_state.username}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation tabs
    tabs = ['🏠 Home', '📊 Overview', '⚙️ Configs', '📁 Projects', '🎛️ Control Panel']
    
    for tab in tabs:
        if st.sidebar.button(tab, key=f"nav_{tab}", use_container_width=True):
            st.session_state.current_tab = tab.split(' ', 1)[1]
            if 'selected_project_id' in st.session_state:
                del st.session_state.selected_project_id
            st.rerun()
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['logged_in', 'username']:
                del st.session_state[key]
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

def render_home_tab():
    """Render the Home tab"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ExaCoda Enterprise Platform</h1>
        <p>Your AI-powered business process automation solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="tab-card">
            <h3>📊 Analytics Dashboard</h3>
            <p>Monitor your business processes and get insights from AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tab-card">
            <h3>🤖 AI Processing</h3>
            <p>Leverage OpenAI and Google Gemini for document analysis and automation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tab-card">
            <h3>📁 Project Management</h3>
            <p>Organize and manage your automation projects efficiently</p>
        </div>
        """, unsafe_allow_html=True)

def render_overview_tab():
    """Render the Overview tab"""
    st.markdown("# 📊 Overview Dashboard")
    
    try:
        mongo_uri = st.secrets["credentials"]["mongo_uri"]
        db_manager = DatabaseManager(mongo_uri)
        projects = db_manager.get_projects(st.session_state.username)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Projects", len(projects))
        
        with col2:
            completed_projects = sum(1 for p in projects if all(step.get('completed', False) for step in p.get('steps', {}).values()))
            st.metric("Completed Projects", completed_projects)
        
        with col3:
            active_projects = len(projects) - completed_projects
            st.metric("Active Projects", active_projects)
        
        if projects:
            st.markdown("## Recent Projects")
            df = pd.DataFrame([
                {
                    'Title': p['title'],
                    'Created': p['created_at'].strftime('%Y-%m-%d'),
                    'Status': 'Completed' if all(step.get('completed', False) for step in p.get('steps', {}).values()) else 'In Progress'
                }
                for p in projects[-5:]  # Last 5 projects
            ])
            st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Unable to load overview: {e}")

def render_configs_tab():
    """Render the Configs tab"""
    st.markdown("# ⚙️ Configuration Settings")
    
    try:
        openai_key = st.secrets["credentials"].get("openai_api_key", "Not configured")
        gemini_key = st.secrets["credentials"].get("gemini_api_key", "Not configured")
        mongo_uri = st.secrets["credentials"].get("mongo_uri", "Not configured")
        
        st.markdown("## API Keys Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### OpenAI Configuration")
            if openai_key != "Not configured":
                st.success("✅ OpenAI API Key: Configured")
            else:
                st.error("❌ OpenAI API Key: Not configured")
        
        with col2:
            st.markdown("### Google Gemini Configuration")
            if gemini_key != "Not configured":
                st.success("✅ Gemini API Key: Configured")
            else:
                st.error("❌ Gemini API Key: Not configured")
        
        st.markdown("### Database Configuration")
        if mongo_uri != "Not configured":
            st.success("✅ MongoDB Atlas: Connected")
        else:
            st.error("❌ MongoDB Atlas: Not configured")
        
        st.markdown("## LLM Provider Selection")
        provider = st.selectbox(
            "Choose your preferred LLM provider:",
            ["OpenAI", "Gemini"],
            index=0 if st.session_state.llm_provider == "OpenAI" else 1
        )
        
        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.success(f"LLM provider changed to {provider}")
        
    except Exception as e:
        st.error(f"Configuration error: {e}")

def render_projects_tab():
    """Render the Projects tab"""
    st.markdown("# 📁 Projects Management")
    
    try:
        mongo_uri = st.secrets["credentials"]["mongo_uri"]
        db_manager = DatabaseManager(mongo_uri)
        
        # Create new project section
        with st.expander("➕ Create New Project", expanded=False):
            with st.form("new_project_form"):
                project_title = st.text_input("Project Title")
                project_description = st.text_area("Project Description")
                
                if st.form_submit_button("Create Project"):
                    if project_title.strip():
                        project_data = {
                            'title': project_title.strip(),
                            'description': project_description.strip()
                        }
                        project_id = db_manager.save_project(st.session_state.username, project_data)
                        if project_id:
                            st.success(f"✅ Project '{project_title}' created successfully!")
                            st.rerun()
                    else:
                        st.error("❌ Project title is required")
        
        # Display existing projects
        projects = db_manager.get_projects(st.session_state.username)
        
        if not projects:
            st.info("📝 No projects found. Create your first project above!")
            return
        
        st.markdown("## Your Projects")
        
        cols = st.columns(3)
        for idx, project in enumerate(projects):
            with cols[idx % 3]:
                completed_steps = sum(1 for step in project.get('steps', {}).values() if step.get('completed', False))
                total_steps = len(project.get('steps', {}))
                progress = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
                
                st.markdown(f"""
                <div class="project-card">
                    <h4>{project['title']}</h4>
                    <p>{project.get('description', 'No description')}</p>
                    <p><strong>Progress:</strong> {completed_steps}/{total_steps} steps ({progress:.0f}%)</p>
                    <p><small>Created: {project['created_at'].strftime('%Y-%m-%d')}</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                col_select, col_delete = st.columns(2)
                with col_select:
                    if st.button("📂 Open", key=f"open_{project['_id']}", use_container_width=True):
                        st.session_state.selected_project_id = project['_id']
                        st.rerun()
                
                with col_delete:
                    if st.button("🗑️ Delete", key=f"delete_{project['_id']}", use_container_width=True):
                        if db_manager.delete_project(project['_id']):
                            st.success("Project deleted!")
                            st.rerun()
        
        # Project details view
        if st.session_state.selected_project_id:
            render_project_details(db_manager)
            
    except Exception as e:
        st.error(f"Projects error: {e}")

def render_project_details(db_manager):
    """Render detailed project view with 5 steps"""
    project = db_manager.get_project(st.session_state.selected_project_id)
    
    if not project:
        st.error("Project not found!")
        return
    
    st.markdown("---")
    st.markdown(f"# 📊 Project: {project['title']}")
    
    if st.button("🔙 Back to Projects List"):
        del st.session_state.selected_project_id
        st.rerun()
    
    # Create tabs for the 5 steps
    step_tabs = st.tabs([
        "1️⃣ Upload & Setup",
        "2️⃣ LLM Processing",
        "3️⃣ Business Processes",
        "4️⃣ Test Scenarios",
        "5️⃣ Test Scripts"
    ])
    
    try:
        openai_key = st.secrets["credentials"].get("openai_api_key")
        gemini_key = st.secrets["credentials"].get("gemini_api_key")
        llm_manager = LLMManager(openai_key, gemini_key)
    except:
        llm_manager = None
        st.error("LLM credentials not configured properly")
    
    # Step 1: Upload & Setup
    with step_tabs[0]:
        render_step1_upload(project, db_manager)
    
    # Step 2: LLM Processing
    with step_tabs[1]:
        render_step2_llm(project, db_manager, llm_manager)
    
    # Step 3: Business Processes
    with step_tabs[2]:
        render_step3_processes(project, db_manager)
    
    # Step 4: Test Scenarios
    with step_tabs[3]:
        render_step4_scenarios(project, db_manager, llm_manager)
    
    # Step 5: Test Scripts
    with step_tabs[4]:
        render_step5_scripts(project, db_manager, llm_manager)

def render_step1_upload(project, db_manager):
    """Step 1: File Upload and Prompts Setup"""
    st.markdown("### 📤 File Upload & Prompt Configuration")
    
    step_data = project.get('steps', {}).get('1_upload', {})
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'csv'],
        help="Upload documents for business process analysis"
    )
    
    # Default prompts
    default_user_prompt = """From the attached document identify and extract the business processes. For each process, provide:
1. Process name/title
2. Detailed description
3. Key stakeholders involved
4. Input requirements
5. Output deliverables
6. Process steps

Return the response as a JSON array of objects with the following structure:
{
  "title": "Process Name",
  "description": "Detailed process description",
  "stakeholders": ["stakeholder1", "stakeholder2"],
  "inputs": ["input1", "input2"],
  "outputs": ["output1", "output2"],
  "steps": ["step1", "step2", "step3"]
}"""

    default_system_prompt = """You are an expert business process analyst. Your task is to analyze business documents and extract structured information about business processes. Always respond with valid JSON format. Be thorough and accurate in your analysis."""
    
    # User prompt
    user_prompt = st.text_area(
        "User Prompt",
        value=step_data.get('user_prompt', default_user_prompt),
        height=20,
        help="This prompt will be sent to the LLM along with your documents"
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value=step_data.get('system_prompt', default_system_prompt),
        height=10,
        help="This system prompt defines the AI's behavior and response format"
    )
    
    if st.button("💾 Save Step 1", type="primary"):
        file_info = []
        if uploaded_files:
            for file in uploaded_files:
                file_info.append({
                    'name': file.name,
                    'size': file.size,
                    'type': file.type
                })
        
        step1_data = {
            'completed': True,
            'files': file_info,
            'user_prompt': user_prompt,
            'system_prompt': system_prompt,
            'uploaded_files_content': {}
        }
        
        # Store file contents (for demo purposes, in production you'd store in file storage)
        if uploaded_files:
            for file in uploaded_files:
                try:
                    content = file.read().decode('utf-8', errors='ignore')
                    step1_data['uploaded_files_content'][file.name] = content
                except:
                    step1_data['uploaded_files_content'][file.name] = f"[Binary file: {file.name}]"
        
        if db_manager.update_project_step(project['_id'], '1_upload', step1_data):
            st.success("✅ Step 1 saved successfully!")
            st.rerun()

def render_step2_llm(project, db_manager, llm_manager):
    """Step 2: LLM Processing"""
    st.markdown("### 🤖 LLM Processing")
    
    step1_data = project.get('steps', {}).get('1_upload', {})
    step2_data = project.get('steps', {}).get('2_llm_response', {})
    
    if not step1_data.get('completed'):
        st.warning("⚠️ Please complete Step 1 first")
        return
    
    # LLM Provider selection
    provider = st.selectbox("Select LLM Provider", ["OpenAI", "Gemini"], 
                           index=0 if st.session_state.llm_provider == "OpenAI" else 1)
    
    if st.button("🚀 Process with LLM", type="primary"):
        if not llm_manager:
            st.error("LLM manager not properly configured")
            return
            
        try:
            with st.spinner(f"Processing with {provider}..."):
                # Combine uploaded files content
                files_content = step1_data.get('uploaded_files_content', {})
                combined_content = "\n\n".join([f"=== {filename} ===\n{content}" 
                                              for filename, content in files_content.items()])
                
                full_prompt = f"{step1_data.get('user_prompt', '')}\n\nDocument Content:\n{combined_content}"
                
                response = llm_manager.call_llm(
                    provider,
                    full_prompt,
                    step1_data.get('system_prompt', ''),
                    step2_data.get()
                )
                
                # Try to parse JSON response
                try:
                    parsed_response = json.loads(response)
                    if not isinstance(parsed_response, list):
                        parsed_response = [parsed_response]
                except json.JSONDecodeError:
                    parsed_response = []
                    st.warning("Response was not valid JSON, storing as raw text")
                
                step2_data_new = {
                    'completed': True,
                    'response_data': {
                        'raw_response': response,
                        'parsed_processes': parsed_response,
                        'provider_used': provider,
                        'processed_at': datetime.utcnow().isoformat()
                    }
                }
                
                if db_manager.update_project_step(project['_id'], '2_llm_response', step2_data_new):
                    st.success("✅ LLM processing completed!")
                    st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing with LLM: {e}")
    
    # Display results if available
    if step2_data.get('completed') and step2_data.get('response_data'):
        st.markdown("### 📋 LLM Response")
        response_data = step2_data['response_data']
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Provider: {response_data.get('provider_used', 'Unknown')}")
        with col2:
            st.info(f"Processed: {response_data.get('processed_at', 'Unknown')}")
        
        # Raw response
        with st.expander("📄 Raw Response"):
            st.text_area("", value=response_data.get('raw_response', ''), height=300, disabled=True)
        
        # Raw response
        with st.expander("📄 Text Response"):
            st.text_area("", value=response_data.get('raw_response', ''), height=300, disabled=True)


        
        # Parsed processes
        parsed_processes = response_data.get('parsed_processes', [])

        if parsed_processes:
            st.markdown("### 📊 Extracted Business Processes")
            df = pd.DataFrame(parsed_processes)
            st.dataframe(df, use_container_width=True)

def render_step3_processes(project, db_manager):
    """Step 3: Business Processes Management"""
    st.markdown("### 📋 Business Processes Management")
    
    step2_data = project.get('steps', {}).get('2_llm_response', {})
    step3_data = project.get('steps', {}).get('3_business_processes', {})
    
    if not step2_data.get('completed'):
        st.warning("⚠️ Please complete Step 2 first")
        return 
    
    # Get parsed processes from Step 2
    parsed_processes = step2_data.get('response_data', {}).get('parsed_processes', [])
    current_processes = step3_data.get('processes', parsed_processes.copy() if parsed_processes else [])
    
    if not current_processes:
        st.info("No business processes found from LLM response")
        return
    
    st.markdown("### ✏️ Edit Business Processes")
    st.markdown("You can modify, add, or remove business processes below:")
    
    # Process editing interface
    edited_processes = []
    
    for i, process in enumerate(current_processes):
        with st.expander(f"📋 Process {i+1}: {process.get('title', 'Untitled')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                title = st.text_input(f"Title", value=process.get('title', ''), key=f"proc_title_{i}")
                description = st.text_area(f"Description", value=process.get('description', ''), key=f"proc_desc_{i}", height=100)
                
                # Handle stakeholders as list or string
                stakeholders_val = process.get('stakeholders', [])
                if isinstance(stakeholders_val, list):
                    stakeholders_str = ', '.join(stakeholders_val)
                else:
                    stakeholders_str = str(stakeholders_val)
                
                stakeholders = st.text_input(f"Stakeholders (comma-separated)", 
                                           value=stakeholders_str, key=f"proc_stakeholders_{i}")
                
                # Handle inputs as list or string
                inputs_val = process.get('inputs', [])
                if isinstance(inputs_val, list):
                    inputs_str = ', '.join(inputs_val)
                else:
                    inputs_str = str(inputs_val)
                
                inputs = st.text_input(f"Inputs (comma-separated)", 
                                     value=inputs_str, key=f"proc_inputs_{i}")
                
                # Handle outputs as list or string
                outputs_val = process.get('outputs', [])
                if isinstance(outputs_val, list):
                    outputs_str = ', '.join(outputs_val)
                else:
                    outputs_str = str(outputs_val)
                
                outputs = st.text_input(f"Outputs (comma-separated)", 
                                      value=outputs_str, key=f"proc_outputs_{i}")
                
                # Handle steps as list or string
                steps_val = process.get('steps', [])
                if isinstance(steps_val, list):
                    steps_str = '\n'.join(steps_val)
                else:
                    steps_str = str(steps_val)
                
                steps = st.text_area(f"Process Steps (one per line)", 
                                   value=steps_str, key=f"proc_steps_{i}", height=150)
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"🗑️ Delete", key=f"delete_proc_{i}", type="secondary"):
                    continue  # Skip this process (effectively deleting it)
            
            # Create updated process object
            edited_process = {
                'title': title,
                'description': description,
                'stakeholders': [s.strip() for s in stakeholders.split(',') if s.strip()],
                'inputs': [i.strip() for i in inputs.split(',') if i.strip()],
                'outputs': [o.strip() for o in outputs.split(',') if o.strip()],
                'steps': [s.strip() for s in steps.split('\n') if s.strip()]
            }
            edited_processes.append(edited_process)
    
    # Add new process section
    with st.expander("➕ Add New Process", expanded=False):
        new_title = st.text_input("New Process Title", key="new_proc_title")
        new_description = st.text_area("New Process Description", key="new_proc_desc", height=100)
        new_stakeholders = st.text_input("Stakeholders (comma-separated)", key="new_proc_stakeholders")
        new_inputs = st.text_input("Inputs (comma-separated)", key="new_proc_inputs")
        new_outputs = st.text_input("Outputs (comma-separated)", key="new_proc_outputs")
        new_steps = st.text_area("Process Steps (one per line)", key="new_proc_steps", height=150)
        
        if st.button("✅ Add Process", type="primary"):
            if new_title.strip():
                new_process = {
                    'title': new_title,
                    'description': new_description,
                    'stakeholders': [s.strip() for s in new_stakeholders.split(',') if s.strip()],
                    'inputs': [i.strip() for i in new_inputs.split(',') if i.strip()],
                    'outputs': [o.strip() for o in new_outputs.split(',') if o.strip()],
                    'steps': [s.strip() for s in new_steps.split('\n') if s.strip()]
                }
                edited_processes.append(new_process)
                st.success("Process added! Click 'Save Changes' to persist.")
    
    # Save changes
    if st.button("💾 Save Changes", type="primary"):
        step3_data_new = {
            'completed': True,
            'processes': edited_processes,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        if db_manager.update_project_step(project['_id'], '3_business_processes', step3_data_new):
            st.success("✅ Business processes saved successfully!")
            st.rerun()
    
    # Display current processes summary
    if edited_processes:
        st.markdown("### 📊 Current Processes Summary")
        summary_data = []
        for i, proc in enumerate(edited_processes):
            summary_data.append({
                '#': i+1,
                'Title': proc.get('title', 'Untitled'),
                'Stakeholders': len(proc.get('stakeholders', [])),
                'Steps': len(proc.get('steps', [])),
                'Description': proc.get('description', '')[:100] + '...' if len(proc.get('description', '')) > 100 else proc.get('description', '')
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

def render_step4_scenarios(project, db_manager, llm_manager):
    """Step 4: Test Scenarios Generation"""
    st.markdown("### 🧪 Test Scenarios Generation")
    
    step3_data = project.get('steps', {}).get('3_business_processes', {})
    step4_data = project.get('steps', {}).get('4_test_scenarios', {})
    
    if not step3_data.get('completed'):
        st.warning("⚠️ Please complete Step 3 first")
        return
    
    processes = step3_data.get('processes', [])
    if not processes:
        st.info("No business processes available")
        return
    
    # LLM Provider selection
    provider = st.selectbox("Select LLM Provider for Test Scenarios", ["OpenAI", "Gemini"], 
                           index=0 if st.session_state.llm_provider == "OpenAI" else 1, key="scenarios_provider")
    
    # Generate test scenarios
    if st.button("🚀 Generate Test Scenarios", type="primary"):
        if not llm_manager:
            st.error("LLM manager not properly configured")
            return
        
        try:
            with st.spinner("Generating test scenarios..."):
                all_scenarios = []
                
                for process in processes:
                    # Create prompt for test scenario generation
                    scenario_prompt = f"""
                    Based on the following business process, generate comprehensive test scenarios.
                    
                    Process: {process.get('title', '')}
                    Description: {process.get('description', '')}
                    Steps: {', '.join(process.get('steps', []))}
                    Inputs: {', '.join(process.get('inputs', []))}
                    Outputs: {', '.join(process.get('outputs', []))}
                    
                    Generate test scenarios that cover:
                    1. Happy path scenarios
                    2. Edge cases
                    3. Error conditions
                    4. Boundary conditions
                    5. Integration scenarios
                    
                    Return as JSON array with objects containing:
                    - scenario_name: Brief name for the test scenario
                    - scenario_description: Detailed description
                    - test_type: Type of test (functional, integration, etc.)
                    - priority: High/Medium/Low
                    - expected_outcome: What should happen
                    - test_data: Required test data
                    """
                    
                    system_prompt = "You are a QA expert specializing in business process testing. Generate comprehensive test scenarios in valid JSON format."
                    
                    response = llm_manager.call_llm(provider, scenario_prompt, system_prompt)
                    
                    try:
                        scenarios = json.loads(response)
                        if not isinstance(scenarios, list):
                            scenarios = [scenarios]
                    except json.JSONDecodeError:
                        scenarios = [{
                            'scenario_name': 'Generated Scenario',
                            'scenario_description': response,
                            'test_type': 'Manual',
                            'priority': 'Medium',
                            'expected_outcome': 'To be defined',
                            'test_data': 'To be defined'
                        }]
                    
                    process_scenarios = {
                        'process_title': process.get('title', ''),
                        'process_id': processes.index(process),
                        'scenarios': scenarios
                    }
                    all_scenarios.append(process_scenarios)
                
                step4_data_new = {
                    'completed': True,
                    'scenarios': all_scenarios,
                    'provider_used': provider,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                if db_manager.update_project_step(project['_id'], '4_test_scenarios', step4_data_new):
                    st.success("✅ Test scenarios generated successfully!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error generating test scenarios: {e}")
    
    # Display and edit existing scenarios
    if step4_data.get('completed') and step4_data.get('scenarios'):
        st.markdown("### 📝 Test Scenarios Management")
        
        scenarios_data = step4_data.get('scenarios', [])
        
        for proc_scenarios in scenarios_data:
            st.markdown(f"#### 📋 Scenarios for: {proc_scenarios.get('process_title', 'Unknown Process')}")
            
            scenarios = proc_scenarios.get('scenarios', [])
            
            for i, scenario in enumerate(scenarios):
                with st.expander(f"🧪 {scenario.get('scenario_name', f'Scenario {i+1}')}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        name = st.text_input("Scenario Name", 
                                           value=scenario.get('scenario_name', ''), 
                                           key=f"scen_name_{proc_scenarios.get('process_id')}_{i}")
                        
                        description = st.text_area("Description", 
                                                 value=scenario.get('scenario_description', ''), 
                                                 key=f"scen_desc_{proc_scenarios.get('process_id')}_{i}",
                                                 height=100)
                        
                        col_type, col_priority = st.columns(2)
                        with col_type:
                            test_type = st.selectbox("Test Type", 
                                                   ["Functional", "Integration", "Performance", "Security", "Usability"],
                                                   index=["Functional", "Integration", "Performance", "Security", "Usability"].index(
                                                       scenario.get('test_type', 'Functional')) if scenario.get('test_type') in 
                                                       ["Functional", "Integration", "Performance", "Security", "Usability"] else 0,
                                                   key=f"scen_type_{proc_scenarios.get('process_id')}_{i}")
                        
                        with col_priority:
                            priority = st.selectbox("Priority", 
                                                   ["High", "Medium", "Low"],
                                                   index=["High", "Medium", "Low"].index(scenario.get('priority', 'Medium')) 
                                                   if scenario.get('priority') in ["High", "Medium", "Low"] else 1,
                                                   key=f"scen_priority_{proc_scenarios.get('process_id')}_{i}")
                        
                        expected_outcome = st.text_area("Expected Outcome", 
                                                      value=scenario.get('expected_outcome', ''), 
                                                      key=f"scen_outcome_{proc_scenarios.get('process_id')}_{i}",
                                                      height=80)
                        
                        test_data = st.text_area("Test Data Requirements", 
                                                value=scenario.get('test_data', ''), 
                                                key=f"scen_data_{proc_scenarios.get('process_id')}_{i}",
                                                height=80)
                        
                        # Update scenario data
                        scenario.update({
                            'scenario_name': name,
                            'scenario_description': description,
                            'test_type': test_type,
                            'priority': priority,
                            'expected_outcome': expected_outcome,
                            'test_data': test_data
                        })
                    
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button(f"🗑️ Delete", key=f"delete_scen_{proc_scenarios.get('process_id')}_{i}"):
                            scenarios.remove(scenario)
                            st.success("Scenario deleted! Save changes to persist.")
        
        if st.button("💾 Save Scenario Changes", type="primary"):
            if db_manager.update_project_step(project['_id'], '4_test_scenarios', step4_data):
                st.success("✅ Test scenarios updated successfully!")
                st.rerun()

def render_step5_scripts(project, db_manager, llm_manager):
    """Step 5: Test Scripts Generation"""
    st.markdown("### 📜 Test Scripts Generation")
    
    step4_data = project.get('steps', {}).get('4_test_scenarios', {})
    step5_data = project.get('steps', {}).get('5_test_scripts', {})
    
    if not step4_data.get('completed'):
        st.warning("⚠️ Please complete Step 4 first")
        return
    
    scenarios_data = step4_data.get('scenarios', [])
    if not scenarios_data:
        st.info("No test scenarios available")
        return
    
    # Technology stack selection
    st.markdown("### 🛠️ Technology Stack Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_framework = st.selectbox("Test Framework", 
                                    ["Selenium WebDriver", "Cypress", "Playwright", "RestAssured", "Postman", "JUnit", "TestNG", "PyTest"])
    
    with col2:
        programming_language = st.selectbox("Programming Language", 
                                          ["Java", "Python", "JavaScript", "C#", "TypeScript"])
    
    with col3:
        app_type = st.selectbox("Application Type", 
                              ["Web Application", "Mobile App", "API", "Desktop App", "Database"])
    
    # LLM Provider selection
    provider = st.selectbox("Select LLM Provider for Test Scripts", ["OpenAI", "Gemini"], 
                           index=0 if st.session_state.llm_provider == "OpenAI" else 1, key="scripts_provider")
    
    # Generate test scripts
    if st.button("🚀 Generate Test Scripts", type="primary"):
        if not llm_manager:
            st.error("LLM manager not properly configured")
            return
        
        try:
            with st.spinner("Generating test scripts..."):
                all_scripts = []
                
                for proc_scenarios in scenarios_data:
                    process_scripts = {
                        'process_title': proc_scenarios.get('process_title', ''),
                        'process_id': proc_scenarios.get('process_id', 0),
                        'scripts': []
                    }
                    
                    for scenario in proc_scenarios.get('scenarios', []):
                        # Create prompt for test script generation
                        script_prompt = f"""
                        Generate a test script for the following test scenario using the specified technology stack:
                        
                        Technology Stack:
                        - Test Framework: {test_framework}
                        - Programming Language: {programming_language}
                        - Application Type: {app_type}
                        
                        Test Scenario:
                        - Name: {scenario.get('scenario_name', '')}
                        - Description: {scenario.get('scenario_description', '')}
                        - Test Type: {scenario.get('test_type', '')}
                        - Expected Outcome: {scenario.get('expected_outcome', '')}
                        - Test Data: {scenario.get('test_data', '')}
                        
                        Generate a complete, executable test script that includes:
                        1. Proper imports and setup
                        2. Test data preparation
                        3. Test execution steps
                        4. Assertions and validations
                        5. Cleanup and teardown
                        6. Comments explaining each step
                        
                        Make the script production-ready with proper error handling and logging.
                        """
                        
                        system_prompt = f"You are a test automation expert specializing in {test_framework} and {programming_language}. Generate complete, executable test scripts with best practices."
                        
                        response = llm_manager.call_llm(provider, script_prompt, system_prompt)
                        
                        script_data = {
                            'scenario_name': scenario.get('scenario_name', ''),
                            'framework': test_framework,
                            'language': programming_language,
                            'app_type': app_type,
                            'script_code': response,
                            'generated_at': datetime.utcnow().isoformat()
                        }
                        
                        process_scripts['scripts'].append(script_data)
                    
                    all_scripts.append(process_scripts)
                
                step5_data_new = {
                    'completed': True,
                    'scripts': all_scripts,
                    'tech_stack': {
                        'framework': test_framework,
                        'language': programming_language,
                        'app_type': app_type
                    },
                    'provider_used': provider,
                    'generated_at': datetime.utcnow().isoformat()
                }
                
                if db_manager.update_project_step(project['_id'], '5_test_scripts', step5_data_new):
                    st.success("✅ Test scripts generated successfully!")
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error generating test scripts: {e}")
    
    # Display and manage existing scripts
    if step5_data.get('completed') and step5_data.get('scripts'):
        st.markdown("### 📜 Generated Test Scripts")
        
        tech_stack = step5_data.get('tech_stack', {})
        st.info(f"📋 Technology Stack: {tech_stack.get('framework', 'Unknown')} | "
               f"{tech_stack.get('language', 'Unknown')} | {tech_stack.get('app_type', 'Unknown')}")
        
        scripts_data = step5_data.get('scripts', [])
        
        for proc_scripts in scripts_data:
            st.markdown(f"#### 📁 Scripts for: {proc_scripts.get('process_title', 'Unknown Process')}")
            
            scripts = proc_scripts.get('scripts', [])
            
            for i, script in enumerate(scripts):
                with st.expander(f"📜 {script.get('scenario_name', f'Script {i+1}')}", expanded=False):
                    st.markdown(f"**Framework:** {script.get('framework', 'Unknown')}")
                    st.markdown(f"**Language:** {script.get('language', 'Unknown')}")
                    st.markdown(f"**Generated:** {script.get('generated_at', 'Unknown')}")
                    
                    # Script code editor
                    script_code = st.text_area(
                        "Script Code",
                        value=script.get('script_code', ''),
                        height=400,
                        key=f"script_code_{proc_scripts.get('process_id')}_{i}",
                        help="You can edit the generated script code here"
                    )
                    
                    # Update script code
                    script['script_code'] = script_code
                    
                    # Download button for individual script
                    script_filename = f"{script.get('scenario_name', 'script').replace(' ', '_').lower()}.{get_file_extension(script.get('language', 'txt'))}"
                    st.download_button(
                        label="📥 Download Script",
                        data=script_code,
                        file_name=script_filename,
                        mime="text/plain",
                        key=f"download_script_{proc_scripts.get('process_id')}_{i}"
                    )
        
        # Save changes button
        if st.button("💾 Save Script Changes", type="primary"):
            if db_manager.update_project_step(project['_id'], '5_test_scripts', step5_data):
                st.success("✅ Test scripts updated successfully!")
                st.rerun()
        
        # Download all scripts as zip
        if st.button("📦 Download All Scripts", type="secondary"):
            zip_buffer = create_scripts_zip(scripts_data)
            st.download_button(
                label="📥 Download Scripts ZIP",
                data=zip_buffer,
                file_name=f"test_scripts_{project['title'].replace(' ', '_')}.zip",
                mime="application/zip"
            )

def get_file_extension(language):
    """Get file extension based on programming language"""
    extensions = {
        'Java': 'java',
        'Python': 'py',
        'JavaScript': 'js',
        'TypeScript': 'ts',
        'C#': 'cs'
    }
    return extensions.get(language, 'txt')

def create_scripts_zip(scripts_data):
    """Create a ZIP file containing all test scripts"""
    import zipfile
    
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for proc_scripts in scripts_data:
            process_name = proc_scripts.get('process_title', 'Unknown_Process').replace(' ', '_')
            
            for i, script in enumerate(proc_scripts.get('scripts', [])):
                scenario_name = script.get('scenario_name', f'Script_{i+1}').replace(' ', '_')
                file_extension = get_file_extension(script.get('language', 'txt'))
                filename = f"{process_name}/{scenario_name}.{file_extension}"
                
                zip_file.writestr(filename, script.get('script_code', ''))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def render_control_panel():
    """Render the Control Panel tab"""
    st.markdown("# 🎛️ Control Panel")
    
    try:
        mongo_uri = st.secrets["credentials"]["mongo_uri"]
        db_manager = DatabaseManager(mongo_uri)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 System Statistics")
            
            # Get all projects for current user
            projects = db_manager.get_projects(st.session_state.username)
            
            total_projects = len(projects)
            completed_projects = sum(1 for p in projects if all(step.get('completed', False) for step in p.get('steps', {}).values()))
            
            st.metric("Total Projects", total_projects)
            st.metric("Completed Projects", completed_projects)
            st.metric("In Progress", total_projects - completed_projects)
            
            # Project completion rate
            completion_rate = (completed_projects / total_projects * 100) if total_projects > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        with col2:
            st.markdown("### ⚙️ System Controls")
            
            # Bulk operations
            if st.button("🗑️ Delete All Projects", type="secondary"):
                if st.checkbox("⚠️ I understand this will delete ALL my projects"):
                    try:
                        client = db_manager.get_client()
                        db = client['enterprise']
                        collection = db['projects']
                        result = collection.delete_many({"username": st.session_state.username})
                        client.close()
                        st.success(f"✅ Deleted {result.deleted_count} projects")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting projects: {e}")
            
            # Export data
            if st.button("📤 Export All Data", type="primary"):
                try:
                    export_data = {
                        'username': st.session_state.username,
                        'export_date': datetime.utcnow().isoformat(),
                        'projects': projects
                    }
                    
                    json_str = json.dumps(export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="📥 Download Export",
                        data=json_str,
                        file_name=f"exacoda_export_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        # Recent activity
        if projects:
            st.markdown("### 📈 Recent Activity")
            
            # Sort projects by updated date
            sorted_projects = sorted(projects, key=lambda x: x.get('updated_at', x.get('created_at')), reverse=True)
            
            activity_data = []
            for project in sorted_projects[:10]:  # Last 10 projects
                last_updated = project.get('updated_at', project.get('created_at'))
                if isinstance(last_updated, str):
                    try:
                        last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    except:
                        pass
                
                activity_data.append({
                    'Project': project['title'],
                    'Last Updated': last_updated.strftime('%Y-%m-%d %H:%M') if hasattr(last_updated, 'strftime') else str(last_updated),
                    'Progress': f"{sum(1 for step in project.get('steps', {}).values() if step.get('completed', False))}/5 steps"
                })
            
            df_activity = pd.DataFrame(activity_data)
            st.dataframe(df_activity, use_container_width=True)
        
    except Exception as e:
        st.error(f"Control panel error: {e}")

def main():
    """Main application function"""
    initialize_session_state()
    
    if not st.session_state.logged_in:
        render_login_page()
    else:
        render_sidebar()
        
        # Route to appropriate tab
        if st.session_state.current_tab == 'Home':
            render_home_tab()
        elif st.session_state.current_tab == 'Overview':
            render_overview_tab()
        elif st.session_state.current_tab == 'Configs':
            render_configs_tab()
        elif st.session_state.current_tab == 'Projects':
            render_projects_tab()
        elif st.session_state.current_tab == 'Control Panel':
            render_control_panel()
        else:
            render_home_tab()

if __name__ == "__main__":
    main()