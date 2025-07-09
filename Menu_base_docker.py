import streamlit as st
import paramiko
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Docker Remote Manager",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e86c1 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2e86c1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .error-card {
        background: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .info-card {
        background: #cce7ff;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    .docker-command {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def create_ssh_client(ip, user, pwd):
    """Create and return an SSH client."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=ip, username=user, password=pwd, timeout=10)
        return ssh
    except Exception as e:
        return None, str(e)

def execute_docker_command(ssh, command):
    """Execute a Docker command via SSH and return output and error."""
    try:
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        return output, error
    except Exception as e:
        return "", str(e)

def test_connection(ip, user, pwd):
    """Test SSH connection and return status."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(hostname=ip, username=user, password=pwd, timeout=5)
        ssh.close()
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🐳 Docker Remote Manager</h1>
    <p>Manage Docker containers on remote RHEL9 servers with ease</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for SSH Configuration
with st.sidebar:
    st.markdown("### 🔐 SSH Configuration")
    
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        rhel_ip = st.text_input("🌐 RHEL9 IP Address", value="10.167.24.14", help="Enter the IP address of your RHEL9 server")
        username = st.text_input("👤 SSH Username", value="root", help="SSH username for authentication")
        password = st.text_input("🔑 SSH Password", type="password", help="SSH password for authentication")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Connection test
    if st.button("🔍 Test Connection", type="secondary"):
        if rhel_ip and username and password:
            with st.spinner("Testing connection..."):
                success, message = test_connection(rhel_ip, username, password)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        else:
            st.warning("Please fill in all connection details")
    
    st.markdown("---")
    
    # Connection status
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = None
    
    if st.session_state.connection_status:
        st.markdown("### 📊 Connection Status")
        st.success("🟢 Connected to remote server")
        st.info(f"**Server:** {rhel_ip}")
        st.info(f"**User:** {username}")
        st.info(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Docker Operations")
    
    # Docker operation selection
    docker_options = {
        "🚀 Launch Container": "launch",
        "⏹️ Stop Container": "stop", 
        "🗑️ Remove Container": "remove",
        "▶️ Start Container": "start",
        "📋 List Docker Images": "images",
        "📦 List Running Containers": "containers",
        "⬇️ Pull Docker Image": "pull",
        "🔄 Container Status": "status"
    }
    
    choice = st.selectbox("Choose Docker Operation", list(docker_options.keys()))
    selected_operation = docker_options[choice]
    
    # Dynamic inputs based on selection
    container_name = ""
    image_name = ""
    
    if selected_operation in ["launch", "stop", "remove", "start", "status"]:
        container_name = st.text_input("📦 Container Name", placeholder="Enter container name", help="Name of the Docker container")
    
    if selected_operation in ["launch", "pull"]:
        image_name = st.text_input("🐳 Image Name", placeholder="e.g., nginx:latest", help="Docker image name with tag")
    
    # Additional options for launch
    if selected_operation == "launch":
        with st.expander("🔧 Advanced Launch Options"):
            port_mapping = st.text_input("🔌 Port Mapping", placeholder="e.g., 8080:80", help="Host:Container port mapping")
            volume_mapping = st.text_input("💾 Volume Mapping", placeholder="e.g., /host/path:/container/path", help="Host:Container volume mapping")
            environment_vars = st.text_input("🌿 Environment Variables", placeholder="e.g., ENV_VAR=value", help="Environment variables")

with col2:
    st.markdown("### 💡 Quick Actions")
    
    # Quick action buttons
    if st.button("🔄 Refresh Status", type="secondary"):
        st.rerun()
    
    if st.button("📊 System Info", type="secondary"):
        if rhel_ip and username and password:
            ssh_result = create_ssh_client(rhel_ip, username, password)
            if isinstance(ssh_result, tuple):
                ssh, error = ssh_result
                st.error(f"Connection failed: {error}")
            else:
                ssh = ssh_result
                output, error = execute_docker_command(ssh, "docker system df")
                if output:
                    st.code(output)
                ssh.close()
    
    st.markdown("---")
    st.markdown("### 📚 Command Reference")
    st.markdown("""
    - **Launch**: Create and start a new container
    - **Stop**: Stop a running container
    - **Remove**: Delete a container
    - **Start**: Start a stopped container
    - **Images**: List all Docker images
    - **Containers**: List running containers
    - **Pull**: Download an image from registry
    - **Status**: Check container status
    """)

# Execute button
st.markdown("---")
if st.button("🚀 Execute Docker Command", type="primary", use_container_width=True):
    # Validation
    if not rhel_ip or not username or not password:
        st.error("❌ Please fill in all SSH connection details")
        st.stop()
    
    if selected_operation in ["launch", "stop", "remove", "start", "status"] and not container_name:
        st.error("❌ Container name is required for this operation")
        st.stop()
    
    if selected_operation in ["launch", "pull"] and not image_name:
        st.error("❌ Image name is required for this operation")
        st.stop()
    
    # Create SSH connection
    with st.spinner("🔗 Connecting to remote server..."):
        ssh_result = create_ssh_client(rhel_ip, username, password)
        
        if isinstance(ssh_result, tuple):
            ssh, error = ssh_result
            st.error(f"❌ SSH connection failed: {error}")
            st.stop()
        else:
            ssh = ssh_result
            st.session_state.connection_status = True
    
    # Build Docker command
    docker_cmd = ""
    
    if selected_operation == "launch":
        docker_cmd = f"docker run -dit --name {container_name}"
        if 'port_mapping' in locals() and port_mapping:
            docker_cmd += f" -p {port_mapping}"
        if 'volume_mapping' in locals() and volume_mapping:
            docker_cmd += f" -v {volume_mapping}"
        if 'environment_vars' in locals() and environment_vars:
            docker_cmd += f" -e {environment_vars}"
        docker_cmd += f" {image_name}"
    elif selected_operation == "stop":
        docker_cmd = f"docker stop {container_name}"
    elif selected_operation == "remove":
        docker_cmd = f"docker rm {container_name}"
    elif selected_operation == "start":
        docker_cmd = f"docker start {container_name}"
    elif selected_operation == "images":
        docker_cmd = "docker images"
    elif selected_operation == "containers":
        docker_cmd = "docker ps"
    elif selected_operation == "pull":
        docker_cmd = f"docker pull {image_name}"
    elif selected_operation == "status":
        docker_cmd = f"docker ps -a --filter name={container_name}"
    
    # Display command being executed
    st.markdown(f'<div class="docker-command"><strong>Executing:</strong> {docker_cmd}</div>', unsafe_allow_html=True)
    
    # Execute command
    with st.spinner("⚡ Executing Docker command..."):
        output, error = execute_docker_command(ssh, docker_cmd)
    
    # Display results
    if output:
        st.markdown('<div class="status-card success-card">', unsafe_allow_html=True)
        st.markdown("**✅ Command executed successfully!**")
        st.code(output, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if error:
        st.markdown('<div class="status-card error-card">', unsafe_allow_html=True)
        st.markdown("**❌ Command completed with errors:**")
        st.code(error, language="bash")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not output and not error:
        st.markdown('<div class="status-card info-card">', unsafe_allow_html=True)
        st.markdown("**ℹ️ Command executed but produced no output**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Close SSH connection
    ssh.close()
    
    # Success message
    st.success("🎉 Operation completed successfully!")
    
    # Auto-refresh option
    if st.checkbox("🔄 Auto-refresh after operations"):
        time.sleep(1)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🐳 Docker Remote Manager | Built with Streamlit | 
    <a href="https://docs.docker.com/" target="_blank">Docker Documentation</a> | 
    <a href="https://docs.streamlit.io/" target="_blank">Streamlit Docs</a></p>
</div>
""", unsafe_allow_html=True)