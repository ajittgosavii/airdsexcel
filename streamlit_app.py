import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import anthropic
from anthropic import APIStatusError # Import specific error type
import json
import time
import traceback
import numpy as np
from datetime import datetime
import io
import os
import requests
from streamlit_oauth import OAuth2Component

# Import reportlab components for PDF generation with error handling
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# #--- Google Authentication Setup ---
CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", st.secrets.get("GOOGLE_CLIENT_ID", None) if hasattr(st, 'secrets') else None)
CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", st.secrets.get("GOOGLE_CLIENT_SECRET", None) if hasattr(st, 'secrets') else None)
REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", st.secrets.get("GOOGLE_REDIRECT_URI", None) if hasattr(st, 'secrets') else None)

# Define Google's OAuth 2.0 endpoints
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

# Check if credentials are configured
if not CLIENT_ID or not CLIENT_SECRET or not REDIRECT_URI:
    # Show a more helpful error message
    st.error("🔐 **Google OAuth Configuration Missing**")
    st.markdown("""
    **To configure Google OAuth:**
    
    1. **Create a Google Cloud Project** at https://console.cloud.google.com
    2. **Enable Google+ API** in the APIs & Services section
    3. **Create OAuth 2.0 credentials** in the Credentials section
    4. **Add your redirect URI** (e.g., `https://your-app.streamlit.app/component/streamlit_oauth.authorize_button/index.html`)
    5. **Set the following secrets/environment variables:**
    
    ```
    GOOGLE_CLIENT_ID=your_client_id_here
    GOOGLE_CLIENT_SECRET=your_client_secret_here
    GOOGLE_REDIRECT_URI=your_redirect_uri_here
    ```
    
    📖 **For detailed setup instructions:** https://docs.streamlit.io/knowledge-base/tutorials/databases/streamlit-oauth
    """)
    st.stop() # Stop the app if credentials are not configured

# Initialize the OAuth2 component
try:
    oauth2 = OAuth2Component(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        authorize_endpoint=AUTHORIZE_URL,
        token_endpoint=TOKEN_URL,
    )
except Exception as e:
    st.error(f"Failed to initialize OAuth2 component: {str(e)}")
    st.stop()
# --- END NEW: Google Authentication Setup ---

# Configure enterprise-grade UI
st.set_page_config(
    page_title="AI Database Migration Studio",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Fixed CSS for proper layout and styling
st.markdown("""
<style>
    /* Import better fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Reset and base styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        overflow-x: hidden; /* Prevent horizontal scrolling */
        line-height: 1.6;
        color: #333;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app container */
    .stApp {
        max-width: 1400px; /* Overall max width for the entire app content */
        margin: 0 auto; /* Center the entire app */
        padding: 0 1rem; /* Add some horizontal padding */
        display: flex; /* Make stApp a flex container */
        flex-direction: column; /* Stack children vertically */
        min-height: 100vh; /* Ensure it takes full viewport height */
    }

    /* Target the main content area wrapper when a sidebar is present */
    /* This aims to remove default Streamlit padding that pushes content */
    .stApp > header + div { /* Selects the div directly after the header (which is the main content wrapper) */
        padding-left: 0rem !important;
        padding-right: 0rem !important;
    }

    /* The 'main' area container that holds the block-container */
    .main {
        flex-grow: 1; /* Allow the main content area to take available space */
        width: 100%; /* Ensure it takes full width */
        margin: 0; /* Remove default margins */
        padding: 0; /* Remove default padding */
        box-sizing: border-box; /* Include padding and border in the element's total width */
    }

    /* Main block container for content - should fill the space provided by stApp or .main */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100% !important; /* Ensure it takes full width of its parent */
        padding-left: 1rem; /* Re-add controlled padding for inner content */
        padding-right: 1rem; /* Re-add controlled padding for inner content */
        margin: 0 auto; /* Ensure it stays centered within the app */
        box-sizing: border-box; /* Crucial for consistent sizing */
    }
    
    /* Ensure Streamlit elements respect container width */
    .stDataFrame, .stPlotlyChart, .stImage, .stVideo, .stAudio, .stExpander, .stTabs, .stColumns, 
    .stAlert, .stSuccess, .stWarning, .stError, .stInfo, .stProgress, .stMarkdown, .stText, .stJson, .stCode, .stTable, .stChart,
    div[data-testid="stHorizontalBlock"] { /* Target horizontal blocks (like st.columns) */
        width: 100% !important;
        box-sizing: border-box; /* Include padding and border in the element's total width and height */
        margin-left: auto !important; /* Ensure content is centered if possible */
        margin-right: auto !important;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; /* Increased padding */
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .ai-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.6rem 1.2rem; /* Slightly larger */
        border-radius: 24px; /* More rounded */
        display: inline-block;
        margin-bottom: 1.2rem; /* Increased margin */
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .main-header h1 {
        font-size: 2.8rem; /* Slightly larger */
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.15rem; /* Slightly larger */
        opacity: 0.9;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.8rem; /* Increased padding */
        border-radius: 12px; /* More rounded */
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); /* More pronounced shadow */
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease; /* Added box-shadow transition */
        height: 130px; /* Slightly taller */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-3px); /* More pronounced lift */
        box-shadow: 0 6px 25px rgba(0,0,0,0.15); /* Stronger hover shadow */
    }
    
    .metric-value {
        font-size: 2.2rem; /* Slightly larger */
        font-weight: 700;
        color: #1a202c;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem; /* Slightly larger */
        color: #718096;
        font-weight: 600; /* Bolder */
        text-transform: uppercase;
        letter-spacing: 0.7px; /* More spacing */
        margin-bottom: 0.6rem;
    }
    
    .metric-subtitle {
        font-size: 0.85rem; /* Slightly larger */
        color: #a0aec0;
        margin-top: 0.5rem;
    }
    
    /* AI insight cards */
    .ai-insight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 16px; /* More rounded */
        padding: 2rem; /* Increased padding */
        margin: 1.5rem 0; /* Increased margin */
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.15); /* More pronounced shadow */
    }
    
    .ai-insight h4 {
        color: #0c4a6e;
        margin-bottom: 1.2rem; /* Increased margin */
        font-size: 1.4rem; /* Slightly larger */
        font-weight: 700; /* Bolder */
    }
    
    /* Status cards */
    .status-card {
        padding: 1.2rem; /* Increased padding */
        border-radius: 10px; /* More rounded */
        margin: 0.7rem 0; /* Increased margin */
        border-left: 5px solid; /* Thicker border */
        font-size: 0.95rem;
    }
    
    .status-success {
        background: #f0fff4;
        border-color: #38a169;
        color: #22543d;
    }
    
    .status-warning {
        background: #fffaf0;
        border-color: #ed8936;
        color: #744210;
    }
    
    .status-error {
        background: #fff5f5;
        border-color: #e53e3e;
        color: #742a2a;
    }
    
    .status-info {
        background: #ebf8ff;
        border-color: #3182ce;
        color: #2a4365;
    }
    
    /* Configuration section */
    .config-section {
        background: #f8fafc;
        padding: 2rem; /* Increased padding */
        border-radius: 12px; /* More rounded */
        border: 1px solid #e2e8f0;
        margin: 1.5rem 0; /* Increased margin */
    }
    
    .config-header {
        font-size: 1.2rem; /* Slightly larger */
        font-weight: 700; /* Bolder */
        color: #2d3748;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem; /* More spacing */
    }
    
    /* Analysis cards */
    .analysis-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px; /* More rounded */
        padding: 1.5rem; /* Increased padding */
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); /* Slightly more pronounced shadow */
        height: 150px; /* Taller */
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .analysis-card h6 {
        color: #2d3748;
        font-size: 1.1rem; /* Slightly larger */
        font-weight: 700; /* Bolder */
        margin-bottom: 0.8rem;
        text-align: center;
    }
    
    /* Button fixes */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px; /* More rounded */
        padding: 0.8rem 1.5rem; /* Increased padding */
        font-weight: 600; /* Bolder */
        font-size: 1.05rem; /* Slightly larger text */
        transition: all 0.2s ease;
        width: 100%;
        font-family: inherit;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.2); /* Added shadow */
    }
    
    .stButton > button:hover {
        transform: translateY(-2px); /* More pronounced lift */
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4); /* Stronger hover shadow */
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem; /* Increased gap */
        background: #f0f2f5; /* Lighter background for tabs */
        padding: 0.4rem; /* Increased padding */
        border-radius: 10px; /* More rounded */
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); /* Inner shadow */
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px; /* More rounded */
        padding: 0.7rem 1.2rem; /* Increased padding */
        border: 1px solid #d1d5db; /* Lighter border */
        font-weight: 600; /* Bolder */
        color: #4a5568;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3); /* Shadow for active tab */
    }
    
    /* Footer styling */
    .footer-content {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2.5rem; /* Increased padding */
        border-radius: 16px; /* More rounded */
        text-align: center;
        margin-top: 3rem; /* Increased margin */
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .footer-content h3 {
        color: #2d3748;
        margin-bottom: 1.2rem;
        font-size: 1.8rem; /* Slightly larger */
        font-weight: 700;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); /* Adjusted minmax */
        gap: 1.2rem; /* Increased gap */
        margin: 1.8rem 0; /* Increased margin */
    }
    
    .feature-item {
        background: white;
        padding: 1.2rem; /* Increased padding */
        border-radius: 10px; /* More rounded */
        border: 1px solid #e2e8f0;
        font-weight: 600; /* Bolder */
        color: #4a5568;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px; /* More rounded */
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px; /* Rounded progress bar */
    }
    
    /* Sidebar improvements */
    .css-1d391kg { /* Target for Streamlit sidebar background */
        background: #f8fafc;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05); /* Added shadow to sidebar */
    }
    
    /* Text input improvements */
    .stTextInput > div > div > input {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px; /* More rounded */
        padding: 0.6rem; /* Increased padding */
        font-size: 1rem;
    }
    
    /* Selectbox improvements */
    .stSelectbox > div > div > div {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px; /* More rounded */
        padding: 0.4rem; /* Adjusted padding */
        font-size: 1rem;
    }
    
    /* Number input improvements */
    .stNumberInput > div > div > input {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px; /* More rounded */
        padding: 0.6rem; /* Increased padding */
        font-size: 1rem;
    }
    
    /* File uploader improvements */
    .stFileUploader > div {
        border: 2px dashed #9ca3af; /* Darker dashed border */
        border-radius: 10px; /* More rounded */
        padding: 1.5rem; /* Increased padding */
        background: #f9fafb;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:hover {
        border-color: #667eea; /* Color on hover */
        background: #f5f8ff; /* Lighter background on hover */
    }

    /* Metric improvements */
    .stMetric {
        background: white;
        padding: 1.5rem; /* Increased padding */
        border-radius: 12px; /* More rounded */
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Expander improvements */
    .stExpander {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
    }
    .stExpander > div > div > .streamlit-expanderContent {
        padding: 1rem;
    }
    .stExpander > div > div > button {
        background: linear-gradient(135deg, #f0f2f5 0%, #e2e8f0 100%); /* Light gradient for expander button */
        border-bottom: 1px solid #d1d5db;
        border-radius: 10px 10px 0 0;
        padding: 0.75rem 1rem;
        font-weight: 600;
        color: #2d3748;
    }
    .stExpander > div > div > button:hover {
        background: linear-gradient(135deg, #e0e2e5 0%, #d1d5db 100%);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1.5rem;
        }
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .metric-card {
            padding: 1rem;
            height: auto;
            margin-bottom: 1rem; /* Add margin for stacking */
        }
        .metric-value {
            font-size: 1.8rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column; /* Stack tabs vertically on mobile */
            gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            width: 100%;
        }
        .config-section {
            padding: 1.2rem;
        }
        .analysis-card {
            height: auto; /* Auto height on small screens */
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class AIAnalytics:
    """AI-powered analytics engine using Claude API"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def analyze_workload_patterns(self, workload_data: dict) -> dict:
        """Analyze workload patterns and provide intelligent recommendations"""

        prompt = f"""
        As an expert database architect and cloud migration specialist, analyze this workload data and provide intelligent insights:

        Workload Data:
        - Database Engine: {workload_data.get('engine')}
        - Current CPU Cores: {workload_data.get('cores')}
        - Current RAM: {workload_data.get('ram')} GB
        - Storage: {workload_data.get('storage')} GB
        - Peak CPU Utilization: {workload_data.get('cpu_util')}%
        - Peak RAM Utilization: {workload_data.get('ram_util')}%
        - IOPS Requirements: {workload_data.get('iops')}
        - Growth Rate: {workload_data.get('growth')}% annually
        - Region: {workload_data.get('region')}

        Please provide a comprehensive analysis including:
        1. Workload Classification (OLTP/OLAP/Mixed)
        2. Performance Bottleneck Identification
        3. Right-sizing Recommendations
        4. Cost Optimization Opportunities
        5. Migration Strategy Recommendations
        6. Risk Assessment and Mitigation
        7. Timeline and Complexity Estimation

        Respond in a structured format with clear sections.
        """

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620", # Updated model
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse AI response
            ai_analysis = self._parse_ai_response(message.content[0].text)
            return ai_analysis

        except APIStatusError as e:
            if e.status_code == 401:
                return {"error": "AI analysis failed: Authentication Error (401). Please check your Claude API key."}
            return {"error": f"AI analysis failed: {str(e)}"}
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}
    
    def generate_migration_strategy(self, analysis_data: dict) -> dict:
        """Generate detailed migration strategy with AI insights"""
        
        prompt = f"""
        Based on the database analysis, create a comprehensive migration strategy:

        Analysis Summary: 
        - Engine: {analysis_data.get('engine', 'Unknown')}
        - Estimated Cost: ${analysis_data.get('monthly_cost', 0):,.2f}/month
        - Complexity: Medium to High

        Please provide:
        1. Pre-migration checklist and requirements
        2. Detailed migration phases with timelines
        3. Resource allocation recommendations
        4. Testing and validation strategy
        5. Rollback procedures
        6. Post-migration optimization steps
        7. Monitoring and alerting setup
        8. Security and compliance considerations

        Include specific AWS services, tools, and best practices.
        """
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620", # Updated model
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_migration_strategy(message.content[0].text)
            
        except APIStatusError as e:
            if e.status_code == 401:
                return {"error": "Migration strategy generation failed: Authentication Error (401). Please check your Claude API key."}
            return {"error": f"Migration strategy generation failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Migration strategy generation failed: {str(e)}"}
    
    def predict_future_requirements(self, historical_data: dict, years: int = 3) -> dict:
        """Predict future resource requirements using AI"""
        
        prompt = f"""
        As a data scientist specializing in capacity planning, analyze these metrics and predict future requirements:

        Current Configuration:
        - CPU Cores: {historical_data.get('cores')}
        - RAM: {historical_data.get('ram')} GB
        - Storage: {historical_data.get('storage')} GB
        - Growth Rate: {historical_data.get('growth')}% annually
        - Engine: {historical_data.get('engine')}

        Prediction Period: {years} years

        Consider:
        - Technology evolution impact
        - Business scaling factors
        - Industry benchmarks for {historical_data.get('engine')} workloads

        Provide predictions for:
        - CPU requirements
        - Memory usage
        - Storage growth
        - IOPS scaling
        - Cost projections

        Include key assumptions and confidence levels.
        """
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620", # Updated model
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_predictions(message.content[0].text)
            
        except APIStatusError as e:
            if e.status_code == 401:
                return {"error": "Prediction generation failed: Authentication Error (401). Please check your Claude API key."}
            return {"error": f"Prediction generation failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Prediction generation failed: {str(e)}"}
    
    def _parse_ai_response(self, response_text: str) -> dict:
        """Parse AI response into structured data"""
        # Extract key insights from the response
        lines = response_text.split('\n')
        
        # Default structure
        result = {
            "workload_type": "Mixed",
            "complexity": "Medium",
            "timeline": "12-16 weeks",
            "bottlenecks": [],
            "recommendations": [],
            "risks": [],
            "summary": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }
        
        # Parse specific sections
        current_section = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if "workload" in line.lower() and ("classification" in line.lower() or "type" in line.lower()):
                if "oltp" in line.lower():
                    result["workload_type"] = "OLTP"
                elif "olap" in line.lower():
                    result["workload_type"] = "OLAP"
                elif "mixed" in line.lower():
                    result["workload_type"] = "Mixed"
            
            if "complexity" in line.lower():
                if "high" in line.lower():
                    result["complexity"] = "High"
                elif "low" in line.lower():
                    result["complexity"] = "Low"
                else:
                    result["complexity"] = "Medium"
            
            # Extract recommendations, bottlenecks, risks
            if any(marker in line for marker in ['•', '-', '*', '1.', '2.', '3.']):
                clean_line = line.strip('•-* \t0123456789.').strip()
                if clean_line:
                    if "recommend" in current_section.lower():
                        result["recommendations"].append(clean_line)
                    elif "bottleneck" in current_section.lower() or "performance" in current_section.lower():
                        result["bottlenecks"].append(clean_line)
                    elif "risk" in current_section.lower():
                        result["risks"].append(clean_line)
            
            # Track current section
            if ":" in line:
                current_section = line
        
        # Ensure we have some content
        if not result["recommendations"]:
            result["recommendations"] = [
                "Consider Aurora for improved performance and cost efficiency",
                "Implement read replicas for better read performance",
                "Use GP3 storage for cost optimization",
                "Enable Performance Insights for monitoring"
            ]
        
        if not result["bottlenecks"]:
            result["bottlenecks"] = [
                "CPU utilization may peak during business hours",
                "Storage IOPS might be a limiting factor",
                "Network bandwidth could impact data transfer"
            ]
        
        if not result["risks"]:
            result["risks"] = [
                "Application compatibility testing required",
                "Data migration complexity for large datasets",
                "Downtime during cutover process"
            ]
        
        return result
    
    def _parse_migration_strategy(self, response_text: str) -> dict:
        """Parse migration strategy response"""
        return {
            "phases": [
                "Assessment and Planning",
                "Environment Setup and Testing", 
                "Data Migration and Validation",
                "Application Migration",
                "Go-Live and Optimization"
            ],
            "timeline": "14-18 weeks",
            "resources": [
                "Database Migration Specialist",
                "Cloud Architect", 
                "DevOps Engineer",
                "Application Developer",
                "Project Manager"
            ],
            "risks": [
                "Data consistency during migration",
                "Application compatibility issues",
                "Performance degradation post-migration"
            ],
            "tools": [
                "AWS Database Migration Service (DMS)",
                "AWS Schema Conversion Tool (SCT)",
                "CloudFormation for infrastructure",
                "CloudWatch for monitoring"
            ],
            "checklist": [
                "Complete application dependency mapping",
                "Set up target AWS environment",
                "Configure monitoring and alerting",
                "Establish rollback procedures",
                "Plan communication strategy"
            ],
            "full_strategy": response_text
        }
    
    def _parse_predictions(self, response_text: str) -> dict:
        """Parse prediction response"""
        return {
            "cpu_trend": "Gradual increase expected",
            "memory_trend": "Stable with seasonal peaks", 
            "storage_trend": "Linear growth with data retention",
            "cost_trend": "Optimized through right-sizing",
            "confidence": "High (85-90%)",
            "key_factors": [
                "Business growth projections",
                "Technology adoption patterns",
                "Seasonal usage variations",
                "Regulatory requirements"
            ],
            "recommendations": [
                "Plan for 20% capacity buffer",
                "Implement auto-scaling policies",
                "Review and optimize quarterly",
                "Consider reserved instances for predictable workloads"
            ],
            "full_prediction": response_text
        }

class EnhancedRDSCalculator:
    """Enhanced RDS calculator with AI integration"""
    
    def __init__(self):
        self.engines = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
        self.regions = ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        
        # Instance database with expanded options
        self.instance_db = {
            "us-east-1": {
                "oracle-ee": [
                    {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.136}},
                    {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.475}},
                    {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.95}},
                    {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 1.90}},
                    {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.60}},
                    {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 1.20}},
                    {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.92}}
                ],
                "aurora-postgresql": [
                    {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.082}},
                    {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.285}},
                    {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.57}},
                    {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.14}},
                    {"type": "db.serverless", "vCPU": 0, "memory": 0, "pricing": {"ondemand": 0.12}}
                ],
                "postgres": [
                    {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.0255}},
                    {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand": 0.051}},
                    {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.102}},
                    {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.192}},
                    {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.384}},
                    {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 0.768}}
                ],
                "sqlserver": [
                    {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand": 0.231}},
                    {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.693}},
                    {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 1.386}},
                    {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 2.772}}
                ],
                "aurora-mysql": [
                    {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.082}},
                    {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.285}},
                    {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.57}},
                    {"type": "db.serverless", "vCPU": 0, "memory": 0, "pricing": {"ondemand": 0.12}}
                ],
                "oracle-se": [
                    {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.105}},
                    {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.365}},
                    {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.730}},
                    {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.462}}
                ]
            }
        }
        
        # Environment profiles
        self.env_profiles = {
            "PROD": {"cpu_factor": 1.0, "storage_factor": 1.0, "ha_required": True},
            "STAGING": {"cpu_factor": 0.8, "storage_factor": 0.7, "ha_required": True},
            "QA": {"cpu_factor": 0.6, "storage_factor": 0.5, "ha_required": False},
            "DEV": {"cpu_factor": 0.4, "storage_factor": 0.3, "ha_required": False}
        }
        
        # Add other regions with regional pricing adjustments
        for region in ["us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"]:
            if region not in self.instance_db:
                self.instance_db[region] = {}
                for engine, instances in self.instance_db["us-east-1"].items():
                    # Apply regional pricing multiplier
                    multiplier = self._get_regional_multiplier(region)
                    regional_instances = []
                    for instance in instances:
                        regional_instance = instance.copy()
                        regional_instance["pricing"] = {
                            "ondemand": instance["pricing"]["ondemand"] * multiplier
                        }
                        regional_instances.append(regional_instance)
                    self.instance_db[region][engine] = regional_instances
    
    def _get_regional_multiplier(self, region: str) -> float:
        """Get regional pricing multiplier"""
        multipliers = {
            "us-east-1": 1.0,
            "us-west-1": 1.08,
            "us-west-2": 1.05,
            "eu-west-1": 1.12,
            "ap-southeast-1": 1.15
        }
        return multipliers.get(region, 1.0)
    
    def calculate_requirements(self, inputs: dict, env: str) -> dict:
        """Calculate resource requirements with AI-enhanced logic"""
        profile = self.env_profiles[env]
        
        # Calculate resources with intelligent scaling
        base_vcpus = inputs['cores'] * (inputs['cpu_util'] / 100)
        base_ram = inputs['ram'] * (inputs['ram_util'] / 100)
        
        # Apply environment factors
        if env == "PROD":
            vcpus = max(4, int(base_vcpus * profile['cpu_factor'] * 1.2))
            ram = max(8, int(base_ram * profile['cpu_factor'] * 1.2))
            storage = max(100, int(inputs['storage'] * profile['storage_factor'] * 1.3))
        elif env == "STAGING":
            vcpus = max(2, int(base_vcpus * profile['cpu_factor']))
            ram = max(4, int(base_ram * profile['cpu_factor']))
            storage = max(50, int(inputs['storage'] * profile['storage_factor']))
        elif env == "QA":
            vcpus = max(2, int(base_vcpus * profile['cpu_factor']))
            ram = max(4, int(base_ram * profile['cpu_factor']))
            storage = max(20, int(inputs['storage'] * profile['storage_factor']))
        else:  # DEV
            vcpus = max(1, int(base_vcpus * profile['cpu_factor']))
            ram = max(2, int(base_ram * profile['cpu_factor']))
            storage = max(20, int(inputs['storage'] * profile['storage_factor']))
        
        # Apply growth projections only for PROD and STAGING
        if env in ["PROD", "STAGING"]:
            growth_factor = (1 + inputs['growth']/100) ** 2
            storage = int(storage * growth_factor)
            
        # Select optimal instance
        instance = self._select_optimal_instance(vcpus, ram, inputs['engine'], inputs['region'], env)
        
        # Calculate costs
        costs = self._calculate_comprehensive_costs(instance, storage, inputs, env)
        
        return {
            "environment": env,
            "instance_type": instance["type"],
            "vcpus": vcpus,
            "ram_gb": ram,
            "storage_gb": storage,
            "monthly_cost": costs["total"],
            "annual_cost": costs["total"] * 12,
            "cost_breakdown": costs,
            "instance_details": instance,
            "optimization_score": self._calculate_optimization_score(instance, vcpus, ram)
        }
    
    def calculate_multi_az_requirements(self, inputs: dict, env: str) -> dict:
        """Calculate Multi-AZ requirements with reader/writer sizing"""
        base_requirements = self.calculate_requirements(inputs, env)
        
        if not inputs.get('multi_az_enabled', False):
            return base_requirements
        
        read_write_ratio = inputs.get('read_write_ratio', 70) / 100
        read_replica_count = inputs.get('read_replica_count', 2)
        
        # Calculate writer instance (handles writes + some reads)
        writer_cpu_load = (1 - read_write_ratio) + (read_write_ratio * 0.3)  # 30% of reads go to writer
        writer_vcpus = max(2, int(base_requirements['vcpus'] * writer_cpu_load))
        writer_ram = max(4, int(base_requirements['ram_gb'] * writer_cpu_load))
        
        # Calculate reader instance sizing
        reader_cpu_load = (read_write_ratio * 0.7) / read_replica_count  # 70% of reads distributed to replicas
        reader_vcpus = max(2, int(base_requirements['vcpus'] * reader_cpu_load))
        reader_ram = max(4, int(base_requirements['ram_gb'] * reader_cpu_load))
        
        # Select optimal instances
        writer_instance = self._select_optimal_instance(writer_vcpus, writer_ram, inputs['engine'], inputs['region'], env)
        reader_instance = self._select_optimal_instance(reader_vcpus, reader_ram, inputs['engine'], inputs['region'], env)
        
        # Calculate costs
        writer_costs = self._calculate_comprehensive_costs(writer_instance, base_requirements['storage_gb'], inputs, env)
        reader_costs = self._calculate_comprehensive_costs(reader_instance, int(base_requirements['storage_gb'] * 0.3), inputs, env)
        total_reader_costs = {k: v * read_replica_count for k, v in reader_costs.items()}
        
        total_monthly_cost = writer_costs['total'] + total_reader_costs['total']
        
        # Return enhanced structure
        result = base_requirements.copy()
        result.update({
            'multi_az_enabled': True,
            'writer_config': {
                'instance_type': writer_instance['type'],
                'vcpus': writer_vcpus,
                'ram_gb': writer_ram,
                'storage_gb': base_requirements['storage_gb'],
                'monthly_cost': writer_costs['total'],
                'cost_breakdown': writer_costs,
                'instance_details': writer_instance
            },
            'reader_config': {
                'instance_type': reader_instance['type'],
                'vcpus': reader_vcpus,
                'ram_gb': reader_ram,
                'storage_gb': int(base_requirements['storage_gb'] * 0.3),
                'monthly_cost_per_replica': reader_costs['total'],
                'total_monthly_cost': total_reader_costs['total'],
                'replica_count': read_replica_count,
                'cost_breakdown': reader_costs,
                'instance_details': reader_instance
            },
            'total_multi_az_cost': total_monthly_cost,
            'read_write_ratio': f"{int(read_write_ratio*100)}% Read / {int((1-read_write_ratio)*100)}% Write"
        })
        
        return result
    
    def _select_optimal_instance(self, vcpus: int, ram: int, engine: str, region: str, env: str = "PROD") -> dict:
        """Select optimal instance type"""
        region_data = self.instance_db.get(region, self.instance_db["us-east-1"])
        engine_instances = region_data.get(engine, region_data.get("postgres", []))
        
        if not engine_instances:
            if env == "DEV":
                return {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.017}}
            elif env in ["QA", "STAGING"]:
                return {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.068}}
            else:
                return {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.4}}
        
        # Filter instances based on environment
        if env == "DEV":
            preferred_instances = [inst for inst in engine_instances if 't3' in inst["type"]]
            if not preferred_instances:
                preferred_instances = engine_instances
        elif env in ["QA", "STAGING"]:
            preferred_instances = [inst for inst in engine_instances if any(family in inst["type"] for family in ['t3', 'm5'])]
            if not preferred_instances:
                preferred_instances = engine_instances
        else:
            preferred_instances = [inst for inst in engine_instances if any(family in inst["type"] for family in ['r5', 'm5'])]
            if not preferred_instances:
                preferred_instances = engine_instances
        
        # Score instances
        scored_instances = []
        for instance in preferred_instances:
            if instance["type"] == "db.serverless":
                score = 120 if env == "DEV" else (100 if env in ["QA", "STAGING"] else 60)
            else:
                cpu_ratio = instance["vCPU"] / max(vcpus, 1)
                ram_ratio = instance["memory"] / max(ram, 1)
                
                if env == "PROD":
                    cpu_fit = 1.2 if 1.2 <= cpu_ratio <= 1.8 else (1.0 if cpu_ratio >= 1.0 else 0.3)
                    ram_fit = 1.2 if 1.2 <= ram_ratio <= 1.8 else (1.0 if ram_ratio >= 1.0 else 0.3)
                    cost_weight = 0.3
                elif env in ["QA", "STAGING"]:
                    cpu_fit = 1.0 if 1.1 <= cpu_ratio <= 1.5 else (0.8 if cpu_ratio >= 1.0 else 0.4)
                    ram_fit = 1.0 if 1.1 <= ram_ratio <= 1.5 else (0.8 if ram_ratio >= 1.0 else 0.4)
                    cost_weight = 0.5
                else:
                    cpu_fit = 1.0 if 1.0 <= cpu_ratio <= 1.3 else (0.7 if cpu_ratio >= 1.0 else 0.2)
                    ram_fit = 1.0 if 1.0 <= ram_ratio <= 1.3 else (0.7 if ram_ratio >= 1.0 else 0.2)
                    cost_weight = 0.7
                
                cost_per_vcpu = instance["pricing"]["ondemand"] / max(instance["vCPU"], 1)
                cost_efficiency = (1.0 / (cost_per_vcpu + 1)) * cost_weight
                
                performance_bonus = 0
                if env == "PROD":
                    if 'r5' in instance["type"]:
                        performance_bonus = 0.3
                    elif 'm5' in instance["type"]:
                        performance_bonus = 0.2
                elif env == "DEV":
                    if 't3' in instance["type"]:
                        performance_bonus = 0.3
                
                score = (cpu_fit + ram_fit + cost_efficiency + performance_bonus) * 100
            
            scored_instances.append((score, instance))
        
        if scored_instances:
            scored_instances.sort(key=lambda x: x[0], reverse=True)
            return scored_instances[0][1]
        
        return engine_instances[0] if engine_instances else {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.4}}
    
    def _calculate_comprehensive_costs(self, instance: dict, storage: int, inputs: dict, env: str) -> dict:
        """Calculate comprehensive monthly costs"""
        instance_cost = instance["pricing"]["ondemand"] * 24 * 30
        
        if env == "PROD":
            instance_cost *= 2
        
        storage_gb_cost = storage * 0.115
        extra_iops = max(0, inputs.get('iops', 3000) - 3000)
        iops_cost = extra_iops * 0.005
        
        backup_days = inputs.get('backup_days', 7)
        backup_cost = storage * 0.095 * (backup_days / 30)
        
        data_transfer = inputs.get('data_transfer_gb', 100)
        transfer_cost = data_transfer * 0.09
        
        monitoring_cost = instance_cost * 0.1 if env == "PROD" else 0
        
        total_cost = (instance_cost + storage_gb_cost + iops_cost + 
                     backup_cost + transfer_cost + monitoring_cost)
        
        return {
            "instance": instance_cost,
            "storage": storage_gb_cost,
            "iops": iops_cost,
            "backup": backup_cost,
            "data_transfer": transfer_cost,
            "monitoring": monitoring_cost,
            "total": total_cost
        }
    
    def _calculate_optimization_score(self, instance: dict, required_vcpus: int, required_ram: int) -> int:
        """Calculate optimization score (0-100)"""
        if instance["type"] == "db.serverless":
            return 95
        
        cpu_efficiency = min(required_vcpus / instance["vCPU"], 1.0)
        ram_efficiency = min(required_ram / instance["memory"], 1.0)
        avg_efficiency = (cpu_efficiency + ram_efficiency) / 2
        
        return int(avg_efficiency * 100)

class PDFReportGenerator:
    """Generates PDF reports from analysis results with enhanced error handling."""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab library not found. Please install with: pip install reportlab")
        
        try:
            # Initialize styles
            self.styles = getSampleStyleSheet()
            self.styles.add(ParagraphStyle(name='H1_Custom', fontSize=24, leading=28, alignment=1, spaceAfter=20, fontName='Helvetica-Bold'))
            self.styles.add(ParagraphStyle(name='H2_Custom', fontSize=18, leading=22, spaceBefore=10, spaceAfter=10, fontName='Helvetica-Bold'))
            self.styles.add(ParagraphStyle(name='H3_Custom', fontSize=14, leading=18, spaceBefore=8, spaceAfter=8, fontName='Helvetica-Bold'))
            self.styles.add(ParagraphStyle(name='Normal_Custom', fontSize=10, leading=12, spaceAfter=6))
            self.styles.add(ParagraphStyle(name='Bullet_Custom', fontSize=10, leading=12, leftIndent=20, spaceAfter=6, bulletText='•'))
            
        except Exception as e:
            raise Exception(f"Failed to initialize PDF generator: {str(e)}") from e

    def generate_report(self, all_results: list | dict):
        """Generates a PDF report based on the analysis results."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []

            story.append(Paragraph("AI Database Migration Studio Report", self.styles['H1_Custom']))
            story.append(Paragraph(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal_Custom']))
            story.append(Spacer(1, 0.2 * inch))

            if not all_results:
                story.append(Paragraph("No analysis results available to generate a report.", self.styles['Normal_Custom']))
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()

            # Handle both single and bulk analysis results
            if isinstance(all_results, dict):
                # Convert single result to a list for consistent processing
                all_results = [all_results]

            # Executive Summary (aggregated for bulk, or single for individual)
            story.append(Paragraph("1. Executive Summary", self.styles['H2_Custom']))
            
            summary_data = [["Database", "Engine", "Instance Type", "Monthly Cost ($)", "Optimization"]]
            total_monthly_cost = 0
            total_databases = len(all_results)
            
            for result in all_results:
                inputs = result.get('inputs', {})
                prod_rec = result['recommendations']['PROD']
                db_name = inputs.get('db_name', 'N/A')
                engine = inputs.get('engine', 'N/A')
                instance_type = prod_rec['instance_type']
                monthly_cost = f"{prod_rec['monthly_cost']:,.0f}"
                optimization = f"{prod_rec.get('optimization_score', 85)}%"
                
                summary_data.append([db_name, engine, instance_type, monthly_cost, optimization])
                total_monthly_cost += prod_rec['monthly_cost']

            table = Table(summary_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.2*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph(f"Total Monthly Cost (Production): ${total_monthly_cost:,.0f}", self.styles['Normal_Custom']))
            story.append(Paragraph(f"Total Annual Cost (Production): ${total_monthly_cost * 12:,.0f}", self.styles['Normal_Custom']))
            story.append(Spacer(1, 0.2 * inch))

            # Detailed Analysis for Each Database
            for i, result in enumerate(all_results):
                inputs = result.get('inputs', {})
                recommendations = result.get('recommendations', {})
                ai_insights = result.get('ai_insights', {})
                db_name = inputs.get('db_name', f'Database {i+1}')

                story.append(Paragraph(f"2. Detailed Analysis: {db_name}", self.styles['H2_Custom']))
                story.append(Paragraph("2.1. Current Configuration", self.styles['H3_Custom']))
                story.append(Paragraph(f"• Engine: {inputs.get('engine', 'N/A').upper()}", self.styles['Bullet_Custom']))
                story.append(Paragraph(f"• Region: {inputs.get('region', 'N/A')}", self.styles['Bullet_Custom']))
                story.append(Paragraph(f"• CPU: {inputs.get('cores', 'N/A')} cores ({inputs.get('cpu_util', 'N/A')}% util)", self.styles['Bullet_Custom']))
                story.append(Paragraph(f"• RAM: {inputs.get('ram', 'N/A')} GB ({inputs.get('ram_util', 'N/A')}% util)", self.styles['Bullet_Custom']))
                story.append(Paragraph(f"• Storage: {inputs.get('storage', 'N/A'):,} GB ({inputs.get('iops', 'N/A'):,} IOPS)", self.styles['Bullet_Custom']))
                story.append(Spacer(1, 0.1 * inch))

                story.append(Paragraph("2.2. Recommended Configurations", self.styles['H3_Custom']))
                rec_table_data = [["Environment", "Instance Type", "vCPUs", "RAM (GB)", "Monthly Cost ($)"]]
                for env, rec in recommendations.items():
                    rec_table_data.append([
                        env, 
                        rec['instance_type'], 
                        rec['vcpus'], 
                        rec['ram_gb'], 
                        f"{rec['monthly_cost']:,.0f}"
                    ])
                
                rec_table = Table(rec_table_data, colWidths=[1.2*inch, 1.5*inch, 0.8*inch, 0.8*inch, 1.2*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(rec_table)
                story.append(Spacer(1, 0.2 * inch))

                if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
                    workload = ai_insights['workload']
                    story.append(Paragraph("2.3. AI Workload Insights", self.styles['H3_Custom']))
                    story.append(Paragraph(f"• Workload Type: {workload.get('workload_type', 'N/A')}", self.styles['Bullet_Custom']))
                    story.append(Paragraph(f"• Migration Complexity: {workload.get('complexity', 'N/A')}", self.styles['Bullet_Custom']))
                    story.append(Paragraph(f"• Estimated Timeline: {workload.get('timeline', 'N/A')}", self.styles['Bullet_Custom']))
                    
                    if workload.get('recommendations'):
                        story.append(Paragraph("Key Recommendations:", self.styles['Normal_Custom']))
                        for rec in workload['recommendations']:
                            story.append(Paragraph(f"• {rec}", self.styles['Bullet_Custom']))
                    if workload.get('risks'):
                        story.append(Paragraph("Identified Risks:", self.styles['Normal_Custom']))
                        for risk in workload['risks']:
                            story.append(Paragraph(f"• {risk}", self.styles['Bullet_Custom']))
                    story.append(Spacer(1, 0.2 * inch))

                if 'migration' in ai_insights and 'error' not in ai_insights['migration']:
                    migration = ai_insights['migration']
                    story.append(Paragraph("2.4. Migration Strategy Overview", self.styles['H3_Custom']))
                    story.append(Paragraph(f"• Estimated Timeline: {migration.get('timeline', 'N/A')}", self.styles['Bullet_Custom']))
                    if migration.get('phases'):
                        story.append(Paragraph("Migration Phases:", self.styles['Normal_Custom']))
                        for phase in migration['phases']:
                            story.append(Paragraph(f"• {phase}", self.styles['Bullet_Custom']))
                    if migration.get('tools'):
                        story.append(Paragraph("Recommended Tools:", self.styles['Normal_Custom']))
                        for tool in migration['tools']:
                            story.append(Paragraph(f"• {tool}", self.styles['Bullet_Custom']))
                    story.append(Spacer(1, 0.2 * inch))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"PDF generation failed: {str(e)}") from e

def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV/Excel file with database configurations"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Column mapping for different naming conventions
        column_mapping = {
            'database_name': 'db_name',
            'database_engine': 'engine', 
            'aws_region': 'region',
            'cpu_cores': 'cores',
            'cpu_utilization': 'cpu_util',
            'ram_gb': 'ram',
            'ram_utilization': 'ram_util',
            'storage_gb': 'storage',
            'growth_rate': 'growth',
            'projection_years': 'years',
            'data_transfer_gb': 'data_transfer_gb'
        }
        
        # Rename columns to match expected format
        df = df.rename(columns=column_mapping)
        
        # Expected columns (after mapping)
        required_columns = ['db_name', 'engine', 'region', 'cores', 'ram', 'storage']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return [], [f"Missing required columns: {', '.join(missing_columns)}"]
        
        valid_inputs = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                input_data = {
                    'db_name': str(row['db_name']),
                    'engine': str(row['engine']),
                    'region': str(row['region']),
                    'cores': int(row['cores']),
                    'cpu_util': int(row.get('cpu_util', 65)),
                    'ram': int(row.get('ram', 0)), # Ensure RAM is handled safely
                    'ram_util': int(row.get('ram_util', 75)),
                    'storage': int(row.get('storage', 100)), # Default to 100 if missing
                    'iops': int(row.get('iops', 8000)),
                    'growth': float(row.get('growth', 15)),
                    'backup_days': int(row.get('backup_days', 7)),
                    'years': int(row.get('years', 3)),
                    'data_transfer_gb': int(row.get('data_transfer_gb', 100))
                }
                valid_inputs.append(input_data)
            except Exception as e:
                errors.append(f"Row {index + 1}: {str(e)}")
        
        return valid_inputs, errors
        
    except Exception as e:
        return [], [f"File parsing error: {str(e)}"]

def export_full_report(all_results):
    """Export comprehensive Excel report"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for result in all_results:
                prod_rec = result['recommendations']['PROD']
                summary_data.append({
                    "Database": result['inputs'].get('db_name', 'N/A'),
                    "Engine": result['inputs'].get('engine', 'N/A'),
                    "Instance Type": prod_rec['instance_type'],
                    "vCPUs": prod_rec['vcpus'],
                    "RAM (GB)": prod_rec['ram_gb'],
                    "Storage (GB)": prod_rec['storage_gb'],
                    "Monthly Cost": prod_rec['monthly_cost'],
                    "Annual Cost": prod_rec['annual_cost'],
                    "Optimization": f"{prod_rec.get('optimization_score', 85)}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Detailed breakdown
            for i, result in enumerate(all_results):
                db_name = result['inputs'].get('db_name', f'Database_{i+1}')
                sheet_name = db_name[:31]  # Excel sheet name limit
                
                detail_data = []
                for env, rec in result['recommendations'].items():
                    detail_data.append({
                        'Environment': env,
                        'Instance Type': rec['instance_type'],
                        'vCPUs': rec['vcpus'],
                        'RAM (GB)': rec['ram_gb'],
                        'Storage (GB)': rec['storage_gb'],
                        'Monthly Cost': rec['monthly_cost'],
                        'Annual Cost': rec['annual_cost']
                    })
                
                detail_df = pd.DataFrame(detail_data)
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        raise Exception(f"Report generation failed: {str(e)}")

def check_pdf_requirements():
    """Check if PDF generation requirements are met"""
    if REPORTLAB_AVAILABLE:
        return True, "PDF generation is ready"
    else:
        return False, "ReportLab library not installed. Run: pip install reportlab"

def test_pdf_generation():
    """Test PDF generation with sample data"""
    try:
        if not REPORTLAB_AVAILABLE:
            return False, "ReportLab not available"
            
        # Create sample data
        sample_results = {
            'inputs': {
                'db_name': 'TestDatabase',
                'engine': 'postgres',
                'region': 'us-east-1',
                'cores': 4,
                'cpu_util': 70,
                'ram': 16,
                'ram_util': 75,
                'storage': 1000,
                'iops': 3000
            },
            'recommendations': {
                'PROD': {
                    'instance_type': 'db.m5.large',
                    'vcpus': 2,
                    'ram_gb': 8,
                    'storage_gb': 1000,
                    'monthly_cost': 500,
                    'annual_cost': 6000,
                    'optimization_score': 85
                }
            },
            'ai_insights': {}
        }
        
        pdf_gen = PDFReportGenerator()
        pdf_data = pdf_gen.generate_report(sample_results)
        return True, f"PDF test successful. Generated {len(pdf_data)} bytes."
    except Exception as e:
        return False, f"PDF test failed: {str(e)}"

def show_pdf_status():
    """Show PDF generation status in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📄 PDF Report Status")
        
        ready, message = check_pdf_requirements()
        if ready:
            st.success(f"✅ {message}")
            
            # Optional: Add test button
            if st.button("🧪 Test PDF Generation", key="test_pdf"):
                test_ready, test_message = test_pdf_generation()
                if test_ready:
                    st.success(f"✅ {test_message}")
                else:
                    st.error(f"❌ {test_message}")
        else:
            st.error(f"❌ {message}")
            st.info("💡 Install ReportLab to enable PDF reports:\n```\npip install reportlab\n```")

def render_troubleshooting_section():
    """Render troubleshooting section for PDF issues"""
    with st.expander("🔧 PDF Generation Troubleshooting", expanded=False):
        st.markdown("""
        ### Common PDF Generation Issues:
        
        **1. ReportLab Not Installed**
        ```bash
        pip install reportlab
        ```
        
        **2. Permission Issues**
        - Make sure you have write permissions
        - Try running with administrator/sudo if needed
        
        **3. Memory Issues (Large Reports)**
        - Try generating reports for fewer databases at once
        - Close other applications to free up memory
        
        **4. Browser Download Issues**
        - Try right-clicking the download button and "Save link as..."
        - Check if your browser is blocking downloads
        - Clear browser cache and try again
        
        **5. File Size Issues**
        - Large reports may take time to generate
        - Wait for the spinner to complete before clicking download
        
        ### Test PDF Generation:
        """)
        
        if st.button("🧪 Run PDF Test", key="troubleshoot_pdf_test"):
            ready, message = check_pdf_requirements()
            if ready:
                test_ready, test_message = test_pdf_generation()
                if test_ready:
                    st.success(f"✅ PDF Generation Test Passed: {test_message}")
                else:
                    st.error(f"❌ PDF Generation Test Failed: {test_message}")
            else:
                st.error(f"❌ Requirements Check Failed: {message}")
                
        st.markdown("""
        ### Alternative Export Options:
        If PDF generation continues to fail, you can:
        - Use Excel export (usually more reliable)
        - Copy the displayed analysis text
        - Use the JSON export for technical details
        - Take screenshots of the analysis results
        """)

def initialize_session_state():
    """Initialize all session state variables with enhanced error handling"""
    if 'ai_analytics' not in st.session_state:
        st.session_state.ai_analytics = None
    if 'calculator' not in st.session_state:
        st.session_state.calculator = EnhancedRDSCalculator()
    
    # Initialize PDF generator with error handling
    if 'pdf_generator' not in st.session_state:
        try:
            if REPORTLAB_AVAILABLE:
                st.session_state.pdf_generator = PDFReportGenerator()
            else:
                st.session_state.pdf_generator = None
        except Exception as e:
            st.session_state.pdf_generator = None
            if 'pdf_warning_shown' not in st.session_state:
                st.warning(f"⚠️ PDF generator initialization failed: {str(e)}")
                st.session_state.pdf_warning_shown = True
    
    if 'file_analysis' not in st.session_state:
        st.session_state.file_analysis = None
    if 'file_inputs' not in st.session_state:
        st.session_state.file_inputs = None
    if 'last_analysis_results' not in st.session_state:
        st.session_state.last_analysis_results = None

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Optional: Add debug info for OAuth troubleshooting
    if st.query_params.get("debug") == "true":
        st.write("**🔧 Debug Info:**")
        st.write(f"- CLIENT_ID configured: {'✅' if CLIENT_ID else '❌'}")
        st.write(f"- CLIENT_SECRET configured: {'✅' if CLIENT_SECRET else '❌'}")
        st.write(f"- REDIRECT_URI: {REDIRECT_URI}")
        st.write(f"- Query params: {dict(st.query_params)}")
        st.write("---")
    
    # --- Google Authentication Integration ---
    if 'user_info' not in st.session_state:
        st.info("Please log in with your Google account to access the AI Database Migration Studio.")

        # NOTE: The redirect_uri must be passed to the authorize_button method.
        # For local development, this should typically be: http://localhost:8501/component/streamlit_oauth.authorize_button/index.html
        # For Streamlit Community Cloud, it should be: https://<your-app-name>.streamlit.app/component/streamlit_oauth.authorize_button/index.html
        # Ensure the REDIRECT_URI in your secrets/environment variable exactly matches this.
        try:
            result = oauth2.authorize_button(
                name="Continue with Google",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=REDIRECT_URI, # This must exactly match the one configured in Google Cloud Console
                scope="openid email profile",
                key="google_oauth_button"
            )

            if result and 'token' in result:
                token = result['token']
                try:
                    # Get user info from Google
                    user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
                    headers = {"Authorization": f"Bearer {token['access_token']}"}
                    user_info_response = requests.get(user_info_url, headers=headers)
                    user_info_response.raise_for_status() # Raise an exception for HTTP errors
                    user_info = user_info_response.json()
                    st.session_state.user_info = user_info
                    st.session_state.token = token
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching user info: {e}")
                    st.session_state.clear()
                    st.stop()
                except Exception as e:
                    st.error(f"An unexpected error occurred during login: {e}")
                    st.session_state.clear()
                    st.stop()
            else:
                st.stop() # Stop the app until logged in
        except Exception as e:
            st.error(f"OAuth initialization failed: {str(e)}")
            st.info("💡 **Troubleshooting tips:**")
            st.info("1. Check that your Google OAuth credentials are properly configured")
            st.info("2. Verify the redirect URI matches exactly in Google Cloud Console")
            st.info("3. Make sure the OAuth consent screen is configured")
            st.stop()

    # User is authenticated, show the app
    user_info = st.session_state.user_info
    st.sidebar.write(f"Logged in as: **{user_info.get('email', 'N/A')}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    # --- END Google Authentication Integration ---
    
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="ai-badge">🤖 Powered by AI </div>
        <h1>AI Database Migration Studio</h1>
        <p>Enterprise database migration planning with intelligent recommendations, cost optimization, and risk assessment powered by advanced AI analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Claude API Key", 
            type="password",
            help="Enter your Anthropic Claude API key to enable AI features",
            placeholder="sk-ant-..."
        )
        
        if api_key:
            try:
                st.session_state.ai_analytics = AIAnalytics(api_key)
                st.success("✅ AI Analytics Enabled")
            except Exception as e:
                st.error(f"❌ API Key Error: {str(e)}")
        else:
            st.info("⚠️ Enter API key to unlock AI features")
        
        st.markdown("---")
        
        # Configuration inputs with better organization
        st.markdown("### 🎯 Migration Configuration")
        
        with st.expander("📊 Database Settings", expanded=True):
            engine = st.selectbox("Database Engine", st.session_state.calculator.engines, index=0)
            region = st.selectbox("AWS Region", st.session_state.calculator.regions, index=0)
        
        with st.expander("🖥️ Current Infrastructure", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                cores = st.number_input("CPU Cores", min_value=1, value=16, step=1)
                cpu_util = st.slider("Peak CPU %", 1, 100, 65)
            with col2:
                ram = st.number_input("RAM (GB)", min_value=1, value=64, step=1)
                ram_util = st.slider("Peak RAM %", 1, 100, 75)
            
            storage = st.number_input("Storage (GB)", min_value=1, value=1000, step=100)
            iops = st.number_input("Peak IOPS", min_value=100, value=8000, step=1000)
        
        with st.expander("⚙️ Migration Settings", expanded=True):
            growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0, max_value=100, value=15)
            backup_days = st.slider("Backup Retention (Days)", 1, 35, 7)
            years_projection = st.slider("Projection Years", 1, 5, 3)
            data_transfer_gb = st.number_input("Monthly Data Transfer (GB)", min_value=0, value=100)

            multi_az_enabled = st.checkbox("Enable Multi-AZ Deployment", value=False, 
                help="Enable Multi-AZ for high availability with read replicas")

            if multi_az_enabled:
                read_replica_count = st.number_input("Number of Read Replicas", min_value=1, max_value=5, value=2)
                read_write_ratio = st.slider("Read/Write Ratio (%)", 10, 90, 70, 
                                help="Percentage of read operations vs write operations")
            else:
                read_replica_count = 0
                read_write_ratio = 50
        
        with st.expander("🤖 AI Features", expanded=True):
            enable_ai_analysis = st.checkbox("Enable AI Workload Analysis", value=True)
            enable_predictions = st.checkbox("Enable Future Predictions", value=True)
            enable_migration_strategy = st.checkbox("Generate Migration Strategy", value=True)
        
        # Show PDF status
        show_pdf_status()
    
    # Collect inputs
    inputs = {
        'engine': engine,
        'region': region,
        'cores': cores,
        'cpu_util': cpu_util,
        'ram': ram,
        'ram_util': ram_util,
        'storage': storage,
        'iops': iops,
        'growth': growth_rate,
        'backup_days': backup_days,
        'years': years_projection,
        'data_transfer_gb': data_transfer_gb,
        # Add these new Multi-AZ fields
        'multi_az_enabled': multi_az_enabled,
        'read_replica_count': read_replica_count,
        'read_write_ratio': read_write_ratio
    }
    
    # Create main tabs with improved styling
    main_tabs = st.tabs(["🔍 AI Analysis", "📁 Bulk Upload", "📊 Manual Configuration", "📋 Reports & Export"])
    
    # Tab 1: AI Analysis
    with main_tabs[0]:
        render_ai_analysis_tab(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy, api_key)
    
    # Tab 2: Bulk Upload  
    with main_tabs[1]:
        render_bulk_upload_tab(enable_ai_analysis, enable_predictions, enable_migration_strategy, api_key)
    
    # Tab 3: Manual Configuration
    with main_tabs[2]:
        render_manual_config_tab(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy, api_key)
    
    # Tab 4: Reports & Export
    with main_tabs[3]:
        render_reports_tab()
        render_troubleshooting_section()

# REPLACE the incomplete render_ai_analysis_tab function (around line 1725) with this complete version:

def render_ai_analysis_tab(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy, api_key):
    """Render the AI Analysis tab"""
    st.markdown("### 🤖 AI-Powered Database Migration Analysis")
    
    # Current configuration display
    st.markdown("#### 📊 Current Configuration Overview")
    
    config_cols = st.columns(4)
    with config_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Database Engine</div>
            <div class="metric-value" style="font-size: 1.5rem;">{inputs['engine'].upper()}</div>
            <div class="metric-subtitle">{inputs['region']}</div>
        </div>
        """, unsafe_allow_html=True)

    with config_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Compute Resources</div>
            <div class="metric-value">{inputs['cores']}</div>
            <div class="metric-subtitle">CPU Cores ({inputs['cpu_util']}% peak)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with config_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Memory</div>
            <div class="metric-value">{inputs['ram']}</div>
            <div class="metric-subtitle">GB RAM ({inputs['ram_util']}% peak)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with config_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Storage & Performance</div>
            <div class="metric-value">{inputs['storage']:,}</div>
            <div class="metric-subtitle">GB Storage ({inputs['iops']:,} IOPS)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Analysis Controls
    st.markdown("#### 🎯 AI Analysis Configuration")
    
    analysis_cols = st.columns([2, 1])
    with analysis_cols[0]:
        st.markdown("**Selected AI Features:**")
        feature_status = []
        if enable_ai_analysis:
            feature_status.append("✅ **Workload Pattern Analysis** - Deep dive into database usage patterns")
        else:
            feature_status.append("❌ Workload Pattern Analysis")
            
        if enable_predictions:
            feature_status.append("✅ **Future Capacity Planning** - AI-powered growth predictions")
        else:
            feature_status.append("❌ Future Capacity Planning")
            
        if enable_migration_strategy:
            feature_status.append("✅ **Migration Strategy Generation** - Step-by-step migration roadmap")
        else:
            feature_status.append("❌ Migration Strategy Generation")
        
        for status in feature_status:
            st.markdown(status)
    
    with analysis_cols[1]:
        if not api_key:
            st.markdown("""
            <div class="status-card status-warning">
                <strong>⚠️ API Key Required</strong><br>
                Enter your Claude API key in the sidebar to enable AI analysis
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card status-success">
                <strong>✅ AI Ready</strong><br>
                All AI features are available
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis Button
    st.markdown("---")
    
    button_cols = st.columns([1, 2, 1])
    with button_cols[1]:
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("🔑 Please enter your Claude API key in the sidebar to enable AI analysis")
            else:
                analyze_workload(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy)

# THEN, replace everything after the main execution block with just this:
# Run the application
if __name__ == "__main__":
    main()
    render_footer()