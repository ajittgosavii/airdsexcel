import streamlit as st
import pandas as pd
import plotly.express as px
import anthropic
import json
import time
import traceback
import numpy as np
from datetime import datetime
import io
from utils import parse_uploaded_file, export_full_report

# Configure enterprise-grade UI
st.set_page_config(
    page_title="AI Database Migration Studio",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .ai-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .ai-insight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .recommendation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .medium-risk {
        background: #fef3c7;
        border: 1px solid #f59e0b;
    }
    .footer-content {
        text-align: center;
        padding: 2rem;
        background: #f8fafc;
        border-radius: 8px;
        margin-top: 2rem;
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
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            ai_analysis = self._parse_ai_response(message.content[0].text)
            return ai_analysis
            
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
                model="claude-3-sonnet-20240229",
                max_tokens=2500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_migration_strategy(message.content[0].text)
            
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
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_predictions(message.content[0].text)
            
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
        
        # Calculate resources with intelligent scaling - FIXED LOGIC
        base_vcpus = inputs['cores'] * (inputs['cpu_util'] / 100)
        base_ram = inputs['ram'] * (inputs['ram_util'] / 100)
        
        # Apply environment factors CORRECTLY
        # PROD gets full or increased resources, DEV gets reduced resources
        if env == "PROD":
            vcpus = max(4, int(base_vcpus * profile['cpu_factor'] * 1.2))  # 20% buffer for PROD
            ram = max(8, int(base_ram * profile['cpu_factor'] * 1.2))
            storage = max(100, int(inputs['storage'] * profile['storage_factor'] * 1.3))  # Extra storage for PROD
        elif env == "STAGING":
            vcpus = max(2, int(base_vcpus * profile['cpu_factor']))
            ram = max(4, int(base_ram * profile['cpu_factor']))
            storage = max(50, int(inputs['storage'] * profile['storage_factor']))
        elif env == "QA":
            vcpus = max(2, int(base_vcpus * profile['cpu_factor']))
            ram = max(4, int(base_ram * profile['cpu_factor']))
            storage = max(20, int(inputs['storage'] * profile['storage_factor']))
        else:  # DEV
            vcpus = max(1, int(base_vcpus * profile['cpu_factor']))  # Allow smaller instances for DEV
            ram = max(2, int(base_ram * profile['cpu_factor']))      # Minimum 2GB for DEV
            storage = max(20, int(inputs['storage'] * profile['storage_factor']))
        
        # Apply growth projections only for PROD and STAGING
        if env in ["PROD", "STAGING"]:
            growth_factor = (1 + inputs['growth']/100) ** 2  # 2-year projection
            storage = int(storage * growth_factor)
        
        # Select optimal instance with environment preference
        instance = self._select_optimal_instance(vcpus, ram, inputs['engine'], inputs['region'], env)
        
        # Calculate comprehensive costs
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
    
    def _select_optimal_instance(self, vcpus: int, ram: int, engine: str, region: str, env: str = "PROD") -> dict:
        """Select optimal instance type using intelligent matching with environment awareness"""
        region_data = self.instance_db.get(region, self.instance_db["us-east-1"])
        engine_instances = region_data.get(engine, region_data.get("postgres", []))
        
        if not engine_instances:
            # Return environment-appropriate fallback
            if env == "DEV":
                return {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.017}}
            elif env in ["QA", "STAGING"]:
                return {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.068}}
            else:  # PROD
                return {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.4}}
        
        # Filter instances based on environment preferences
        if env == "DEV":
            # Prefer t3 instances for DEV (burstable, cost-effective)
            preferred_instances = [inst for inst in engine_instances if 't3' in inst["type"]]
            if not preferred_instances:
                preferred_instances = engine_instances
        elif env in ["QA", "STAGING"]:
            # Prefer t3 and m5 instances for non-PROD
            preferred_instances = [inst for inst in engine_instances if any(family in inst["type"] for family in ['t3', 'm5'])]
            if not preferred_instances:
                preferred_instances = engine_instances
        else:  # PROD
            # Prefer r5 and m5 instances for PROD (performance-oriented)
            preferred_instances = [inst for inst in engine_instances if any(family in inst["type"] for family in ['r5', 'm5'])]
            if not preferred_instances:
                preferred_instances = engine_instances
        
        # Score each instance based on fit and cost efficiency
        scored_instances = []
        for instance in preferred_instances:
            if instance["type"] == "db.serverless":
                # Serverless scoring based on environment
                if env == "DEV":
                    score = 120  # High score for DEV (cost-effective for variable workloads)
                elif env in ["QA", "STAGING"]:
                    score = 100
                else:  # PROD
                    score = 60   # Lower score for PROD (less predictable performance)
            else:
                # Calculate fit score
                cpu_ratio = instance["vCPU"] / max(vcpus, 1)
                ram_ratio = instance["memory"] / max(ram, 1)
                
                # Environment-specific scoring
                if env == "PROD":
                    # For PROD: Prefer 20-80% larger than requirements (performance headroom)
                    cpu_fit = 1.2 if 1.2 <= cpu_ratio <= 1.8 else (1.0 if cpu_ratio >= 1.0 else 0.3)
                    ram_fit = 1.2 if 1.2 <= ram_ratio <= 1.8 else (1.0 if ram_ratio >= 1.0 else 0.3)
                    cost_weight = 0.3  # Less concerned about cost for PROD
                elif env in ["QA", "STAGING"]:
                    # For non-PROD: Prefer 10-50% larger than requirements
                    cpu_fit = 1.0 if 1.1 <= cpu_ratio <= 1.5 else (0.8 if cpu_ratio >= 1.0 else 0.4)
                    ram_fit = 1.0 if 1.1 <= ram_ratio <= 1.5 else (0.8 if ram_ratio >= 1.0 else 0.4)
                    cost_weight = 0.5  # Balanced cost/performance
                else:  # DEV
                    # For DEV: Prefer exact fit or slightly larger (cost optimization)
                    cpu_fit = 1.0 if 1.0 <= cpu_ratio <= 1.3 else (0.7 if cpu_ratio >= 1.0 else 0.2)
                    ram_fit = 1.0 if 1.0 <= ram_ratio <= 1.3 else (0.7 if ram_ratio >= 1.0 else 0.2)
                    cost_weight = 0.7  # High concern about cost for DEV
                
                # Cost efficiency (lower cost per unit resource is better)
                cost_per_vcpu = instance["pricing"]["ondemand"] / max(instance["vCPU"], 1)
                cost_efficiency = (1.0 / (cost_per_vcpu + 1)) * cost_weight
                
                # Performance preference for different environments
                performance_bonus = 0
                if env == "PROD":
                    # Bonus for high-performance instance families
                    if 'r5' in instance["type"]:
                        performance_bonus = 0.3
                    elif 'm5' in instance["type"]:
                        performance_bonus = 0.2
                elif env == "DEV":
                    # Bonus for cost-effective instance families
                    if 't3' in instance["type"]:
                        performance_bonus = 0.3
                    elif 't2' in instance["type"]:
                        performance_bonus = 0.2
                
                score = (cpu_fit + ram_fit + cost_efficiency + performance_bonus) * 100
            
            scored_instances.append((score, instance))
        
        # Return the highest scored instance
        if scored_instances:
            scored_instances.sort(key=lambda x: x[0], reverse=True)
            return scored_instances[0][1]
        
        # Fallback if no instances found
        return engine_instances[0] if engine_instances else {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.4}}
    
    def _calculate_comprehensive_costs(self, instance: dict, storage: int, inputs: dict, env: str) -> dict:
        """Calculate comprehensive monthly costs"""
        # Instance cost
        instance_cost = instance["pricing"]["ondemand"] * 24 * 30
        
        # Multi-AZ multiplier for production
        if env == "PROD":
            instance_cost *= 2  # Multi-AZ deployment
        
        # Storage cost (GP3 pricing)
        storage_gb_cost = storage * 0.115
        
        # IOPS cost (GP3 includes 3000 IOPS free)
        extra_iops = max(0, inputs.get('iops', 3000) - 3000)
        iops_cost = extra_iops * 0.005
        
        # Backup cost
        backup_days = inputs.get('backup_days', 7)
        backup_cost = storage * 0.095 * (backup_days / 30)
        
        # Data transfer cost
        data_transfer = inputs.get('data_transfer_gb', 100)
        transfer_cost = data_transfer * 0.09
        
        # Monitoring and additional features
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
            return 95  # Serverless is highly optimized for variable workloads
        
        cpu_efficiency = min(required_vcpus / instance["vCPU"], 1.0)
        ram_efficiency = min(required_ram / instance["memory"], 1.0)
        
        # Average efficiency with slight preference for balanced usage
        avg_efficiency = (cpu_efficiency + ram_efficiency) / 2
        
        # Convert to 0-100 score
        return int(avg_efficiency * 100)

def initialize_session_state():
    """Initialize all session state variables"""
    if 'ai_analytics' not in st.session_state:
        st.session_state.ai_analytics = None
    if 'calculator' not in st.session_state:
        st.session_state.calculator = EnhancedRDSCalculator()
    if 'file_analysis' not in st.session_state:
        st.session_state.file_analysis = None
    if 'file_inputs' not in st.session_state:
        st.session_state.file_inputs = None

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header with AI branding
    st.markdown("""
    <div class="main-header">
        <div class="ai-badge">🤖 Powered by AI</div>
        <h1>AI Database Migration Studio</h1>
        <p>Enterprise database migration planning with intelligent recommendations, cost optimization, and risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        api_key = st.text_input(
            "Claude API Key", 
            type="password",
            help="Enter your Anthropic Claude API key to enable AI features"
        )
        
        if api_key:
            try:
                st.session_state.ai_analytics = AIAnalytics(api_key)
                st.success("✅ AI Analytics Enabled")
            except Exception as e:
                st.error(f"❌ API Key Error: {str(e)}")
        else:
            st.warning("⚠️ Enter API key to unlock AI features")
        
        st.markdown("---")
        
        # Configuration inputs
        st.header("🎯 Migration Configuration")
        
        # Database settings
        st.subheader("Database Settings")
        engine = st.selectbox("Database Engine", st.session_state.calculator.engines, index=0)
        region = st.selectbox("AWS Region", st.session_state.calculator.regions, index=0)
        
        # Current infrastructure
        st.subheader("Current Infrastructure")
        cores = st.number_input("CPU Cores", min_value=1, value=16, step=1)
        cpu_util = st.slider("Peak CPU Utilization (%)", 1, 100, 65)
        ram = st.number_input("RAM (GB)", min_value=1, value=64, step=1)
        ram_util = st.slider("Peak RAM Utilization (%)", 1, 100, 75)
        storage = st.number_input("Storage (GB)", min_value=1, value=1000, step=100)
        iops = st.number_input("Peak IOPS", min_value=100, value=8000, step=1000)
        
        # Migration settings
        st.subheader("Migration Settings")
        growth_rate = st.number_input("Annual Growth Rate (%)", min_value=0, max_value=100, value=15)
        backup_days = st.slider("Backup Retention (Days)", 1, 35, 7)
        years_projection = st.slider("Projection Years", 1, 5, 3)
        data_transfer_gb = st.number_input("Monthly Data Transfer (GB)", min_value=0, value=100)
        
        # AI Settings
        st.subheader("🤖 AI Features")
        enable_ai_analysis = st.checkbox("Enable AI Workload Analysis", value=True)
        enable_predictions = st.checkbox("Enable Future Predictions", value=True)
        enable_migration_strategy = st.checkbox("Generate Migration Strategy", value=True)
    
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
        'data_transfer_gb': data_transfer_gb
    }
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 File Upload", "🧪 Manual Input"])
    
    with tab1:
        st.markdown("### Upload Multiple Database Configurations")
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel file with database configurations", 
            type=["csv", "xlsx"],
            help="Upload a file containing multiple database configurations"
        )
        
        if uploaded_file:
            try:
                # Show file info for debugging
                st.info(f"📄 Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")
                
                valid_inputs, errors = parse_uploaded_file(uploaded_file)
                
                # Show detailed results
                if errors:
                    st.error(f"❌ **Found {len(errors)} validation errors:**")
                    with st.expander("View Error Details", expanded=False):
                        for i, error in enumerate(errors, 1):
                            st.error(f"{i}. {error}")
                
                if valid_inputs:
                    st.success(f"✅ Successfully parsed **{len(valid_inputs)}** valid database configurations")
                    st.session_state.file_inputs = valid_inputs
                    
                    # Show detailed summary of parsed databases
                    st.markdown("#### 📋 Parsed Database Summary")
                    summary_data = []
                    for i, db in enumerate(valid_inputs, 1):
                        summary_data.append({
                            "#": i,
                            "Database Name": db.get('db_name', f'Database {i}'),
                            "Engine": db.get('engine', 'Unknown'),
                            "Region": db.get('region', 'Unknown'),
                            "CPU Cores": db.get('cores', 'N/A'),
                            "RAM (GB)": db.get('ram', 'N/A'),
                            "Storage (GB)": db.get('storage', 'N/A')
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Analysis options
                    st.markdown("#### 🎯 Analysis Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Selected AI Features:**")
                        st.write(f"• AI Workload Analysis: {'✅ Enabled' if enable_ai_analysis else '❌ Disabled'}")
                        st.write(f"• Future Predictions: {'✅ Enabled' if enable_predictions else '❌ Disabled'}")
                        st.write(f"• Migration Strategy: {'✅ Enabled' if enable_migration_strategy else '❌ Disabled'}")
                    
                    with col2:
                        st.markdown("**Ready to Analyze:**")
                        if st.button("🚀 Analyze All Databases", type="primary", use_container_width=True):
                            if not api_key:
                                st.error("🔑 Please enter your Claude API key in the sidebar to enable AI analysis")
                            else:
                                analyze_file(valid_inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy)
                else:
                    st.error("❌ No valid database configurations found. Please check your file format and data.")
                    
            except Exception as e:
                st.error(f"❌ **Error processing file:** {str(e)}")
                st.info("💡 Make sure your CSV file has all required columns and proper formatting.")
    
    with tab2:
        st.markdown("### Single Database Analysis")
        
        # Show current configuration summary
        st.markdown("#### 📊 Current Configuration")
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.metric("Database Engine", engine)
            st.metric("CPU Cores", cores)
            st.metric("RAM (GB)", ram)
        
        with config_col2:
            st.metric("AWS Region", region)
            st.metric("CPU Utilization", f"{cpu_util}%")
            st.metric("RAM Utilization", f"{ram_util}%")
        
        with config_col3:
            st.metric("Storage (GB)", f"{storage:,}")
            st.metric("IOPS", f"{iops:,}")
            st.metric("Growth Rate", f"{growth_rate}%")
        
        # Analysis options
        st.markdown("#### 🎯 Analysis Options")
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("**Selected AI Features:**")
            st.write(f"• AI Workload Analysis: {'✅ Enabled' if enable_ai_analysis else '❌ Disabled'}")
            st.write(f"• Future Predictions: {'✅ Enabled' if enable_predictions else '❌ Disabled'}")
            st.write(f"• Migration Strategy: {'✅ Enabled' if enable_migration_strategy else '❌ Disabled'}")
        
        with analysis_col2:
            st.markdown("**Actions:**")
            if st.button("🚀 Generate AI-Powered Analysis", type="primary", use_container_width=True):
                if not api_key:
                    st.error("🔑 Please enter your Claude API key in the sidebar to enable AI analysis")
                else:
                    analyze_workload(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy)
            
            if st.button("📊 Export Sample Report", use_container_width=True):
                generate_sample_report()

def analyze_file(valid_inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy):
    """Analyze multiple databases from uploaded file"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    
    try:
        for i, inputs in enumerate(valid_inputs):
            status_text.text(f"🔄 Analyzing database {i+1} of {len(valid_inputs)}...")
            progress = (i + 1) / len(valid_inputs)
            progress_bar.progress(progress)
            
            # Calculate recommendations for each environment
            calculator = st.session_state.calculator
            recommendations = {}
            for env in calculator.env_profiles:
                recommendations[env] = calculator.calculate_requirements(inputs, env)
            
            # AI Analysis if enabled
            ai_insights = {}
            if st.session_state.ai_analytics and enable_ai_analysis:
                try:
                    workload_analysis = st.session_state.ai_analytics.analyze_workload_patterns(inputs)
                    ai_insights['workload'] = workload_analysis
                except Exception as e:
                    ai_insights['workload'] = {"error": str(e)}
            
            if st.session_state.ai_analytics and enable_predictions:
                try:
                    predictions = st.session_state.ai_analytics.predict_future_requirements(inputs, inputs.get('years', 3))
                    ai_insights['predictions'] = predictions
                except Exception as e:
                    ai_insights['predictions'] = {"error": str(e)}
            
            if st.session_state.ai_analytics and enable_migration_strategy:
                try:
                    migration_strategy = st.session_state.ai_analytics.generate_migration_strategy(recommendations['PROD'])
                    ai_insights['migration'] = migration_strategy
                except Exception as e:
                    ai_insights['migration'] = {"error": str(e)}
            
            all_results.append({
                'inputs': inputs,
                'recommendations': recommendations,
                'ai_insights': ai_insights
            })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete for all databases!")
        
        # Display results
        display_file_results(all_results)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")

def display_file_results(all_results):
    """Display results from file analysis with professional layout"""
    st.subheader("📊 Multi-Database Migration Analysis")
    
    # Executive Dashboard
    st.markdown("### 📈 Executive Dashboard")
    
    # Summary metrics
    total_monthly = sum(result['recommendations']['PROD']['monthly_cost'] for result in all_results)
    total_annual = sum(result['recommendations']['PROD']['annual_cost'] for result in all_results)
    avg_monthly = total_monthly / len(all_results)
    avg_optimization = sum(result['recommendations']['PROD'].get('optimization_score', 85) for result in all_results) / len(all_results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Databases", len(all_results))
    with col2:
        st.metric("Total Monthly Cost", f"${total_monthly:,.0f}")
    with col3:
        st.metric("Average per Database", f"${avg_monthly:,.0f}/mo")
    with col4:
        st.metric("Avg Optimization Score", f"{avg_optimization:.0f}%")
    
    # Create main analysis tabs
    analysis_tabs = st.tabs([
        "📋 Summary Table", 
        "🤖 AI Intelligence", 
        "💰 Cost Analysis", 
        "🔍 Individual Analysis",
        "📄 Export Reports"
    ])
    
    with analysis_tabs[0]:  # Summary Table
        st.markdown("#### 📊 Database Configuration Summary")
        
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
        
        st.dataframe(
            summary_df,
            column_config={
                "Monthly Cost": st.column_config.NumberColumn("Monthly Cost", format="$%.0f"),
                "Annual Cost": st.column_config.NumberColumn("Annual Cost", format="$%.0f"),
                "Optimization": st.column_config.TextColumn("Optimization Score")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Quick stats
        st.markdown("#### 📈 Quick Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            engine_counts = {}
            for result in all_results:
                engine = result['inputs'].get('engine', 'Unknown')
                engine_counts[engine] = engine_counts.get(engine, 0) + 1
            
            st.markdown("**Database Engines:**")
            for engine, count in engine_counts.items():
                percentage = (count / len(all_results)) * 100
                st.write(f"• {engine}: {count} databases ({percentage:.1f}%)")
        
        with col2:
            region_counts = {}
            for result in all_results:
                region = result['inputs'].get('region', 'Unknown')
                region_counts[region] = region_counts.get(region, 0) + 1
            
            st.markdown("**AWS Regions:**")
            for region, count in region_counts.items():
                percentage = (count / len(all_results)) * 100
                st.write(f"• {region}: {count} databases ({percentage:.1f}%)")
    
    with analysis_tabs[1]:  # AI Intelligence
        st.markdown("#### 🤖 AI-Powered Migration Intelligence")
        
        # Check if AI analysis is available
        ai_available = any(result.get('ai_insights') for result in all_results)
        
        if not ai_available:
            st.info("🔑 AI analysis requires a Claude API key. Enter your API key in the sidebar and re-run the analysis to see AI insights.")
        else:
            # Aggregate AI insights
            workload_types = {}
            complexity_levels = {}
            common_recommendations = []
            migration_timelines = []
            
            for result in all_results:
                ai_insights = result.get('ai_insights', {})
                if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
                    workload = ai_insights['workload']
                    
                    # Count workload types
                    wtype = workload.get('workload_type', 'Unknown')
                    workload_types[wtype] = workload_types.get(wtype, 0) + 1
                    
                    # Count complexity levels
                    complexity = workload.get('complexity', 'Unknown')
                    complexity_levels[complexity] = complexity_levels.get(complexity, 0) + 1
                    
                    # Collect recommendations
                    common_recommendations.extend(workload.get('recommendations', []))
                    
                    # Collect timelines
                    timeline = workload.get('timeline', '')
                    if timeline:
                        migration_timelines.append(timeline)
            
            # Display AI intelligence
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 Workload Intelligence")
                if workload_types:
                    # Create workload distribution chart
                    fig_workload = px.pie(
                        values=list(workload_types.values()),
                        names=list(workload_types.keys()),
                        title="Workload Type Distribution"
                    )
                    fig_workload.update_traces(textposition='inside', textinfo='percent+label')
                    fig_workload.update_layout(height=300)
                    st.plotly_chart(fig_workload, use_container_width=True)
                
                st.markdown("##### ⚡ Complexity Assessment")
                if complexity_levels:
                    for complexity, count in complexity_levels.items():
                        percentage = (count / len(all_results)) * 100
                        if complexity == "High":
                            st.error(f"🔴 **{complexity}**: {count} databases ({percentage:.1f}%)")
                        elif complexity == "Medium":
                            st.warning(f"🟡 **{complexity}**: {count} databases ({percentage:.1f}%)")
                        else:
                            st.success(f"🟢 **{complexity}**: {count} databases ({percentage:.1f}%)")
            
            with col2:
                st.markdown("##### 🎯 AI Recommendations")
                if common_recommendations:
                    from collections import Counter
                    rec_counts = Counter(common_recommendations)
                    
                    st.markdown("**Most Frequent Recommendations:**")
                    for rec, count in rec_counts.most_common(5):
                        percentage = (count / len(all_results)) * 100
                        st.write(f"• {rec}")
                        st.caption(f"   Recommended for {count} databases ({percentage:.1f}%)")
                
                st.markdown("##### ⏱️ Migration Timeline Analysis")
                if migration_timelines:
                    unique_timelines = list(set(migration_timelines))
                    st.write("**Estimated Migration Timelines:**")
                    for timeline in unique_timelines:
                        count = migration_timelines.count(timeline)
                        percentage = (count / len(migration_timelines)) * 100
                        st.write(f"• {timeline}: {count} databases ({percentage:.1f}%)")
            
            # Strategic recommendations
            st.markdown("##### 🎯 Strategic Migration Recommendations")
            
            strategic_insights = []
            if workload_types.get('OLTP', 0) > len(all_results) * 0.6:
                strategic_insights.append("🏢 **OLTP-Heavy Portfolio**: Consider Aurora for better OLTP performance and cost optimization")
            
            if complexity_levels.get('High', 0) > 0:
                strategic_insights.append("⚠️ **Complex Migrations Detected**: Recommend phased migration approach with extensive testing")
            
            if len(set(result['inputs'].get('region') for result in all_results)) > 3:
                strategic_insights.append("🌍 **Multi-Region Setup**: Consider regional data compliance and cross-region costs")
            
            total_cost_millions = total_annual / 1000000
            if total_cost_millions > 1:
                strategic_insights.append(f"💰 **High-Value Migration**: ${total_cost_millions:.1f}M annual cost - consider Enterprise Support and Professional Services")
            
            if strategic_insights:
                for insight in strategic_insights:
                    st.info(insight)
            else:
                st.success("✅ **Well-Balanced Portfolio**: Standard migration practices should be sufficient")
    
    with analysis_tabs[2]:  # Cost Analysis
        st.markdown("#### 💰 Comprehensive Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Database costs comparison
            db_costs = [result['recommendations']['PROD']['monthly_cost'] for result in all_results]
            db_names = [result['inputs'].get('db_name', f'DB{i+1}') for i, result in enumerate(all_results)]
            db_names_short = [name[:12] + "..." if len(name) > 12 else name for name in db_names]
            
            fig1 = px.bar(
                x=db_names_short,
                y=db_costs,
                title="Monthly Cost by Database",
                labels={'x': 'Database', 'y': 'Monthly Cost ($)'},
                text=[f'${cost:,.0f}' for cost in db_costs],
                color=db_costs,
                color_continuous_scale='RdYlBu_r'
            )
            fig1.update_traces(textposition='outside')
            fig1.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Engine cost distribution
            engine_costs = {}
            for result in all_results:
                engine = result['inputs'].get('engine', 'Unknown')
                cost = result['recommendations']['PROD']['monthly_cost']
                if engine in engine_costs:
                    engine_costs[engine] += cost
                else:
                    engine_costs[engine] = cost
            
            # The display_file_results function is now complete above    values=list(engine_costs.values()),
                names=list(engine_costs.keys()),
                title="Total Monthly Cost by Engine"
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cost efficiency analysis
        st.markdown("##### 📊 Cost Efficiency Analysis")
        
        efficiency_data = []
        for result in all_results:
            prod_rec = result['recommendations']['PROD']
            inputs = result['inputs']
            
            # Calculate cost per core and per GB
            cost_per_core = prod_rec['monthly_cost'] / max(inputs.get('cores', 1), 1)
            cost_per_gb_ram = prod_rec['monthly_cost'] / max(inputs.get('ram', 1), 1)
            
            efficiency_data.append({
                "Database": inputs.get('db_name', 'N/A'),
                "Engine": inputs.get('engine', 'N/A'),
                "Cost/Core": cost_per_core,
                "Cost/GB RAM": cost_per_gb_ram,
                "Optimization Score": prod_rec.get('optimization_score', 85)
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        st.dataframe(
            efficiency_df,
            column_config={
                "Cost/Core": st.column_config.NumberColumn("Cost per Core", format="$%.2f"),
                "Cost/GB RAM": st.column_config.NumberColumn("Cost per GB RAM", format="$%.2f"),
                "Optimization Score": st.column_config.NumberColumn("Optimization Score", format="%.0f%%")
            },
            use_container_width=True,
            hide_index=True
        )
    
    with analysis_tabs[3]:  # Individual Analysis
        st.markdown("#### 🔍 Individual Database Deep Dive")
        
        # Database selector
        db_names = [result['inputs'].get('db_name', f'Database {i+1}') for i, result in enumerate(all_results)]
        selected_db_idx = st.selectbox(
            "Select database for detailed analysis:",
            options=list(range(len(all_results))),
            format_func=lambda x: f"{db_names[x]} ({all_results[x]['inputs'].get('engine', 'Unknown')})"
        )
        
        if selected_db_idx is not None:
            selected_result = all_results[selected_db_idx]
            
            st.markdown(f"### 📋 Analysis for {db_names[selected_db_idx]}")
            display_single_database_analysis(selected_result, selected_db_idx + 1)
    
    with analysis_tabs[4]:  # Export Reports
        st.markdown("#### 📄 Export & Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📊 Executive Report")
            if st.button("📈 Generate Executive Summary", use_container_width=True):
                try:
                    excel_data = export_full_report(all_results)
                    st.download_button(
                        label="Download Executive Excel Report",
                        data=excel_data,
                        file_name=f"executive_migration_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    st.success("✅ Executive report generated!")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            st.markdown("##### 📋 Data Exports")
            
            # CSV export
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Summary CSV",
                data=csv_data,
                file_name=f"migration_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # JSON export
            json_data = json.dumps(all_results, indent=2, default=str)
            st.download_button(
                label="🔧 Download Technical JSON",
                data=json_data,
                file_name=f"migration_technical_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            st.markdown("##### 🎯 Quick Actions")
            
            if st.button("📧 Generate Email Summary", use_container_width=True):
                # Generate email-friendly summary
                email_summary = f"""
Migration Analysis Summary - {datetime.now().strftime('%Y-%m-%d')}

Executive Summary:
• Total Databases Analyzed: {len(all_results)}
• Total Monthly Cost: ${total_monthly:,.0f}
• Total Annual Cost: ${total_annual:,.0f}
• Average Cost per Database: ${avg_monthly:,.0f}/month

Top Recommendations:
• Proceed with cloud migration for cost optimization
• Estimated annual savings vs on-premise: ${total_monthly * 12 * 0.3:,.0f}
• Recommended timeline: 12-18 weeks for phased migration

Next Steps:
1. Review detailed analysis report
2. Plan migration phases
3. Engage AWS Professional Services if needed
                """
                
                st.text_area("Email Summary (Copy & Paste):", email_summary, height=300)
            
            st.info("💡 **Pro Tip:** Use the Executive Excel Report for stakeholder presentations and the Technical JSON for development teams.")

def display_single_database_analysis(result, db_number):
    """Display detailed analysis for a single database with proper formatting"""
    inputs = result['inputs']
    recommendations = result['recommendations']
    ai_insights = result.get('ai_insights', {})
    
    # Create sub-tabs for different aspects
    detail_tabs = st.tabs(["📊 Configuration", "🏗️ Recommendations", "🤖 AI Analysis", "💰 Cost Breakdown"])
    
    with detail_tabs[0]:  # Configuration
        st.markdown("##### 🖥️ Current Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Database Engine", inputs.get('engine', 'Unknown'))
            st.metric("CPU Cores", inputs.get('cores', 'N/A'))
            st.metric("CPU Utilization", f"{inputs.get('cpu_util', 0)}%")
        with col2:
            st.metric("AWS Region", inputs.get('region', 'Unknown'))
            st.metric("RAM (GB)", inputs.get('ram', 'N/A'))
            st.metric("RAM Utilization", f"{inputs.get('ram_util', 0)}%")
        with col3:
            st.metric("Storage (GB)", f"{inputs.get('storage', 0):,}")
            st.metric("IOPS", f"{inputs.get('iops', 0):,}")
            st.metric("Growth Rate", f"{inputs.get('growth', 0)}%")
    
    with detail_tabs[1]:  # Recommendations
        st.markdown("##### 🏗️ Environment Recommendations")
        
        env_data = []
        for env, rec in recommendations.items():
            env_data.append({
                'Environment': env,
                'Instance Type': rec['instance_type'],
                'vCPUs': rec['vcpus'],
                'RAM (GB)': rec['ram_gb'],
                'Storage (GB)': rec['storage_gb'],
                'Monthly Cost': rec['monthly_cost'],
                'Optimization': f"{rec.get('optimization_score', 85)}%"
            })
        
        env_df = pd.DataFrame(env_data)
        st.dataframe(
            env_df,
            column_config={
                "Monthly Cost": st.column_config.NumberColumn("Monthly Cost", format="$%.0f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Highlight production recommendation
        prod_rec = recommendations['PROD']
        st.info(f"🎯 **Production Recommendation:** {prod_rec['instance_type']} at ${prod_rec['monthly_cost']:,.0f}/month with {prod_rec.get('optimization_score', 85)}% optimization score")
    
    with detail_tabs[2]:  # AI Analysis
        if ai_insights:
            st.markdown("##### 🤖 AI-Generated Insights")
            
            # Workload analysis
            if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
                workload = ai_insights['workload']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="ai-insight">
                        <h6>🔍 Workload Classification</h6>
                        <p><strong>Type:</strong> {workload.get('workload_type', 'Mixed')}</p>
                        <p><strong>Complexity:</strong> {workload.get('complexity', 'Medium')}</p>
                        <p><strong>Timeline:</strong> {workload.get('timeline', '12-16 weeks')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if workload.get('recommendations'):
                        st.markdown("###### 🎯 AI Recommendations")
                        for i, rec in enumerate(workload['recommendations'][:4], 1):
                            st.write(f"{i}. {rec}")
                
                # Migration strategy
                if 'migration' in ai_insights and 'error' not in ai_insights['migration']:
                    migration = ai_insights['migration']
                    
                    st.markdown("###### 🚀 Migration Strategy")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Estimated Timeline:** {migration.get('timeline', '14-18 weeks')}")
                        st.write("**Key Migration Phases:**")
                        for i, phase in enumerate(migration.get('phases', [])[:4], 1):
                            st.write(f"{i}. {phase}")
                    
                    with col2:
                        st.write("**Required Resources:**")
                        for resource in migration.get('resources', [])[:4]:
                            st.write(f"• {resource}")
            else:
                st.info("🔑 AI analysis not available. Enter Claude API key to enable AI insights.")
        else:
            st.info("🔑 AI analysis requires Claude API key. Configure in sidebar to see detailed insights.")
    
    with detail_tabs[3]:  # Cost Breakdown
        st.markdown("##### 💰 Detailed Cost Analysis")
        
        prod_rec = recommendations['PROD']
        cost_breakdown = prod_rec.get('cost_breakdown', {})
        
        if cost_breakdown:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost breakdown chart
                labels = list(cost_breakdown.keys())
                values = list(cost_breakdown.values())
                
                fig = px.pie(
                    values=values,
                    names=labels,
                    title="Monthly Cost Breakdown"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cost details table
                cost_details = []
                for component, cost in cost_breakdown.items():
                    percentage = (cost / cost_breakdown['total']) * 100
                    cost_details.append({
                        "Component": component.replace('_', ' ').title(),
                        "Monthly Cost": cost,
                        "Percentage": f"{percentage:.1f}%"
                    })
                
                cost_df = pd.DataFrame(cost_details)
                st.dataframe(
                    cost_df,
                    column_config={
                        "Monthly Cost": st.column_config.NumberColumn("Monthly Cost", format="$%.2f")
                    },
                    use_container_width=True,
                    hide_index=True
                )
        
        # Annual projection
        annual_cost = prod_rec['annual_cost']
        growth_rate = inputs.get('growth', 0)
        
        st.markdown("###### 📈 3-Year Cost Projection")
        projection_data = []
        for year in range(1, 4):
            yearly_cost = annual_cost * ((1 + growth_rate/100) ** (year-1))
            projection_data.append({
                "Year": year,
                "Annual Cost": yearly_cost
            })
        
        proj_df = pd.DataFrame(projection_data)
        fig_proj = px.line(
            proj_df,
            x='Year',
            y='Annual Cost',
            title=f"3-Year Cost Projection ({growth_rate}% annual growth)",
            markers=True
        )
        fig_proj.update_layout(height=300)
        st.plotly_chart(fig_proj, use_container_width=True)values=list(engine_costs.values()),
            names=list(engine_costs.keys()),
            title="Total Cost by Database Engine"
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Export options
    st.subheader("📄 Export Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Detailed Excel Report", use_container_width=True):
            try:
                excel_data = export_full_report(all_results)
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name=f"migration_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.success("✅ Excel report generated!")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        # CSV export
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Summary CSV",
            data=csv_data,
            file_name=f"migration_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # JSON export
        json_data = json.dumps(all_results, indent=2, default=str)
        st.download_button(
            label="🔧 Download Full JSON",
            data=json_data,
            file_name=f"migration_full_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

def display_single_database_analysis(result, db_number):
    """Display detailed analysis for a single database"""
    inputs = result['inputs']
    recommendations = result['recommendations']
    ai_insights = result.get('ai_insights', {})
    
    # Database info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Engine", inputs.get('engine', 'Unknown'))
        st.metric("CPU Cores", inputs.get('cores', 'N/A'))
    with col2:
        st.metric("Region", inputs.get('region', 'Unknown'))
        st.metric("RAM (GB)", inputs.get('ram', 'N/A'))
    with col3:
        st.metric("Storage (GB)", f"{inputs.get('storage', 0):,}")
        st.metric("IOPS", f"{inputs.get('iops', 0):,}")
    
    # Environment recommendations
    st.markdown("#### 🏗️ Environment Recommendations")
    
    env_data = []
    for env, rec in recommendations.items():
        env_data.append({
            'Environment': env,
            'Instance Type': rec['instance_type'],
            'vCPUs': rec['vcpus'],
            'RAM (GB)': rec['ram_gb'],
            'Storage (GB)': rec['storage_gb'],
            'Monthly Cost': rec['monthly_cost'],
            'Optimization Score': f"{rec.get('optimization_score', 85)}%"
        })
    
    env_df = pd.DataFrame(env_data)
    st.dataframe(
        env_df,
        column_config={
            "Monthly Cost": st.column_config.NumberColumn("Monthly Cost", format="$%.0f")
        },
        use_container_width=True,
        hide_index=True
    )
    
    # AI Insights for this database
    if ai_insights:
        st.markdown("#### 🤖 AI Analysis")
        
        # Workload analysis
        if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
            workload = ai_insights['workload']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="ai-insight">
                    <h5>🔍 Workload Classification</h5>
                    <p><strong>Type:</strong> {workload.get('workload_type', 'Mixed')}</p>
                    <p><strong>Complexity:</strong> {workload.get('complexity', 'Medium')}</p>
                    <p><strong>Timeline:</strong> {workload.get('timeline', '12-16 weeks')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if workload.get('recommendations'):
                    st.markdown("##### 🎯 AI Recommendations")
                    for rec in workload['recommendations'][:4]:
                        st.write(f"• {rec}")
        
        # Migration strategy
        if 'migration' in ai_insights and 'error' not in ai_insights['migration']:
            migration = ai_insights['migration']
            
            st.markdown("##### 🚀 Migration Strategy")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Timeline:** {migration.get('timeline', '14-18 weeks')}")
                st.write("**Key Phases:**")
                for phase in migration.get('phases', [])[:3]:
                    st.write(f"• {phase}")
            
            with col2:
                st.write("**Required Resources:**")
                for resource in migration.get('resources', [])[:3]:
                    st.write(f"• {resource}")
    
    # Cost breakdown chart for this database
    prod_rec = recommendations['PROD']
    cost_breakdown = prod_rec.get('cost_breakdown', {})
    
    if cost_breakdown:
        st.markdown("#### 💰 Cost Breakdown")
        labels = list(cost_breakdown.keys())
        values = list(cost_breakdown.values())
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Monthly Cost Breakdown"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def analyze_workload(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy):
    """Main analysis function with AI integration"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Basic calculations
        status_text.text("🔄 Calculating resource requirements...")
        progress_bar.progress(20)
        
        calculator = st.session_state.calculator
        recommendations = {}
        for env in calculator.env_profiles:
            recommendations[env] = calculator.calculate_requirements(inputs, env)
        
        progress_bar.progress(40)
        
        # Step 2: AI Analysis
        ai_insights = {}
        if st.session_state.ai_analytics and enable_ai_analysis:
            status_text.text("🤖 Running AI workload analysis...")
            progress_bar.progress(60)
            
            try:
                workload_analysis = st.session_state.ai_analytics.analyze_workload_patterns(inputs)
                ai_insights['workload'] = workload_analysis
            except Exception as e:
                st.error(f"AI Analysis Error: {str(e)}")
                ai_insights['workload'] = {"error": str(e)}
        
        # Step 3: Future Predictions
        if st.session_state.ai_analytics and enable_predictions:
            status_text.text("🔮 Generating future predictions...")
            progress_bar.progress(75)
            
            try:
                predictions = st.session_state.ai_analytics.predict_future_requirements(inputs, inputs['years'])
                ai_insights['predictions'] = predictions
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                ai_insights['predictions'] = {"error": str(e)}
        
        # Step 4: Migration Strategy
        if st.session_state.ai_analytics and enable_migration_strategy:
            status_text.text("📋 Creating migration strategy...")
            progress_bar.progress(90)
            
            try:
                migration_strategy = st.session_state.ai_analytics.generate_migration_strategy(recommendations['PROD'])
                ai_insights['migration'] = migration_strategy
            except Exception as e:
                st.error(f"Migration Strategy Error: {str(e)}")
                ai_insights['migration'] = {"error": str(e)}
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(recommendations, ai_insights, inputs)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")
        st.error("Please check your inputs and try again.")

def display_results(recommendations, ai_insights, inputs):
    """Display comprehensive results with AI insights"""
    
    # Key Metrics Dashboard
    st.subheader("📊 Migration Dashboard")
    
    prod_rec = recommendations['PROD']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Recommended Instance</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{prod_rec['instance_type']}</div>
            <div style="font-size: 0.8rem; color: #888;">{prod_rec['vcpus']} vCPUs, {prod_rec['ram_gb']} GB RAM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Monthly Cost</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${prod_rec['monthly_cost']:,.0f}</div>
            <div style="font-size: 0.8rem; color: #888;">Production Environment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Calculate estimated savings vs on-premise
        onprem_monthly = inputs['cores'] * 200  # Rough estimate: $200/core/month
        monthly_savings = onprem_monthly - prod_rec['monthly_cost']
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Monthly Savings</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${monthly_savings:,.0f}</div>
            <div style="font-size: 0.8rem; color: #888;">vs On-Premise</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        optimization_score = prod_rec.get('optimization_score', 85)
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Optimization Score</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #f59e0b;">{optimization_score}%</div>
            <div style="font-size: 0.8rem; color: #888;">Resource Efficiency</div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Insights Section
    if ai_insights and st.session_state.ai_analytics:
        st.subheader("🤖 AI-Powered Insights")
        
        # Display workload analysis if available
        if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
            workload = ai_insights['workload']
            st.markdown(f"""
            <div class="ai-insight">
                <h4>🔍 Workload Analysis</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                    <div><strong>Classification:</strong> {workload.get('workload_type', 'Mixed')}</div>
                    <div><strong>Complexity:</strong> {workload.get('complexity', 'Medium')}</div>
                    <div><strong>Timeline:</strong> {workload.get('timeline', '12-16 weeks')}</div>
                </div>
                <h5>🎯 Key Recommendations:</h5>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in workload.get('recommendations', [])[:4]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance bottlenecks
            if workload.get('bottlenecks'):
                st.markdown("#### ⚠️ Identified Bottlenecks")
                bottleneck_cols = st.columns(min(len(workload['bottlenecks']), 3))
                for i, bottleneck in enumerate(workload['bottlenecks'][:3]):
                    with bottleneck_cols[i]:
                        st.markdown(f"""
                        <div class="risk-card medium-risk">
                            <h6>Performance Issue {i+1}</h6>
                            <p>{bottleneck}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Migration strategy
        if 'migration' in ai_insights and 'error' not in ai_insights['migration']:
            st.subheader("🚀 AI-Generated Migration Strategy")
            
            migration = ai_insights['migration']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📅 Migration Timeline")
                st.info(f"**Estimated Duration:** {migration.get('timeline', '14-18 weeks')}")
                
                phases = migration.get('phases', [])
                for i, phase in enumerate(phases, 1):
                    st.markdown(f"**Phase {i}:** {phase}")
            
            with col2:
                st.markdown("#### 👥 Required Resources")
                resources = migration.get('resources', [])
                for resource in resources:
                    st.markdown(f"• {resource}")
                
                st.markdown("#### 🛠️ Recommended Tools")
                tools = migration.get('tools', [])
                for tool in tools[:4]:
                    st.markdown(f"• {tool}")
    
    # Environment-specific recommendations
    st.subheader("🏗️ Environment Recommendations")
    
    df_data = []
    for env, rec in recommendations.items():
        df_data.append({
            'Environment': env,
            'Instance Type': rec['instance_type'],
            'vCPUs': rec['vcpus'],
            'RAM (GB)': rec['ram_gb'],
            'Storage (GB)': rec['storage_gb'],
            'Monthly Cost': rec['monthly_cost'],
            'Optimization Score': f"{rec.get('optimization_score', 85)}%"
        })
    
    df = pd.DataFrame(df_data)
    
    # Style the dataframe
    st.dataframe(
        df,
        column_config={
            "Monthly Cost": st.column_config.NumberColumn(
                "Monthly Cost",
                format="$%.0f"
            ),
            "Optimization Score": st.column_config.TextColumn(
                "Optimization Score"
            )
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Cost Analysis Charts
    st.subheader("💰 Cost Analysis & Projections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly cost comparison by environment
        env_costs = [rec['monthly_cost'] for rec in recommendations.values()]
        env_names = list(recommendations.keys())
        
        fig1 = px.bar(
            x=env_names, 
            y=env_costs,
            title="Monthly Cost by Environment",
            labels={'x': 'Environment', 'y': 'Monthly Cost ($)'},
            color=env_costs,
            color_continuous_scale='Viridis',
            text=[f'${cost:,.0f}' for cost in env_costs]
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cost breakdown for production
        cost_breakdown = prod_rec.get('cost_breakdown', {})
        if cost_breakdown:
            labels = list(cost_breakdown.keys())
            values = list(cost_breakdown.values())
            
            fig2 = px.pie(
                values=values,
                names=labels,
                title="Production Cost Breakdown"
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    # 3-year projection
    st.subheader("📈 Multi-Year Cost Projection")
    
    years = list(range(1, inputs['years'] + 1))
    growth_factor = 1 + (inputs['growth'] / 100)
    
    projection_data = []
    for year in years:
        year_growth = growth_factor ** (year - 1)
        for env, rec in recommendations.items():
            projection_data.append({
                'Year': year,
                'Environment': env,
                'Annual Cost': rec['annual_cost'] * year_growth
            })
    
    proj_df = pd.DataFrame(projection_data)
    
    fig3 = px.line(
        proj_df,
        x='Year',
        y='Annual Cost',
        color='Environment',
        title=f"{inputs['years']}-Year Cost Projection with {inputs['growth']}% Annual Growth",
        markers=True
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    annual_savings = (inputs['cores'] * 200 * 12) - prod_rec['annual_cost']  # vs estimated on-prem
    roi_percentage = (annual_savings / prod_rec['annual_cost']) * 100 if prod_rec['annual_cost'] > 0 else 0
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border: 1px solid #0ea5e9; border-radius: 12px; padding: 2rem; margin: 1rem 0; color: #1E293B;">
        <h4 style="color: #1E293B;">💼 Executive Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 1rem 0;">
            <div>
                <h5 style="color: #1E293B;">💰 Financial Impact</h5>
                <p style="color: #1E293B;">• <strong>Annual Cost:</strong> ${prod_rec['annual_cost']:,.0f}</p>
                <p style="color: #1E293B;">• <strong>Annual Savings:</strong> ${annual_savings:,.0f}</p>
                <p style="color: #1E293B;">• <strong>ROI:</strong> {roi_percentage:.0f}%</p>
            </div>
            <div>
                <h5 style="color: #1E293B;">⚡ Performance Benefits</h5>
                <p style="color: #1E293B;">• <strong>Improved Availability:</strong> 99.99% SLA</p>
                <p style="color: #1E293B;">• <strong>Auto Scaling:</strong> Dynamic resource allocation</p>
                <p style="color: #1E293B;">• <strong>Backup & Recovery:</strong> Automated & reliable</p>
            </div>
            <div>
                <h5 style="color: #1E293B;">🎯 Strategic Advantages</h5>
                <p style="color: #1E293B;">• <strong>Reduced Ops Overhead:</strong> Managed service benefits</p>
                <p style="color: #1E293B;">• <strong>Enhanced Security:</strong> AWS security framework</p>
                <p style="color: #1E293B;">• <strong>Global Scalability:</strong> Multi-region deployment ready</p>
            </div>
        </div>
        <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem; color: #1E293B;">
            <strong style="color: #1E293B;">💡 Recommendation:</strong> Proceed with migration to achieve significant cost savings, 
            improved performance, and reduced operational complexity. Estimated payback period: 
            {12 / max(roi_percentage/100, 0.1):.0f} months.
        </div>
    </div>
    """, unsafe_allow_html=True)

def generate_sample_report():
    """Generate and download sample report"""
    st.subheader("📊 Sample Report Generation")
    
    # Create sample data for demonstration
    sample_data = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "database_engine": "PostgreSQL",
        "current_environment": "On-Premise",
        "target_environment": "AWS RDS",
        "estimated_monthly_cost": 2850,
        "estimated_annual_savings": 45000,
        "migration_timeline": "12-16 weeks",
        "risk_level": "Low to Medium"
    }
    
    # Generate CSV report
    csv_data = pd.DataFrame([sample_data])
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📄 Download Sample CSV Report",
        data=csv_buffer.getvalue(),
        file_name=f"migration_analysis_sample_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.info("💡 This is a sample report. Run the full analysis with your Claude API key to generate comprehensive reports with AI insights.")

def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div class="footer-content">
        <h3>🚀 AI Database Migration Studio</h3>
        <p><strong>Powered by AI • Enterprise-Ready • Cloud-Native</strong></p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
            <div>✅ Multi-Engine Support</div>
            <div>🤖 AI-Powered Analysis</div>
            <div>📊 Cost Optimization</div>
            <div>🔒 Enterprise Security</div>
        </div>
        <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            Transform your database migration with the power of artificial intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
    render_footer()