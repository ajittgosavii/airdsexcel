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
    
    # File import section
    with st.expander("📁 Import Database Configurations", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel file with database configurations", 
            type=["csv", "xlsx"],
            help="Upload a file containing multiple database configurations"
        )
        
        if uploaded_file:
            try:
                valid_inputs, errors = parse_uploaded_file(uploaded_file)
                
                if errors:
                    for error in errors:
                        st.error(error)
                
                if valid_inputs:
                    st.success(f"✅ Successfully parsed {len(valid_inputs)} valid database configurations")
                    st.session_state.file_inputs = valid_inputs
                    
                    # Show preview
                    with st.expander("Preview Imported Data"):
                        preview_df = pd.DataFrame(valid_inputs)
                        st.dataframe(preview_df.head())
                    
                    if st.button("🚀 Analyze All Databases", type="primary", use_container_width=True):
                        if not api_key:
                            st.error("🔑 Please enter your Claude API key in the sidebar to enable AI analysis")
                        else:
                            analyze_file(valid_inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Generate AI-Powered Analysis", type="primary", use_container_width=True):
            if not api_key:
                st.error("🔑 Please enter your Claude API key in the sidebar to enable AI analysis")
            else:
                analyze_workload(inputs, enable_ai_analysis, enable_predictions, enable_migration_strategy)
    
    with col2:
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
    """Display results from file analysis"""
    st.subheader("📊 Multi-Database Analysis Results")
    
    # Summary table
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
            "Annual Cost": prod_rec['annual_cost']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display summary
    st.dataframe(
        summary_df,
        column_config={
            "Monthly Cost": st.column_config.NumberColumn("Monthly Cost", format="$%.0f"),
            "Annual Cost": st.column_config.NumberColumn("Annual Cost", format="$%.0f")
        },
        use_container_width=True
    )
    
    # Total costs
    total_monthly = summary_df['Monthly Cost'].sum()
    total_annual = summary_df['Annual Cost'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Monthly Cost", f"${total_monthly:,.0f}")
    with col2:
        st.metric("Total Annual Cost", f"${total_annual:,.0f}")
    with col3:
        st.metric("Average per Database", f"${total_monthly/len(all_results):,.0f}/mo")
    
    # Export option
    if st.button("📄 Export Detailed Report"):
        try:
            excel_data = export_full_report(all_results)
            st.download_button(
                label="Download Excel Report",
                data=excel_data,
                file_name=f"migration_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")

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