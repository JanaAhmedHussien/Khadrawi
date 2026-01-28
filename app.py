import os
import json
import datetime
import asyncio
from typing import Dict, List, Optional, Any
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum

# Gemini
from google import genai
from dotenv import load_dotenv

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="🤖 Agentic Plant Disease Detection",
    layout="wide"
)

st.title("🤖 Agentic Plant Disease Detection with Explainable AI")

# ------------------------------
# Agent State and Enums
# ------------------------------
class AgentState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    DIAGNOSING = "diagnosing"
    EXPLAINING = "explaining"
    RECOMMENDING = "recommending"
    FOLLOWUP = "followup"

class DiseaseSeverity(Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class AgentMemory:
    """Memory for persistent agent knowledge"""
    farmer_id: str
    history: List[Dict] = field(default_factory=list)
    field_conditions: Dict = field(default_factory=dict)
    previous_diagnoses: List = field(default_factory=list)
    
@dataclass
class PlantCase:
    """Represents a plant disease case"""
    image: Any
    prediction: str
    confidence: float
    severity: DiseaseSeverity
    timestamp: str
    location: Optional[str] = None
    plant_type: Optional[str] = None
    season: Optional[str] = None
    
# ------------------------------
# Agentic Tools Definition
# ------------------------------
class AgenticTools:
    """Collection of specialized tools for the agent"""
    
    @staticmethod
    def analyze_severity(disease_name: str, confidence: float) -> DiseaseSeverity:
        """Analyze disease severity based on type and confidence"""
        severe_diseases = ["late_blight", "bacterial_spot", "leaf_mold_critical"]
        
        if confidence < 0.3:
            return DiseaseSeverity.NONE
        elif confidence < 0.6:
            return DiseaseSeverity.MILD
        elif disease_name in severe_diseases and confidence > 0.7:
            return DiseaseSeverity.SEVERE
        elif confidence > 0.8:
            return DiseaseSeverity.CRITICAL
        else:
            return DiseaseSeverity.MODERATE
    
    @staticmethod
    def calculate_risk_score(severity: DiseaseSeverity, season: str) -> float:
        """Calculate risk score based on severity and environmental factors"""
        season_risk = {"rainy": 0.8, "humid": 0.7, "dry": 0.3, "winter": 0.4}
        
        severity_multiplier = {
            DiseaseSeverity.NONE: 0.1,
            DiseaseSeverity.MILD: 0.3,
            DiseaseSeverity.MODERATE: 0.6,
            DiseaseSeverity.SEVERE: 0.8,
            DiseaseSeverity.CRITICAL: 1.0
        }
        
        base_risk = severity_multiplier.get(severity, 0.5)
        seasonal_risk = season_risk.get(season.lower(), 0.5)
        
        return (base_risk * 0.7) + (seasonal_risk * 0.3)
    
    @staticmethod
    def generate_action_plan(severity: DiseaseSeverity, risk_score: float) -> List[str]:
        """Generate action plan based on analysis"""
        actions = []
        
        if severity == DiseaseSeverity.SEVERE or risk_score > 0.7:
            actions.append("🚨 Immediate treatment required")
            actions.append("📢 Alert nearby farmers")
            actions.append("🩺 Schedule expert consultation")
        
        if severity == DiseaseSeverity.MODERATE or risk_score > 0.5:
            actions.append("💊 Apply recommended treatment")
            actions.append("📅 Monitor daily for progression")
            actions.append("📸 Take follow-up images in 3 days")
        
        if severity in [DiseaseSeverity.MILD, DiseaseSeverity.MODERATE]:
            actions.append("🌱 Apply preventive measures")
            actions.append("📊 Log in disease tracker")
            actions.append("🔔 Set reminder for re-check")
        
        actions.append("📚 Review educational materials")
        
        return actions

# ------------------------------
# Plant Disease Agent
# ------------------------------
class PlantDiseaseAgent:
    """Main agent orchestrating the disease detection pipeline"""
    
    def __init__(self, model, class_names, device, gemini_client=None):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.gemini_client = gemini_client
        self.state = AgentState.IDLE
        self.memory = {}  # farmer_id -> AgentMemory
        self.tools = AgenticTools()
        
        # Initialize memory for demo
        self._init_demo_memory()
    
    def _init_demo_memory(self):
        """Initialize with demo farmer data"""
        demo_memory = AgentMemory(
            farmer_id="demo_farmer_001",
            history=[
                {"date": "2024-01-15", "disease": "leaf_mold", "severity": "mild", "treated": True},
                {"date": "2024-02-10", "disease": "healthy", "severity": "none", "treated": False}
            ],
            field_conditions={
                "location": "California",
                "soil_type": "loamy",
                "irrigation": "drip",
                "last_spray": "2024-02-01"
            }
        )
        self.memory["demo_farmer_001"] = demo_memory
    
    def set_state(self, new_state: AgentState):
        """Update agent state with logging"""
        st.session_state.agent_state = new_state.value
        self.state = new_state
    
    async def process_case(self, image, farmer_id="demo_farmer_001", context=None) -> Dict:
        """Main agent pipeline - processes a plant disease case"""
        
        # Start agent pipeline
        self.set_state(AgentState.ANALYZING)
        
        # Create plant case
        plant_case = await self._analyze_image(image, context)
        
        # Get farmer memory
        farmer_memory = self.memory.get(farmer_id)
        if farmer_memory:
            plant_case.season = self._get_current_season(farmer_memory.field_conditions.get("location", ""))
        
        # Diagnose with context
        self.set_state(AgentState.DIAGNOSING)
        diagnosis = await self._diagnose_with_context(plant_case, farmer_memory)
        
        # Generate explanations
        self.set_state(AgentState.EXPLAINING)
        explanations = await self._generate_explanations(plant_case)
        
        # Generate recommendations
        self.set_state(AgentState.RECOMMENDING)
        recommendations = await self._generate_recommendations(plant_case, diagnosis, farmer_memory)
        
        # Plan follow-up
        self.set_state(AgentState.FOLLOWUP)
        followup_plan = self._plan_followup(plant_case, diagnosis)
        
        # Update memory
        self._update_memory(farmer_id, plant_case, diagnosis)
        
        # Return to idle
        self.set_state(AgentState.IDLE)
        
        return {
            "plant_case": plant_case,
            "diagnosis": diagnosis,
            "explanations": explanations,
            "recommendations": recommendations,
            "followup_plan": followup_plan,
            "agent_state": self.state.value
        }
    
    async def _analyze_image(self, image, context=None) -> PlantCase:
        """Analyze uploaded image"""
        # Transform image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax().item()
            confidence = probs[0, pred_idx].item()
        
        predicted_class = self.class_names[pred_idx]
        
        # Analyze severity
        severity = self.tools.analyze_severity(predicted_class, confidence)
        
        return PlantCase(
            image=image,
            prediction=predicted_class,
            confidence=confidence,
            severity=severity,
            timestamp=datetime.datetime.now().isoformat(),
            plant_type=context.get("plant_type") if context else None
        )
    
    async def _diagnose_with_context(self, plant_case: PlantCase, farmer_memory=None) -> Dict:
        """Diagnose with historical context"""
        risk_score = self.tools.calculate_risk_score(
            plant_case.severity,
            plant_case.season or "unknown"
        )
        
        diagnosis = {
            "primary_diagnosis": plant_case.prediction,
            "confidence": plant_case.confidence,
            "severity": plant_case.severity.value,
            "risk_score": risk_score,
            "is_recurring": False,
            "historical_context": None
        }
        
        # Check if recurring issue
        if farmer_memory:
            previous_cases = [h for h in farmer_memory.history 
                            if h.get("disease") == plant_case.prediction]
            diagnosis["is_recurring"] = len(previous_cases) > 0
            diagnosis["historical_context"] = f"Found {len(previous_cases)} previous cases"
        
        return diagnosis
    
    async def _generate_explanations(self, plant_case: PlantCase) -> Dict:
        """Generate multiple explanations"""
        explanations = {
            "visual": None,
            "textual": None,
            "comparative": None
        }
        
        # Visual explanation (Grad-CAM)
        if 'gradcam' in st.session_state:
            try:
                cam = st.session_state.gradcam.generate(
                    self._preprocess_image(plant_case.image).unsqueeze(0).to(self.device),
                    self.class_names.index(plant_case.prediction)
                )
                explanations["visual"] = cam.cpu().numpy()
            except:
                explanations["visual"] = "Visual explanation unavailable"
        
        # Textual explanation from Gemini
        if self.gemini_client:
            try:
                prompt = f"""
                Provide a farmer-friendly explanation for plant disease: {plant_case.prediction}
                Confidence: {plant_case.confidence:.2%}
                Severity: {plant_case.severity.value}
                
                Include:
                1. Simple description
                2. Key visual symptoms
                3. Likely causes
                4. Why it might be occurring now
                """
                
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt]
                )
                explanations["textual"] = response.text
            except:
                explanations["textual"] = "Text explanation unavailable"
        
        return explanations
    
    async def _generate_recommendations(self, plant_case: PlantCase, diagnosis: Dict, farmer_memory=None) -> Dict:
        """Generate personalized recommendations"""
        recommendations = {
            "immediate_actions": [],
            "treatments": [],
            "prevention": [],
            "monitoring": []
        }
        
        # Immediate actions based on severity
        recommendations["immediate_actions"] = self.tools.generate_action_plan(
            plant_case.severity,
            diagnosis["risk_score"]
        )
        
        # Generate treatments using Gemini if available
        if self.gemini_client:
            try:
                prompt = f"""
                Suggest specific treatments for {plant_case.prediction} (Severity: {plant_case.severity.value})
                Provide:
                1. Organic treatment options
                2. Chemical treatments (if severe)
                3. Application instructions
                """
                
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt]
                )
                recommendations["treatments"] = response.text.split('\n')
            except:
                recommendations["treatments"] = ["Consult local agricultural expert"]
        
        # Prevention based on history
        if farmer_memory and diagnosis["is_recurring"]:
            recommendations["prevention"] = [
                "This is a recurring issue - consider crop rotation",
                "Improve soil drainage",
                "Use resistant plant varieties",
                "Schedule regular preventive sprays"
            ]
        
        # Monitoring schedule
        if plant_case.severity in [DiseaseSeverity.MODERATE, DiseaseSeverity.SEVERE, DiseaseSeverity.CRITICAL]:
            recommendations["monitoring"] = [
                "Daily visual inspection for 1 week",
                "Take photos every 2 days to track progress",
                "Monitor weather conditions",
                "Check neighboring plants"
            ]
        
        return recommendations
    
    def _plan_followup(self, plant_case: PlantCase, diagnosis: Dict) -> Dict:
        """Create follow-up plan"""
        followup_days = {
            DiseaseSeverity.MILD: 7,
            DiseaseSeverity.MODERATE: 3,
            DiseaseSeverity.SEVERE: 1,
            DiseaseSeverity.CRITICAL: 0  # Immediate
        }
        
        days = followup_days.get(plant_case.severity, 7)
        followup_date = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        return {
            "next_check": followup_date,
            "urgency": "high" if diagnosis["risk_score"] > 0.7 else "medium",
            "check_items": [
                "Disease progression",
                "Treatment effectiveness",
                "New symptoms",
                "Spread to other plants"
            ]
        }
    
    def _update_memory(self, farmer_id: str, plant_case: PlantCase, diagnosis: Dict):
        """Update agent memory with new case"""
        if farmer_id not in self.memory:
            self.memory[farmer_id] = AgentMemory(farmer_id=farmer_id)
        
        memory_entry = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "disease": plant_case.prediction,
            "confidence": plant_case.confidence,
            "severity": plant_case.severity.value,
            "risk_score": diagnosis["risk_score"],
            "treated": False,
            "followup_date": self._plan_followup(plant_case, diagnosis)["next_check"]
        }
        
        self.memory[farmer_id].history.append(memory_entry)
        
        # Keep only last 10 entries
        if len(self.memory[farmer_id].history) > 10:
            self.memory[farmer_id].history = self.memory[farmer_id].history[-10:]
    
    def _preprocess_image(self, image):
        """Preprocess image for Grad-CAM"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image)
    
    def _get_current_season(self, location: str) -> str:
        """Simple season detection based on month"""
        month = datetime.datetime.now().month
        if location.lower() in ["california", "florida", "texas"]:
            # US seasons
            if month in [12, 1, 2]: return "winter"
            elif month in [3, 4, 5]: return "spring"
            elif month in [6, 7, 8]: return "summer"
            else: return "fall"
        else:
            # Tropical/Indian seasons
            if month in [6, 7, 8, 9]: return "monsoon"
            elif month in [10, 11]: return "post-monsoon"
            elif month in [12, 1, 2]: return "winter"
            else: return "summer"

# ------------------------------
# Original Model Definition (unchanged)
# ------------------------------
class GeneralizedPlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def conv_block(in_c, out_c, drop):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(drop)
            )

        self.stage1 = conv_block(3, 32, 0.10)
        self.stage2 = conv_block(32, 64, 0.15)
        self.stage3 = conv_block(64, 128, 0.20)
        self.stage4 = conv_block(128, 256, 0.25)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

# ------------------------------
# Enhanced Grad-CAM
# ------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        target = output[:, class_idx]
        target.backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze()

# ------------------------------
# Load Models and Initialize
# ------------------------------
@st.cache_resource
def load_model_and_classes():
    try:
        class_names = torch.load("class_names.pth", map_location=torch.device('cpu'))
        num_classes = len(class_names)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GeneralizedPlantCNN(num_classes)
        
        state_dict = torch.load("best_crop_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Initialize Grad-CAM
        try:
            target_layer = model.stage4[3]
            gradcam = GradCAM(model, target_layer)
            st.session_state.gradcam = gradcam
        except:
            st.session_state.gradcam = None
        
        return model, class_names, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load Gemini
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("GEMINI_API_KEY not found. AI reports disabled.")
        gemini_client = None
    else:
        gemini_client = genai.Client(api_key=api_key)
except ImportError:
    st.warning("Google Gemini API not installed.")
    gemini_client = None

# Load model
model, class_names, device = load_model_and_classes()

if model is None:
    st.error("Failed to load model. Please ensure model files exist.")
    st.stop()

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = PlantDiseaseAgent(model, class_names, device, gemini_client)

# ------------------------------
# Streamlit UI
# ------------------------------
st.sidebar.title("🕹️ Agent Controls")

# Agent mode selection
mode = st.sidebar.radio(
    "Select Mode:",
    ["🤖 Agentic Mode (Full Pipeline)", "👨‍🌾 Traditional Mode (Simple Detection)"],
    index=0
)

# Context input for agentic mode
context = {}
if mode == "🤖 Agentic Mode (Full Pipeline)":
    st.sidebar.subheader("🌾 Farm Context")
    context["plant_type"] = st.sidebar.selectbox(
        "Plant Type",
        ["Tomato", "Potato", "Corn", "Rice", "Wheat", "Other"]
    )
    context["location"] = st.sidebar.text_input("Location (optional)", "California")
    context["season"] = st.sidebar.selectbox(
        "Current Season",
        ["Spring", "Summer", "Monsoon", "Fall", "Winter", "Unknown"]
    )

# Upload image
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if mode == "👨‍🌾 Traditional Mode":
        # Traditional mode (original functionality)
        with st.spinner("Analyzing image..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = probs.argmax().item()
                confidence = probs[0, pred_idx].item()
            
            predicted_class = class_names[pred_idx]
        
        with col2:
            st.subheader("🧠 Model Prediction")
            st.metric(label="Predicted Disease", value=predicted_class)
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            
            top_probs, top_indices = torch.topk(probs, 3)
            st.subheader("Top 3 Predictions:")
            for i in range(3):
                st.write(f"**{class_names[top_indices[0][i].item()]}**: {top_probs[0][i].item():.2%}")
        
        # Grad-CAM Visualization
        if 'gradcam' in st.session_state and st.session_state.gradcam is not None:
            st.subheader("🔥 Grad-CAM Explanation")
            try:
                cam = st.session_state.gradcam.generate(img_tensor, pred_idx)
                cam_np = cam.cpu().numpy()
                cam_np = cv2.resize(cam_np, (224, 224))
                img_np = np.array(image.resize((224, 224)))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.image(img_np, caption="Original", use_container_width=True)
                with col2: st.image(heatmap, caption="Heatmap", use_container_width=True)
                with col3: st.image(overlay, caption="Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {e}")
        
        # Gemini Report
        if gemini_client:
            st.subheader("📄 AI Medical Report")
            with st.spinner("Generating report..."):
                try:
                    prompt = f"""
                    You are an agricultural disease expert. Provide a concise report.

                    Disease: {predicted_class}
                    Confidence: {confidence:.2%}

                    Explain:
                    1. What this disease is
                    2. Common symptoms
                    3. Causes
                    4. Treatment options
                    5. Prevention measures
                    """
                    
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt]
                    )
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    else:
        # Agentic Mode
        with st.spinner("🤖 Agent is processing your case..."):
            result = asyncio.run(
                st.session_state.agent.process_case(image, "demo_farmer_001", context)
            )
        
        plant_case = result["plant_case"]
        diagnosis = result["diagnosis"]
        explanations = result["explanations"]
        recommendations = result["recommendations"]
        followup_plan = result["followup_plan"]
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Diagnosis", "🔍 Explanations", "💡 Recommendations", 
            "📅 Follow-up", "🧠 Agent Memory"
        ])
        
        with tab1:
            st.subheader("🩺 Comprehensive Diagnosis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Disease", plant_case.prediction)
            with col2:
                st.metric("Confidence", f"{plant_case.confidence:.2%}")
            with col3:
                st.metric("Severity", plant_case.severity.value)
            with col4:
                st.metric("Risk Score", f"{diagnosis['risk_score']:.1%}")
            
            if diagnosis["is_recurring"]:
                st.warning("⚠️ This appears to be a recurring issue for this farmer")
            
            st.subheader("📊 Prediction Details")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
            
            # Show top predictions
            top_probs, top_indices = torch.topk(probs, 5)
            for i in range(5):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.progress(top_probs[0][i].item())
                with col2:
                    st.write(f"**{class_names[top_indices[0][i].item()]}**: {top_probs[0][i].item():.2%}")
        
        with tab2:
            st.subheader("🔍 Multi-Modal Explanations")
            
            # Visual Explanation
            if explanations["visual"] is not None and isinstance(explanations["visual"], np.ndarray):
                st.subheader("🔥 Visual Heatmap")
                cam_np = explanations["visual"]
                cam_np = cv2.resize(cam_np, (224, 224))
                img_np = np.array(image.resize((224, 224)))
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                
                col1, col2, col3 = st.columns(3)
                with col1: st.image(img_np, caption="Original", use_container_width=True)
                with col2: st.image(heatmap, caption="Heatmap", use_container_width=True)
                with col3: st.image(overlay, caption="Affected Areas", use_container_width=True)
            
            # Textual Explanation
            if explanations["textual"]:
                st.subheader("📝 Disease Explanation")
                st.write(explanations["textual"])
        
        with tab3:
            st.subheader("💡 Personalized Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚨 Immediate Actions")
                for action in recommendations["immediate_actions"]:
                    st.write(f"• {action}")
            
            with col2:
                st.subheader("💊 Treatment Options")
                if isinstance(recommendations["treatments"], list):
                    for treatment in recommendations["treatments"]:
                        st.write(f"• {treatment}")
                else:
                    st.write(recommendations["treatments"])
            
            if recommendations["prevention"]:
                st.subheader("🛡️ Prevention Measures")
                for prevention in recommendations["prevention"]:
                    st.write(f"• {prevention}")
            
            if recommendations["monitoring"]:
                st.subheader("👀 Monitoring Schedule")
                for monitor in recommendations["monitoring"]:
                    st.write(f"• {monitor}")
        
        with tab4:
            st.subheader("📅 Follow-up Plan")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Next Check", followup_plan["next_check"])
                st.metric("Urgency", followup_plan["urgency"].upper())
            
            with col2:
                st.subheader("Checklist for Next Visit")
                for item in followup_plan["check_items"]:
                    st.checkbox(item)
        
        with tab5:
            st.subheader("🧠 Agent Memory & History")
            
            farmer_memory = st.session_state.agent.memory.get("demo_farmer_001")
            if farmer_memory:
                st.subheader("📊 Field Conditions")
                for key, value in farmer_memory.field_conditions.items():
                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                
                st.subheader("📈 Case History")
                if farmer_memory.history:
                    history_df = []
                    for entry in farmer_memory.history[-5:]:  # Show last 5
                        history_df.append({
                            "Date": entry["date"],
                            "Disease": entry["disease"],
                            "Severity": entry["severity"],
                            "Risk Score": f"{entry.get('risk_score', 0):.1%}"
                        })
                    st.table(history_df)
                else:
                    st.info("No previous cases recorded")
                
                # Export memory option
                if st.button("📤 Export Case History"):
                    json_str = json.dumps(farmer_memory.history, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"farmer_history_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )

else:
    st.info("👈 Please upload a leaf image to begin analysis")
    
    # Show agent status
    if 'agent' in st.session_state:
        st.sidebar.subheader("Agent Status")
        st.sidebar.write(f"**State**: {st.session_state.agent.state.value}")
        st.sidebar.write(f"**Farmers in memory**: {len(st.session_state.agent.memory)}")
        
        if st.sidebar.button("🔄 Reset Agent Memory"):
            st.session_state.agent.memory = {}
            st.session_state.agent._init_demo_memory()
            st.sidebar.success("Agent memory reset!")

# ------------------------------
# Footer
# ------------------------------
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Agentic AI Features:**
    - 🤖 Autonomous multi-step reasoning
    - 🧠 Persistent memory across sessions
    - 🔄 Context-aware recommendations
    - 📅 Proactive follow-up planning
    - 📊 Historical analysis
    """
)