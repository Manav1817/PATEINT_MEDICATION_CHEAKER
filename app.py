import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING (Option 1: Modern Minimalist)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Med-Safety | Clinical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Soft & Clean Custom CSS
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Modern Card Container */
    .med-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Stat Badge Metrics */
    .stat-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    /* Alert Status Badges */
    .badge-safe {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MOCK DATA GENERATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_data():
    patients = pd.DataFrame({
        "Patient ID": [f"P-233{i:03d}" for i in range(1, 9)],
        "Patient Name": ["Anya Sharma", "John Doe", "Sarah Jenkins", "Michael Chen", "Emily Watson", "Robert Taylor", "Lisa Anderson", "David Miller"],
        "Age": [34, 52, 29, 61, 45, 68, 38, 50],
        "Prescribed Medication": ["Warfarin", "Ibuprofen", "Lisinopril", "Metformin", "Aspirin", "Atorvastatin", "Amoxicillin", "Sertraline"],
        "Potential Interaction": ["Ibuprofen", "Warfarin", "Spironolactone", "Contrast Dye", "Clopidogrel", "Clarithromycin", "Methotrexate", "MAOIs"],
        "Risk Score": [88, 88, 45, 20, 72, 15, 30, 85],
        "Risk Level": ["Warning", "Warning", "Moderate", "Safe", "Warning", "Safe", "Safe", "Warning"],
        "Last Check": ["10 mins ago", "25 mins ago", "1 hr ago", "2 hrs ago", "3 hrs ago", "5 hrs ago", "6 hrs ago", "1 day ago"]
    })
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=7, freq='D')
    trends = pd.DataFrame({
        "Date": dates.strftime("%a"),
        "Safe Prescriptions": [42, 48, 55, 50, 62, 58, 65],
        "Moderate Risks": [12, 15, 10, 14, 8, 11, 9],
        "High Risks Flagged": [5, 3, 6, 2, 4, 1, 3]
    })
    return patients, trends

patients_df, trends_df = load_sample_data()


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-heart.png", width=50)
    st.title("MedSafety AI")
    st.caption("EHR Prescription Verification System")
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["Overview", "Patients Directory", "Safety Analytics", "Settings"],
        index=0
    )
    
    st.divider()
    st.markdown("**Active Physician:**")
    st.markdown("👨‍⚕️ Dr. Anya Sharma")
    st.caption("Department: Cardiology")


# -----------------------------------------------------------------------------
# 4. PAGE IMPLEMENTATIONS
# -----------------------------------------------------------------------------

# PAGE 1: OVERVIEW (OPTION 1 DESIGN)
if page == "Overview":
    st.title("Prescription Safety Overview")
    st.caption("Real-time clinical drug safety monitoring and interaction predictions")
    st.write("")
    
    # Header Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="med-card">
            <div class="stat-label">System Safety Score</div>
            <div class="stat-value" style="color: #00A86B;">91%</div>
            <small style="color: #00A86B;">↑ 2.4% from last week</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="med-card">
            <div class="stat-label">Active Alerts (24h)</div>
            <div class="stat-value" style="color: #E06D53;">4</div>
            <small style="color: #E06D53;">2 High Priority</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="med-card">
            <div class="stat-label">Pending Validations</div>
            <div class="stat-value">12</div>
            <small style="color: #64748B;">Requires review</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="med-card">
            <div class="stat-label">Total Verified Today</div>
            <div class="stat-value">77</div>
            <small style="color: #00A86B;">100% evaluated</small>
        </div>
        """, unsafe_allow_html=True)

    # Visual Analytics Row
    col_gauge, col_trend = st.columns([1, 1.5])
    
    with col_gauge:
        st.subheader("Prescription Safety Score Gauge")
        
        # Soft Minimalist Gauge Arc
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=88,
            number={'suffix': "%", 'font': {'size': 36, 'color': "#1E293B"}},
            title={'text': "Selected Prescription Risk", 'font': {'size': 14, 'color': "#64748B"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': "#0F172A", 'width': 0.15},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 40], 'color': '#D1FAE5'},   # Safe Soft Green
                    {'range': [40, 70], 'color': '#FEF3C7'},  # Moderate Soft Yellow
                    {'range': [70, 100], 'color': '#FEE2E2'}  # Warning Soft Red
                ],
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_trend:
        st.subheader("Safety Trends")
        
        fig_trend = px.line(
            trends_df, 
            x="Date", 
            y=["Safe Prescriptions", "Moderate Risks", "High Risks Flagged"],
            color_discrete_sequence=["#00A86B", "#F59E0B", "#E06D53"],
            markers=True
        )
        fig_trend.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_trend.update_xaxes(showgrid=False)
        fig_trend.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Recent Validations Section
    st.subheader("Recent Patient Prescription Validations")
    
    # Formatted Dataframe View
    def highlight_risk(val):
        if val == "Warning":
            return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
        elif val == "Moderate":
            return 'background-color: #FEF3C7; color: #92400E;'
        return 'background-color: #D1FAE5; color: #065F46;'

    st.dataframe(
        patients_df.style.map(highlight_risk, subset=["Risk Level"]),
        use_container_width=True,
        hide_index=True
    )


# PAGE 2: PATIENTS DIRECTORY
elif page == "Patients Directory":
    st.title("Patient Directory")
    st.caption("Search patient records and check drug compatibility")
    
    search_term = st.text_input("🔍 Search patient name or medication...", "")
    
    filtered_df = patients_df[
        patients_df["Patient Name"].str.contains(search_term, case=False) |
        patients_df["Prescribed Medication"].str.contains(search_term, case=False)
    ]
    
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)


# PAGE 3: SAFETY ANALYTICS
elif page == "Safety Analytics":
    st.title("Safety Analytics & Insights")
    st.caption("Aggregate clinical data analysis")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk Distribution")
        fig_pie = px.pie(
            patients_df, 
            names="Risk Level", 
            color="Risk Level",
            color_discrete_map={"Safe": "#00A86B", "Moderate": "#F59E0B", "Warning": "#E06D53"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b:
        st.subheader("Risk Score by Patient Age")
        fig_scatter = px.scatter(
            patients_df, 
            x="Age", 
            y="Risk Score", 
            color="Risk Level",
            size="Risk Score",
            color_discrete_map={"Safe": "#00A86B", "Moderate": "#F59E0B", "Warning": "#E06D53"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# PAGE 4: SETTINGS
elif page == "Settings":
    st.title("System Settings")
    st.subheader("Threshold Configurations")
    st.slider("High Risk Flag Threshold Score", min_value=50, max_value=95, value=75)
    st.checkbox("Enable Automated Interaction Notifications", value=True)
    st.checkbox("Require Secondary Pharmacist Sign-Off on High Risk Warnings", value=True)
    st.button("Save Settings")
