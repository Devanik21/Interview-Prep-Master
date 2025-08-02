import streamlit as st
import google.generativeai as genai
import random
import json
from datetime import datetime
import time

# Configure page
st.set_page_config(
    page_title="CrackAnyJob - Interview Prep Master",
    page_icon="🎯",
    layout="wide"
)

# Job roles and their comprehensive topics
JOB_ROLES = {
    # Data/ML/AI Roles (Primary Focus)
    "Data Scientist": {
        "topics": [
            "Statistics & Probability", "Machine Learning Algorithms", "Deep Learning",
            "Feature Engineering", "Data Preprocessing", "Model Evaluation & Validation",
            "Python (NumPy, Pandas, Scikit-learn)", "R Programming", "SQL & Databases",
            "Data Visualization", "A/B Testing", "Time Series Analysis",
            "Natural Language Processing", "Computer Vision", "MLOps",
            "Big Data Technologies", "Cloud Platforms", "Business Intelligence"
        ]
    },
    "Machine Learning Engineer": {
        "topics": [
            "ML Algorithms & Implementation", "Deep Learning Frameworks", "Model Deployment",
            "MLOps & CI/CD", "Feature Stores", "Model Monitoring", "Scalable ML Systems",
            "Distributed Computing", "Cloud ML Services", "Kubernetes & Docker",
            "Python Advanced", "Model Optimization", "A/B Testing for ML",
            "Data Pipeline Architecture", "Real-time ML", "AutoML", "Model Versioning"
        ]
    },
    "AI Research Scientist": {
        "topics": [
            "Deep Learning Theory", "Neural Network Architectures", "Optimization Theory",
            "Reinforcement Learning", "Generative Models", "Transformer Architecture",
            "Computer Vision Advanced", "NLP Research", "Graph Neural Networks",
            "Meta Learning", "Few-shot Learning", "Mathematical Foundations",
            "Research Methodology", "Paper Implementation", "Experimental Design",
            "PyTorch/TensorFlow Advanced", "CUDA Programming", "Distributed Training"
        ]
    },
    "Data Analyst": {
        "topics": [
            "SQL Advanced", "Excel Advanced", "Tableau/Power BI", "Python (Pandas, NumPy)",
            "Statistical Analysis", "Data Cleaning", "Data Visualization", "Reporting",
            "Business Intelligence", "A/B Testing", "Cohort Analysis", "KPI Development",
            "Web Scraping", "APIs", "ETL Processes", "Database Design", "R Programming"
        ]
    },
    "Data Engineer": {
        "topics": [
            "SQL Advanced", "Python/Scala", "Apache Spark", "Hadoop Ecosystem",
            "Data Warehousing", "ETL/ELT Pipelines", "Apache Kafka", "Apache Airflow",
            "Cloud Platforms (AWS/GCP/Azure)", "Data Modeling", "NoSQL Databases",
            "Stream Processing", "Data Lakes", "Docker & Kubernetes", "Monitoring & Logging",
            "Data Governance", "Performance Optimization", "Distributed Systems"
        ]
    },
    "Computer Vision Engineer": {
        "topics": [
            "Image Processing", "CNN Architectures", "Object Detection", "Image Segmentation",
            "OpenCV", "Deep Learning Frameworks", "Transfer Learning", "GANs",
            "3D Computer Vision", "Video Analysis", "Medical Imaging", "Edge Deployment",
            "Model Optimization", "CUDA Programming", "Python Advanced", "Mathematics",
            "Dataset Creation", "Annotation Tools"
        ]
    },
    "NLP Engineer": {
        "topics": [
            "Natural Language Processing", "Transformers", "BERT/GPT Models", "Text Preprocessing",
            "Named Entity Recognition", "Sentiment Analysis", "Language Models",
            "Hugging Face", "spaCy/NLTK", "Text Classification", "Information Extraction",
            "Question Answering", "Machine Translation", "Speech Processing",
            "Python Advanced", "Deep Learning", "Deployment Strategies"
        ]
    },
    "MLOps Engineer": {
        "topics": [
            "ML Pipeline Design", "Model Deployment", "Kubernetes", "Docker", "CI/CD for ML",
            "Model Monitoring", "A/B Testing Infrastructure", "Feature Stores", "Model Serving",
            "Cloud Platforms", "Infrastructure as Code", "Logging & Monitoring",
            "Version Control for ML", "Automated Testing", "Performance Optimization",
            "Security in ML", "Cost Optimization", "Compliance & Governance"
        ]
    },
    "Business Intelligence Analyst": {
        "topics": [
            "SQL Advanced", "Data Warehousing", "ETL Processes", "Tableau/Power BI Advanced",
            "Business Analysis", "KPI Development", "Dashboard Design", "Data Modeling",
            "Excel Advanced", "Statistical Analysis", "Reporting", "Data Governance",
            "Business Requirements", "Stakeholder Management", "Data Storytelling"
        ]
    },
    "Research Scientist": {
        "topics": [
            "Research Methodology", "Statistical Analysis", "Experimental Design",
            "Scientific Writing", "Data Analysis", "Literature Review", "Hypothesis Testing",
            "Machine Learning", "Python/R", "Academic Publishing", "Grant Writing",
            "Peer Review Process", "Collaboration", "Ethics in Research"
        ]
    },
    
    # Software Development Roles
    "Software Developer": {
        "topics": [
            "Data Structures & Algorithms", "System Design", "Object-Oriented Programming",
            "Database Management Systems", "Operating Systems", "Computer Networks",
            "Programming Languages", "Version Control", "Testing", "Design Patterns",
            "Web Development", "API Design", "Security", "Performance Optimization"
        ]
    },
    "Frontend Developer": {
        "topics": [
            "HTML/CSS Advanced", "JavaScript ES6+", "React/Angular/Vue", "TypeScript",
            "Responsive Design", "Web Performance", "Browser APIs", "Testing",
            "Build Tools", "State Management", "UI/UX Principles", "Accessibility",
            "Progressive Web Apps", "Cross-browser Compatibility"
        ]
    },
    "Backend Developer": {
        "topics": [
            "Server-side Programming", "Database Design", "API Development", "Microservices",
            "System Design", "Caching", "Message Queues", "Security", "Testing",
            "Performance Optimization", "DevOps Basics", "Cloud Services", "Monitoring"
        ]
    },
    "Full Stack Developer": {
        "topics": [
            "Frontend Technologies", "Backend Development", "Database Management",
            "System Design", "API Design", "DevOps", "Testing", "Security",
            "Performance Optimization", "Version Control", "Project Management"
        ]
    },
    "DevOps Engineer": {
        "topics": [
            "Linux/Unix", "Docker & Kubernetes", "CI/CD Pipelines", "Infrastructure as Code",
            "Cloud Platforms", "Monitoring & Logging", "Configuration Management",
            "Scripting", "Security", "Networking", "Database Administration", "Performance Tuning"
        ]
    },
    "Product Manager": {
        "topics": [
            "Product Strategy", "Market Research", "User Experience", "Analytics",
            "Roadmap Planning", "Stakeholder Management", "Agile/Scrum", "A/B Testing",
            "Business Analysis", "Technical Understanding", "Communication", "Leadership"
        ]
    },
    "Cybersecurity Analyst": {
        "topics": [
            "Network Security", "Threat Analysis", "Incident Response", "Risk Assessment",
            "Security Frameworks", "Penetration Testing", "Cryptography", "Compliance",
            "Security Tools", "Forensics", "Malware Analysis", "Security Policies"
        ]
    },
    "Cloud Architect": {
        "topics": [
            "Cloud Platforms", "Architecture Design", "Security", "Cost Optimization",
            "Migration Strategies", "Disaster Recovery", "Monitoring", "Compliance",
            "Microservices", "Containerization", "Networking", "Identity Management"
        ]
    },
    "QA Engineer": {
        "topics": [
            "Test Planning", "Manual Testing", "Automated Testing", "Test Frameworks",
            "Bug Reporting", "Performance Testing", "Security Testing", "API Testing",
            "Mobile Testing", "Cross-browser Testing", "Continuous Testing", "Quality Metrics"
        ]
    },
    "UI/UX Designer": {
        "topics": [
            "Design Principles", "User Research", "Wireframing", "Prototyping",
            "Design Tools", "Usability Testing", "Information Architecture",
            "Interaction Design", "Visual Design", "Accessibility", "Design Systems"
        ]
    }
}

# Question types and difficulty levels
QUESTION_TYPES = [
    "Conceptual", "Practical", "Problem-solving", "Case Study", 
    "Coding", "System Design", "Behavioral", "Scenario-based"
]

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

def initialize_session_state():
    if 'current_questions' not in st.session_state:
        st.session_state.current_questions = []
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []
    if 'current_role' not in st.session_state:
        st.session_state.current_role = None
    if 'interview_mode' not in st.session_state:
        st.session_state.interview_mode = False

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None

def generate_question(model, role, topic, question_type, difficulty):
    prompt = f"""
    Generate a comprehensive {difficulty.lower()} level {question_type.lower()} interview question for a {role} position, 
    specifically focusing on {topic}.
    
    Requirements:
    - Make the question challenging and industry-relevant
    - Include follow-up questions if applicable
    - For coding questions, specify the programming language
    - For case studies, provide realistic scenarios
    - For system design, include scale and constraints
    - Ensure the question tests deep understanding, not just memorization
    
    Format your response as:
    **Question:** [Main question]
    **Follow-up:** [Additional probing questions]
    **Expected Topics to Cover:** [Key areas the answer should address]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating question: {str(e)}"

def generate_comprehensive_question_set(model, role, num_questions=10):
    topics = JOB_ROLES[role]["topics"]
    questions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_questions):
        topic = random.choice(topics)
        question_type = random.choice(QUESTION_TYPES)
        difficulty = random.choice(DIFFICULTY_LEVELS)
        
        status_text.text(f"Generating question {i+1}/{num_questions}: {topic} - {question_type}")
        
        question = generate_question(model, role, topic, question_type, difficulty)
        questions.append({
            "topic": topic,
            "type": question_type,
            "difficulty": difficulty,
            "question": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        progress_bar.progress((i + 1) / num_questions)
        time.sleep(0.5)  # Avoid rate limiting
    
    progress_bar.empty()
    status_text.empty()
    return questions

def main():
    st.title("🎯 CrackAnyJob - Interview Preparation Master")
    st.markdown("**Comprehensive interview preparation for 20+ job roles with focus on Data, ML & AI**")
    
    initialize_session_state()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio")
    
    if not api_key:
        st.sidebar.warning("Please enter your Gemini API key to continue")
        st.info("🔑 **Get Started:**\n1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Enter it in the sidebar\n3. Select your target job role\n4. Start practicing!")
        return
    
    model = configure_gemini(api_key)
    if not model:
        return
    
    st.sidebar.success("✅ API Key configured successfully!")
    
    # Role selection
    st.sidebar.header("🎯 Job Role Selection")
    selected_role = st.sidebar.selectbox("Choose your target role:", list(JOB_ROLES.keys()))
    
    # Display role topics
    if selected_role:
        st.sidebar.subheader(f"📚 {selected_role} Topics")
        topics = JOB_ROLES[selected_role]["topics"]
        for i, topic in enumerate(topics, 1):
            st.sidebar.text(f"{i}. {topic}")
    
    # Question generation options
    st.sidebar.header("⚙️ Question Settings")
    num_questions = st.sidebar.slider("Number of questions to generate", 5, 25, 10)
    
    # Interview modes
    interview_mode = st.sidebar.selectbox(
        "Interview Mode",
        ["Mixed Topics", "Topic-focused", "Rapid Fire", "Deep Dive"]
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🚀 Interview Prep: {selected_role}")
        
        # Generate questions button
        if st.button("🎲 Generate New Question Set", type="primary"):
            with st.spinner("Generating comprehensive question set..."):
                questions = generate_comprehensive_question_set(model, selected_role, num_questions)
                st.session_state.current_questions = questions
                st.session_state.current_role = selected_role
                st.success(f"Generated {len(questions)} questions for {selected_role}!")
        
        # Display questions
        if st.session_state.current_questions:
            st.subheader("📝 Interview Questions")
            
            for i, q in enumerate(st.session_state.current_questions, 1):
                with st.expander(f"Question {i}: {q['topic']} - {q['type']} ({q['difficulty']})"):
                    st.markdown(q['question'])
                    
                    # Add practice features
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"Mark as Practiced", key=f"practice_{i}"):
                            st.success("Marked as practiced!")
                    with col_b:
                        if st.button(f"Need Review", key=f"review_{i}"):
                            st.warning("Added to review list!")
                    with col_c:
                        if st.button(f"Mastered", key=f"master_{i}"):
                            st.success("Great job! 🎉")
    
    with col2:
        st.header("📊 Progress Tracker")
        
        if st.session_state.current_questions:
            # Statistics
            total_questions = len(st.session_state.current_questions)
            topics_covered = len(set(q['topic'] for q in st.session_state.current_questions))
            
            st.metric("Total Questions", total_questions)
            st.metric("Topics Covered", topics_covered)
            st.metric("Unique Question Types", len(set(q['type'] for q in st.session_state.current_questions)))
            
            # Difficulty distribution
            st.subheader("📈 Difficulty Distribution")
            difficulty_counts = {}
            for q in st.session_state.current_questions:
                difficulty_counts[q['difficulty']] = difficulty_counts.get(q['difficulty'], 0) + 1
            
            for diff, count in difficulty_counts.items():
                st.write(f"**{diff}:** {count} questions")
            
            # Topic coverage
            st.subheader("🎯 Topic Coverage")
            topic_counts = {}
            for q in st.session_state.current_questions:
                topic_counts[q['topic']] = topic_counts.get(q['topic'], 0) + 1
            
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                st.write(f"**{topic}:** {count}")
        
        # Tips section
        st.header("💡 Interview Tips")
        tips = [
            "Practice explaining concepts in simple terms",
            "Prepare real-world examples for each topic",
            "Practice coding on a whiteboard",
            "Research the company's tech stack",
            "Prepare questions to ask the interviewer",
            "Practice system design on paper",
            "Review your resume thoroughly",
            "Mock interview with peers"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")
    
    # Footer
    st.markdown("---")
    st.markdown("**CrackAnyJob** - Master your interview preparation with AI-powered questions 🚀")

if __name__ == "__main__":
    main()
