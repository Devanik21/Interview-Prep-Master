import streamlit as st
import google.generativeai as genai
import random
import json
from datetime import datetime
import time
# Add these lines after line 7
import markdown2
import weasyprint
from base64 import b64encode

# Configure page
st.set_page_config(
    page_title="CrackAnyJob - Interview Prep Master",
    page_icon="🎯",
    layout="wide"
)

# Famous AI/ML/Data Certification Exams
FAMOUS_EXAMS = {
    "AWS Machine Learning Specialty": {
        "topics": [
            "Data Engineering for ML", "Exploratory Data Analysis", "Data Preprocessing",
            "Feature Engineering", "Model Training & Tuning", "Model Evaluation",
            "Amazon SageMaker", "AWS ML Services", "Model Deployment", "MLOps on AWS",
            "Security & Compliance", "Cost Optimization", "Monitoring & Logging",
            "AutoML", "Built-in Algorithms", "Custom Algorithms", "Batch Transform",
            "Real-time Inference", "A/B Testing", "Data Labeling"
        ],
        "exam_info": {
            "duration": "180 minutes",
            "questions": "65 questions",
            "format": "Multiple choice and multiple response",
            "passing_score": "750/1000",
            "cost": "$300 USD"
        }
    },
    "Google Cloud Professional ML Engineer": {
        "topics": [
            "ML Problem Framing", "ML Solution Architecture", "Data Preparation",
            "ML Model Development", "ML Pipeline Orchestration", "Model Monitoring",
            "Vertex AI", "BigQuery ML", "AutoML", "TensorFlow Extended (TFX)",
            "Kubeflow", "ML Security", "Responsible AI", "Model Serving",
            "Feature Store", "Hyperparameter Tuning", "Distributed Training",
            "MLOps Best Practices", "Cost Management", "Performance Optimization"
        ],
        "exam_info": {
            "duration": "120 minutes",
            "questions": "50-60 questions",
            "format": "Multiple choice and multiple select",
            "passing_score": "Not disclosed",
            "cost": "$200 USD"
        }
    },
    "Microsoft Azure AI Engineer Associate": {
        "topics": [
            "Azure Cognitive Services", "Computer Vision", "Natural Language Processing",
            "Speech Services", "Azure Machine Learning", "Responsible AI",
            "Knowledge Mining", "Document Intelligence", "Bot Framework",
            "Azure OpenAI Service", "Custom Vision", "Form Recognizer",
            "Language Understanding (LUIS)", "QnA Maker", "Translator",
            "Content Moderator", "Anomaly Detector", "Personalizer"
        ],
        "exam_info": {
            "duration": "120 minutes",
            "questions": "40-60 questions",
            "format": "Multiple choice, multiple response, scenarios",
            "passing_score": "700/1000",
            "cost": "$165 USD"
        }
    },
    "TensorFlow Developer Certificate": {
        "topics": [
            "TensorFlow Fundamentals", "Neural Network & Deep Learning",
            "Image Classification", "Natural Language Processing", "Time Series",
            "Sequences & Predictions", "Computer Vision", "Convolutional Neural Networks",
            "Transfer Learning", "Multiclass Classifications", "Binary Classifications",
            "Regression", "Overfitting & Underfitting", "Using Real-world Images",
            "Understanding ImageDataGenerator", "Strategies to Prevent Overfitting"
        ],
        "exam_info": {
            "duration": "300 minutes",
            "questions": "5 coding tasks",
            "format": "Hands-on coding in PyCharm",
            "passing_score": "5/5 tasks correct",
            "cost": "$100 USD"
        }
    },
    "Databricks Certified ML Associate": {
        "topics": [
            "Databricks ML Runtime", "MLflow", "Feature Store", "AutoML",
            "Model Registry", "ML Workflows", "Collaborative Notebooks",
            "Delta Lake for ML", "Hyperopt", "Distributed ML", "Model Serving",
            "A/B Testing", "Model Monitoring", "Data Preparation", "Feature Engineering",
            "Model Training", "Model Tuning", "Model Deployment", "MLOps"
        ],
        "exam_info": {
            "duration": "90 minutes",
            "questions": "45 questions",
            "format": "Multiple choice",
            "passing_score": "70%",
            "cost": "$200 USD"
        }
    },
    "IBM Data Science Professional Certificate": {
        "topics": [
            "Data Science Methodology", "Python for Data Science", "Databases & SQL",
            "Data Analysis with Python", "Data Visualization", "Machine Learning",
            "Applied Data Science Capstone", "IBM Watson Studio", "Jupyter Notebooks",
            "Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn", "Statistics",
            "Hypothesis Testing", "Regression Analysis", "Classification", "Clustering"
        ],
        "exam_info": {
            "duration": "Self-paced",
            "questions": "Multiple projects",
            "format": "Hands-on projects and peer reviews",
            "passing_score": "Project completion",
            "cost": "Coursera subscription"
        }
    },
    "Snowflake SnowPro Core Certification": {
        "topics": [
            "Snowflake Architecture", "Virtual Warehouses", "Storage & Data Protection",
            "Data Movement", "Account & Resource Monitoring", "Performance Optimization",
            "Data Sharing", "Data Marketplace", "Semi-structured Data", "Time Travel",
            "Fail-safe", "Cloning", "Tasks & Streams", "Stored Procedures", "UDFs",
            "Security Features", "Role-based Access Control", "Network Policies"
        ],
        "exam_info": {
            "duration": "115 minutes",
            "questions": "100 questions",
            "format": "Multiple choice and multiple select",
            "passing_score": "750/1000",
            "cost": "$175 USD"
        }
    },
    "Apache Spark Developer Certification": {
        "topics": [
            "Spark Architecture", "RDDs", "DataFrames", "Datasets", "Spark SQL",
            "Data Sources", "Spark Streaming", "MLlib", "GraphX", "Cluster Managers",
            "Performance Tuning", "Memory Management", "Caching", "Partitioning",
            "Joins", "Aggregations", "Window Functions", "UDFs", "Broadcast Variables",
            "Accumulators", "Spark Submit", "Configuration", "Monitoring"
        ],
        "exam_info": {
            "duration": "180 minutes",
            "questions": "40 hands-on problems",
            "format": "Live coding environment",
            "passing_score": "70%",
            "cost": "$300 USD"
        }
    },
    "Tableau Desktop Specialist": {
        "topics": [
            "Connecting to Data", "Data Preparation", "Data Exploration",
            "Data Analysis", "Sharing Insights", "Basic Charts", "Formatting",
            "Calculations", "Mapping", "Analytics", "Dashboards", "Stories",
            "Data Blending", "Joins", "Unions", "Pivoting", "Splitting",
            "Parameters", "Sets", "Groups", "Hierarchies", "Table Calculations"
        ],
        "exam_info": {
            "duration": "60 minutes",
            "questions": "30 questions",
            "format": "Multiple choice",
            "passing_score": "75%",
            "cost": "$100 USD"
        }
    },
    "Alteryx Designer Core Certification": {
        "topics": [
            "Designer Interface", "Data Connections", "Data Preparation",
            "Data Blending", "Data Parsing", "Spatial Analytics", "Predictive Analytics",
            "Workflow Documentation", "Analytic Apps", "Macros", "Interface Tools",
            "In-Database Tools", "Reporting Tools", "Spatial Tools", "Predictive Tools",
            "Time Series Tools", "AB Analysis Tools", "Optimization Tools"
        ],
        "exam_info": {
            "duration": "180 minutes",
            "questions": "80 questions",
            "format": "Multiple choice and hands-on",
            "passing_score": "73%",
            "cost": "$150 USD"
        }
    },
    "Cloudera Data Platform (CDP) Certification": {
        "topics": [
            "CDP Architecture", "Data Engineering", "Data Warehousing",
            "Machine Learning", "Operational Database", "Data Hub", "Data Catalog",
            "Replication Manager", "Workload XM", "Management Console",
            "Security & Governance", "Apache Hive", "Apache Impala", "Apache Spark",
            "Apache Kafka", "Apache NiFi", "Cloudera Machine Learning"
        ],
        "exam_info": {
            "duration": "120 minutes",
            "questions": "60 questions",
            "format": "Multiple choice",
            "passing_score": "70%",
            "cost": "$295 USD"
        }
    },
    "H2O.ai Certified AI/ML Specialist": {
        "topics": [
            "H2O-3 Platform", "AutoML", "Driverless AI", "H2O Flow", "Machine Learning",
            "Deep Learning", "Ensemble Methods", "Model Interpretability", "MLOps",
            "Feature Engineering", "Hyperparameter Tuning", "Model Validation",
            "Deployment Strategies", "Monitoring & Management", "Explainable AI",
            "Responsible AI", "Business Applications"
        ],
        "exam_info": {
            "duration": "120 minutes",
            "questions": "50 questions",
            "format": "Multiple choice and scenario-based",
            "passing_score": "80%",
            "cost": "$200 USD"
        }
    },
    "NVIDIA Deep Learning Institute (DLI) Certification": {
        "topics": [
            "Fundamentals of Deep Learning", "Computer Vision", "Natural Language Processing",
            "Accelerated Computing", "CUDA Programming", "Deep Learning Frameworks",
            "Model Optimization", "Inference Deployment", "Multi-GPU Training",
            "Distributed Training", "TensorRT", "Triton Inference Server", "RAPIDS",
            "Jetson Platform", "Edge AI", "Conversational AI", "Recommender Systems"
        ],
        "exam_info": {
            "duration": "120 minutes",
            "questions": "Hands-on assessment",
            "format": "Practical coding and implementation",
            "passing_score": "Competency-based",
            "cost": "$90 USD"
        }
    }
}

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

DIFFICULTY_LEVELS = ["Foundational: For building a solid base.", "Intermediate: For typical interview problems.", "Competitive (Hard): For challenging rounds at top companies."]


# Place this entire block after the DIFFICULTY_LEVELS list (around line 433)

# CSS for styling the PDF report to make it look professional and colored
PDF_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
body {
    font-family: 'Roboto', 'Helvetica', sans-serif;
    color: #333;
    line-height: 1.6;
}
h1, h2, h3 {
    color: #0d47a1; /* Dark Blue */
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 8px;
    margin-top: 24px;
}
h1 { font-size: 28px; }
h2 { font-size: 22px; }
h3 { font-size: 18px; }
strong {
    color: #c62828; /* Dark Red */
}
pre {
    background-color: #282c34; /* Dark background for code blocks */
    color: #abb2bf; /* Light text for code */
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 14px;
}
code {
    font-family: 'Courier New', Courier, monospace;
    background-color: #f0f0f0;
    padding: 2px 5px;
    border-radius: 4px;
}
pre > code {
    background-color: transparent;
    padding: 0;
}
ul {
    padding-left: 20px;
}
li {
    margin-bottom: 8px;
}
"""

def create_pdf_report(questions, role_or_exam, is_exam):
    """Generates a styled PDF report from the list of questions."""
    report_title = "Certification Exam Prep" if is_exam else "Interview Prep"
    
    # Start building the markdown content
    markdown_content = f"# {report_title}: {role_or_exam}\n"
    
    for i, q in enumerate(questions, 1):
        question_type_emoji = "📜" if q.get('context') == 'exam' else "🎤"
        markdown_content += f"## {question_type_emoji} Question {i}: {q['topic']}\n"
        markdown_content += f"**Type:** {q['type']} | **Difficulty:** {q['difficulty']}\n\n"
        markdown_content += f"{q['question']}\n\n"
        markdown_content += "---\n\n"
        
    # Convert markdown to styled HTML
    html_body = markdown2.markdown(markdown_content, extras=["fenced-code-blocks", "code-friendly"])
    
    html_string = f"""
    <html>
        <head>
            <style>{PDF_CSS}</style>
        </head>
        <body>
            {html_body}
        </body>
    </html>
    """
    
    # Generate PDF from HTML
    pdf_bytes = weasyprint.HTML(string=html_string).write_pdf()
    return pdf_bytes

def get_pdf_download_link(pdf_bytes, filename="CrackAnyJob_Prep_Report.pdf"):
    """Generates a download link for the PDF."""
    b64_pdf = b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" style="float: right; background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">📥 Download PDF Report</a>'

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
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini API: {str(e)}")
        return None

def generate_question(model, role, topic, question_type, difficulty):
    prompt = f"""
    Generate a comprehensive {difficulty.lower()} level {question_type.lower()} interview question for a {role} position, 
    specifically focusing on {topic}.
    
    Requirements:
    -Foundational: Perfect for starting a topic and building a solid base without feeling overwhelmed.

    -Intermediate: This is the sweet spot. The majority of questions asked in companies like TCS, Infosys, Wipro, and even product-based companies will fall into this category.

    -Competitive (Hard): This is your target for top-tier companies (like FAANG, top startups) and challenging online assessment rounds.
    - Make the question industry-relevant but for undergraduate fresher level( example - bachelors degree ) and should be suitable for a final year student.
    - Ensure the question is clear 
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

def generate_exam_question(model, exam_name, topic, difficulty):
    exam_info = FAMOUS_EXAMS[exam_name]
    prompt = f"""
    Generate a {difficulty.lower()} level certification exam question for {exam_name}, 
    specifically focusing on {topic}.
    
    Exam Context:
    - Duration: {exam_info['exam_info']['duration']}
    - Format: {exam_info['exam_info']['format']}
    - Focus: Real-world application and practical knowledge
    
    Requirements:
    -Foundational: Perfect for starting a topic and building a solid base without feeling overwhelmed.

    -Intermediate: This is the sweet spot. The majority of questions asked in companies like TCS, Infosys, Wipro, and even product-based companies will fall into this category.

    -Competitive (Hard): This is your target for top-tier companies (like FAANG, top startups) and challenging online assessment rounds.
    - Make the question realistic and exam-style
    - Include multiple choice options (A, B, C, D) if applicable
    - Add detailed explanation for the correct answer
    - Include common misconceptions as distractors
    - Focus on practical application, not just theory
    
    Format your response as:
    **Question:** [Main question]
    **Options:** [If multiple choice]
    **Correct Answer:** [Answer with explanation]
    **Key Concepts:** [Important concepts tested]
    **Study Tips:** [How to prepare for this topic]
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating question: {str(e)}"

def generate_comprehensive_question_set(model, role_or_exam, num_questions=10, is_exam=False):
    if is_exam:
        topics = FAMOUS_EXAMS[role_or_exam]["topics"]
        context = "exam"
    else:
        topics = JOB_ROLES[role_or_exam]["topics"]
        context = "interview"
    
    questions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_questions):
        topic = random.choice(topics)
        difficulty = random.choice(DIFFICULTY_LEVELS)
        
        if is_exam:
            status_text.text(f"Generating exam question {i+1}/{num_questions}: {topic}")
            question = generate_exam_question(model, role_or_exam, topic, difficulty)
            question_type = "Certification"
        else:
            question_type = random.choice(QUESTION_TYPES)
            status_text.text(f"Generating question {i+1}/{num_questions}: {topic} - {question_type}")
            question = generate_question(model, role_or_exam, topic, question_type, difficulty)
        
        questions.append({
            "topic": topic,
            "type": question_type,
            "difficulty": difficulty,
            "question": question,
            "context": context,
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
        st.info("🔑 **Get Started:**\n1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Enter it in the sidebar\n3. Select your preparation mode\n4. Start practicing!")
        return
    
    model = configure_gemini(api_key)
    if not model:
        return
    
    st.sidebar.success("✅ API Key configured successfully!")
    
    # Preparation mode selection
    st.sidebar.header("🎯 Preparation Mode")
    prep_mode = st.sidebar.selectbox(
        "Choose your preparation focus:",
        ["🎤 Job Interview Prep", "📜 Certification Exam Prep"]
    )
    
    # Dynamic selection based on mode
    if prep_mode == "🎤 Job Interview Prep":
        st.sidebar.subheader("🎯 Job Role Selection")
        selected_item = st.sidebar.selectbox("Choose your target role:", list(JOB_ROLES.keys()))
        item_topics = JOB_ROLES[selected_item]["topics"]
        is_exam_mode = False
        
        st.sidebar.subheader(f"📚 {selected_item} Topics")
        for i, topic in enumerate(item_topics, 1):
            st.sidebar.text(f"{i}. {topic}")
            
    else:  # Certification Exam Prep
        st.sidebar.subheader("📜 Certification Selection")
        selected_item = st.sidebar.selectbox("Choose your target certification:", list(FAMOUS_EXAMS.keys()))
        item_topics = FAMOUS_EXAMS[selected_item]["topics"]
        is_exam_mode = True
        
        # Display exam info
        exam_info = FAMOUS_EXAMS[selected_item]["exam_info"]
        st.sidebar.subheader(f"📋 {selected_item} Info")
        st.sidebar.text(f"⏱️ Duration: {exam_info['duration']}")
        st.sidebar.text(f"❓ Questions: {exam_info['questions']}")
        st.sidebar.text(f"📝 Format: {exam_info['format']}")
        st.sidebar.text(f"🎯 Passing: {exam_info['passing_score']}")
        st.sidebar.text(f"💰 Cost: {exam_info['cost']}")
        
        st.sidebar.subheader(f"📚 Exam Topics")
        for i, topic in enumerate(item_topics, 1):
            st.sidebar.text(f"{i}. {topic}")
    
    # Question generation options
    st.sidebar.header("⚙️ Question Settings")
    num_questions = st.sidebar.slider("Number of questions to generate", 5, 25, 10)
    
    if not is_exam_mode:
        # Interview modes (only for job interviews)
        interview_mode = st.sidebar.selectbox(
            "Interview Mode",
            ["Mixed Topics", "Topic-focused", "Rapid Fire", "Deep Dive"]
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_exam_mode:
            st.header(f"📜 Exam Prep: {selected_item}")
            exam_info = FAMOUS_EXAMS[selected_item]["exam_info"]
            
            # Exam overview
            st.subheader("📋 Exam Overview")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Duration", exam_info['duration'])
            with col_b:
                st.metric("Questions", exam_info['questions'])
            with col_c:
                st.metric("Cost", exam_info['cost'])
            
            st.info(f"**Format:** {exam_info['format']}")
            st.info(f"**Passing Score:** {exam_info['passing_score']}")
            
        else:
            st.header(f"🚀 Interview Prep: {selected_item}")
        
        # Generate questions button
        button_text = "🎲 Generate Certification Questions" if is_exam_mode else "🎲 Generate Interview Questions"
        if st.button(button_text, type="primary"):
            with st.spinner("Generating comprehensive question set..."):
                questions = generate_comprehensive_question_set(model, selected_item, num_questions, is_exam_mode)
                st.session_state.current_questions = questions
                st.session_state.current_role = selected_item
                st.session_state.is_exam_mode = is_exam_mode
                context = "certification questions" if is_exam_mode else "interview questions"
                st.success(f"Generated {len(questions)} {context} for {selected_item}!")
        
        # Display questions
        if st.session_state.current_questions:
            question_context = "Certification Questions" if st.session_state.get('is_exam_mode', False) else "Interview Questions"
            st.subheader(f"📝 {question_context}")
            
            for i, q in enumerate(st.session_state.current_questions, 1):
                question_type_emoji = "📜" if q.get('context') == 'exam' else "🎤"
                with st.expander(f"{question_type_emoji} Question {i}: {q['topic']} - {q['type']} ({q['difficulty']})"):
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
            
            # --- PDF Export Section ---
            st.markdown("---")
            st.subheader("📄 Export Report")
            st.write("Click the button below to generate a colored PDF of the questions above.")
            
            if st.button("Generate and Download PDF", type="primary"):
                with st.spinner("🎨 Creating your styled PDF report..."):
                    is_exam = st.session_state.get('is_exam_mode', False)
                    role_or_exam = st.session_state.current_role
                    
                    pdf_data = create_pdf_report(st.session_state.current_questions, role_or_exam, is_exam)
                    
                    # Generate a clean filename
                    filename = "".join(c for c in role_or_exam if c.isalnum() or c in (' ', '_')).rstrip()
                    filename = f"CrackAnyJob_{filename.replace(' ', '_')}_Prep.pdf"
                    # Create and display the download link
                    download_link = get_pdf_download_link(pdf_data, filename)
                    st.markdown(download_link, unsafe_allow_html=True)
                    st.success("Your PDF is ready for download! Check the link above.")

    
    with col2:
        st.header("📊 Progress Tracker")
        
        if st.session_state.current_questions:
            # Statistics
            total_questions = len(st.session_state.current_questions)
            topics_covered = len(set(q['topic'] for q in st.session_state.current_questions))
            
            st.metric("Total Questions", total_questions)
            st.metric("Topics Covered", topics_covered)
            st.metric("Unique Question Types", len(set(q['type'] for q in st.session_state.current_questions)))
            
            # Context-specific metrics
            if st.session_state.get('is_exam_mode', False):
                st.metric("Exam Focus", "Certification Prep")
            else:
                st.metric("Interview Focus", "Job Preparation")
            
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
        
        # Dynamic tips based on mode
        if st.session_state.get('is_exam_mode', False):
            st.header("📜 Certification Tips")
            tips = [
                "Read official documentation thoroughly",
                "Take practice exams regularly",
                "Focus on hands-on experience",
                "Join study groups and forums",
                "Schedule your exam strategically",
                "Practice time management",
                "Review exam objectives carefully",
                "Use official practice materials",
                "Understand the exam format",
                "Plan for retakes if needed"
            ]
        else:
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
    
    # Additional exam resources section
    if st.session_state.get('is_exam_mode', False) and st.session_state.current_questions:
        st.markdown("---")
        st.header("📚 Additional Resources")
        
        current_exam = st.session_state.current_role
        if current_exam:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🔗 Official Links")
                exam_links = {
                    "AWS Machine Learning Specialty": "https://aws.amazon.com/certification/certified-machine-learning-specialty/",
                    "Google Cloud Professional ML Engineer": "https://cloud.google.com/certification/machine-learning-engineer",
                    "Microsoft Azure AI Engineer Associate": "https://docs.microsoft.com/en-us/learn/certifications/azure-ai-engineer/",
                    "TensorFlow Developer Certificate": "https://www.tensorflow.org/certificate",
                    "Databricks Certified ML Associate": "https://www.databricks.com/learn/certification",
                }
                
                if current_exam in exam_links:
                    st.markdown(f"[Official Exam Page]({exam_links[current_exam]})")
            
            with col2:
                st.subheader("📖 Study Materials")
                st.write("• Official documentation")
                st.write("• Hands-on labs")
                st.write("• Practice exams")
                st.write("• Video courses")
            
            with col3:
                st.subheader("⏰ Study Plan")
                st.write("• 4-8 weeks preparation")
                st.write("• Daily practice sessions")
                st.write("• Weekly mock exams")
                st.write("• Hands-on projects") #Topic coverage
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
