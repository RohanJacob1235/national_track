import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Apply custom styling
st.markdown("""
    <style>
        /* General styling */
        body {
            background-color: #F4F4F9; /* Light grey background */
            font-family: 'Roboto', sans-serif;
        }
        
        /* Title styling - Make it HUGE */
        .title {
            font-size: 72px;  /* Increased font size */
            font-weight: bold;
            color: #6C63FF;  /* Purple accent color */
            text-align: center;
            font-family: 'Roboto', sans-serif;
            margin-bottom: 10px;
        }
        
        /* Subtitle styling */
        .subtitle {
            font-size: 24px;
            color: #6A6A6A;  /* Muted dark gray */
            text-align: center;
            font-family: 'Roboto', sans-serif;
            margin-bottom: 30px;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #6C63FF;  /* Purple sidebar */
            color: white;
        }
        
        .sidebar .sidebar-content h1 {
            color: #FFFFFF; /* White title in the sidebar */
        }

        /* Custom button styles */
        .stButton>button {
            background-color: #6C63FF; /* Purple buttons */
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #5A52D1; /* Darker purple on hover */
        }

        /* Image styles */
        .stImage>img {
            border-radius: 15px; /* Rounded corners for images */
            border: 3px solid #6C63FF; /* Border around images */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow for images */
        }
    </style>
""", unsafe_allow_html=True)

# Generate Synthetic Dataset
def generate_dataset():
    tracks = ["AI & ML", "Cloud & Cybersecurity", "Data Science", "IoT & Embedded"]
    states = ["Kerala", "Tamil Nadu", "Karnataka", "Maharashtra"]
    colleges = ["College A", "College B", "College C", "College D"]
    feedback_samples = [
        "Great session, learned a lot!", "Could be more interactive", "Amazing experience!", "Too technical, needed basics"
    ]
    
    data = {
        "Participant_ID": [f"P{i+1}" for i in range(400)],
        "Track": [random.choice(tracks) for _ in range(400)],
        "Day": [random.randint(1, 4) for _ in range(400)],
        "State": [random.choice(states) for _ in range(400)],
        "College": [random.choice(colleges) for _ in range(400)],
        "Feedback": [random.choice(feedback_samples) for _ in range(400)]
    }
    
    return pd.DataFrame(data)

df = generate_dataset()

# Streamlit Dashboard
st.markdown("<h1 class='title'>Krita</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>- A National Poster Presentation Event</h2>", unsafe_allow_html=True)

# Filters
track_filter = st.sidebar.multiselect("Select Tracks", df["Track"].unique(), default=df["Track"].unique())
day_filter = st.sidebar.multiselect("Select Days", df["Day"].unique(), default=df["Day"].unique())
state_filter = st.sidebar.multiselect("Select States", df["State"].unique(), default=df["State"].unique())
college_filter = st.sidebar.multiselect("Select Colleges", df["College"].unique(), default=df["College"].unique())

filtered_df = df[ 
    (df["Track"].isin(track_filter)) & 
    (df["Day"].isin(day_filter)) & 
    (df["State"].isin(state_filter)) & 
    (df["College"].isin(college_filter))
]

# Participation Trends Visualization
st.subheader("Participation Trends")
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
sns.countplot(data=filtered_df, x="Track", ax=ax[0, 0])
ax[0, 0].set_title("Track-wise Participation")
sns.countplot(data=filtered_df, x="Day", ax=ax[0, 1])
ax[0, 1].set_title("Day-wise Participation")
sns.countplot(data=filtered_df, x="State", ax=ax[1, 0])
ax[1, 0].set_title("State-wise Participation")
sns.countplot(data=filtered_df, x="College", ax=ax[1, 1])
ax[1, 1].set_title("College-wise Participation")
st.pyplot(fig)

# Text Analysis - Word Cloud
def generate_wordcloud(track):
    text = " ".join(df[df["Track"] == track]["Feedback"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

st.subheader("Feedback Word Clouds")
selected_track = st.selectbox("Select Track for Word Cloud", df["Track"].unique())
fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(generate_wordcloud(selected_track), interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# Text Similarity Analysis
def feedback_similarity(track):
    feedback_texts = df[df["Track"] == track]["Feedback"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(feedback_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

st.subheader("Feedback Similarity Analysis")
sim_matrix = feedback_similarity(selected_track)
st.write("Feedback Similarity Matrix:")
st.dataframe(pd.DataFrame(sim_matrix, columns=range(1, len(sim_matrix)+1), index=range(1, len(sim_matrix)+1)))

# Image Processing - Gallery
def load_images_for_day(day):
    # Ensure the day is an integer, then format the file path
    day = int(day)  # Convert to integer explicitly if it's a string
    image_path = f"images/day{day}.jpg"  # Image file format: day1.jpeg, day2.jpeg, etc.
    
    if os.path.exists(image_path):  # Check if the file exists
        return Image.open(image_path)
    else:
        return None  # Return None if the image doesn't exist

st.subheader("Day-wise Image Gallery")
day_selected = st.selectbox("Select Day", df["Day"].unique())

# Load the image for the selected day
image = load_images_for_day(day_selected)

if image:
    st.image(image, caption=f"Day {day_selected}", use_container_width=True)
else:
    st.warning(f"No image found for Day {day_selected}.")

# Custom Image Processing
def apply_grayscale(img_path):
    img = Image.open(img_path).convert("L")
    return img

st.subheader("Custom Image Processing")
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    st.image(uploaded_image, caption="Original Image", use_container_width=True)
    processed_img = apply_grayscale(uploaded_image)
    st.image(processed_img, caption="Grayscale Image", use_container_width=True)

st.success("Dashboard is ready for analysis!")
