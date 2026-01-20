1. chatgpt.ipynb — Data Processing & Model Development
In the Jupyter Notebook,I focused on the core "behind-the-scenes" work: preparing the data and training the machine learning model.

Exploratory Data Analysis (EDA): I loaded the dataset and performed initial checks using df.info(), df.shape, and df.isnull().sum() to 
understand the data structure and identify missing values.

Data Cleaning:

Datetime Handling: I identified and replaced placeholder values like "########" in the date column with NaN and converted the column to 
a proper datetime format for time-based analysis.

Text Preprocessing: I built a robust cleaning pipeline for the review column. This included converting text to lowercase, removing URLs, 
stripping special characters/numbers, removing English stopwords, and applying WordNet Lemmatization to reduce words to their base form.

Feature Engineering & Model Training:

TF-IDF Vectorization: I converted the cleaned text data into numerical format using TfidfVectorizer so the model could process it.

Classification: I trained a Logestic Regression ,  Random Forest, Multinomial Naive Bayes model to classify the sentiment of the reviews.

Evaluation: I evaluated the model using accuracy scores, classification reports, and confusion matrices.

Model Deployment Preparation: Finally, I exported the trained model (nlp_model.pkl) and the vectorizer (tfidf_vectorizer.pkl) using pickle 
so they could be used in the live application.

2. chat.py — Streamlit Dashboard Application
In the Python script, you built the user-facing interface using Streamlit to showcase the project’s insights and provide a live sentiment analysis tool.

Model Integration: I set up the app to load the pre-trained Naive Bayes model and TF-IDF vectorizer at startup.

Interactive Sidebar: I implemented a navigation menu using streamlit_option_menu with two primary sections: "AI Echo Sentiment Analysis" and 
"Sentiment Analysis" (Insights).

Live Sentiment Predictor: I created a functional text area where users can type their own reviews. The app processes this input through the cleaning 
pipeline and uses the loaded model to predict the sentiment in real-time.

Data Visualization (Dashboard): I built an extensive analytics dashboard using matplotlib and seaborn to answer key business questions, including:

Word Clouds: Visualizing the most frequent words in positive and negative reviews.

Sentiment Trends: Showing how sentiment has changed over time and across different ChatGPT versions.

Feedback Themes: Using a CountVectorizer to extract the most common keywords specifically from negative feedback to identify areas for improvement.

Platform & Rating Analysis: Stripplots and frequency charts to see which platforms (Google Play, App Store, etc.) have the highest engagement and how ratings
relate to sentiment.
