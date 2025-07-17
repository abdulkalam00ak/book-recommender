import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from model_training import BookRecommendationModel

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .book-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .book-author {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .book-details {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .rating-badge {
        background: #10B981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .buy-button {
        background: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .buy-button:hover {
        background: #DC2626;
        text-decoration: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('book_recommendations_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please run the dataset generator first.")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = BookRecommendationModel()
        model.load_model('book_recommendation_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model not found! Please train the model first.")
        return None

def display_book_card(book, index):
    """Display a book recommendation card"""
    st.markdown(f"""
    <div class="book-card">
        <div class="book-title">{index}. {book['title']}</div>
        <div class="book-author">by {book['author']}</div>
        <div class="book-details">
            <strong>Genre:</strong> {book['genre'].replace('_', ' ').title()} | 
            <strong>Pages:</strong> {book['pages']} | 
            <strong>Year:</strong> {book['publication_year']} | 
            <strong>Price:</strong> ₹{book['price']}
        </div>
        <div style="margin-top: 1rem;">
            <span class="rating-badge">⭐ {book['predicted_rating']:.2f}</span>
            <span style="margin-left: 1rem; opacity: 0.8;">
                Avg Rating: {book['avg_rating']:.2f} ({book['review_count']} reviews)
            </span>
        </div>
        <a href="{book['amazon_link']}" target="_blank" class="buy-button">
            🛒 Buy on Amazon
        </a>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    if df is None or model is None:
        st.stop()
    
    # Sidebar for user input
    st.sidebar.header("📋 User Profile")
    
    # User inputs
    user_age = st.sidebar.slider("Age", min_value=3, max_value=80, value=25, step=1)
    user_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # Genre preference (optional filter)
    st.sidebar.header("📖 Genre Preferences")
    available_genres = df['genre'].unique()
    genre_display = {
        'dark_fantasy': 'Dark Fantasy',
        'fairy_tales': 'Fairy Tales',
        'hindi_novels': 'Hindi Novels',
        'sci_fi': 'Science Fiction'
    }
    
    selected_genres = st.sidebar.multiselect(
        "Select preferred genres (optional)",
        available_genres,
        format_func=lambda x: genre_display.get(x, x.title())
    )
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider("Number of recommendations", 1, 10, 5)
    
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding perfect books for you..."):
                try:
                    # Get recommendations
                    recommendations = model.recommend_books(user_age, user_gender, df, num_recommendations * 2)
                    
                    # Filter by genre if selected
                    if selected_genres:
                        recommendations = [r for r in recommendations if r['genre'] in selected_genres]
                    
                    # Limit to requested number
                    recommendations = recommendations[:num_recommendations]
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} perfect books for you!")
                        
                        # Display recommendations
                        for i, book in enumerate(recommendations, 1):
                            display_book_card(book, i)
                    else:
                        st.warning("No books found matching your preferences. Try adjusting your filters.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    # Statistics section
    st.markdown("---")
    st.header("📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", df['book_id'].nunique())
    
    with col2:
        st.metric("Total Ratings", len(df))
    
    with col3:
        st.metric("Genres Available", len(df['genre'].unique()))
    
    with col4:
        st.metric("Average Rating", f"{df['rating'].mean():.2f}")
    
    # Genre distribution
    st.subheader("📈 Genre Distribution")
    genre_counts = df['genre'].value_counts()
    genre_display_counts = {genre_display.get(k, k.title()): v for k, v in genre_counts.items()}
    st.bar_chart(genre_display_counts)
    
    # Age distribution
    st.subheader("👥 Age Distribution of Users")
    age_dist = df['user_age'].value_counts().sort_index()
    st.line_chart(age_dist)
    
    # Sample books by genre
    st.subheader("📚 Sample Books by Genre")
    
    for genre in available_genres:
        with st.expander(f"{genre_display.get(genre, genre.title())} Books"):
            genre_books = df[df['genre'] == genre].drop_duplicates(subset=['book_id']).head(5)
            for _, book in genre_books.iterrows():
                st.write(f"**{book['title']}** by {book['author']}")
                st.write(f"📖 {book['pages']} pages | 📅 {book['publication_year']} | 💰 ₹{book['price']}")
                st.write("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; opacity: 0.7; margin-top: 2rem;">
            <p>📚 Book Recommendation System | Built with Streamlit & Machine Learning</p>
            <p>Discover your next favorite book based on your age, gender, and reading preferences!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
