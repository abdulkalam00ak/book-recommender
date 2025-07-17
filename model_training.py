import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class BookRecommendationModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.genre_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.is_trained = False
        
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Encode categorical variables
        data['genre_encoded'] = self.genre_encoder.fit_transform(data['genre'])
        data['gender_encoded'] = self.gender_encoder.fit_transform(data['user_gender'])
        
        # Feature engineering
        data['age_genre_interaction'] = data['user_age'] * data['genre_encoded']
        data['age_squared'] = data['user_age'] ** 2
        
        # Select features for training
        features = ['user_age', 'user_gender', 'genre', 'publication_year', 'pages', 'price']
        feature_columns = ['user_age', 'genre_encoded', 'gender_encoded', 'publication_year', 
                          'pages', 'price', 'age_genre_interaction', 'age_squared']
        
        X = data[feature_columns]
        y = data['rating']
        
        return X, y, data
    
    def train_model(self, df):
        """Train the recommendation model"""
        print("Preprocessing data...")
        X, y, processed_data = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        self.is_trained = True
        return processed_data
    
    def predict_rating(self, age, gender, genre, publication_year=2020, pages=300, price=500):
        """Predict rating for given user profile and book"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Encode categorical variables
        try:
            genre_encoded = self.genre_encoder.transform([genre])[0]
        except ValueError:
            genre_encoded = 0  # Default for unknown genre
            
        try:
            gender_encoded = self.gender_encoder.transform([gender])[0]
        except ValueError:
            gender_encoded = 0  # Default for unknown gender
        
        # Feature engineering
        age_genre_interaction = age * genre_encoded
        age_squared = age ** 2
        
        # Create feature array
        features = np.array([[age, genre_encoded, gender_encoded, publication_year, 
                            pages, price, age_genre_interaction, age_squared]])
        
        # Predict rating
        rating = self.model.predict(features)[0]
        return max(1.0, min(5.0, rating))  # Ensure rating is between 1 and 5
    
    def recommend_books(self, user_age, user_gender, df, n_recommendations=5):
        """Recommend books based on user profile"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Get unique books
        unique_books = df.drop_duplicates(subset=['book_id']).copy()
        
        # Predict ratings for all books
        predictions = []
        for _, book in unique_books.iterrows():
            try:
                predicted_rating = self.predict_rating(
                    user_age, user_gender, book['genre'], 
                    book['publication_year'], book['pages'], book['price']
                )
                predictions.append({
                    'book_id': book['book_id'],
                    'title': book['title'],
                    'author': book['author'],
                    'genre': book['genre'],
                    'isbn': book['isbn'],
                    'pages': book['pages'],
                    'publication_year': book['publication_year'],
                    'price': book['price'],
                    'amazon_link': book['amazon_link'],
                    'predicted_rating': predicted_rating,
                    'avg_rating': df[df['book_id'] == book['book_id']]['rating'].mean(),
                    'review_count': book['review_count']
                })
            except Exception as e:
                continue
        
        # Sort by predicted rating and return top recommendations
        recommendations = sorted(predictions, key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n_recommendations]
    
    def save_model(self, filepath='book_recommendation_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'genre_encoder': self.genre_encoder,
            'gender_encoder': self.gender_encoder,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='book_recommendation_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.genre_encoder = model_data['genre_encoder']
        self.gender_encoder = model_data['gender_encoder']
        self.is_trained = model_data['is_trained']
        print("Model loaded successfully!")

def train_and_save_model():
    """Main function to train and save the model"""
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('book_recommendations_dataset.csv')
    
    # Initialize and train model
    model = BookRecommendationModel()
    processed_data = model.train_model(df)
    
    # Save model
    model.save_model()
    
    # Test recommendations
    print("\nTesting recommendations...")
    test_recommendations = model.recommend_books(25, 'Male', df, 5)
    
    print("\nSample recommendations for 25-year-old Male:")
    for i, rec in enumerate(test_recommendations, 1):
        print(f"{i}. {rec['title']} by {rec['author']}")
        print(f"   Genre: {rec['genre']}, Predicted Rating: {rec['predicted_rating']:.2f}")
        print(f"   Average Rating: {rec['avg_rating']:.2f}, Reviews: {rec['review_count']}")
        print()
    
    return model, df

if __name__ == "__main__":
    model, df = train_and_save_model()
