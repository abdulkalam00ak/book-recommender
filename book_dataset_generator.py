import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_book_dataset():
    """Generate a comprehensive book dataset with various genres and demographics"""
    
    # Define book data by genre
    book_data = {
        'dark_fantasy': {
            'titles': [
                'The Witcher: Blood of Elves', 'The Name of the Wind', 'The Blade Itself', 
                'The Broken Empire', 'The Poppy War', 'The Priory of the Orange Tree',
                'The Fifth Season', 'The Way of Kings', 'The Lies of Locke Lamora',
                'The Goblin Emperor', 'The City of Brass', 'The Bear and the Nightingale',
                'The Ten Thousand Doors of January', 'The Invisible Life of Addie LaRue',
                'The Midnight Girls', 'The House in the Cerulean Sea', 'Mexican Gothic',
                'The Seven Husbands of Evelyn Hugo', 'The Silent Companions', 'The Graveyard Shift',
                'Dark Waters Rising', 'Shadow and Bone', 'Six of Crows', 'The Cruel Prince',
                'Red Queen', 'Throne of Glass', 'A Court of Thorns and Roses'
            ],
            'authors': [
                'Andrzej Sapkowski', 'Patrick Rothfuss', 'Joe Abercrombie', 'Mark Lawrence',
                'R.F. Kuang', 'Samantha Shannon', 'N.K. Jemisin', 'Brandon Sanderson',
                'Scott Lynch', 'Katherine Addison', 'S.A. Chakraborty', 'Katherine Arden',
                'Alix E. Harrow', 'V.E. Schwab', 'Alicia Jasinska', 'TJ Klune',
                'Silvia Moreno-Garcia', 'Taylor Jenkins Reid', 'Laura Purcell', 'Stephen King',
                'Michael Crichton', 'Leigh Bardugo', 'Leigh Bardugo', 'Holly Black',
                'Victoria Aveyard', 'Sarah J. Maas', 'Sarah J. Maas'
            ],
            'target_age': [18, 25, 30, 35, 40, 45],
            'target_gender': ['Male', 'Female', 'All']
        },
        
        'fairy_tales': {
            'titles': [
                'The Little Prince', 'Charlotte\'s Web', 'The Lion, the Witch and the Wardrobe',
                'Harry Potter and the Sorcerer\'s Stone', 'The Secret Garden', 'Alice in Wonderland',
                'The Wonderful Wizard of Oz', 'Peter Pan', 'The Wind in the Willows',
                'The Velveteen Rabbit', 'Where the Wild Things Are', 'The Giving Tree',
                'The Very Hungry Caterpillar', 'Goodnight Moon', 'The Cat in the Hat',
                'Green Eggs and Ham', 'Oh, the Places You\'ll Go!', 'The Polar Express',
                'The Rainbow Fish', 'Corduline', 'The Tale of Peter Rabbit',
                'The Snow Queen', 'Beauty and the Beast', 'Cinderella', 'Sleeping Beauty',
                'The Little Mermaid', 'Rapunzel', 'Hansel and Gretel', 'Little Red Riding Hood'
            ],
            'authors': [
                'Antoine de Saint-Exupéry', 'E.B. White', 'C.S. Lewis', 'J.K. Rowling',
                'Frances Hodgson Burnett', 'Lewis Carroll', 'L. Frank Baum', 'J.M. Barrie',
                'Kenneth Grahame', 'Margery Williams', 'Maurice Sendak', 'Shel Silverstein',
                'Eric Carle', 'Margaret Wise Brown', 'Dr. Seuss', 'Dr. Seuss', 'Dr. Seuss',
                'Chris Van Allsburg', 'Marcus Pfister', 'Don Freeman', 'Beatrix Potter',
                'Hans Christian Andersen', 'Gabrielle-Suzanne Barbot', 'Charles Perrault',
                'Charles Perrault', 'Hans Christian Andersen', 'Brothers Grimm', 'Brothers Grimm',
                'Charles Perrault'
            ],
            'target_age': [3, 5, 7, 8, 10, 12],
            'target_gender': ['All', 'Male', 'Female']
        },
        
        'hindi_novels': {
            'titles': [
                'Godan', 'Gaban', 'Nirmala', 'Sevasadan', 'Pratigya', 'Karambhoomi',
                'Ghare Baire', 'Chokher Bali', 'Gitanjali', 'Kabuliwala', 'Postmaster',
                'Maila Anchal', 'Parineeta', 'Devdas', 'Srikanta', 'Pather Panchali',
                'Aparajito', 'Apur Sansar', 'Shesher Kabita', 'Char Adhyay',
                'Raag Darbari', 'Chandrakanta', 'Chandrakanta Santati', 'Bhuthnath',
                'Titli', 'Sunita', 'Kamayani', 'Urvashi', 'Jayshankar Prasad Ki Kahaniyan'
            ],
            'authors': [
                'Munshi Premchand', 'Munshi Premchand', 'Munshi Premchand', 'Munshi Premchand',
                'Munshi Premchand', 'Munshi Premchand', 'Rabindranath Tagore', 'Rabindranath Tagore',
                'Rabindranath Tagore', 'Rabindranath Tagore', 'Rabindranath Tagore',
                'Phanishwar Nath Renu', 'Sarat Chandra Chattopadhyay', 'Sarat Chandra Chattopadhyay',
                'Sarat Chandra Chattopadhyay', 'Bibhutibhushan Bandyopadhyay', 
                'Bibhutibhushan Bandyopadhyay', 'Bibhutibhushan Bandyopadhyay',
                'Rabindranath Tagore', 'Rabindranath Tagore', 'Shrilal Shukla',
                'Devaki Nandan Khatri', 'Devaki Nandan Khatri', 'Devaki Nandan Khatri',
                'Jainendra Kumar', 'Jainendra Kumar', 'Jayshankar Prasad', 'Jayshankar Prasad',
                'Jayshankar Prasad'
            ],
            'target_age': [16, 20, 25, 30, 35, 40, 45, 50],
            'target_gender': ['All', 'Male', 'Female']
        },
        
        'sci_fi': {
            'titles': [
                'Dune', 'Foundation', 'Neuromancer', 'The Martian', 'Ender\'s Game',
                'The Hitchhiker\'s Guide to the Galaxy', '1984', 'Brave New World',
                'The Time Machine', 'The War of the Worlds', 'I, Robot', 'Fahrenheit 451',
                'The Left Hand of Darkness', 'Hyperion', 'The Stars My Destination',
                'Childhood\'s End', 'Starship Troopers', 'The Moon Is a Harsh Mistress',
                'Stranger in a Strange Land', 'Do Androids Dream of Electric Sheep?',
                'The Caves of Steel', 'The Naked Sun', 'Solaris', 'The Handmaid\'s Tale',
                'Ready Player One', 'The Fifth Season', 'Station Eleven', 'The Road',
                'World War Z', 'The Expanse: Leviathan Wakes'
            ],
            'authors': [
                'Frank Herbert', 'Isaac Asimov', 'William Gibson', 'Andy Weir', 'Orson Scott Card',
                'Douglas Adams', 'George Orwell', 'Aldous Huxley', 'H.G. Wells', 'H.G. Wells',
                'Isaac Asimov', 'Ray Bradbury', 'Ursula K. Le Guin', 'Dan Simmons',
                'Alfred Bester', 'Arthur C. Clarke', 'Robert A. Heinlein', 'Robert A. Heinlein',
                'Robert A. Heinlein', 'Philip K. Dick', 'Isaac Asimov', 'Isaac Asimov',
                'Stanisław Lem', 'Margaret Atwood', 'Ernest Cline', 'N.K. Jemisin',
                'Emily St. John Mandel', 'Cormac McCarthy', 'Max Brooks', 'James S.A. Corey'
            ],
            'target_age': [14, 16, 18, 20, 25, 30, 35, 40, 45],
            'target_gender': ['All', 'Male', 'Female']
        }
    }
    
    # Generate dataset
    dataset = []
    book_id = 1
    
    for genre, data in book_data.items():
        titles = data['titles']
        authors = data['authors']
        target_ages = data['target_age']
        target_genders = data['target_gender']
        
        for i, (title, author) in enumerate(zip(titles, authors)):
            # Generate multiple entries for each book with different demographics
            for _ in range(random.randint(50, 200)):  # Random number of ratings per book
                age = random.choice(target_ages) + random.randint(-5, 10)
                age = max(3, min(80, age))  # Ensure age is between 3 and 80
                
                gender = random.choice(target_genders)
                if gender == 'All':
                    gender = random.choice(['Male', 'Female'])
                
                # Generate rating based on genre-age-gender fit
                base_rating = 4.0
                if genre == 'dark_fantasy' and age < 16:
                    base_rating = 2.5
                elif genre == 'fairy_tales' and age > 40:
                    base_rating = 3.0
                elif genre == 'hindi_novels' and age < 15:
                    base_rating = 2.0
                elif genre == 'sci_fi' and age < 12:
                    base_rating = 2.5
                
                rating = base_rating + random.uniform(-1.5, 1.5)
                rating = max(1.0, min(5.0, rating))
                
                # Generate book details
                isbn = f"978-{random.randint(100000000, 999999999)}"
                pages = random.randint(150, 800)
                publication_year = random.randint(1950, 2024)
                price = random.randint(200, 1500)
                
                # Generate Amazon-style links
                amazon_link = f"https://www.amazon.in/dp/{random.choice(['B0', 'B1', 'B2'])}{random.randint(1000000, 9999999)}"
                
                dataset.append({
                    'book_id': book_id,
                    'title': title,
                    'author': author,
                    'genre': genre,
                    'isbn': isbn,
                    'pages': pages,
                    'publication_year': publication_year,
                    'price': price,
                    'amazon_link': amazon_link,
                    'user_age': age,
                    'user_gender': gender,
                    'rating': round(rating, 2),
                    'review_count': random.randint(10, 5000)
                })
            
            book_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV
    df.to_csv('book_recommendations_dataset.csv', index=False)
    print(f"Dataset created with {len(df)} records!")
    print(f"Unique books: {df['book_id'].nunique()}")
    print(f"Genres: {df['genre'].unique()}")
    
    return df

# Generate the dataset
if __name__ == "__main__":
    df = generate_book_dataset()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
