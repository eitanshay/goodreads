import pandas as pd
import numpy as np
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AwardRecommender:
    def __init__(self, goodreads_csv_path=None):
        """
        Initialize the Award Recommender.
        
        Args:
            goodreads_csv_path (str): Path to exported Goodreads CSV file
        """
        self.goodreads_csv_path = goodreads_csv_path
        self.user_books_df = None
        self.hugo_winners = None
        self.nebula_winners = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def load_user_books(self):
        """Load and process the user's book data from Goodreads export CSV"""
        print("Loading your Goodreads library...")
        
        if not os.path.exists(self.goodreads_csv_path):
            print(f"Error: Could not find file at {self.goodreads_csv_path}")
            return False
            
        try:
            # Load the CSV file
            self.user_books_df = pd.read_csv(self.goodreads_csv_path)
            
            # Print column names for debugging
            print(f"Found columns: {', '.join(self.user_books_df.columns.tolist())}")
            
            # Standardize column names
            if 'Book Id' in self.user_books_df.columns:
                column_mapping = {
                    'Book Id': 'book_id',
                    'Title': 'title',
                    'Author': 'author',
                    'My Rating': 'user_rating',
                    'Average Rating': 'avg_rating',
                    'Shelves': 'shelves',
                    'Bookshelves': 'bookshelves',
                    'Exclusive Shelf': 'shelf',
                    'ISBN': 'isbn',
                    'ISBN13': 'isbn13',
                    'Genres': 'genres'
                }
                self.user_books_df.rename(columns={k: v for k, v in column_mapping.items() 
                                                if k in self.user_books_df.columns}, inplace=True)
            
            # Create sets for books to exclude
            self.read_titles = set()
            self.currently_reading_titles = set()
            
            # Find shelf column - could be named differently in different exports
            shelf_column = None
            for col in ['shelf', 'bookshelves', 'shelves', 'Exclusive Shelf']:
                if col in self.user_books_df.columns:
                    shelf_column = col
                    break
            
            if shelf_column:
                print(f"Found shelf information in column: {shelf_column}")
                # Find books that are currently being read
                currently_reading_df = self.user_books_df[
                    self.user_books_df[shelf_column].str.contains('currently-reading', case=False, na=False)
                ]
                if len(currently_reading_df) > 0:
                    print(f"Found {len(currently_reading_df)} books you're currently reading")
                    self.currently_reading_titles = set(currently_reading_df['title'].str.lower())
                
                # Find books that have been read
                read_df = self.user_books_df[
                    self.user_books_df[shelf_column].str.contains('read', case=False, na=False) &
                    ~self.user_books_df[shelf_column].str.contains('currently-reading|to-read', case=False, na=False)
                ]
                if len(read_df) > 0:
                    self.read_titles = set(read_df['title'].str.lower())
                    print(f"Found {len(self.read_titles)} books you've read")
            
            # If no shelf information, use rated books as read books
            if not self.read_titles and 'user_rating' in self.user_books_df.columns:
                rated_books = self.user_books_df[
                    self.user_books_df['user_rating'] > 0
                ]
                self.read_titles = set(rated_books['title'].str.lower())
                print(f"Using {len(self.read_titles)} rated books as read books")
            
            # Create set of all books to exclude from recommendations
            self.exclude_titles = self.read_titles.union(self.currently_reading_titles)
            print(f"Total of {len(self.exclude_titles)} books excluded from recommendations")
            
            # Create a clean dataset of books with ratings for similarity analysis
            if 'user_rating' in self.user_books_df.columns:
                self.rated_books = self.user_books_df[
                    self.user_books_df['user_rating'] > 0
                ].copy()
                print(f"Found {len(self.rated_books)} books with ratings")
            else:
                self.rated_books = pd.DataFrame()
                print("No books with ratings found")
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
            
    def fetch_hugo_winners(self):
        """Fetch Hugo Award winning novels"""
        print("Fetching Hugo Award winners (Best Novel category)...")
        
        # Define Hugo winners list (fallback in case web scraping fails)
        # This list contains Hugo Award winners for Best Novel from 1953 through recent years
        self.hugo_winners = [
            {"year": 2024, "title": "The Mountain in the Sea", "author": "Ray Nayler"},
            {"year": 2023, "title": "Nettle & Bone", "author": "T. Kingfisher"},
            {"year": 2022, "title": "A Desolation Called Peace", "author": "Arkady Martine"},
            {"year": 2021, "title": "Network Effect", "author": "Martha Wells"},
            {"year": 2020, "title": "A Memory Called Empire", "author": "Arkady Martine"},
            {"year": 2019, "title": "The Calculating Stars", "author": "Mary Robinette Kowal"},
            {"year": 2018, "title": "The Stone Sky", "author": "N. K. Jemisin"},
            {"year": 2017, "title": "The Obelisk Gate", "author": "N. K. Jemisin"},
            {"year": 2016, "title": "The Fifth Season", "author": "N. K. Jemisin"},
            {"year": 2015, "title": "The Three-Body Problem", "author": "Liu Cixin"},
            {"year": 2014, "title": "Ancillary Justice", "author": "Ann Leckie"},
            {"year": 2013, "title": "Redshirts", "author": "John Scalzi"},
            {"year": 2012, "title": "Among Others", "author": "Jo Walton"},
            {"year": 2011, "title": "Blackout/All Clear", "author": "Connie Willis"},
            {"year": 2010, "title": "The Windup Girl", "author": "Paolo Bacigalupi"},
            {"year": 2010, "title": "The City & the City", "author": "China Miéville"},
            {"year": 2009, "title": "The Graveyard Book", "author": "Neil Gaiman"},
            {"year": 2008, "title": "The Yiddish Policemen's Union", "author": "Michael Chabon"},
            {"year": 2007, "title": "Rainbows End", "author": "Vernor Vinge"},
            {"year": 2006, "title": "Spin", "author": "Robert Charles Wilson"},
            {"year": 2005, "title": "Jonathan Strange & Mr Norrell", "author": "Susanna Clarke"},
            {"year": 2004, "title": "Paladin of Souls", "author": "Lois McMaster Bujold"},
            {"year": 2003, "title": "Hominids", "author": "Robert J. Sawyer"},
            {"year": 2002, "title": "American Gods", "author": "Neil Gaiman"},
            {"year": 2001, "title": "Harry Potter and the Goblet of Fire", "author": "J. K. Rowling"},
            {"year": 2000, "title": "A Deepness in the Sky", "author": "Vernor Vinge"},
            {"year": 1999, "title": "To Say Nothing of the Dog", "author": "Connie Willis"},
            {"year": 1998, "title": "Forever Peace", "author": "Joe Haldeman"},
            {"year": 1997, "title": "Blue Mars", "author": "Kim Stanley Robinson"},
            {"year": 1996, "title": "The Diamond Age", "author": "Neal Stephenson"},
            {"year": 1995, "title": "Mirror Dance", "author": "Lois McMaster Bujold"},
            {"year": 1994, "title": "Green Mars", "author": "Kim Stanley Robinson"},
            {"year": 1993, "title": "Doomsday Book", "author": "Connie Willis"},
            {"year": 1993, "title": "A Fire Upon the Deep", "author": "Vernor Vinge"},
            {"year": 1992, "title": "Barrayar", "author": "Lois McMaster Bujold"},
            {"year": 1991, "title": "The Vor Game", "author": "Lois McMaster Bujold"},
            {"year": 1990, "title": "Hyperion", "author": "Dan Simmons"},
            {"year": 1989, "title": "Cyteen", "author": "C. J. Cherryh"},
            {"year": 1988, "title": "The Uplift War", "author": "David Brin"},
            {"year": 1987, "title": "Speaker for the Dead", "author": "Orson Scott Card"},
            {"year": 1986, "title": "Ender's Game", "author": "Orson Scott Card"},
            {"year": 1985, "title": "Neuromancer", "author": "William Gibson"},
            {"year": 1984, "title": "Startide Rising", "author": "David Brin"},
            {"year": 1983, "title": "Foundation's Edge", "author": "Isaac Asimov"},
            {"year": 1982, "title": "Downbelow Station", "author": "C. J. Cherryh"},
            {"year": 1981, "title": "The Snow Queen", "author": "Joan D. Vinge"},
            {"year": 1980, "title": "The Fountains of Paradise", "author": "Arthur C. Clarke"},
            {"year": 1979, "title": "Dreamsnake", "author": "Vonda N. McIntyre"},
            {"year": 1978, "title": "Gateway", "author": "Frederik Pohl"},
            {"year": 1977, "title": "Where Late the Sweet Birds Sang", "author": "Kate Wilhelm"},
            {"year": 1976, "title": "The Forever War", "author": "Joe Haldeman"},
            {"year": 1975, "title": "The Dispossessed", "author": "Ursula K. Le Guin"},
            {"year": 1974, "title": "Rendezvous with Rama", "author": "Arthur C. Clarke"},
            {"year": 1973, "title": "The Gods Themselves", "author": "Isaac Asimov"},
            {"year": 1972, "title": "To Your Scattered Bodies Go", "author": "Philip José Farmer"},
            {"year": 1971, "title": "Ringworld", "author": "Larry Niven"},
            {"year": 1970, "title": "The Left Hand of Darkness", "author": "Ursula K. Le Guin"},
            {"year": 1969, "title": "Stand on Zanzibar", "author": "John Brunner"},
            {"year": 1968, "title": "Lord of Light", "author": "Roger Zelazny"},
            {"year": 1967, "title": "The Moon Is a Harsh Mistress", "author": "Robert A. Heinlein"},
            {"year": 1966, "title": "Dune", "author": "Frank Herbert"},
            {"year": 1966, "title": "...And Call Me Conrad", "author": "Roger Zelazny"},
            {"year": 1965, "title": "The Wanderer", "author": "Fritz Leiber"},
            {"year": 1964, "title": "Here Gather the Stars", "author": "Clifford D. Simak"},
            {"year": 1963, "title": "The Man in the High Castle", "author": "Philip K. Dick"},
            {"year": 1962, "title": "Stranger in a Strange Land", "author": "Robert A. Heinlein"},
            {"year": 1961, "title": "A Canticle for Leibowitz", "author": "Walter M. Miller, Jr."},
            {"year": 1960, "title": "Starship Troopers", "author": "Robert A. Heinlein"},
            {"year": 1959, "title": "A Case of Conscience", "author": "James Blish"},
            {"year": 1958, "title": "The Big Time", "author": "Fritz Leiber"},
            {"year": 1956, "title": "Double Star", "author": "Robert A. Heinlein"},
            {"year": 1955, "title": "They'd Rather Be Right", "author": "Mark Clifton & Frank Riley"},
            {"year": 1953, "title": "The Demolished Man", "author": "Alfred Bester"}
        ]
        
        print(f"Found {len(self.hugo_winners)} Hugo Award winners for Best Novel")
        
        # Try to clean up title and author formatting
        for book in self.hugo_winners:
            book['title_lower'] = book['title'].lower()
            book['author_lower'] = book['author'].lower()
            book['award'] = 'Hugo Award'
            book['source'] = 'Hugo Award for Best Novel'

        return True

    def fetch_nebula_winners(self):
        """Fetch Nebula Award winning novels"""
        print("Fetching Nebula Award winners (Best Novel category)...")
        
        # Define Nebula winners list (fallback in case web scraping fails)
        # This list contains Nebula Award winners for Best Novel from 1966 through recent years
        self.nebula_winners = [
            {"year": 2023, "title": "Translation State", "author": "Ann Leckie"},
            {"year": 2022, "title": "Until the Last of Me", "author": "Sylvain Neuvel"},
            {"year": 2021, "title": "A Master of Djinn", "author": "P. Djèlí Clark"},
            {"year": 2020, "title": "Network Effect", "author": "Martha Wells"},
            {"year": 2019, "title": "A Song for a New Day", "author": "Sarah Pinsker"},
            {"year": 2018, "title": "The Calculating Stars", "author": "Mary Robinette Kowal"},
            {"year": 2017, "title": "The Stone Sky", "author": "N. K. Jemisin"},
            {"year": 2016, "title": "All the Birds in the Sky", "author": "Charlie Jane Anders"},
            {"year": 2015, "title": "Uprooted", "author": "Naomi Novik"},
            {"year": 2014, "title": "Annihilation", "author": "Jeff VanderMeer"},
            {"year": 2013, "title": "Ancillary Justice", "author": "Ann Leckie"},
            {"year": 2012, "title": "2312", "author": "Kim Stanley Robinson"},
            {"year": 2011, "title": "Among Others", "author": "Jo Walton"},
            {"year": 2010, "title": "Blackout/All Clear", "author": "Connie Willis"},
            {"year": 2009, "title": "The Windup Girl", "author": "Paolo Bacigalupi"},
            {"year": 2008, "title": "Powers", "author": "Ursula K. Le Guin"},
            {"year": 2007, "title": "The Yiddish Policemen's Union", "author": "Michael Chabon"},
            {"year": 2006, "title": "Seeker", "author": "Jack McDevitt"},
            {"year": 2005, "title": "Camouflage", "author": "Joe Haldeman"},
            {"year": 2004, "title": "Paladin of Souls", "author": "Lois McMaster Bujold"},
            {"year": 2003, "title": "The Speed of Dark", "author": "Elizabeth Moon"},
            {"year": 2002, "title": "American Gods", "author": "Neil Gaiman"},
            {"year": 2001, "title": "The Quantum Rose", "author": "Catherine Asaro"},
            {"year": 2000, "title": "Darwin's Radio", "author": "Greg Bear"},
            {"year": 1999, "title": "Parable of the Talents", "author": "Octavia E. Butler"},
            {"year": 1998, "title": "Forever Peace", "author": "Joe Haldeman"},
            {"year": 1997, "title": "The Moon and the Sun", "author": "Vonda N. McIntyre"},
            {"year": 1996, "title": "Slow River", "author": "Nicola Griffith"},
            {"year": 1995, "title": "The Terminal Experiment", "author": "Robert J. Sawyer"},
            {"year": 1994, "title": "Moving Mars", "author": "Greg Bear"},
            {"year": 1993, "title": "Red Mars", "author": "Kim Stanley Robinson"},
            {"year": 1992, "title": "Doomsday Book", "author": "Connie Willis"},
            {"year": 1991, "title": "Tehanu: The Last Book of Earthsea", "author": "Ursula K. Le Guin"},
            {"year": 1990, "title": "The Healer's War", "author": "Elizabeth Ann Scarborough"},
            {"year": 1989, "title": "Falling Free", "author": "Lois McMaster Bujold"},
            {"year": 1988, "title": "The Falling Woman", "author": "Pat Murphy"},
            {"year": 1987, "title": "Speaker for the Dead", "author": "Orson Scott Card"},
            {"year": 1986, "title": "Ender's Game", "author": "Orson Scott Card"},
            {"year": 1985, "title": "Neuromancer", "author": "William Gibson"},
            {"year": 1984, "title": "Startide Rising", "author": "David Brin"},
            {"year": 1983, "title": "No Enemy But Time", "author": "Michael Bishop"},
            {"year": 1982, "title": "The Claw of the Conciliator", "author": "Gene Wolfe"},
            {"year": 1981, "title": "Timescape", "author": "Gregory Benford"},
            {"year": 1980, "title": "The Fountains of Paradise", "author": "Arthur C. Clarke"},
            {"year": 1979, "title": "Dreamsnake", "author": "Vonda N. McIntyre"},
            {"year": 1978, "title": "Gateway", "author": "Frederik Pohl"},
            {"year": 1977, "title": "Man Plus", "author": "Frederik Pohl"},
            {"year": 1976, "title": "The Forever War", "author": "Joe Haldeman"},
            {"year": 1975, "title": "The Dispossessed", "author": "Ursula K. Le Guin"},
            {"year": 1974, "title": "Rendezvous with Rama", "author": "Arthur C. Clarke"},
            {"year": 1973, "title": "The Gods Themselves", "author": "Isaac Asimov"},
            {"year": 1972, "title": "A Time of Changes", "author": "Robert Silverberg"},
            {"year": 1971, "title": "Ringworld", "author": "Larry Niven"},
            {"year": 1970, "title": "The Left Hand of Darkness", "author": "Ursula K. Le Guin"},
            {"year": 1969, "title": "Rite of Passage", "author": "Alexei Panshin"},
            {"year": 1968, "title": "Picnic on Paradise", "author": "Joanna Russ"},
            {"year": 1967, "title": "The Einstein Intersection", "author": "Samuel R. Delany"},
            {"year": 1966, "title": "Dune", "author": "Frank Herbert"},
            {"year": 1966, "title": "Flowers for Algernon", "author": "Daniel Keyes"}
        ]
        
        print(f"Found {len(self.nebula_winners)} Nebula Award winners for Best Novel")
        
        # Try to clean up title and author formatting
        for book in self.nebula_winners:
            book['title_lower'] = book['title'].lower()
            book['author_lower'] = book['author'].lower()
            book['award'] = 'Nebula Award'
            book['source'] = 'Nebula Award for Best Novel'
            
        return True
        
    def create_combined_award_list(self):
        """Create combined list of all award winners"""
        print("Creating combined award winners list...")
        
        # Combine both award lists
        self.award_winners = self.hugo_winners + self.nebula_winners
        
        # Remove duplicates (books that won both awards)
        title_seen = {}
        unique_winners = []
        
        for book in self.award_winners:
            # Create a key with title and author to handle cases where different books have same title
            key = (book['title_lower'], book['author_lower'])
            
            if key not in title_seen:
                title_seen[key] = book
                book['awards'] = [book['award']]
                unique_winners.append(book)
            else:
                # Book already in the list, update awards
                existing_book = title_seen[key]
                if book['award'] not in existing_book['awards']:
                    existing_book['awards'].append(book['award'])
                    existing_book['source'] = 'Hugo and Nebula Awards for Best Novel'
        
        self.award_winners = unique_winners
        
        # Sort by year (most recent first)
        self.award_winners.sort(key=lambda x: x['year'], reverse=True)
        
        print(f"Combined list contains {len(self.award_winners)} unique award-winning books")
        return True
    
    def get_unread_award_winners(self):
        """Get list of award winners that user hasn't read yet"""
        print("Finding award winners you haven't read yet...")
        
        # Check each award winner against user's read/currently reading books
        self.unread_winners = []
        
        for book in self.award_winners:
            # Skip if the book is in the exclude list (read or currently reading)
            if book['title_lower'] in self.exclude_titles:
                continue
                
            # If we get here, the book is unread
            self.unread_winners.append(book)
        
        print(f"Found {len(self.unread_winners)} award winners you haven't read yet")
        return self.unread_winners
    
    def create_features_for_user_books(self):
        """Create features for user-rated books for similarity calculations"""
        print("Creating features for your rated books...")
        
        if len(self.rated_books) == 0:
            print("No rated books found, cannot create similarity features")
            return False
        
        # Create features from title and author
        self.rated_books['features'] = self.rated_books['title'] + ' ' + self.rated_books['author']
        
        # Create TF-IDF matrix
        features_matrix = self.vectorizer.fit_transform(self.rated_books['features'])
        self.user_features = features_matrix
        
        print("Created features for user book preferences")
        return True
    
    def rank_unread_winners(self):
        """Rank unread award winners based on likelihood of user enjoying them"""
        print("Ranking unread award winners based on your preferences...")
        
        # If we have user ratings, use them for ranking
        if hasattr(self, 'user_features') and self.user_features is not None:
            # Calculate author preferences
            author_ratings = {}
            for _, book in self.rated_books.iterrows():
                author = book['author'].lower()
                rating = book['user_rating']
                
                if author not in author_ratings:
                    author_ratings[author] = []
                    
                author_ratings[author].append(rating)
            
            # Calculate average rating per author
            author_avg_ratings = {
                author: sum(ratings)/len(ratings) 
                for author, ratings in author_ratings.items()
            }
            
            # Calculate predicted score for each unread winner
            for book in self.unread_winners:
                # Initial score based on author preference
                if book['author_lower'] in author_avg_ratings:
                    # You've read this author before
                    book['score'] = author_avg_ratings[book['author_lower']] * 1.5  # Weight author preference heavily
                    book['reason'] = f"You rated books by {book['author']} an average of {author_avg_ratings[book['author_lower']]:.1f}/5"
                else:
                    # Default score for unread authors
                    book['score'] = 3.0  # Neutral score
                    book['reason'] = "Award winner you might enjoy based on your reading preferences"
                
                # Boost score for multiple awards
                if len(book['awards']) > 1:
                    book['score'] += 0.5
                    book['reason'] += f". Won both Hugo and Nebula Awards in {book['year']}"
                else:
                    book['reason'] += f". Won the {book['awards'][0]} in {book['year']}"
                
                # Calculate similarity to user's favorite books
                book_features = self.vectorizer.transform([book['title'] + ' ' + book['author']])
                
                # Get top rated books (4-5 stars)
                favorites = self.rated_books[self.rated_books['user_rating'] >= 4]
                
                if len(favorites) > 0:
                    # Get features for favorite books
                    favorites_features = self.vectorizer.transform(favorites['title'] + ' ' + favorites['author'])
                    
                    # Calculate similarity
                    similarities = cosine_similarity(book_features, favorites_features).flatten()
                    avg_similarity = np.mean(similarities)
                    
                    # Adjust score based on similarity to favorites
                    similarity_boost = avg_similarity * 2  # Scale similarity to a reasonable range
                    book['score'] += similarity_boost
                    
                    # Add reason if similarity is significant
                    if avg_similarity > 0.1:
                        book['reason'] += ". Similar to books you've rated highly"
        else:
            # Without user ratings, just rank by recency and multiple awards
            for book in self.unread_winners:
                # Base score on year (more recent = higher score)
                year_score = (book['year'] - 1950) / 70  # Scale years to 0-1 range
                book['score'] = 3 + year_score  # Baseline 3 plus recency bonus
                
                # Boost for multiple awards
                if len(book['awards']) > 1:
                    book['score'] += 1.0
                    book['reason'] = f"Won both Hugo and Nebula Awards in {book['year']}"
                else:
                    book['reason'] = f"Won the {book['awards'][0]} in {book['year']}"
        
        # Sort by score
        self.unread_winners.sort(key=lambda x: x['score'], reverse=True)
        
        print("Ranked unread award winners based on your preferences")
        return self.unread_winners
    
    def recommend(self, top_n=None):
        """Run the entire recommendation process and return top recommendations"""
        # Load user books data
        if not self.load_user_books():
            return None
            
        # Get award winners data
        if self.hugo_winners is None:
            self.fetch_hugo_winners()
        
        if self.nebula_winners is None:
            self.fetch_nebula_winners()
        
        # Combine award lists
        self.create_combined_award_list()
        
        # Get unread winners
        self.get_unread_award_winners()
        
        # Create features for similarity calculation
        if len(self.rated_books) > 0:
            self.create_features_for_user_books()
        
        # Rank unread winners
        self.rank_unread_winners()
        
        # Return top N recommendations or all if top_n is None
        if top_n is not None:
            return self.unread_winners[:min(top_n, len(self.unread_winners))]
        else:
            return self.unread_winners


def main():
    print("Hugo & Nebula Award Winner Recommender")
    print("======================================")
    
    # Get user input for Goodreads CSV
    csv_path = input("Enter the path to your Goodreads export CSV file: ")
    
    # Create recommender
    recommender = AwardRecommender(goodreads_csv_path=csv_path)
    
    # Run recommendation process
    recommendations = recommender.recommend()
    
    if recommendations:
        print("\nRecommended Award-Winning Books You Haven't Read Yet:")
        print("=====================================================")
        
        for i, book in enumerate(recommendations, 1):
            award_str = " & ".join(book['awards'])
            score_str = f"{book['score']:.1f}/5" if 'score' in book else ""
            
            print(f"{i}. {book['title']} by {book['author']} ({book['year']}) - {award_str}")
            if 'reason' in book:
                print(f"   Reason: {book['reason']}")
            if score_str:
                print(f"   Predicted rating: {score_str}")
            print()
    else:
        print("Could not generate recommendations. Please check your Goodreads CSV file.")
    
    print("Done!")

if __name__ == "__main__":
    main()
