import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from collections import defaultdict

class GoodreadsRecommender:
    def __init__(self, user_id=None, csv_path=None):
        """
        Initialize the recommender with either a Goodreads user ID or a path to exported CSV data.
        
        Args:
            user_id (str, optional): Your Goodreads user ID
            csv_path (str, optional): Path to exported Goodreads CSV file
        """
        self.user_id = user_id
        self.csv_path = csv_path
        self.books_df = None
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def load_data(self):
        """Load book data either from Goodreads API/scraping or from CSV file"""
        if self.csv_path and os.path.exists(self.csv_path):
            print(f"Loading data from CSV file: {self.csv_path}")
            try:
                self.books_df = pd.read_csv(self.csv_path)
                
                # Print column names to help debug
                print(f"Found columns: {', '.join(self.books_df.columns.tolist())}")
                
                # Rename columns to standardized format if needed
                if 'Book Id' in self.books_df.columns:
                    column_mapping = {
                        'Book Id': 'book_id',
                        'Title': 'title',
                        'Author': 'author',
                        'My Rating': 'user_rating',
                        'Average Rating': 'avg_rating',
                        'Shelves': 'shelves',
                        'Bookshelves': 'bookshelves',
                        'ISBN': 'isbn',
                        'ISBN13': 'isbn13',
                        'Genres': 'genres'
                    }
                    self.books_df.rename(columns={k: v for k, v in column_mapping.items() 
                                                if k in self.books_df.columns}, inplace=True)
                
                # Handle alternate column names that might be in the export
                if 'user_rating' not in self.books_df.columns:
                    # Try to find rating column with different names
                    rating_columns = [col for col in self.books_df.columns if 'rating' in col.lower()]
                    if rating_columns:
                        print(f"Using '{rating_columns[0]}' as user rating column")
                        self.books_df.rename(columns={rating_columns[0]: 'user_rating'}, inplace=True)
                
                # Convert ratings to numeric
                if 'user_rating' in self.books_df.columns:
                    self.books_df['user_rating'] = pd.to_numeric(self.books_df['user_rating'], errors='coerce')
                else:
                    print("Warning: No user rating column found. Recommendations may be limited.")
                    # Create a default rating column
                    self.books_df['user_rating'] = 3.0  # Default neutral rating
                
                # Make sure title and author columns exist
                if 'title' not in self.books_df.columns and 'Title' in self.books_df.columns:
                    self.books_df.rename(columns={'Title': 'title'}, inplace=True)
                if 'author' not in self.books_df.columns and 'Author' in self.books_df.columns:
                    self.books_df.rename(columns={'Author': 'author'}, inplace=True)
                
                # If we still don't have title or author, try to identify them
                if 'title' not in self.books_df.columns:
                    title_candidates = [col for col in self.books_df.columns if 'title' in col.lower() or 'name' in col.lower()]
                    if title_candidates:
                        print(f"Using '{title_candidates[0]}' as title column")
                        self.books_df.rename(columns={title_candidates[0]: 'title'}, inplace=True)
                    else:
                        print("Error: Could not identify title column in CSV")
                        return False
                
                if 'author' not in self.books_df.columns:
                    author_candidates = [col for col in self.books_df.columns if 'author' in col.lower() or 'writer' in col.lower()]
                    if author_candidates:
                        print(f"Using '{author_candidates[0]}' as author column")
                        self.books_df.rename(columns={author_candidates[0]: 'author'}, inplace=True)
                    else:
                        print("Error: Could not identify author column in CSV")
                        return False
                
                return True
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return False
        
        elif self.user_id:
            print(f"Attempting to fetch data for Goodreads user ID: {self.user_id}")
            # Note: Goodreads discontinued their public API, so we'll need to scrape
            # This is a placeholder - actual implementation would require web scraping
            try:
                # This would be replaced with actual scraping code
                print("Goodreads API is deprecated. Please export your books as CSV from Goodreads")
                print("Instructions: Go to Goodreads -> My Books -> Import/Export -> Export Library")
                return False
            except Exception as e:
                print(f"Error fetching data: {e}")
                return False
        else:
            print("Please provide either a Goodreads user ID or path to exported CSV file")
            return False
    
    def clean_and_prepare_data(self):
        """Clean and prepare data for analysis"""
        if self.books_df is None:
            print("No data loaded. Please call load_data() first.")
            return False
        
        # Create sets for books to exclude from recommendations
        self.read_titles = set()
        self.currently_reading_titles = set()
        
        # Check if there's a bookshelves or shelf column to identify currently reading books
        shelf_column = None
        for col in ['bookshelves', 'shelves', 'shelf', 'Bookshelves', 'Exclusive Shelf']:
            if col in self.books_df.columns:
                shelf_column = col
                break
        
        if shelf_column:
            print(f"Found shelf information in column: {shelf_column}")
            # Find books that are currently being read
            currently_reading_df = self.books_df[
                self.books_df[shelf_column].str.contains('currently-reading', case=False, na=False)
            ]
            if len(currently_reading_df) > 0:
                print(f"Found {len(currently_reading_df)} books you're currently reading")
                self.currently_reading_titles = set(currently_reading_df['title'].str.lower())
            
            # Identify books that have been read (using shelf info if available)
            read_df = self.books_df[
                self.books_df[shelf_column].str.contains('read', case=False, na=False) &
                ~self.books_df[shelf_column].str.contains('currently-reading|to-read', case=False, na=False)
            ]
            if len(read_df) > 0:
                self.read_titles = set(read_df['title'].str.lower())
        
        # Drop books with no ratings (if using for recommendation)
        rated_books = self.books_df.dropna(subset=['user_rating'])
        # Only keep books with ratings > 0 (assumes 0 means unrated)
        self.rated_books = rated_books[rated_books['user_rating'] > 0].copy()
        
        # If we found no read books from shelves, consider all rated books as read
        if not self.read_titles:
            print("Using rated books as read books")
            self.read_titles = set(self.rated_books['title'].str.lower())
        
        # Books to exclude from recommendations are both read books and currently reading books
        self.exclude_titles = self.read_titles.union(self.currently_reading_titles)
        print(f"Total books excluded from recommendations: {len(self.exclude_titles)}")
        
        # Extract additional features if available
        if 'bookshelves' in self.books_df.columns:
            # Create feature text combining title, author, and shelves for content-based filtering
            self.rated_books['features'] = (
                self.rated_books['title'].fillna('') + ' ' +
                self.rated_books['author'].fillna('') + ' ' +
                self.rated_books['bookshelves'].fillna('')
            )
        else:
            # Create feature text combining just title and author
            self.rated_books['features'] = (
                self.rated_books['title'].fillna('') + ' ' +
                self.rated_books['author'].fillna('')
            )
        
        print(f"Prepared {len(self.rated_books)} rated books for analysis")
        return True

    def create_content_features(self):
        """Create content-based features using TF-IDF"""
        if not hasattr(self, 'rated_books') or self.rated_books is None:
            print("No cleaned data. Please call clean_and_prepare_data() first.")
            return False
        
        # Reset index to make sure we have continuous indices
        self.rated_books = self.rated_books.reset_index(drop=True)
        
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(self.rated_books['features'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create a mapping of indices
        self.indices = pd.Series(self.rated_books.index, index=self.rated_books['title']).drop_duplicates()
        
        print("Created content-based features")
        return True
    
    def get_content_based_recommendations(self, title, top_n=5):
        """
        Get content-based recommendations based on book title
        
        Args:
            title (str): Title of the book to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended book titles
        """
        if not hasattr(self, 'cosine_sim'):
            print("Content features not created. Please call create_content_features() first.")
            return []
        
        # Find the index of the book
        if title not in self.indices:
            close_matches = self.rated_books[self.rated_books['title'].str.contains(title, case=False, na=False)]
            if len(close_matches) > 0:
                title = close_matches.iloc[0]['title']
                print(f"Using closest match: '{title}'")
            else:
                print(f"Book '{title}' not found in your library")
                return []
        
        try:
            idx = self.indices[title]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get books not in your read or currently reading libraries
            recommendations = []
            count = 0
            
            # Start from the second most similar (first is the book itself)
            for i, score in sim_scores[1:]:
                # Check if we have enough recommendations
                if count >= top_n:
                    break
                
                book_title = self.rated_books.iloc[i]['title']
                book_author = self.rated_books.iloc[i]['author']
                
                # Skip already read books or currently reading books
                if book_title.lower() in self.exclude_titles:
                    continue
                
                # Otherwise, add to recommendations
                recommendations.append([
                    book_title,
                    book_author, 
                    score  # Use the similarity score
                ])
                count += 1
                
                # If we can't find enough unread similar books, break
                if i >= len(sim_scores) - 2:
                    break
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations for '{title}': {e}")
            return []
    
    def get_collaborative_filtering_recommendations(self, top_n=10):
        """
        A simple collaborative filtering approach based on your ratings
        
        Args:
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended book titles with predicted scores
        """
        # In a real implementation, this would use data from other users
        # Since we don't have that, we'll create a simple approach based on your highest rated authors/genres
        
        if not hasattr(self, 'rated_books') or self.rated_books is None:
            print("No cleaned data. Please call clean_and_prepare_data() first.")
            return []
        
        # Find your favorite authors (those you've rated highly)
        author_ratings = defaultdict(list)
        for _, row in self.rated_books.iterrows():
            author_ratings[row['author']].append(row['user_rating'])
        
        author_avg_ratings = {author: sum(ratings)/len(ratings) 
                             for author, ratings in author_ratings.items() 
                             if len(ratings) >= 1}  # At least one book read
        
        # Find books by your favorite authors that you haven't read yet
        recommendations = []
        for author, avg_rating in sorted(author_avg_ratings.items(), key=lambda x: x[1], reverse=True):
            # Find all books by this author
            author_books = self.books_df[
                (self.books_df['author'] == author)
            ]
            
            for _, book in author_books.iterrows():
                # Skip books you've already read or are currently reading
                if book['title'].lower() in self.exclude_titles:
                    continue
                    
                if 'avg_rating' in book and not pd.isna(book['avg_rating']):
                    book_avg_rating = float(book['avg_rating'])
                    predicted_score = (avg_rating + book_avg_rating) / 2
                else:
                    predicted_score = avg_rating
                
                recommendations.append({
                    'title': book['title'],
                    'author': author,
                    'predicted_score': predicted_score,
                    'reason': f"You rated books by {author} an average of {avg_rating:.1f}/5"
                })
                
                if len(recommendations) >= top_n:
                    break
            
            if len(recommendations) >= top_n:
                break
        
        # Sort recommendations by predicted score
        recommendations.sort(key=lambda x: x['predicted_score'], reverse=True)
        return recommendations[:top_n]
    
    def get_hybrid_recommendations(self, top_n=10):
        """
        Get hybrid recommendations using both content-based and collaborative filtering
        
        Args:
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended books
        """
        # Get your highest rated books (4-5 stars)
        high_rated_books = self.rated_books[self.rated_books['user_rating'] >= 4]
        
        if len(high_rated_books) == 0:
            print("No highly rated books found, using all rated books instead")
            high_rated_books = self.rated_books
        
        # Limit to a sample of books to avoid processing too many
        if len(high_rated_books) > 5:
            print(f"Using a sample of 5 highly rated books from your {len(high_rated_books)} highly rated books")
            high_rated_books = high_rated_books.sample(5)
        
        # Get content-based recommendations for each highly rated book
        all_cb_recommendations = []
        for _, book in high_rated_books.iterrows():
            try:
                recs = self.get_content_based_recommendations(book['title'], top_n=3)
                if recs:  # Only process if we got recommendations
                    for rec in recs:
                        # Check if we've already read this book or currently reading it
                        if rec[0].lower() in self.exclude_titles:
                            continue
                            
                        all_cb_recommendations.append({
                            'title': rec[0],
                            'author': rec[1],
                            'score': float(book['user_rating']) * 0.2,  # Weight by your rating
                            'source': 'content',
                            'reason': f"Similar to '{book['title']}' which you rated {book['user_rating']}/5"
                        })
            except Exception as e:
                print(f"Error getting content recommendations for '{book['title']}': {e}")
                continue
        
        # Get collaborative filtering recommendations
        cf_recommendations = self.get_collaborative_filtering_recommendations(top_n=top_n)
        for rec in cf_recommendations:
            # Skip if we've already read this book or currently reading it
            if rec['title'].lower() in self.exclude_titles:
                continue
                
            rec['source'] = 'collaborative'
            rec['score'] = rec['predicted_score']
        
        # Combine and deduplicate recommendations
        all_recommendations = all_cb_recommendations + cf_recommendations
        
        # If we don't have any recommendations, return an empty list
        if not all_recommendations:
            print("Could not generate any recommendations. Check your data format.")
            return []
        
        # Create a dictionary to store the highest score for each title
        best_recommendations = {}
        for rec in all_recommendations:
            title = rec['title']
            if title not in best_recommendations or rec['score'] > best_recommendations[title]['score']:
                best_recommendations[title] = rec
        
        # Convert back to list and sort by score
        final_recommendations = list(best_recommendations.values())
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the top N recommendations (or all if we have fewer than top_n)
        return final_recommendations[:min(top_n, len(final_recommendations))]

    def recommend_books(self, method='hybrid', top_n=10):
        """
        Get book recommendations using the specified method
        
        Args:
            method (str): Recommendation method - 'content', 'collaborative', or 'hybrid'
            top_n (int): Number of recommendations to return
            
        Returns:
            list: List of recommended books
        """
        if method == 'content':
            # Just use a random highly-rated book for content-based recommendations
            high_rated = self.rated_books[self.rated_books['user_rating'] >= 4]
            if len(high_rated) > 0:
                sample_book = high_rated.sample(1).iloc[0]
                return self.get_content_based_recommendations(sample_book['title'], top_n)
            else:
                # Fall back to the first book if no highly rated books
                return self.get_content_based_recommendations(self.rated_books.iloc[0]['title'], top_n)
        
        elif method == 'collaborative':
            return self.get_collaborative_filtering_recommendations(top_n)
        
        else:  # hybrid
            return self.get_hybrid_recommendations(top_n)


def main():
    print("Goodreads Book Recommender")
    print("==========================")
    
    # Get user input
    csv_path = input("Enter the path to your Goodreads export CSV file: ")
    
    # Create recommender
    recommender = GoodreadsRecommender(csv_path=csv_path)
    
    # Load and prepare data
    if recommender.load_data():
        print("\nPreparing your book data...")
        if not recommender.clean_and_prepare_data():
            print("Error: Could not prepare data. Check your CSV file format.")
            return
        
        print("\nGenerating content features for recommendations...")
        if not recommender.create_content_features():
            print("Error: Could not create content features. Check your data.")
            return
        
        # Get recommendations
        print("\nGenerating book recommendations based on your ratings...")
        try:
            recommendations = recommender.recommend_books(method='hybrid', top_n=10)
            
            if not recommendations:
                print("Could not generate recommendations using hybrid method.")
                print("Attempting to generate recommendations using collaborative filtering only...")
                recommendations = recommender.get_collaborative_filtering_recommendations(top_n=10)
            
            # Display recommendations
            if recommendations:
                # Perform one final filter to ensure we're not recommending any read or currently reading books
                final_recommendations = [rec for rec in recommendations if rec['title'].lower() not in recommender.exclude_titles]
                
                if final_recommendations:
                    print("\nRecommended Books:")
                    print("=================")
                    
                    for i, rec in enumerate(final_recommendations, 1):
                        print(f"{i}. '{rec['title']}' by {rec['author']}")
                        if 'reason' in rec:
                            print(f"   Reason: {rec['reason']}")
                        if 'score' in rec:
                            print(f"   Predicted rating: {rec['score']:.1f}/5")
                        elif 'predicted_score' in rec:
                            print(f"   Predicted rating: {rec['predicted_score']:.1f}/5")
                        print()
                else:
                    print("\nCould not find any new books to recommend.")
                    print("Try adding more books with ratings to your Goodreads library.")
            else:
                print("\nCould not generate any recommendations.")
                print("Please check that your CSV file contains rated books.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Try using a different recommendation method or check your CSV format.")
    
    print("Done!")

if __name__ == "__main__":
    main()
