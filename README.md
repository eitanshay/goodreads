# goodreads
book recommendation based on books i've read (and ranked... on goodreads)

Identified "Currently Reading" Books: Added code to detect books marked as "currently-reading" in your Goodreads export by looking for shelf information.
Created an Exclusion Set: Created a combined set of books to exclude that includes both the books you've read and the books you're currently reading.
Updated Filtering Logic: Modified all recommendation methods to check against this combined exclusion set.
Improved Shelf Detection: The code now looks for various possible column names that might contain shelf information (like "bookshelves", "shelves", or "Exclusive Shelf") to ensure compatibility with different Goodreads export formats.
Added Better Debugging Information: The script now reports how many books it found in your "currently reading" shelf and the total number of books excluded from recommendations.

This version will ensure that your recommendations include only books that you haven't read or started reading yet. Goodreads typically marks books as:

"read" (for completed books)
"currently-reading" (for books in progress)
"to-read" (for books on your want-to-read list)

The script will exclude both "read" and "currently-reading" books from your recommendations, leaving you with fresh suggestions that you haven't started yet.
