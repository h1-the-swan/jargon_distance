import os
fnames = ['./shakespeare/1786.txt.utf-8', './shakespeare/1793.txt.utf-8',
        './sawyer/74-0.txt', './sawyer/76-0.txt',
        './dickens/pg19337.txt', './dickens/pg730.txt']
fnames = [os.path.abspath(os.path.join(os.path.dirname(__file__), fname)) for fname in fnames]

groups = ['William Shakespeare - As You Like It', 'William Shakespeare - Othello', 
        'Mark Twain - The Adventures of Tom Sawyer', 'Mark Twain - The Adventures of Huckleberry Finn',
        'Charles Dickens - A Christmas Carol', 'Charles Dickens - Oliver Twist']

