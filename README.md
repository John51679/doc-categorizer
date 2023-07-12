# doc-categorizer
This project was created as part of "Language Technology" subject in Computer Engineering & Informatics Department (CEID) of University of Patras. The essence of it was to classify a document, or a set of documents in their corresponding category directory.

The uncategorized documents directory is called `UNCATEGORIZED`, and after the code runs, it becomes empty, with each of its documents being moved into their predicted category.

The code `categorizer.py` takes 20 documents from each category directory, and uses them for the classification process. It then constructs a vocabulary from each unique word found and calculates the tf-idf score of each document both in the categorized and the uncategorized collection. That way we tranform our data into vector space where we can perform the cosine similarity method to find the most relevant category that each uncategorized document belongs to (calculate the average cosine similarity between the uncatecorized document and all documents of a given category)
