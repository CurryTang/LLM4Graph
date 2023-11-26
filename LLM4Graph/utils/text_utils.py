import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vector(texts, max_df = 1., min_df = 2, max_features = None):
    """
    Function to get the TF-IDF vector for a list of texts.

    This function returns the TF-IDF vector for a list of texts.

    Parameters:
    texts (list): The list of texts.
    max_df (float): The maximum document frequency.
    min_df (float): The minimum document frequency.

    Returns:
    np.ndarray: The TF-IDF vector.
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return torch.from_numpy(X.toarray()).float()