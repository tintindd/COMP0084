import numpy as np
import matplotlib.pyplot as plt
import collections
import re
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')

# Loading text information
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
        return text

# Data preprocessing

# def data_pre_processing(text):
    # return re.findall(r'\b\w+\b', text)

def data_pre_processing(text):
    tokens = re.findall(r'\b\w+\b', text)

    porter =  PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]

    return tokens


# Calculate word counts
def calculate_word_counts(words_1_gram):
    return collections.Counter(words_1_gram)

# Calculate normalized frequency and word rank
def compute_words_information(word_counts):
    counts_sorted = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    ranks = np.arange(1, len(counts_sorted) + 1)
    frequencies = np.array([count for _, count in counts_sorted])
    normalized_freqs = frequencies / sum(frequencies)
    return counts_sorted, ranks, frequencies, normalized_freqs

# Calculate Zipf-value
def calculate_zipf(counts_sorted, ranks):
    N = len(counts_sorted)
    s = 1
    zipf_value = (1 / (ranks ** s)) / sum(1 / (np.arange(1, N + 1) ** s))
    return zipf_value, N

# Qualitatively justify
def qualitatively_value(ranks, normalized_freqs):
    mean_rank_frquency = np.mean(ranks*normalized_freqs)
    variance_rank_frequency = np.var(ranks*normalized_freqs)

    # print(f"Mean of rank*frequency is:{mean_rank_frquency}\n")
    # print(f"Variance of rank*frequency is:{variance_rank_frequency}\n")
    return mean_rank_frquency, variance_rank_frequency

# Remove stop words and calculate relative informatiom
def calculate_word_counts_remove(words_1_gram):
    stop_words = set(stopwords.words('english'))
    words_remove = [word for word in words_1_gram if word not in stop_words]
    counts_remove = collections.Counter(words_remove)
    return words_remove, counts_remove

def compute_words_remove_information(counts_remove):
    sorted_counts_remove = sorted(counts_remove.items(), key=lambda x: x[1], reverse=True)
    ranks_remove = np.arange(1, len(sorted_counts_remove) + 1)
    frequencies_remove = np.array([count for _, count in sorted_counts_remove])
    normalized_freqs_remove = frequencies_remove / sum(frequencies_remove)
    return sorted_counts_remove, ranks_remove, frequencies_remove, normalized_freqs_remove

def find_word_position(word, counts_sorted):
    for index, (w, freq) in enumerate(counts_sorted):
        if w == word:
            return index + 1
    return None

def task1_function():

    #Loading....
    file_path = "passage-collection.txt"
    text = load_text(file_path)

    # Get 1-gram words and calculate word counts
    words_1_gram = data_pre_processing(text)
    words_counts = calculate_word_counts(words_1_gram)

    # Get basic information about words
    counts_sorted, ranks, frequencies, normalized_freqs = compute_words_information(words_counts)

    # Calculate Zipf-value
    zipf_value, N = calculate_zipf(counts_sorted, ranks)

    # Calculate mean and variance for rank*frequency
    mean_rank_frquency, variance_rank_frequency = qualitatively_value(ranks, normalized_freqs)

    # Create figures
    plt.figure(figsize=(6, 5)) # Figure 1
    plt.plot(ranks, normalized_freqs, label="Normalized Frequency", marker="o")
    plt.plot(ranks, zipf_value, label="Zipf's Law", linestyle="dashed")
    plt.xlabel("Rank")
    plt.ylabel("Normalized Frequency")
    # plt.title("Word Frequency vs. Rank")
    plt.legend()
    plt.savefig("figure1.pdf")

    plt.figure(figsize=(6, 5)) # Figure 2
    plt.loglog(ranks, normalized_freqs, label="Normalized Frequency", marker="o")
    plt.loglog(ranks, zipf_value, label="Zipf's Law", linestyle="dashed")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Normalized Frequency)")
    # plt.title("Zipf's Law - Log-Log Plot")
    plt.legend()
    plt.savefig("figure2.pdf")
    # plt.show()

    # Remove stop words and calculate necessary data again
    words_remove, counts_remove = calculate_word_counts_remove(words_1_gram)
    sorted_counts_remove, ranks_remove, frequencies_remove, normalized_freqs_remove = compute_words_remove_information(counts_remove)


    # Comparison of Zipf distributions after removal of stop words
    plt.figure(figsize=(6, 5)) # Figure 3
    plt.loglog(ranks_remove, normalized_freqs_remove, label="Normalized Frequency", marker="o")
    plt.loglog(ranks_remove, zipf_value[:len(ranks_remove)], label="Zipf's Law", linestyle="dashed")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Normalized Frequency)")
    # plt.title("Effect of Stopword Removal")
    plt.legend()
    plt.savefig("figure3.pdf")
    # plt.show()

    # Save vocabulary to file (includes stop words)
    with open("task1_vocabulary.txt", "w", encoding="utf-8") as vocab_file:
        for word, _ in counts_sorted:
            vocab_file.write(word + "\n")

    # Find the top-ranked word after removing stop words
    # top_word_after_removal = sorted_counts_remove[0][0]
    # Find its position in the original sorted list
    # position = find_word_position(top_word_after_removal, counts_sorted)

    # Print results
    print(f"Mean of rank*frequency is:{mean_rank_frquency}")
    print(f"Variance of rank*frequency is:{variance_rank_frequency}")
    print(f"Number of occurrences of terms: {N}") # 102210
    print(f"First ranked terms:<{counts_sorted[0][0]}>,{normalized_freqs[0]}")
    print(f"Zipf frequency (k = 1):{zipf_value[0]}\n")

    print(f"Number of occurrences of terms when remove stop words:{len(sorted_counts_remove)}") # 102083
    print(f"First ranked terms after remove stop words:<{sorted_counts_remove[0][0]}>,{normalized_freqs_remove[0]}")
    print(f"The original frequency of the first word (after removing stop words):<{counts_sorted[23][0]}>,{normalized_freqs[23]} ")


if __name__ == "__main__":
    task1_function()