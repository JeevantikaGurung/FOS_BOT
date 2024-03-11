import re
def Index(paragraph):
    #paragraph = "This is a big paragraph that contains several sentences. It is used as a sample to demonstrate inverted indexing. Inverted indexing is a technique used to optimize text searches. It is commonly used in search engines to quickly find documents that match a given query."
    # Tokenize the paragraph into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

    # Tokenize the sentences into words
    words = [sentence.lower().split() for sentence in sentences]

    # Create an empty dictionary to store the inverted index
    inverted_index = {}

    # Populate the inverted index with words and their positions in the paragraph
    for i, sentence in enumerate(words):
        for word in sentence:
            if word not in inverted_index:
                inverted_index[word] = [i]
            else:
                inverted_index[word].append(i)

    # Get the prompt word from the user
    prompt_word = input("Enter the word to search for: ").lower()

    # Check if the prompt word exists in the inverted index
    if prompt_word in inverted_index:
        # Print the sentences that contain the prompt word
        sentence_indexes = inverted_index[prompt_word]
        for sentence_index in sentence_indexes:
            print(sentences[sentence_index])
    else:   
        print(f"The word '{prompt_word}' does not appear in the paragraph.")


class Indexing:
    def _init_(self):
        self.para = [""""""]
    def Index(self,text):
        self.para = text
        Index(self.para)


index = Indexing()
index.Index("""A Rebellious descendant of the old aristocracy who is always out on the battlefield.
As one born into the old aristocracy, carrying the bloodline of sinners, Eula has needed a unique approach to the world to navigate the towering walls of prejudice peacefully. Of course, this did not prevent her from severing ties with her clan. As the outstanding Spindrift Knight, she hunts down Mondstadt's enemies in the wild to exact her unique "vengeance".""")