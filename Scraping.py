# %%
import pandas as pd

# Read the Excel file into a DataFrame
df = pd.read_excel('Input.xlsx')

# Print the DataFrame
print(df)

# %% [markdown]
# IData Scraping

# %%
from bs4 import BeautifulSoup
import requests

# %%
import os
import requests
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool

def process_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', class_='td-post-content tagdiv-type')

    if content is None:
        content = soup.select_one('#tdi_117 > div > div.vc_column.tdi_120.wpb_column.vc_column_container.tdc-column.td-pb-span8 > div > div.td_block_wrap.tdb_single_content.tdi_130.td-pb-border-top.td_block_template_1.td-post-content.tagdiv-type > div')

    if content is not None:
        paragraphs = []
        for p_tag in content.find_all('p'):
            paragraphs.append(p_tag.get_text())

        url_id = df.loc[df['URL'] == url, 'URL_ID'].values[0]
        filename = os.path.join(folder_path, f"{url_id}.txt")

        with open(filename, 'w', encoding='utf-8') as file:
            file.write(' '.join(paragraphs))

        return f"File '{filename}' created."
    else:
        return f"No content found for URL: {url}"

if __name__ == '__main__':
    folder_path = 'Extracted_Pages'
    os.makedirs(folder_path, exist_ok=True)

    urls = df['URL'][:]

    with ThreadPool() as pool:
        results = pool.map(process_url, urls)

    for result in results:
        print(result)

# %% [markdown]
# Step 3 Data Cleaning
# 1. Constant arrays jisme stopwords ki files individually assign kr denge
# 2. ek variable array jisme hr file ko assign krenge iteratively
# 3. subtract stop word arrays from the variable array
# 4. save the final variable array in new file cleaned krke

# %% [markdown]
# Importing Stop Words

# %%
import codecs

try:
    negative_words = {word.lower() for word in file.read().splitlines()}
except UnicodeDecodeError:
    # If 'utf-8' encoding fails, try a different encoding
    with codecs.open('MasterDictionary/negative-words.txt', 'r', encoding='latin-1', errors='ignore') as file:
        negative_words = {word.lower() for word in file.read().splitlines()}
def read_stopwords_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        words = file.read().splitlines()
    return words

stopwords_files = [
    'StopWords/StopWords_Auditor.txt',
    'StopWords/StopWords_Currencies.txt',
    'StopWords/StopWords_DatesandNumbers.txt',
    'StopWords/StopWords_Generic.txt',
    'StopWords/StopWords_GenericLong.txt',
    'StopWords/StopWords_Geographic.txt',
    'StopWords/StopWords_Names.txt'
]

stopwords = []

for file in stopwords_files:
    file_words = read_stopwords_file(file)
    stopwords.extend(file_words)

print(stopwords)

import codecs

try:
    negative_words = {word.lower() for word in file.read().splitlines()}
except UnicodeDecodeError:
    # If 'utf-8' encoding fails, try a different encoding
    with codecs.open('MasterDictionary/negative-words.txt', 'r', encoding='latin-1', errors='ignore') as file:
        negative_words = {word.lower() for word in file.read().splitlines()}
# %%
import os

directory_path = 'Extracted_Pages'
output_directory = 'Clean_Text'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        original_file_path = os.path.join(directory_path, filename)
        cleaned_file_path = os.path.join(output_directory, f'cleaned_{filename}')
        
        with open(original_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        cleaned_words = [word for word in text.split() if word.lower() not in stopwords]
        cleaned_text = ' '.join(cleaned_words)
        
        with open(cleaned_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)

# %%
positive_words = []
with open('MasterDictionary/positive-words.txt', 'r', encoding='utf-8') as file:
    positive_words = file.read().splitlines()

import codecs

negative_words = []
with codecs.open('MasterDictionary/negative-words.txt', 'r', encoding='utf-8', errors='ignore') as file:
    negative_words = file.read().splitlines()


# %%
print(positive_words)
print(negative_words)

# %%
#Sentiment Analysis Functions
positivity_scores = []
negativity_scores= []
polarity_scores= []
subjectivity_scores= []

#Positive Score Function
def positivity_score(text):
    positive_word_count = 0
    for word in text.split():
        if word.lower() in positive_words:
            positive_word_count += 1
    positivity_scores.append(positive_word_count)
    return positive_word_count
#Negative Score Function
def negativity_score(text):
    negative_word_count = 0
    for word in text.split():
        if word.lower() in negative_words:
            negative_word_count += 1
    negativity_scores.append(negative_word_count)
    return negative_word_count

#Polarity Score function
def polarity_score(positive_score, negative_score):
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    polarity_scores.append(polarity_score)

def subjectivity_score(positive_score, negative_score, total_words):
    subjective_score = (positive_score + negative_score) / (total_words + 0.000001)
    subjectivity_scores.append(subjective_score)




# %%
import os
def sentiment_analysis():
    clean_text_folder = 'Clean_text'
    for filename in os.listdir(clean_text_folder):
        with open(os.path.join(clean_text_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            positive_score = positivity_score(text)
            negative_score = negativity_score(text)
            polarity_score(positive_score, negative_score)
            subjectivity_score(positive_score,negative_score,len(text))

# %%
sentiment_analysis()
print(positivity_scores)
print(negativity_scores)
print(polarity_scores)
print(subjectivity_scores)



# %%
#complex word counter
complex_word_counter= []
def complex_word_count(text):
    syllabel = ['a', 'e', 'i', 'o', 'u']
    complex_words = 0
    words = text.split()
    for word in words:
        vowel_count = sum(1 for letter in word if letter in syllabel)
        if vowel_count > 2:
            complex_words += 1
    complex_word_counter.append(complex_words)
    return complex_words
    
# Sentence Count function
sentence_counts = []
def count_sentences(text):
    sentence_count = text.count('.') + text.count('?') + text.count('!')
    sentence_counts.append(sentence_count)
    return sentence_count
    
    
#word count (with removed punctuation)
import re
word_count = []
def count_words_in_file(text):
        # Remove punctuation from the text
        text_without_punct = re.sub(r'[^\w\s]', '', text)
        # Split the text into words
        words = text_without_punct.split()
        # Return the number of words
        word_count.append(len(words))
        return(len(words))



# %%


# %%
words_per_sentence = []
def average_words_per_sentence(text):
    word_count = count_words_in_file(text)
    sentence_count = count_sentences(text)
    if sentence_count > 0:  # Check to avoid division by zero
        average = word_count / sentence_count
    else:
        average = 0  # Default value when there are no sentences
    words_per_sentence.append(average)



syllable_per_word= []
def average_syllables_per_word(text):
    syllable_exceptions = ['es', 'ed']
    syllable_count = 0
    words = text.split()
    for word in words:
        if word[-2:] in syllable_exceptions:
            continue
        vowel_count = sum(1 for letter in word if letter in ['a', 'e', 'i', 'o', 'u'])
        syllable_count += max(1, vowel_count)
    total_words = len(words)
    if total_words > 0:  # Check to avoid division by zero
        average_syllables = syllable_count / total_words
    else:
        average_syllables = 0  # Default value when there are no words
    syllable_per_word.append(average_syllables)



personal_pronoun_count=[]
def personal_pronouns(text):
    pronoun_count = 0
    pronouns = ['I', 'we', 'my', 'ours', 'us']
    for pronoun in pronouns:
        pronoun_count += len(re.findall(r'\b' + pronoun + r'\b', text, re.IGNORECASE))
    
    personal_pronoun_count.append(pronoun_count)



avg_word_length = []
def average_word_length(text):
    words = text.split()
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    if total_words > 0:  # Check to avoid division by zero
        average_length = total_characters / total_words
    else:
        average_length = 0  # Default value when there are no words
    avg_word_length.append(average_length)
   


# %%
avg_sentence_length_count = []
def avg_sentence_length(text):
    sentences = text.split('.')
    total_sentences = len(sentences)
    total_characters = sum(len(sentence) for sentence in sentences)
    if total_sentences != 0:  # Avoid division by zero
        average_length = total_characters / total_sentences
        avg_sentence_length_count.append(average_length)
    else:
        avg_sentence_length_count.append(0)

complex_word_percent_counter = []
def percent_complex_words(text):
    syllabel = ['a', 'e', 'i', 'o', 'u']
    complex_words = 0
    words = text.split()
    for word in words:
        vowel_count = sum(1 for letter in word if letter in syllabel)
        if vowel_count > 2:
            complex_words += 1

    total_words = len(words)
    if total_words != 0:  # Avoid division by zero
        percentage_complex = (complex_words / total_words) * 100
        
    else:
        percentage_complex = 0
        
    complex_word_percent_counter.append(percentage_complex)

fog_index_counter=[]
def fog_index(text):
    # Calculate average sentence length
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('?') + text.count('!')
    if sentence_count != 0:
        average_sentence_length = word_count / sentence_count
    else:
        average_sentence_length = 0

    # Calculate percentage of complex words
    syllabel = ['a', 'e', 'i', 'o', 'u']
    complex_words = 0
    words = text.split()
    for word in words:
        vowel_count = sum(1 for letter in word if letter in syllabel)
        if vowel_count > 2:
            complex_words += 1
    total_words = len(words)
    if total_words != 0:
        percentage_complex = (complex_words / total_words) * 100
    else:
        percentage_complex = 0

    # Calculate Fog Index
    fog_index_value = 0.4 * (average_sentence_length + percentage_complex)
    fog_index_counter.append(fog_index_value)

# %%
def readability_analysis():
    clean_text_folder = 'Clean_text'
    for filename in os.listdir(clean_text_folder):
        with open(os.path.join(clean_text_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            # Call analysis functions once per file
            complex_word_count(text)
            count_words_in_file(text)
            count_sentences(text)
            fog_index(text)
            average_word_length(text)
            percent_complex_words(text)
            avg_sentence_length(text)
            average_syllables_per_word(text)
            average_words_per_sentence(text)

def readability_analysis_orignal():
    text_folder = 'Extracted_Pages'
    for filename in os.listdir(text_folder):
        with open(os.path.join(text_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            # Call analysis function once per file
            personal_pronouns(text)
                

# %%
readability_analysis()
print(word_count)
print(sentence_counts)
print(complex_word_counter)
print(avg_word_length)
print(fog_index_counter)
print(complex_word_percent_counter)
print(avg_sentence_length_count)
print(avg_word_length)
print(syllable_per_word)
print(words_per_sentence)
readability_analysis_orignal()
print(personal_pronoun_count)


word_count_len = len(word_count)
sentence_counts_len = len(sentence_counts)
complex_word_counter_len = len(complex_word_counter)
avg_word_length_len = len(avg_word_length)
fog_index_counter_len = len(fog_index_counter)
complex_word_percent_counter_len = len(complex_word_percent_counter)
avg_sentence_length_count_len = len(avg_sentence_length_count)
syllable_per_word_len = len(syllable_per_word)
words_per_sentence_len = len(words_per_sentence)
personal_pronoun_count_len = len(personal_pronoun_count)

print("Number of positivity scores: ", len(positivity_scores))
print("Number of negativity scores: ", len(negativity_scores))
print("Number of polarity scores: ", len(polarity_scores))
print("Number of subjectivity scores: ", len(subjectivity_scores))
print("Word Count Length: ", word_count_len)
print("Sentence Counts Length: ", sentence_counts_len)
print("Complex Word Counter Length: ", complex_word_counter_len)
print("Average Word Length: ", avg_word_length_len)
print("Fog Index Counter Length: ", fog_index_counter_len)
print("Complex Word Percent Counter Length: ", complex_word_percent_counter_len)
print("Average Sentence Length Count: ", avg_sentence_length_count_len)
print("Syllable Per Word Length: ", syllable_per_word_len)
print("Words Per Sentence Length: ", words_per_sentence_len)
print("Personal Pronoun Count Length: ", personal_pronoun_count_len)



# %%
import pandas as pd

data = {
    'URL_ID': df["URL_ID"][:],
    'URL': urls,
    'POSITIVE SCORE': positivity_scores,
    'NEGATIVE SCORE': negativity_scores,
    'POLARITY SCORE': polarity_scores,
    'SUBJECTIVITY SCORE': subjectivity_scores,
    'AVG SENTENCE LENGTH': avg_sentence_length_count,
    'PERCENTAGE OF COMPLEX WORDS': complex_word_percent_counter,
    'FOG INDEX': fog_index_counter,
    'AVG NUMBER OF WORDS PER SENTENCE': words_per_sentence,
    'COMPLEX WORD COUNT': complex_word_counter,
    'WORD COUNT': word_count,
    'SYLLABLE PER WORD': syllable_per_word,
    'PERSONAL PRONOUNS': personal_pronoun_count,
    'AVG WORD LENGTH': avg_word_length
}

df = pd.DataFrame(data)
df.to_excel('readability_analysis.xlsx', index=False)


