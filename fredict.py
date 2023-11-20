import os

import tokenize_uk as tok
import pymorphy2
import sqlite3
from sqlite3 import Error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

#перевіряє слова
def example1(documents):
    answer = []
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    print(tfidf_matrix)
    features_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[1]
    sorted_keywords = [(word, score) for score, word in sorted(zip(tfidf_scores, features_names), reverse=True)]
    #
    for tpl in sorted_keywords:
        answer.append(str(tpl))
    return answer

#перевірка кластерів
def example2(documents):
    answer = []
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    cluster_num = 200
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    kmeans.fit(tfidf_matrix)
    for cluster_id in range(cluster_num):
        tmp = ""
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        print(f"Cluster: {cluster_id + 1}")
        for idx in cluster_indices:
            tmp += str(documents[idx])
            print((documents[idx]))
        answer.append(tmp)
    return answer


NUM_SAMPLES = 20
SAMPLE_SIZE = 20000-1 #скільки слів було обмежено, обрізаємо текст
pos_tags = [ #для генерації частини мови
    "NOUN",  # noun
    "ADJF",  # adjective (full form)
    "ADJS",  # adjective (short form)
    "COMP",  # comparative
    "VERB",  # verb (personal form)
    "INFN",  # verb (infinitive)
    "PRTF",  # participle (full form)
    "PRTS",  # participle (short form)
    "GRND",  # gerund
    "NUMR",  # numeral
    "ADVB",  # adverb
    "NPRO",  # pronoun
    "PRED",  # predicative
    "PREP",  # preposition
    "CONJ",  # conjunction
    "PRCL",  # particle
    "INTJ",  # interjection
    "None"

]

#текст, переводить в нижній регістр, токенізує його
class Sample:
    def __init__(self, text):
        self._text = text
        self._tokens = tok.tokenize_words(self._text.lower())
        self._words_list = [s.lower() for s in self._tokens if s[0].isalpha()][:SAMPLE_SIZE]
#повертає токенізовані речення
    def get_sentences(self):
        return tok.tokenize_sents(self._text)

    def get_words(self):
        words_set = set()
        words = []

        for word_str in self._words_list:
            if word_str in words_set:
                continue

            num = self._words_list.count(word_str)
            words.append(Word.from_token(word_str, num))
        return words

    def show(self, words):
        for word in words:
            word.show()

#створюємо курсор і БД
class SQL:
    def __init__(self, file):
        self._connection = SQL.create_connection(file)
        self._cursor = self._connection.cursor()

    @staticmethod
    def create_connection(db_file):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            print(sqlite3.version)
        except Error as e:
            print(e)

        return conn

    def insert_sentences(self, text_id, sentence):
        insert_sentence_sql = '''
            INSERT INTO sentences (text_id, sentence)
            VALUES (?, ?);
        '''

        self._cursor.execute(insert_sentence_sql, (text_id, sentence))
        self._connection.commit()

    def fill_texts_table(self, texts):
        insert_text_sql = '''
            INSERT INTO texts (text)
            VALUES (?);
        '''
        for text in texts:
            self._cursor.execute(insert_text_sql, (text,))
        self._connection.commit()

    def create_pos_table(self):
        create_poses_table_sql = '''
            CREATE TABLE IF NOT EXISTS poses (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            );
        '''
        self._cursor.execute(create_poses_table_sql)
        self._connection.commit()

    def fill_pos_table(self, pos_tags):
        insert_pos_sql = '''
            INSERT INTO poses (text)
            VALUES (?);
        '''
        # Corrected to insert each POS as a tuple
        for pos in pos_tags:
            self._cursor.execute(insert_pos_sql, (pos,))
        self._connection.commit()

    def insert_into_freq_table(self, word, sentence_id):
        pos_id = self.get_pos_id(word._pos)
        data_tuple = (word._form, word._lemma, sentence_id, pos_id, word._freq)
        insert_word_freq_sql = '''
           INSERT INTO WordFreq (form, lemma,sentence_id, pos_id, freq )
           VALUES(?, ?, ?, ?, ?);
        '''
        self._cursor.execute(insert_word_freq_sql, data_tuple)
        self._connection.commit()

    def select_first_n_chars_from_all_texts(self, n=10):
        # Retrieve all text entries from the database
        query = "SELECT text FROM texts;"
        res = self._cursor.execute(query)
        rows = [row[0] for row in res]

        # Concatenate all texts
        all_texts_combined = "".join(rows)

        # Return the first n characters of the concatenated string
        return all_texts_combined[:n]

    def select_pos_freq(self):
        result = f'''
               SELECT
                   pos_id,
                   COUNT(word_id)
               FROM
                   WordFreq
               GROUP BY
                   pos_id;
               '''
        res = self._cursor.execute(result)
        for row in res:
            print(row)

    def get_last_inserted_id(self, table_name):
        query = f"SELECT last_insert_rowid() FROM {table_name};"
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]

    def create_word_freq_table(self):
        create_word_freq_sql = '''
               CREATE TABLE IF NOT EXISTS WordFreq (
                   word_id INTEGER PRIMARY KEY,
                   form TEXT,
                   lemma TEXT,
                   sentence_id INTEGER,
                   pos_id INTEGER,
                   freq INTEGER,
                   FOREIGN KEY(pos_id) REFERENCES poses(id),
                   FOREIGN KEY(sentence_id) REFERENCES sentences(id)
               );
               '''
        self._cursor.execute(create_word_freq_sql)
        self._connection.commit()

#кількість унікальних слів
    def count_form(self) -> int:

        result = f'''
               SELECT
                   COUNT(DISTINCT form)
               FROM
                   WordFreq;
               '''
        res = self._cursor.execute(result)
        return res.fetchone()[0]

    def table_exists(self, table_name):
        """ Check if a table exists in the database """
        query = f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        self._cursor.execute(query)
        if self._cursor.fetchone()[0] == 1:
            return True
        return False

    def count_rows_in_table(self, table_name):
        """ Count the number of rows in a table """
        query = f"SELECT COUNT(*) FROM {table_name}"
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]

    def create_statistics_table_for_1000_forms_subsample(self):
        """ Create table with statistics for the 1000 * 20 """
        query = '''
            CREATE TABLE IF NOT EXISTS statistics_1000 (
                id INTEGER PRIMARY KEY, 
                text_id INTEGER,      
                form TEXT,
                abs_freq INTEGER,
                sub_sample_1_freq INTEGER,
                sub_sample_2_freq INTEGER,
                sub_sample_3_freq   INTEGER,
                sub_sample_4_freq  INTEGER,
                sub_sample_5_freq INTEGER,
                sub_sample_6_freq INTEGER,
                sub_sample_7_freq INTEGER,  
                sub_sample_8_freq INTEGER,  
                sub_sample_9_freq INTEGER,  
                sub_sample_10_freq INTEGER,  
                sub_sample_11_freq INTEGER,  
                sub_sample_12_freq INTEGER,  
                sub_sample_13_freq INTEGER,  
                sub_sample_14_freq INTEGER, 
                sub_sample_15_freq INTEGER, 
                sub_sample_16_freq INTEGER, 
                sub_sample_17_freq INTEGER, 
                sub_sample_18_freq INTEGER, 
                sub_sample_19_freq INTEGER, 
                sub_sample_20_freq INTEGER,
                FOREIGN KEY(text_id) REFERENCES texts(id)
            );
        '''
        self._cursor.execute(query)
        self._connection.commit()

#беремо одне слово і рахуємо його в кожній підвибірці
    def count_forms_from_index_to_index(self, form: str, x: int, text_id: int, samples, sample_size: int = 20000):
        """ Count the number of forms in the WordFreq table from index to index """
        # Calculate the offset (n-1)*1000 to get the start of the n-th subsample
        size = sample_size // samples
        offset = (x) * size  #відстань для х subsample (координата початку), розмір вікна

        query = """
            SELECT COUNT(*)
            FROM (
                SELECT *
                FROM WordFreq
                JOIN sentences ON WordFreq.sentence_id = sentences.id
                WHERE sentences.text_id = ?
                LIMIT ? OFFSET ? 
            ) AS Subsample
            WHERE form = ?
            """
        # замість ? підставляє значення параметру
        self._cursor.execute(query, (text_id, size, offset, form))

        return self._cursor.fetchone()[0] #поверне кількість для словоформи

#рахує частоту, скільки зустрічається форма слова в одному тексті
    def count_abs_freq_where_text_id_is(self, form: str, text_id: int):
        """ Count the absolute frequency of a form in the WordFreq table for a specific text_id """
        query = """
        SELECT COUNT(WordFreq.form) 
        FROM WordFreq 
        JOIN sentences ON WordFreq.sentence_id = sentences.id 
        WHERE WordFreq.form=? AND sentences.text_id=?
        """
        self._cursor.execute(query, (form, text_id))
        return self._cursor.fetchone()[0]

    def count_texts(self):
        """ Count the number of texts in the database """
        query = "SELECT COUNT(*) FROM texts"
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]

    def get_all_words(self):
        """ Retrieve all words from the WordFreq table """
        query = "SELECT * FROM WordFreq"
        self._cursor.execute(query)
        words = self._cursor.fetchall()
        return words

#створюється текстова таблиця, куди буде записано id тексту і весь повністю текст
    def create_texts_table(self):
        query = '''
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY,
                text TEXT,
                number INTEGER
            );
        '''
        self._cursor.execute(query)
        self._connection.commit()
#бере унікальні значеня форм для тексту (чи першого чи другого) і створює список
    def get_all_distinct_forms_as_list(self, text_id: int):
        """ Retrieve all distinct forms from the WordFreq table for a specific text_id """
        query = """
            SELECT DISTINCT WordFreq.form
            FROM WordFreq
            JOIN sentences ON WordFreq.sentence_id = sentences.id
            WHERE sentences.text_id = ?
        """
        self._cursor.execute(query, (text_id,))
        forms = [form[0] for form in self._cursor.fetchall()]
        return forms

#повертає частоту з поля слів за формою
    def get_num_for_word(self, word: str):
        """ Retrieve the number of a word from the WordFreq table """
        query = "SELECT freq FROM WordFreq WHERE form=?"
        self._cursor.execute(query, (word,))
        return self._cursor.fetchone()[0]

#заповнює останню таблицю для 20 підвибірок сатистикою
    def fill_statistics_table_for_1000_forms_subsample(self, text_id: int, subsamples: int = 20,
                                                       sample_size: int = 20000):
        forms = self.get_all_distinct_forms_as_list(text_id)

        for form in forms:
            statistics_for_form = self.get_freq_for_n_subsamples(form, subsamples, text_id, sample_size)
            abs_freq = self.count_abs_freq_where_text_id_is(form, text_id)
            query = "INSERT INTO statistics_1000 (text_id, form, abs_freq, sub_sample_1_freq," \
                    " sub_sample_2_freq, sub_sample_3_freq, sub_sample_4_freq," \
                    " sub_sample_5_freq, sub_sample_6_freq, sub_sample_7_freq, " \
                    "sub_sample_8_freq, sub_sample_9_freq, sub_sample_10_freq, " \
                    "sub_sample_11_freq, sub_sample_12_freq, sub_sample_13_freq, " \
                    "sub_sample_14_freq, sub_sample_15_freq, sub_sample_16_freq, " \
                    "sub_sample_17_freq, sub_sample_18_freq, sub_sample_19_freq," \
                    " sub_sample_20_freq)  " \
                    "VALUES ( ?,?, ?, ?,?, ?, ?, ?,?, ?, ?, ?,?, ?, ?, ?,?, ?, ?, ?,?, ?, ?)"
            self._cursor.execute(query, (text_id, form, abs_freq, *statistics_for_form))
        self._connection.commit()

#беремо список речень з тексту (або з 1 або з 2)
    def get_sentence_from_text_id(self, text_id: int):
        query = "SELECT sentence FROM sentences WHERE text_id=?"
        self._cursor.execute(query, (text_id,))
        answer = [item[0] for item in self._cursor.fetchall()]
        return answer

#бере всі айді з текстів
    def get_all_texts_ids(self):
        query = "SELECT id FROM texts"
        answer = [item[0] for item in self._cursor.execute(query).fetchall()]
        return answer

# бере всі речення з усіх текстів
    def get_all_sentences(self):
        text_ids = self.get_all_texts_ids()
        answer = []
        for text_id in text_ids:
            answer.append(self.get_sentence_from_text_id(text_id))
        return answer
#показує статистику
    def show_statistics_table_for_1000_forms_subsample(self):
        query = "SELECT * FROM statistics_1000"
        self._cursor.execute(query)
        statistics = self._cursor.fetchall()
        return statistics

#викликає для всіх (по черзі для кожного) сабсемплу обрахунок його частоти
    def get_freq_for_n_subsamples(self, form, n, text_id, sample_size):
        answer = []
        for x in range(n):
            answer.append(self.count_forms_from_index_to_index(form, x, text_id, n, sample_size))
        return answer

#створює таблицю для речень
    def create_sentences_table(self):
        query = '''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY,
                text_id INTEGER,
                sentence TEXT,
                FOREIGN KEY(text_id) REFERENCES texts(id)
            );
        '''
        self._cursor.execute(query)
        self._connection.commit()

#бере всі речення, повертає таблицю речень
    def select_all_sentences(self):
        query = "SELECT * FROM sentences"
        self._cursor.execute(query)
        sentences = self._cursor.fetchall()
        return sentences

#повертає id  частини мови
    def get_pos_id(self, pos):
        """ Retrieve the id of a part of speech """
        query = f"SELECT id FROM poses WHERE text='{pos}'"
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]

#створюємо клас Word , викликаємо пайморфі, розбиваємо текст на форму, лему, чм і частоту
class Word:
    s_morph = pymorphy2.MorphAnalyzer(lang='uk')

    def __init__(self, form, lemma, pos, freq):
        self._form = form
        self._lemma = lemma
        self._pos = pos
        self._freq = freq

    @staticmethod
    def from_token(token, freq):
        parsed = Word.s_morph.parse(token)[0]
        return Word(token, parsed.normal_form, parsed.tag.POS, freq)
#виводить словофоорма, лема, чм, частота
    def show(self):
        print(
            f"Словоформа : {self._form}                Лема : {self._lemma}                Частина мови : {self._pos}                Частота : {self._freq}")

    def get_tuple(self):
        return (self._form, self._lemma, self._pos, self._freq)

#для зручності, якщо треба викликати і видалити БД
def drop_db_if_exists():
    if os.path.exists('freq.db'):
        os.remove('freq.db')

#робимо БД, створюємо таблиці, якщо їх нема
def create_db_from_txts(txt_files):
    # Ensure tables are created
    if not sql.table_exists('texts'):
        sql.create_texts_table()
    if not sql.table_exists('sentences'):
        sql.create_sentences_table()
    if not sql.table_exists('poses'):
        sql.create_pos_table()
        sql.fill_pos_table(pos_tags)
    if not sql.table_exists('WordFreq'):
        sql.create_word_freq_table()
    if not sql.table_exists('statistics_1000'):
        sql.create_statistics_table_for_1000_forms_subsample()
#відкриває для всіх текстових файлів один з наших текстів і для потім виконує подальші дії
    for txt_file in txt_files:
        with open(txt_file, encoding='utf-8', mode='r') as f:
            text = f.read()
            word_count = 0
#викликає метод який заповнює таблицю тексту текстом
            # Insert text and get text ID
            print("Inserting text into DB...")
            sql.fill_texts_table([text])

            text_id = sql.get_last_inserted_id('texts')
#бере з тексту, робить семпл, розбиває його на речення
            # Process sentences
            sample = Sample(text)
            sentences = sample.get_sentences()
            print("Inserting sentences into DB...")
            for sentence in sentences:
                if word_count > SAMPLE_SIZE: #якщо кількість слів більша за розмір семплу - зупиняє
                    break
                sql.insert_sentences(text_id, sentence)
                sentence_id = sql.get_last_inserted_id('sentences')

#тепер беремо окремо слова з кожного речення  і заповнюємо таблицю слів
                # Process words in the sentence
                sample_sentence = Sample(sentence)
                words = sample_sentence.get_words()
                for word in words:
                    if word_count > SAMPLE_SIZE:
                        break
                    sql.insert_into_freq_table(word, sentence_id)
                    word_count += 1 #щоб зупинилось, коли нарахує 20000 тис слів
            print(word_count)
            print("Inserting statistics into DB...") #вставляє слово у таблицю
            sql.fill_statistics_table_for_1000_forms_subsample(text_id, NUM_SAMPLES, SAMPLE_SIZE)

#створює обєкт sql який буде викликати методи для запису та редагування таблиць
#drop_db_if_exists()
sql = SQL('freq.db')
if not os.path.exists('freq.db'):

    create_db_from_txts(['misto_by_valerian_pidmohylny.txt', 'modified_extracted_text.txt'])


sql.select_pos_freq()

print(sql.get_all_words())

print(sql.show_statistics_table_for_1000_forms_subsample())
sentences_list = sql.get_all_sentences()

#обчислюємо TF-IDF
#аналітика для слів
for n,sentences in enumerate(sentences_list):
    with open(f"text{n}_analytics.txt", "w") as f:
        print(f"Info for the text number {n}")
        for line in example1((sentences)):
            f.writelines(line+"\n")


        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#аналітика кластерів
    with open(f"text{n}_analytics_cluster.txt", "w") as f:
        print(f"Info for the text number {n}")
        for n,line in enumerate(example2((sentences))):
            f.writelines(f"Cluster {n}\n"+line+"\n")

        f.writelines("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")