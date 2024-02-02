!pip install tashaphyne
import gensim
from gensim.models import KeyedVectors
from gensim.models import word2vec

!unzip '/kaggle/working/AraVec'
!rm AraVec
import nltk
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tashaphyne.stemming import ArabicLightStemmer
import re
import emoji
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

test=pd.read_csv("/kaggle/input/neural/test_no_label.csv")
df=pd.read_csv("/kaggle/input/neural/train1.csv")
#print(df)
df = df.iloc[:]
# print("print data", df)
# print(len(df))
# Remove duplicated
df.review_description.duplicated().sum()
df.drop(df[df.review_description.duplicated() == True].index, axis=0, inplace=True)


# Remove Punctuation
df.review_description = df.review_description.astype(str)
df.review_description = df.review_description.apply(
    lambda x: re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
df.review_description = df.review_description.apply(lambda x: x.replace('؛', "", ))
# print("print data after pun", df.head())


# Define the function to remove consecutive duplicated Arabic words
def remove_duplicate_arabic_words(text):
    # Tokenize the text into words
    words = text.split()

    # Remove consecutive duplicated words
    unique_words = [words[i] for i in range(len(words)) if i == 0 or words[i] != words[i - 1]]

    # Join the unique words back into a sentence
    modified_text = ' '.join(unique_words)

    return modified_text


df['review_description'] = df['review_description'].apply(remove_duplicate_arabic_words)
# Remove StopWords
stopWords = list(set(stopwords.words("arabic")))  ## To remove duplictes and return to list again

# Some words needed to work with to will remove
for word in ['لا', 'لكن', 'ولكن']:
    stopWords.remove(word)
df.review_description = df.review_description.apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
# print("print data after removing stop words", df.head())

# Replace Emoji by Text
emojis = {"🙂": "يبتسم", "😂": "يضحك", "💔": "قلب حزين", "🙂": "يبتسم", "❤": "حب", "❤": "حب", "😍": "حب", "😭": "يبكي",
          "😢": "حزن", "😔": "حزن", "♥": "حب", "💜": "حب", "😅": "يضحك", "🙁": "حزين", "💕": "حب", "💙": "حب", "😞": "حزين",
          "😊": "سعادة", "👏": "يصفق", "👌": "احسنت", "😴": "ينام", "😀": "يضحك", "😌": "حزين", "🌹": "وردة", "🙈": "حب",
          "😄": "يضحك", "😐": "محايد", "✌": "منتصر", "✨": "نجمه", "🤔": "تفكير", "😏": "يستهزء", "😒": "يستهزء", "🙄": "ملل",
          "😕": "عصبية", "😃": "يضحك", "🌸": "وردة", "😓": "حزن", "💞": "حب", "💗": "حب", "😑": "منزعج", "💭": "تفكير",
          "😎": "ثقة", "💛": "حب", "😩": "حزين", "💪": "عضلات", "👍": "موافق", "🙏🏻": "رجاء طلب", "😳": "مصدوم", "👏🏼": "تصفيق",
          "🎶": "موسيقي", "🌚": "صمت", "💚": "حب", "🙏": "رجاء طلب", "💘": "حب", "🍃": "سلام", "☺": "يضحك", "🐸": "ضفدع",
          "😶": "مصدوم", "✌": "مرح", "✋🏻": "توقف", "😉": "غمزة", "🌷": "حب", "🙃": "مبتسم", "😫": "حزين", "😨": "مصدوم",
          "🎼 ": "موسيقي", "🍁": "مرح", "🍂": "مرح", "💟": "حب", "😪": "حزن", "😆": "يضحك", "😣": "استياء", "☺": "حب",
          "😱": "كارثة", "😁": "يضحك", "😖": "استياء", "🏃🏼": "يجري", "😡": "غضب", "🚶": "يسير", "🤕": "مرض", "‼": "تعجب",
          "🕊": "طائر", "👌🏻": "احسنت", "❣": "حب", "🙊": "مصدوم", "💃": "سعادة مرح", "💃🏼": "سعادة مرح", "😜": "مرح",
          "👊": "ضربة", "😟": "استياء", "💖": "حب", "😥": "حزن", "🎻": "موسيقي", "✒": "يكتب", "🚶🏻": "يسير", "💎": "الماظ",
          "😷": "وباء مرض", "☝": "واحد", "🚬": "تدخين", "💐": "ورد", "🌞": "شمس", "👆": "الاول", "⚠": "تحذير",
          "🤗": "احتواء", "✖": "غلط", "📍": "مكان", "👸": "ملكه", "👑": "تاج", "✔": "صح", "💌": "قلب", "😲": "مندهش",
          "💦": "ماء", "🚫": "خطا", "👏🏻": "برافو", "🏊": "يسبح", "👍🏻": "تمام", "⭕": "دائره كبيره", "🎷": "ساكسفون",
          "👋": "تلويح باليد", "✌🏼": "علامه النصر", "🌝": "مبتسم", "➿": "عقده مزدوجه", "💪🏼": "قوي", "📩": "تواصل معي",
          "☕": "قهوه", "😧": "قلق و صدمة", "🗨": "رسالة", "❗": "تعجب", "🙆🏻": "اشاره موافقه", "👯": "اخوات", "©": "رمز",
          "👵🏽": "سيده عجوزه", "🐣": "كتكوت", "🙌": "تشجيع", "🙇": "شخص ينحني", "👐🏽": "ايدي مفتوحه", "👌🏽": "بالظبط",
          "⁉": "استنكار", "⚽": "كوره", "🕶": "حب", "🎈": "بالون", "🎀": "ورده", "💵": "فلوس", "😋": "جائع", "😛": "يغيظ",
          "😠": "غاضب", "✍🏻": "يكتب", "🌾": "ارز", "👣": "اثر قدمين", "❌": "رفض", "🍟": "طعام", "👬": "صداقة", "🐰": "ارنب",
          "☂": "مطر", "⚜": "مملكة فرنسا", "🐑": "خروف", "🗣": "صوت مرتفع", "👌🏼": "احسنت", "☘": "مرح", "😮": "صدمة",
          "😦": "قلق", "⭕": "الحق", "✏": "قلم", "ℹ": "معلومات", "🙍🏻": "رفض", "⚪": "نضارة نقاء", "🐤": "حزن", "💫": "مرح",
          "💝": "حب", "🍔": "طعام", "❤": "حب", "✈": "سفر", "🏃🏻‍♀": "يسير", "🍳": "ذكر", "🎤": "مايك غناء", "🎾": "كره",
          "🐔": "دجاجة", "🙋": "سؤال", "📮": "بحر", "💉": "دواء", "🙏🏼": "رجاء طلب", "💂🏿 ": "حارس", "🎬": "سينما",
          "♦": "مرح", "💡": "قكرة", "‼": "تعجب", "👼": "طفل", "🔑": "مفتاح", "♥": "حب", "🕋": "كعبة", "🐓": "دجاجة",
          "💩": "معترض", "👽": "فضائي", "☔": "مطر", "🍷": "عصير", "🌟": "نجمة", "☁": "سحب", "👃": "معترض", "🌺": "مرح",
          "🔪": "سكينة", "♨": "سخونية", "👊🏼": "ضرب", "✏": "قلم", "🚶🏾‍♀": "يسير", "👊": "ضربة", "◾": "وقف", "😚": "حب",
          "🔸": "مرح", "👎🏻": "لا يعجبني", "👊🏽": "ضربة", "😙": "حب", "🎥": "تصوير", "👉": "جذب انتباه", "👏🏽": "يصفق",
          "💪🏻": "عضلات", "🏴": "اسود", "🔥": "حريق", "😬": "عدم الراحة", "👊🏿": "يضرب", "🌿": "ورقه شجره", "✋🏼": "كف ايد",
          "👐": "ايدي مفتوحه", "☠": "وجه مرعب", "🎉": "يهنئ", "🔕": "صامت", "😿": "وجه حزين", "☹": "وجه يائس", "😘": "حب",
          "😰": "خوف و حزن", "🌼": "ورده", "💋": "بوسه", "👇": "لاسفل", "❣": "حب", "🎧": "سماعات", "📝": "يكتب", "😇": "دايخ",
          "😈": "رعب", "🏃": "يجري", "✌🏻": "علامه النصر", "🔫": "يضرب", "❗": "تعجب", "👎": "غير موافق", "🔐": "قفل",
          "👈": "لليمين", "™": "رمز", "🚶🏽": "يتمشي", "😯": "متفاجأ", "✊": "يد مغلقه", "😻": "اعجاب", "🙉": "قرد",
          "👧": "طفله صغيره", "🔴": "دائره حمراء", "💪🏽": "قوه", "💤": "ينام", "👀": "ينظر", "✍🏻": "يكتب", "❄": "تلج",
          "💀": "رعب", "😤": "وجه عابس", "🖋": "قلم", "🎩": "كاب", "☕": "قهوه", "😹": "ضحك", "💓": "حب", "☄ ": "نار",
          "👻": "رعب", "❎": "خطء", "🤮": "حزن", '🏻': "احمر"}
emoticons_to_emoji = {":)": "🙂", ":(": "🙁", "xD": "😆", ":=(": "😭", ":'(": "😢", ":'‑(": "😢", "XD": "😂", ":D": "🙂",
                      "♬": "موسيقي", "♡": "❤", "☻": "🙂"}


def checkemojie(text):
    emojistext = []
    for char in text:
        if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
            emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
    return " ".join(emojistext)


def emojiTextTransform(text):
    cleantext = re.sub(r'[^\w\s]', '', text)
    return cleantext + " " + checkemojie(text)


# Apply checkemojie and emojiTextTransform
df['review_description'] = df['review_description'].apply(lambda x: emojiTextTransform(x))
# print("print data after changing the emoji to text", df['review_description'].head())

# Remove Numbers
df.review_description = df.review_description.apply(lambda x: ''.join([word for word in x if not word.isdigit()]))

# Apply Stemming
arabic_stemmer = ArabicLightStemmer()
# Apply stemming to the 'review_description' column
df['review_description'] = df['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))


hidden_dim =264 
output_size = 3 
num_layers = 2  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializations and corrections to your code
df.dropna(subset=['review_description'], inplace=True)
review_description = df['review_description']
y = df['rating']  # 1, 0, -1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_description)
seq = tokenizer.texts_to_sequences(review_description)
seq_pad = pad_sequences(seq, maxlen=150, padding="post", truncating="post")
vocab_size = len(tokenizer.word_index) + 1

w2v_embeddings_index = {}
TOTAL_EMBEDDING_DIM = 300
embeddings_file = '/kaggle/working/full_grams_cbow_300_twitter.mdl'
w2v_model = KeyedVectors.load(embeddings_file)

for word in w2v_model.wv.index_to_key:
    w2v_embeddings_index[word] = w2v_model.wv[word]

embedding_matrix = np.zeros((vocab_size, TOTAL_EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    embedding_vector = w2v_embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LSTMModel(nn.Module):
    def _init_(self, vocab_size, embed_dim, hidden_dim, output_size, num_layers, embedding_matrix):
        super(LSTMModel, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_size)  

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        out = lstm_out[:, -1] 
        out = self.fc(out)  

        return out 
model = LSTMModel(vocab_size, TOTAL_EMBEDDING_DIM, hidden_dim, output_size, num_layers, embedding_matrix).to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau


optimizer = optim.Adam(model.parameters(),0.1)
lossfun = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

def ApplyLstm(model, train_loader, val_loader, optimizer, lossfun, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for text, labels in train_loader:
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = lossfun(output, labels)
            total_loss += loss.item()
            predicted_classes = torch.argmax(output, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for text, labels in val_loader:
                text, labels = text.to(device), labels.to(device)
                output = model(text)
                loss = lossfun(output, labels)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        accuracy = correct_predictions / total_predictions * 100

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        scheduler.step(avg_val_loss)

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

y_encoded = le.fit_transform(y)  # This will transform -1, 0, 1 to 0, 1, 2

X = torch.from_numpy(seq_pad).long().to(device)
y_encoded = torch.tensor(y_encoded).long().to(device)
dataset = TensorDataset(X, y_encoded)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

ApplyLstm(model, train_loader, val_loader, optimizer, lossfun, scheduler, epochs=50)
test=pd.read_csv("/kaggle/input/neural/test _no_label.csv")
test['review_description'] = test['review_description'].apply(remove_duplicate_arabic_words)
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([word for word in x.split() if word not in stopWords]))
test['review_description'] = test['review_description'].apply(lambda x: emojiTextTransform(x))
test['review_description'] = test['review_description'].apply(lambda x: ''.join([word for word in x if not word.isdigit()]))
test['review_description'] = test['review_description'].apply(
    lambda x: " ".join([arabic_stemmer.light_stem(word) for word in x.split()]))

test_seq = tokenizer.texts_to_sequences(test['review_description'])
test_seq_pad = pad_sequences(test_seq, maxlen=150, padding="post", truncating="post")
X_test = torch.from_numpy(test_seq_pad).long().to(device)

model.eval()
with torch.no_grad():
    output = model(X_test)
    predicted_classes = torch.argmax(output, dim=1)

predicted_labels = le.inverse_transform(predicted_classes.cpu().numpy())
test['predicted_label'] = predicted_labels


test.to_csv('predicted_labels4.csv', columns=['ID', 'review_description', 'predicted_label'], index=False, encoding='utf-8-sig')