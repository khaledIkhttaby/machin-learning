import os
import emoji
import re
import pandas as pd

settings_dir = os.path.dirname(__file__)
settings_dir = os.path.abspath(os.path.dirname(settings_dir))
project_root = os.path.abspath(os.path.dirname(settings_dir))
resources = project_root + "/resources/"


def split_emoji(txt):
    x = ' '.join(c for c in txt.split() if c not in emoji.UNICODE_EMOJI)
    return x


stop_word = pd.read_csv(resources + "data/stopwords.csv", names=['data'])['data'].values


def len_sent(text):
    return len(text.split())


def preprocessing(data):
    data = str(data)
    data = split_emoji(data)
    data = str(' '.join(re.sub("([٠١٢٣٤٥٦٧٨٩]+)|([0-9]+)|([A-Za-z]+)|\_+|(\#)+|(\/)+|(\:)+", " ", data).split()))
    data = re.sub("[إأٱآا]", "ا", data)
    data = re.sub("ة", "ه", data)
    # remove duplicate
    noise = re.compile(""" ّ    | # Tashdid
                            َ    | # Fatha
                            ً    | # Tanwin Fath
                            ُ    | # Damma
                            ٌ    | # Tanwin Damm
                            ِ    | # Kasra
                            ٍ    | # Tanwin Kasr
                            ْ    | # Sukun
                            ـ   |  # Tatwil/Kashida
                            ،   |
                        """, re.VERBOSE)
    flagsUs = re.compile("["
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         "]+", flags=re.UNICODE)
    dirtyChars = re.compile("["
                            "\u0600-\u0620"
                            "\u063B-\u0640"
                            "\u064B-\u065F"
                            "\u066A-\u06FF"
                            "\u0750-\u077F"
                            "\u08A0-\u08FF"
                            "\uFB50-\uFBE9"
                            "\uFBF0-\uFBFB"
                            "\uFC5B-\uFC63"
                            "\uFCF2-\uFCF4"
                            "\uFD3C-\uFD4F"
                            "\uFD90-\uFD91"
                            "\uFDC8-\uFDFF"
                            "\uFE70-\uFE7F"
                            "\uFEFD-\uFEFF"
                            "]+", flags=re.UNICODE)
    data = str(re.sub(flagsUs, '', data))
    data = str(re.sub(dirtyChars, '', data))
    data = str(re.sub(noise, '', data))
    #     data = str(''.join(c for c in data if c not in punctuation))
    data = re.sub(
        '\/+|\●+|\◽+|\٪+|\▪+|\»+|\«+|\_+|\ʚïɞ+|\▐+|\►+|\"+|\*+|\▁+|\》+|\《+|\[+|\Ещё+|\]+|\|+|\;+|\'+|\<+|\>+|\\+|\`+|\{+|\}+|\~+|\"+|\-+|\:+|\@+|\#+|\$+|\ﷺ+|\%+|\^+|\&+|\(+|\)+|\.+|\,+|\?+|\=+|\++|\؛+\“+|\”+',
        ' ', data)
    data = re.sub('\!', '  ', data)
    data = re.sub('\⚘', '  ', data)
    data = re.sub('\��', '  ', data)
    data = re.sub('\؟ ', ' ', data)
    data = re.sub('\.', ' ', data)
    data = re.sub('\s+', ' ', data)
    data = re.sub('\\\+', ' ', data)
    data = re.sub('\"+', ' ', data)
    data = re.sub(' ا ', ' ', data)
    data = re.sub(' ب ', ' ', data)
    data = re.sub(' ت ', ' ', data)
    data = re.sub(' ث ', ' ', data)
    data = re.sub(' ج ', ' ', data)
    data = re.sub(' ح ', ' ', data)
    data = re.sub(' خ ', ' ', data)
    data = re.sub(' د ', ' ', data)
    data = re.sub(' ذ ', ' ', data)
    data = re.sub(' ر ', ' ', data)
    data = re.sub(' ز ', ' ', data)
    data = re.sub(' س ', ' ', data)
    data = re.sub(' ش ', ' ', data)
    data = re.sub(' ص ', ' ', data)
    data = re.sub(' ض ', ' ', data)
    data = re.sub(' ط ', ' ', data)
    data = re.sub(' ظ ', ' ', data)
    data = re.sub(' ع ', ' ', data)
    data = re.sub(' غ ', ' ', data)
    data = re.sub(' ف ', ' ', data)
    data = re.sub(' ق ', ' ', data)
    data = re.sub(' ك ', ' ', data)
    data = re.sub(' ل ', ' ', data)
    data = re.sub(' م ', ' ', data)
    data = re.sub(' ن ', ' ', data)
    data = re.sub(' ه ', ' ', data)
    data = re.sub(' و ', ' ', data)
    data = re.sub(' ي ', ' ', data)
    data = re.sub(' ئ ', ' ', data)
    data = re.sub(' ؤ ', ' ', data)
    data = re.sub(' ء ', ' ', data)
    data = re.sub('\_+', ' ', data)
    data = re.sub('\…+', ' ', data)
    data = re.sub('\“|\”', '', data)
    data = re.sub(r'([\u0600-\u06FF])\1{3,}', r'\1\1\1', data)
    data = re.sub(r'[\u2066]', ' ', data)
    data = re.sub(r'[\u2069]', ' ', data)
    data = re.sub(r'[\uFE0F]', ' ', data)
    data = re.sub(r'[\u25a0]', ' ', data)
    data = re.sub(r'[\u2022]', ' ', data)
    data = re.sub(r'[\u2592]', ' ', data)
    data = re.sub(
        '[\u1ea0]|[\u1e97]|[\u1ea1]|[\u02bf]|[\u1e97]|[\u1ea1]|[\u1e97]|[\u1ea1]|[\u1ea1]|[\u02be]|[\u1ea1]|[\u1ea1]',
        ' ', data)
    data = ' '.join([word for word in data.split() if word not in stop_word])
    data = " ".join(data.split())
    return data


# In[14]:


def len_text(text):
    return len(text)



def prepare_data():
    path_data = resources + "data/Posts.csv"
    all_data = pd.read_csv(path_data)

    my_data = all_data[["haha", "sad", "love", "angry", "wow", "content", "page_name"]]
    my_data['sumreactions'] = my_data['haha'] + my_data['sad'] + my_data['love'] + my_data['angry'] + my_data['wow']

    df_age_negative = my_data[my_data['sumreactions'] < 1]  # Step 1
    all_data = all_data.drop(df_age_negative.index, axis=0)
    all_data['contentclean'] = all_data['content'].apply(preprocessing)
    all_data['len_content'] = all_data['contentclean'].apply(len_text)
    my_data['len_text'] = all_data['len_content']
    my_data['contentclean'] = all_data['contentclean']
    my_data[my_data['len_text'] > 3].to_csv(resources + "data/Data_cleaning_with_content.csv", index=False)
