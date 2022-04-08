import re,unicodedata
import time,datetime

def unicodeToAscii(s):
    """
    convert to ascii string
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

replace_dict = {"don't": "do not","shouldn't": "should not",
         "can't": "cannot", "I'm": "Iam", "I've": "I have",
         "haven't": "have not",
         "hasn't": "has not", 
         "didn't": "did not"}

def clean_eng_text(text,replace_dict=replace_dict):
      """
      cleans english text
      Args:
           text: text to clean
           replace_dict: dict to replace words such as didn't wth did not
      Returns:
           clean text
      """
      text = unicodeToAscii(text.lower().strip())
      # remove some special charecters
      text = re.sub('[-?<>.!]','',text)
      # remove extra spaces
      text = re.sub('\s+',' ',text)
      text = ' '.join([replace_dict[word] if word in replace_dict.keys() else word for word in text.split()])
      text = [re.sub('[^A-Za-z]','',word) for word in text.split()]
      return ' '.join(text).lower()
      
def clean_ger_text(text):
    """
    cleans german text
    Args:
       text: germen text
    Returns:
       clean text
    """
    text = text.strip()
    text = re.sub("[.?]"," ",text)
    text = re.sub('\s+',' ',text)
    text = text.strip()
    return text
    
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))