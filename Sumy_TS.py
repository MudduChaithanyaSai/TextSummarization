import sumy
from nltk.tokenize import punkt
import streamlit
import docx2txt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.tokenize import sent_tokenize

# taking input DATA
document1 = docx2txt.process("input_text.docx")

# counting number of sentences present in original text
sent_count = len(sent_tokenize(document1))
#Using parsers PlaintextParser method:
# For Strings
parser = PlaintextParser.from_string(document1,Tokenizer("english"))

# USING LSA
#Based on term frequency techniques with singular value decomposition to summarize texts.
#Let us summarize the text to 30% its original size or number of sentences    
##Method using stopwords
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
summarizer_lsa2 = LsaSummarizer()
summarizer_lsa2 = LsaSummarizer(Stemmer("english"))
summarizer_lsa2.stop_words = get_stop_words("english")
for sentence in summarizer_lsa2(parser.document,sent_count * 0.30):
    print(sentence)
