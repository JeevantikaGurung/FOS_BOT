#with open("venv\summ.txt", "r") as file:
    #text1 = file.read()
from transformers import pipeline
import os
import warnings


model_name = "sshleifer/distilbart-cnn-12-6"
model_revision = "a4f8f3e"
default_max_length = 1000

def summarize(text):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    summarizer = pipeline("summarization", model=model_name, revision=model_revision)
    max_l = None
    # If max_length is not specified, calculate it based on the length of the input text
    if max_l is None:
        input_length = len(text)
        max_l = min(default_max_length, input_length)

    summary_text = summarizer(text, max_length=max_l, min_length=5, do_sample=False)[0]['summary_text']
    return summary_text

class Summarizer:
    def _init_(self):
        self.text = """"""
    
    def set_text(self,text):
        self.text = text

    def summarize(self):
        return summarize(self.text)
    
#sum = Summarizer()
#sum.set_text("""A Rebellious descendant of the old aristocracy who is always out on the battlefield.
#As one born into the old aristocracy, carrying the bloodline of sinners, 
#Eula has needed a unique approach to the world to navigate the towering walls of prejudice peacefully. 
#Of course, this did not prevent her from severing ties with her clan. As the outstanding Spindrift Knight, 
#she hunts down Mondstadt's enemies in the wild to exact her unique "vengeance".""")
#print(sum.summarize())