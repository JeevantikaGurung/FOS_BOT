import openai

openai.api_key = "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"

def paraphrase(text):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Paraphrase: {text}",
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text

class Paraphraser:
    def _init_(self):
        self.text = """"""
    
    def set_text(self,text):
        self.text = text

    def paraphrase(self):
        return paraphrase(self.text)

##sent = input()
##sent = """Eula Lawrence is a playable Cryo character in Genshin Impact.Although a descendant of the infamous and tyrannical Lawrence Clan, Eula severed her ties with the clan and became the captain of the Reconnaissance Company with the Knights of Favonius."""
##paraphrased_text = paraphrase(sent)
##print(paraphrased_text)