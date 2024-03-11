import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def generate_answer(question, context):
    # Tokenize the inputs
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    # Generate the answer
    start_scores, end_scores = model(**inputs,return_dict=False)
    
    start_scores = start_scores.clone().detach()
    end_scores = end_scores.clone().detach()

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer_tokens = inputs["input_ids"][0][start_index:end_index]
    answer_tokens = answer_tokens.tolist()
    # Convert the answer tokens back to text
    answer = tokenizer.decode(answer_tokens)
    answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()
    return answer






class Query:
    def __init__(self):
        self.context = [""""""]
    def generate_answer(self,query,text):
        self.context = text
        return generate_answer(query,self.context)

# Example usage

#q = Query()
#context = """Eula Lawrence is a playable Cryo character in Genshin Impact.
#Although a descendant of the infamous and tyrannical Lawrence Clan, Eula severed her ties with the clan and became the captain of the Reconnaissance Company with the Knights of Favonius."""
#question = "What element is Eula?"   # Don't forget to include '?' at the end of the question
#answer = q.generate_answer(question,context)



#answer = generate_answer(question, context)
#print(answer) 