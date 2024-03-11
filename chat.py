import random
import json
import Paraphraser
import Inverted_Indexing
import torch
import pyttsx3
import Speech
import Query_Indexing
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import summarizer

def get_response():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('venv\intents.json', 'r') as json_data:
        intents = json.load(json_data)

    FILE = "venv\data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    engine = pyttsx3.init()

    bot_name = "FOS"
    print(f"{bot_name}:Let's chat! (Type 'Quit' to exit)")
    Speech.SpeakText("Let's chat! FOS at your service")

    while True:
        ##sentence = "do you use credit cards?"
        sentence = input("You: ")
        ##Speech.Activate
        if sentence == "Quit":
            break
            
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        #print(tag)
        
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:

            if tag == "Paraphrase":
                print(f"{bot_name}: Enter as many lines of text as you want.")
                print(f"{bot_name}: When you're done, enter a single period on a line by itself.")
                buffer = []
                while True:
                    print("> ", end="")
                    line = input()
                    if line == ".":
                        break
                    buffer.append(line)
                multiline_string = "\n".join(buffer)
                para = Paraphraser.Paraphraser()
                para.set_text(multiline_string)
                paraphrased_text = para.paraphrase()
                print(f"{bot_name}: " + paraphrased_text)
                engine.say(paraphrased_text)
                engine.runAndWait()
                continue


            if tag == "Summarizer":
                print(f"{bot_name}: Enter as many lines of text as you want.")
                print(f"{bot_name}: When you're done, enter a single period on a line by itself.")
                buffer = []
                while True:
                    print("> ", end="")
                    line = input()
                    if line == ".":
                        break
                    buffer.append(line)
                multiline_string = "\n".join(buffer)
                summ = summarizer.Summarizer()
                summ.set_text(multiline_string)
                summarized_text = summ.summarize()
                print(f"{bot_name}: " + summarized_text)
                engine.say(summarized_text)
                engine.runAndWait()
                continue


            if tag == "Inverted Index":
                print(f"{bot_name}: Enter as many lines of text as you want.")
                print(f"{bot_name}: When you're done, enter a single period on a line by itself.")
                buffer = []
                while True:
                    print("> ", end="")
                    line = input()
                    if line == ".":
                        break
                    buffer.append(line)
                multiline_string = "\n".join(buffer)
                index = Inverted_Indexing.Indexing()
                sent = index.Index(multiline_string)
                print(f"{bot_name}: " + sent)
                engine.say(sent)
                engine.runAndWait()
                continue


            if tag == "Query retrieval":
                print(f"{bot_name}: Enter as many lines of text as you want.")
                print(f"{bot_name}: When you're done, enter a single period on a line by itself.")
                buffer = []
                while True:
                    print("> ", end="")
                    line = input()
                    if line == ".":
                        break
                    buffer.append(line)
                multiline_string = "\n".join(buffer)
                query = Query_Indexing.Query()
                question  = input(f"{bot_name}: I have processed the document/paragraph, Shoot your questions lad I am here to help\n")
                answer = query.generate_answer(question,multiline_string)
                print(f"{bot_name}: " + answer)
                engine.say(answer)
                engine.runAndWait()
                continue
        
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    print(f"{bot_name}: " + response)
                    engine.say(response)
                    engine.runAndWait()
        else:
            print(f"{bot_name}: I do not understand...")

get_response()
