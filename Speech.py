import speech_recognition as sr
import pyttsx3

r = sr.Recognizer()
def SpeakText(command):
    engine = pyttsx3.init()
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)
    engine.say(command)
    engine.runAndWait()

def Activate():
    while(1):   
        try:
            SpeakText("Speak my son")
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say ",MyText)
                SpeakText(MyText)
                break
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))  
        except sr.UnknownValueError:
            print("Unknown error occurred")