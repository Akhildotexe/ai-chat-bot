from tkinter import *
from tkinter import ttk
import requests


url = "https://random-stuff-api.p.rapidapi.com/ai"


headers = {
    'authorization': "HZYidwPaIcaL",
    'x-rapidapi-host': "https://v6.rsa-api.xyz/",
    'x-rapidapi-key': "0e046e3bd0mshc505d671972904bp17d339jsn7f2e09ae4501"
    }


# Make a window with a title of "ChatBot" and a width of 1920 and a height of 1080
window = Tk()
window.title("ChatBot")
window.geometry("1920x1080")
label = Label(window, text="Hello, I am ChatBot")
label.pack()
e = ttk.Entry(window, width=50)
e.pack()

# Function to get the response from the API when user types something into the entry box.
def myclick():
    hello = e.get()
    session = requests.request(f'GET', params = {"msg" : hello} , headers = headers, url = url)
    chatting = session.json()["message"]
    mylabel = Label(window, text=chatting)
    mylabel.pack()

#create a new text box where the chatbot can send the response
# mybutton = ttk.textbox(window, text="Click Me", command=myclick)
# mybutton.pack()

b = ttk.Button(window, text="Talk to the Chatbot!", command=myclick)
b.pack()


# Make a button that says "Quit" and when you click it, it will close the window
button = Button(window, text="Quit", command=window.destroy)
button.pack()


while True:
    window.mainloop()
    break 