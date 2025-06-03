# GUI-based color input handler (conceptual)
from tkinter import Tk, Label, Button, colorchooser

def choose_aura_color():
    def color_chosen():
        color_code = colorchooser.askcolor(title="Choose your aura color")[1]
        label.config(text=f"Selected Color: {color_code}")
        # Future: save color_code to a user profile or pass to another module

    root = Tk()
    root.title("Aura Color Selector")
    label = Label(root, text="No Color Selected")
    label.pack(pady=20)

    choose_color_btn = Button(root, text="Choose Color", command=color_chosen)
    choose_color_btn.pack(pady=10)

    root.mainloop()
