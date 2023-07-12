import tkinter as tk
from tkinter import ttk, filedialog
from load_data import running
import winsound

data_file1 = ""
model_folder = ""
data_folder2 = ""
selected_field = ""
training_size = 0
testing_size = 0
cv_value = 0

def validate_integer(text):
    if text.isdigit():
        return True
    elif text == "":
        return True
    else:
        return False

def on_file1_button_click():
    global data_file1  # Use the global data_file1 variable
    selected_file = filedialog.askopenfilename(filetypes=[("Microsoft Excel Comma Separated Values File", "*.csv")])
    data_file1 = selected_file  # Assign the selected file to the data_file1 variable
    selected_file_label1.config(text="Selected file 1: " + selected_file)

def on_file2_button_click():
    global model_folder  # Use the global data_file1 variable
    selected_file = filedialog.askdirectory()
    model_folder = selected_file  # Assign the selected file to the data_file1 variable
    selected_file_label2.config(text="Selected file 2: " + selected_file)

def on_checkbox_change():
    if checkbox_var.get() == 1:
        file2_button.configure(state='normal')
    else:
        file2_button.configure(state='disabled')

def on_folder2_button_click():
    global data_folder2
    selected_folder = filedialog.askdirectory()
    data_folder2 = selected_folder
    selected_folder_label2.config(text="Selected folder 2: " + selected_folder)

def on_submit_button_click():
    global selected_field, training_size, testing_size, cv_value
    selected_field = dropdown1.get()
    training_size = int(entry_var.get())
    testing_size = int(entry_var2.get())
    cv_value = int(entry_var3.get())
    show_info_window()

def import_from_file():
    global selected_field, training_size, testing_size, data_file1, data_folder2, cv_value, model_folder
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            selected_field = lines[0].strip()
            data_file1 = lines[1].strip()
            data_folder2 = lines[2].strip()
            training_size = int(lines[3].strip())
            testing_size = int(lines[4].strip())
            cv_value = int(lines[5].strip())
            if len(lines) >= 7:
                model_folder = lines[6].strip()
            enable_fields(False)
    show_info_window()

def enable_fields(enabled=True):
    dropdown1.configure(state='normal' if enabled else 'disabled')
    file1_button.configure(state='normal' if enabled else 'disabled')
    folder2_button.configure(state='normal' if enabled else 'disabled')
    entry.configure(state='normal' if enabled else 'disabled')
    entry2.configure(state='normal' if enabled else 'disabled')
    entry3.configure(state='normal' if enabled else 'disabled')
    submit_button.configure(state='normal' if enabled else 'disabled')

# Create the main window
window = tk.Tk()
window.title("Data Selection")

# Set the window size
window.geometry("700x700")  # Width x Height

# Create a label for Dropdown 1
label1 = ttk.Label(window, text="Select an option for Interpretive Field:")
label1.pack()

# Create the first dropdown list
options1 = ['Bridge Detail', 'First Harmful Event', 'Intersection Related','Manner of Collision','Object Struck','Other Factor','Roadway Part','Road Class','Direction of Travel','First Harmful Event Involvement','Roadway Relation','Physical Features','Pedestrian Action','Pedalcyclist Action','E-scooter','Autonomous Unit','PBCAT Pedestrian','PBCAT Pedalcyclist']
dropdown1 = ttk.Combobox(window, values=options1)
dropdown1.pack()

# Create the first button to select the data file
file1_button = ttk.Button(window, text="Narratives csv", command=on_file1_button_click)
file1_button.pack()

# Create a label to display the selected file path
selected_file_label1 = ttk.Label(window, text="Narratives csv: ")
selected_file_label1.pack()

# Create the second button to select the data folder
folder2_button = ttk.Button(window, text="Fields Folder", command=on_folder2_button_click)
folder2_button.pack()

# Create a label to display the selected folder path
selected_folder_label2 = ttk.Label(window, text="Fields Folder: ")
selected_folder_label2.pack()

# Create a label
label = ttk.Label(window, text="Training Data Size:")
label.pack()

# Create an entry widget for integer input
entry_var = tk.StringVar()
entry = ttk.Entry(window, textvariable=entry_var, validate="key", validatecommand=(window.register(validate_integer), "%P"))
entry.pack()

# Create a label
label2 = ttk.Label(window, text="Number of Test Cases:")
label2.pack()

# Create an entry widget for integer input
entry_var2 = tk.StringVar()
entry2 = ttk.Entry(window, textvariable=entry_var2, validate="key", validatecommand=(window.register(validate_integer), "%P"))
entry2.pack()

# Create a label
label3 = ttk.Label(window, text="Cross Check Value:")
label3.pack()

# Create an entry widget for integer input
entry_var3 = tk.StringVar()
entry3 = ttk.Entry(window, textvariable=entry_var3, validate="key", validatecommand=(window.register(validate_integer), "%P"))
entry3.pack()

# Create a checkbox
checkbox_var = tk.IntVar()
checkbox = ttk.Checkbutton(window, text="Enable Option", variable=checkbox_var, command=on_checkbox_change)
checkbox.pack()

# Create the first button to select the data file
file2_button = ttk.Button(window, text="Previous Trained Model", command=on_file2_button_click)
file2_button.configure(state='disabled')
file2_button.pack()

# Create a label to display the selected file path
selected_file_label2 = ttk.Label(window, text="Previous Trained Model Folder: ")
selected_file_label2.pack()

# Create a submit button
submit_button = ttk.Button(window, text="Submit", command=on_submit_button_click)
submit_button.pack()

# Create an import button
import_button = ttk.Button(window, text="Import from File", command=import_from_file)
import_button.pack()

# Check if values have been imported from the file
if selected_field and data_file1 and model_folder and data_folder2 and training_size and testing_size and cv_value:
    enable_fields(False)
    dropdown1.set(selected_field)
    selected_file_label1.config(text="Narratives File: " + data_file1)
    selected_file_label2.config(text="Trained Model File: " + model_folder)
    selected_folder_label2.config(text="Crashes Folder: " + data_folder2)
    entry_var.set(int(training_size))
    entry_var2.set(int(testing_size))
    entry_var3.set(int(cv_value))


def show_info_window():
    info_window = tk.Toplevel()
    info_window.title("Inputted Information")

    label1 = ttk.Label(info_window, text="Interpretive Field: " + selected_field)
    label1.pack()
    label2 = ttk.Label(info_window, text="Narratives: " + data_file1)
    label2.pack()
    label3 = ttk.Label(info_window, text="Crash Folder: " + data_folder2)
    label3.pack()
    label4 = ttk.Label(info_window, text="Training Size: " + str(training_size))
    label4.pack()
    label5 = ttk.Label(info_window, text="Testing Size: " + str(testing_size))
    label5.pack()
    label6 = ttk.Label(info_window, text="Cross Check Value: " + str(cv_value))
    label6.pack()
    label7 = ttk.Label(info_window, text="Trained Model Folder: " + model_folder)
    label7.pack()

    def run_button_command():
        info_window.destroy()  # Close the information window
        window.destroy()  # Close the input window

    run_button = ttk.Button(info_window, text="Run", command=run_button_command)
    run_button.pack()


# Start the main loop
window.mainloop()

# Print the values after the main loop is done
print("Selected options:")
print("Interpretive Field:", selected_field)
print("Narratives File:", data_file1)
print("Fields File:", data_folder2)
print("Training Size:", training_size)
print("Testing Size:", testing_size)
print("Cross Check Value:", cv_value)
print("Trained Model Folder:", model_folder)

# while training_size <= 100000:
running(data_file1,data_folder2,testing_size,training_size,cv_value,selected_field,model_folder)
    # training_size = training_size + 5000
winsound.PlaySound('6-[AudioTrimmer.com].WAV', winsound.SND_FILENAME)