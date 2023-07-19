import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from load_data import running
import sys
from halo import Halo
import time
start = time.time()

spinner = Halo(text='', spinner='dots')
spinner.start()

file_path_to_narratives_file = ""
file_path_to_fields_folder = ""
file_path_to_damages_folder = ""
file_path_to_units_folder = ""
selected_field = ""
model_folder = ""
sample_size = 0
test_percent = 0
test_values = 0
cv_value = 0
model_loading = "False"

def validate_integer(text):
    if text.isdigit():
        return True
    elif text == "":
        return True
    else:
        return False

def on_file1_button_click():
    global file_path_to_narratives_file  # Use the global file_path_to_narratives_file variable
    selected_file = filedialog.askopenfilename(filetypes=[("Microsoft Excel Comma Separated Values File", "*.csv")])
    file_path_to_narratives_file = selected_file  # Assign the selected file to the file_path_to_narratives_file variable
    selected_file_label1.config(text="Selected file 1: " + selected_file)

def on_file2_button_click():
    global model_folder  # Use the global file_path_to_narratives_file variable
    selected_file = filedialog.askdirectory()
    model_folder = selected_file  # Assign the selected file to the file_path_to_narratives_file variable
    selected_file_label2.config(text="Selected file 2: " + selected_file)

def on_checkbox_change():
    if checkbox_var.get() == 1:
        file2_button.configure(state='normal')
        label4.configure(state='normal')
        selected_file_label2.configure(state='normal')
        entry4.configure(state='normal')
        entry.configure(state='disabled')
        entry2.configure(state='disabled')
        entry3.configure(state='disabled')
        label.configure(state='disabled')
        label2.configure(state='disabled')
        label3.configure(state='disabled')
    else:
        file2_button.configure(state='disabled')
        label4.configure(state='disabled')
        selected_file_label2.configure(state='disabled')
        entry4.configure(state='disabled')
        entry.configure(state='normal')
        entry2.configure(state='normal')
        entry3.configure(state='normal')
        label.configure(state='normal')
        label2.configure(state='normal')
        label3.configure(state='normal')

def on_folder2_button_click():
    global file_path_to_fields_folder
    selected_folder = filedialog.askdirectory()
    file_path_to_fields_folder = selected_folder
    selected_folder_label2.config(text="Selected folder 2: " + selected_folder)

def on_folder3_button_click():
    global file_path_to_damages_folder
    selected_folder = filedialog.askdirectory()
    file_path_to_damages_folder = selected_folder
    selected_folder_label3.config(text="Selected folder 3: " + selected_folder)

def on_folder4_button_click():
    global file_path_to_units_folder
    selected_folder = filedialog.askdirectory()
    file_path_to_units_folder = selected_folder
    selected_folder_label4.config(text="Selected folder 4: " + selected_folder)

def on_submit_button_click():
    global selected_field, sample_size, test_percent, cv_value
    selected_field = dropdown1.get()
    sample_size = int(entry_var.get())
    test_percent = int(entry_var2.get())
    cv_value = int(entry_var3.get())
    show_info_window()

def import_from_file():
    global model_loading, selected_field, sample_size, test_percent, file_path_to_narratives_file, file_path_to_fields_folder, file_path_to_damages_folder, file_path_to_units_folder, cv_value, model_folder, test_values
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            selected_field = lines[0].strip()
            file_path_to_narratives_file = lines[1].strip()
            file_path_to_fields_folder = lines[2].strip()
            file_path_to_damages_folder = lines[3].strip()
            file_path_to_units_folder = lines[4].strip()
            model_loading = lines[5].strip()
            if model_loading == "True":
                model_folder = lines[6].strip()
                test_values = int(lines[7].strip())
            else:
                sample_size = int(lines[6].strip())
                test_percent = int(lines[7].strip())
                cv_value = int(lines[8].strip())
            enable_fields(False)
    show_info_window()

def enable_fields(enabled=True):
    dropdown1.configure(state='normal' if enabled else 'disabled')
    file1_button.configure(state='normal' if enabled else 'disabled')
    folder2_button.configure(state='normal' if enabled else 'disabled')
    folder3_button.configure(state='normal' if enabled else 'disabled')
    folder4_button.configure(state='normal' if enabled else 'disabled')
    entry.configure(state='normal' if enabled else 'disabled')
    entry2.configure(state='normal' if enabled else 'disabled')
    entry3.configure(state='normal' if enabled else 'disabled')
    submit_button.configure(state='normal' if enabled else 'disabled')
    import_button.configure(state='normal' if enabled else 'disabled')
    checkbox.configure(state='normal' if enabled else 'disabled')
    if checkbox_var.get() == 1:
        file2_button.configure(state='normal' if enabled else 'disabled')
        entry4.configure(state='normal' if enabled else 'disabled')

def on_main_window_close():
    # This function will be called when the user clicks the X button on the main window
    # Add any cleanup or additional actions you want before closing the main window
    print("Main window is being closed")
    # You can add any other actions you want before closing the main window here.
    window.destroy()  # Close the main window
    sys.exit(0)  # End the program gracefully

# Create the main window
window = tk.Tk()
window.title("Data Selection")

# Set the window size
window.geometry("700x700")  # Width x Height

# Bind the 'WM_DELETE_WINDOW' event to the on_main_window_close function
window.protocol("WM_DELETE_WINDOW", on_main_window_close)

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

# Create the second button to select the data folder
folder3_button = ttk.Button(window, text="Damages Folder", command=on_folder3_button_click)
folder3_button.pack()

# Create a label to display the selected folder path
selected_folder_label3 = ttk.Label(window, text="Damages Folder: ")
selected_folder_label3.pack()

# Create the second button to select the data folder
folder4_button = ttk.Button(window, text="Units Folder", command=on_folder4_button_click)
folder4_button.pack()

# Create a label to display the selected folder path
selected_folder_label4 = ttk.Label(window, text="Units Folder: ")
selected_folder_label4.pack()

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
selected_file_label2.configure(state='disabled')
selected_file_label2.pack()

# Create a label
label4 = ttk.Label(window, text="Number of Tests for the Previous Model:")
label4.configure(state='disabled')
label4.pack()

# Create an entry widget for integer input
entry_var4 = tk.StringVar()
entry4 = ttk.Entry(window, textvariable=entry_var4, validate="key", validatecommand=(window.register(validate_integer), "%P"))
entry4.configure(state='disabled')
entry4.pack()

# Create a submit button
submit_button = ttk.Button(window, text="Submit", command=on_submit_button_click)
submit_button.pack()

# Create an import button
import_button = ttk.Button(window, text="Import from File", command=import_from_file)
import_button.pack()

# Check if values have been imported from the file
if selected_field and file_path_to_narratives_file and model_folder and file_path_to_fields_folder and file_path_to_damages_folder and file_path_to_units_folder and sample_size and cv_value:
    enable_fields(False)
    dropdown1.set(selected_field)
    selected_file_label1.config(text="Narratives File: " + file_path_to_narratives_file)
    selected_file_label2.config(text="Trained Model File: " + model_folder)
    selected_folder_label2.config(text="Crashes Folder: " + file_path_to_fields_folder)
    entry_var.set(int(sample_size))
    entry_var2.set(int(test_percent))
    entry_var3.set(int(cv_value))


def show_info_window():
    info_window = tk.Toplevel()
    info_window.title("Inputted Information")
    if model_loading == "True":
        label0 = ttk.Label(info_window, text="TESTING MODEL")
        label0.pack()
        label1 = ttk.Label(info_window, text="Interpretive Field: " + selected_field)
        label1.pack()
        label2 = ttk.Label(info_window, text="Narratives: " + file_path_to_narratives_file)
        label2.pack()
        label3 = ttk.Label(info_window, text="Fields Folder: " + file_path_to_fields_folder)
        label3.pack()
        label4 = ttk.Label(info_window, text="Damages Folder: " + file_path_to_damages_folder)
        label4.pack()
        label5 = ttk.Label(info_window, text="Units Folder: " + file_path_to_units_folder)
        label5.pack()
        label9 = ttk.Label(info_window, text="Trained Model Folder: " + model_folder)
        label9.pack()
        label10  = ttk.Label(info_window, text="Number of Tests: " + str(test_values))
        label10.pack()
    else:
        label0 = ttk.Label(info_window, text="TRAINING MODEL")
        label0.pack()
        label1 = ttk.Label(info_window, text="Interpretive Field: " + selected_field)
        label1.pack()
        label2 = ttk.Label(info_window, text="Narratives: " + file_path_to_narratives_file)
        label2.pack()
        label3 = ttk.Label(info_window, text="Fields Folder: " + file_path_to_fields_folder)
        label3.pack()
        label4 = ttk.Label(info_window, text="Damages Folder: " + file_path_to_damages_folder)
        label4.pack()
        label5 = ttk.Label(info_window, text="Units Folder: " + file_path_to_units_folder)
        label5.pack()
        label6 = ttk.Label(info_window, text="Training Size: " + str(sample_size))
        label6.pack()
        label7 = ttk.Label(info_window, text="Testing Percentage: " + str(test_percent))
        label7.pack()
        label8 = ttk.Label(info_window, text="Cross Check Value: " + str(cv_value))
        label8.pack()

    def run_button_command():
        info_window.destroy()  # Close the information window
        window.destroy()  # Close the input window

    run_button = ttk.Button(info_window, text="Run", command=run_button_command)
    run_button.pack()

# Dont touch this I will kill you
image_path = "el_gato.png"  # Replace with the actual image file path
image = Image.open(image_path)
photo_image = ImageTk.PhotoImage(image=image)

image_label = tk.Label(window, image=photo_image)
image_label.pack(side=tk.BOTTOM, padx=10, pady=10)

# Start the main loop
window.mainloop()

# Print the values after the main loop is done
if model_loading == "False":
    print("Selected options:")
    print("TRAINING NEW DATA")
    print("Interpretive Field:", selected_field)
    print("Narratives File:", file_path_to_narratives_file)
    print("Fields Folder:", file_path_to_fields_folder)
    print("Damages Folder:", file_path_to_damages_folder)
    print("Units Folder:", file_path_to_units_folder)
    print("Training Size:", sample_size)
    print("Testing Size:", test_percent)
    print("Cross Check Value:", cv_value)
    from plyer import notification
    notification.notify(
        title="Code Started for " + str(test_percent) + " with test percent "+str(sample_size),
        message="Your code has started running!",
    )
    running(file_path_to_narratives_file,file_path_to_fields_folder,file_path_to_damages_folder,file_path_to_units_folder,sample_size,test_percent,cv_value,model_folder,selected_field)
    # Code execution completed
    notification.notify(
        title="Code Done for " + str(test_percent) + " with test percent "+str(sample_size),
        message="Your code has finished running!",
    )
else:
    print("Selected options:")
    print("TESTING PREVIOUS MODEL")
    print("Interpretive Field:", selected_field)
    print("Narratives File:", file_path_to_narratives_file)
    print("Fields Folder:", file_path_to_fields_folder)
    print("Damages Folder:", file_path_to_damages_folder)
    print("Units Folder:", file_path_to_units_folder)
    print("Trained Model Folder:", model_folder)
    print("Number of Tests:", test_values)

    from plyer import notification
    notification.notify(
        title="Code Started for test value size " + str(test_values),
        message="Your code has started running!",
    )
    test_percent = 0
    sample_size = test_values
    cv_value = 0
    running(file_path_to_narratives_file,file_path_to_fields_folder,file_path_to_damages_folder,file_path_to_units_folder,sample_size,test_percent,cv_value,model_folder,selected_field)
    # Code execution completed
    notification.notify(
        title="Code Done for test value size " + str(test_values),
        message="Your code has finished running!",
    )
spinner.stop()
end = time.time()
print("Entire program completed in", end - start, "seconds.")