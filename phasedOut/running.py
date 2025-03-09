import tkinter as tk
from tkinter import ttk, filedialog, font, messagebox
import threading
from PIL import Image, ImageTk
import sys
from halo import Halo
import time
from PipelineHandler import PipelineHandler
from InterpretedField import InterpretedField
from plyer import notification
import os
from itertools import count, cycle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Exception ignored in atexit callback: <bound method Gcf.destroy_all of <class 'matplotlib._pylab_helpers.Gcf'>>")
start = time.time()
global restart, root, single_window
restart = True

class CloseError(Exception):
    pass

while restart == True:
    file_path_to_data_file = ""
    selected_field = ""
    model_folder = ""
    sample_size = 0
    test_size = 0
    cv_value = 0
    model_loading = "False"

    import io

    class TextRedirector(io.StringIO):
        def __init__(self, text_widget):
            self.text_widget = text_widget
            super().__init__()

        def write(self, s):
            self.text_widget.config(state='normal')
            self.text_widget.insert('end', s)
            self.text_widget.see('end')
            self.text_widget.config(state='disabled')

    class ImageLabel(tk.Label):
        """
        A Label that displays images, and plays them if they are gifs
        :im: A PIL Image instance or a string filename
        """
        def load(self, im):
            if isinstance(im, str):
                im = Image.open(im)
            frames = []
            try:
                for i in count(1):
                    frames.append(ImageTk.PhotoImage(im.copy()))
                    im.seek(i)
            except EOFError:
                pass
            self.frames = cycle(frames)
            try:
                self.delay = im.info['duration']
            except:
                self.delay = 100
            if len(frames) == 1:
                self.update_image(next(self.frames))
            else:
                self.next_frame()
        
        def unload(self):
            self.config(image=None)
            self.frames = None
        
        def start_animation(self):
            # Start the animation by scheduling the first frame update
            self.next_frame()
        
        def next_frame(self):
            if self.frames:
                self.update_image(next(self.frames))
                self.after_id = self.after(self.delay, self.next_frame)
        
        def update_image(self, image):
            self.config(image=image)
            self.image = image

        def stop_animation(self):
            if hasattr(self, 'after_id'):
                self.after_cancel(self.after_id)
                self.unload()
                self.frames = None  # Reset frames to stop further animation

    def validate_integer(text):
        if text.isdigit():
            return True
        elif text == "":
            return True
        else:
            return False
        
    def on_data_click():
        global file_path_to_data_file
        selected_file = filedialog.askopenfilename(filetypes=[("Microsoft Excel Comma Separated Values File", "*.csv")])
        file_path_to_data_file = selected_file
        selected_data_file_label.config(text="Data CSV: "+selected_file)

    def on_model_click():
        global model_folder
        "selected file is a .joblib file"
        selected_file = filedialog.askopenfilename(filetypes=[("Model File", "*.joblib")])
        model_folder = selected_file
        selected_model_file_label.config(text="Model: "+selected_file)


    def on_checkbox_change():
        if checkbox_var.get() == 1:
            model_button.config(state="normal")
            training_entry.config(state="disabled")
            cv_entry.config(state="disabled")
        else:
            model_button.config(state="disabled")
            training_entry.config(state="normal")
            cv_entry.config(state="normal")

    def import_from_file():
        global file_path_to_data_file, selected_field, model_folder, sample_size, test_size, cv_value, model_loading
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                model_loading = lines[0].strip()
                if model_loading == "True":
                    selected_field = lines[1].strip()
                    file_path_to_data_file = lines[2].strip()
                    test_size = int(lines[3].strip())
                    model_folder = lines[4].strip()
                else:
                    selected_field = lines[1].strip()
                    file_path_to_data_file = lines[2].strip()
                    sample_size = int(lines[3].strip())
                    test_size = int(lines[4].strip())
                    cv_value = int(lines[5].strip())
                
                
                enable_fields(False)
        show_info_window()
        
    def enable_fields(enabled=True):
        dropdown1.config(state="normal" if enabled else "disabled")
        data_button.config(state="normal" if enabled else "disabled")
        training_entry.config(state="normal" if enabled else "disabled")
        testing_entry.config(state="normal" if enabled else "disabled")
        checkbox.config(state="normal" if enabled else "disabled")
        model_button.config(state="normal" if enabled else "disabled")
        submit_button.config(state="normal" if enabled else "disabled")
        cv_entry.config(state="normal" if enabled else "disabled")

    def on_submit_button_click():
        global selected_field, sample_size, test_size, cv_value, model_loading, model_folder
        selected_field = dropdown1.get()
        sample_size = int(training_entry_var.get())
        test_size = int(testing_entry_var.get())
        cv_value = int(cv_entry_var.get())
        show_info_window()


    def on_single_test_click():
        global single_field, single_data_file, single_model_file, single_window

        def single_test_close():
            print("Closing Single Test Window")
            single_window.destroy()
            sys.exit(0)

        single_window = tk.Tk()
        single_window.title("Singular Test")
        single_window.geometry("500x500")
        single_window.protocol("WM_DELETE_WINDOW", single_test_close)

        single_title_label = tk.Label(single_window, text="Singular Test", font=("Arial", 20))
        single_title_label.pack()

        single_field_label = tk.Label(single_window, text="Field: ")
        single_field_label.pack()

        # Create the first dropdown list
        options1 = ['Bridge Detail', 'First Harmful Event', 'Intersection Related','Manner of Collision','Object Struck','Other Factor','Roadway Part','Road Class','Direction of Travel','First Harmful Event Involvement','Roadway Relation','Physical Features','Pedestrian Action','Pedalcyclist Action','E-scooter','Autonomous Unit','PBCAT Pedestrian','PBCAT Pedalcyclist']
        single_dropdown1 = ttk.Combobox(single_window, values=options1)
        single_dropdown1.pack()

        data_label = tk.Label(single_window, text="Data: "+file_path_to_data_file)
        data_label.pack()

        def on_single_data_click():
            global single_data_file
            selected_file = filedialog.askopenfilename(filetypes=[("Microsoft Excel Comma Separated Values File", "*.csv")])
            single_data_file = selected_file
            data_label.config(text="Data CSV: "+selected_file)

        # Create the button to select the data file
        data_button = tk.Button(single_window, text="Select Data File", command=on_single_data_click)
        data_button.pack()

        model_label = tk.Label(single_window, text="Model: "+model_folder)
        model_label.pack()

        def on_single_model_click():
            global single_model_file
            "selected file is a .joblib file"
            selected_file = filedialog.askopenfilename(filetypes=[("Model File", "*.joblib")])
            single_model_file = selected_file
            model_label.config(text="Model: "+selected_file)

        # Create the button to select the model file
        model_button = tk.Button(single_window, text="Select Model File", command=on_single_model_click)
        model_button.pack()

        def show_info_single():
            window_check = tk.Toplevel()
            window_check.title("Singular Test Inputs")
            window_check.geometry("600x600")

            bold_font2 = font.Font(family="Helvetica", size=12, weight="bold")

            single_field_label = tk.Label(window_check, text="Field: ", font=(bold_font2))
            single_field_label.pack()

            single_label = tk.Label(window_check, text=single_field)
            single_label.pack()

            single_data_label = tk.Label(window_check, text="Data: ", font=(bold_font2))
            single_data_label.pack()

            single_data = tk.Label(window_check, text=single_data_file)
            single_data.pack()

            single_model_label = tk.Label(window_check, text="Model: ", font=(bold_font2))
            single_model_label.pack()

            single_model = tk.Label(window_check, text=single_model_file)
            single_model.pack()

            "Make run button"
            def run_single_test():
                print("Running Single Test")
                single_window.destroy()
                single_run()

            run_button = tk.Button(window_check, text="Run", command=run_single_test)
            run_button.pack()
            

        def on_submit_button_click2():
            global single_field, single_data_file, single_model_file
            single_field = single_dropdown1.get()
            show_info_single()
            

        # Create the button to submit the data
        submit_button = tk.Button(single_window, text="Submit", command=on_submit_button_click2)
        submit_button.pack()


    def show_images_window():
        # Function to handle the click event of the hyperlink
        new_window = tk.Toplevel(window)
        new_window.title("Example Inputs")

        # Add the two images to the new window
        # You can replace the image paths with your actual image paths
        image_path1 = r"important_images\train_example.PNG"
        image_path2 = r"important_images\test_example.PNG"

        # Resize the images to a certain size (e.g., 200x200)
        desired_size = (800, 300)
        image1 = Image.open(image_path1).resize(desired_size, Image.LANCZOS)
        image2 = Image.open(image_path2).resize(desired_size, Image.LANCZOS)

        photo_image1 = ImageTk.PhotoImage(image=image1)
        photo_image2 = ImageTk.PhotoImage(image=image2)

        # Create labels and use the grid layout manager to place them side by side
        label1 = tk.Label(new_window, image=photo_image1)
        label1.grid(row=0, column=0, padx=10, pady=10)

        label2 = tk.Label(new_window, image=photo_image2)
        label2.grid(row=0, column=1, padx=10, pady=10)

        # Keep a reference to the images to avoid garbage collection
        label1.image = photo_image1
        label2.image = photo_image2


    def on_main_window_close():
        try:
            global restart, window
            print("Main window is being closed")
            restart = False
            window.destroy()  # Close the main window
            # sys.exit(0)  # End the program gracefully     
        except:
            folder_path = "temp"
            if os.path.exists(folder_path):
                folder_contents = os.listdir(folder_path)
                for item in folder_contents:
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        os.rmdir(item_path)
                os.rmdir(folder_path)
            sys.exit(0)
        

    "Create the window"
    window = tk.Tk()
    window.title("Machine Learning GUI")
    window.geometry("800x600")
    # Bind the 'WM_DELETE_WINDOW' event to the on_main_window_close function
    window.protocol("WM_DELETE_WINDOW", on_main_window_close)

    title_label = ttk.Label(window, text="Machine Learning GUI", font=("Arial", 20))
    title_label.pack()

    # Create a checkbox
    checkbox_var = tk.IntVar()
    checkbox = ttk.Checkbutton(window, text="Would you like to use a trained model?", variable=checkbox_var, command=on_checkbox_change)
    checkbox.pack()

    def single_run():
        window.destroy()
        on_single_test_click()

    single_click = ttk.Button(window, text="Single Test", command=single_run)
    single_click.pack()

    # Create a label for Dropdown 1
    dropdown_label = ttk.Label(window, text="Select an option for Interpretive Field:")
    dropdown_label.pack()

    # Create the first dropdown list
    options1 = ['Bridge Detail', 'First Harmful Event', 'Intersection Related','Manner of Collision','Object Struck','Other Factor','Roadway Part','Road Class','Direction of Travel','First Harmful Event Involvement','Roadway Relation','Physical Features','Pedestrian Action','Pedalcyclist Action','E-scooter','Autonomous Unit','PBCAT Pedestrian','PBCAT Pedalcyclist']
    dropdown1 = ttk.Combobox(window, values=options1)
    dropdown1.pack()

    # SELECTING DATA FILE
    data_button = ttk.Button(window, text="Select Data CSV", command=on_data_click)
    data_button.pack()

    selected_data_file_label = ttk.Label(window, text="Data CSV: ")
    selected_data_file_label.pack()

    # TRAINING DATA SIZE
    training_size_label = ttk.Label(window, text="Training Data Size:")
    training_size_label.pack()

    # Create an entry widget for integer input
    training_entry_var = tk.StringVar()
    training_entry = ttk.Entry(window, textvariable=training_entry_var, validate="key", validatecommand=(window.register(validate_integer), "%P"))
    training_entry.pack()

    # Create a label
    testing_size_label = ttk.Label(window, text="Number of Test Cases:")
    testing_size_label.pack()

    # Create an entry widget for integer input
    testing_entry_var = tk.StringVar()
    testing_entry = ttk.Entry(window, textvariable=testing_entry_var, validate="key", validatecommand=(window.register(validate_integer), "%P"))
    testing_entry.pack()

    # CV VALUE
    cv_label = ttk.Label(window, text="Cross Check Value:")
    cv_label.pack()

    # Create an entry widget for integer input
    cv_entry_var = tk.StringVar()
    cv_entry = ttk.Entry(window, textvariable=cv_entry_var, validate="key", validatecommand=(window.register(validate_integer), "%P"))
    cv_entry.pack()

    # SELECTING MODEL FILE
    model_button = ttk.Button(window, text="Select Model", command=on_model_click)
    model_button.config(state="disabled")
    model_button.pack()

    selected_model_file_label = ttk.Label(window, text="Selected Model: ")
    selected_model_file_label.pack()

    # Create a submit button
    submit_button = ttk.Button(window, text="Submit", command=on_submit_button_click)
    submit_button.pack()

    # Create an import button
    import_button = ttk.Button(window, text="Import from File", command=import_from_file)
    import_button.pack()

    # Create a hyperlink label
    hyperlink_label = tk.Label(window, text="What do input files look like?", fg="blue", cursor="hand2")
    hyperlink_label.pack()

    # Bind the click event of the hyperlink to the function show_images_window
    hyperlink_label.bind("<Button-1>", lambda event: show_images_window())


    def show_info_window():
        info_window = tk.Toplevel()
        info_window.title("Inputted Information")

        # Create a bold font
        bold_font1 = font.Font(family="Helvetica", size=12, weight="bold")
        bold_font2 = font.Font(family="Helvetica", size=8, weight="bold")

        label0_text = "TESTING MODEL:" if model_loading == "True" else "TRAINING MODEL:"
        label0 = ttk.Label(info_window, text=label0_text, font=bold_font1)
        label0.pack()

        label1_1 = ttk.Label(info_window, text="Interpretive Field:", font=bold_font2)
        label1_1.pack()
        label1_2 = ttk.Label(info_window, text=selected_field)
        label1_2.pack()

        label2_1 = ttk.Label(info_window, text="Data CSV:", font=bold_font2)
        label2_1.pack()
        label2_2 = ttk.Label(info_window, text=file_path_to_data_file)
        label2_2.pack()

        label4_1 = ttk.Label(info_window, text="Number of Test Cases:", font=bold_font2)
        label4_1.pack()
        label4_2 = ttk.Label(info_window, text=str(test_size))
        label4_2.pack()

        if model_loading == "True":
            label5_1 = ttk.Label(info_window, text="Model Folder:", font=bold_font2)
            label5_2 = ttk.Label(info_window, text=model_folder)
        else:
            label3_1 = ttk.Label(info_window, text="Training Data Size:", font=bold_font2)
            label3_2 = ttk.Label(info_window, text=str(sample_size))
            label3_1.pack()
            label3_2.pack()
            label5_1 = ttk.Label(info_window, text="Cross Check Value:", font=bold_font2)
            label5_2 = ttk.Label(info_window, text=str(cv_value))

        label5_1.pack()
        label5_2.pack()

        def run_button_command():
            global root, lbl
            info_window.destroy()  # Close the information window
            window.destroy()  # Close the input window

            def on_root_close():
                global restart, lbl
                if lbl: 
                    lbl.stop_animation()
                root.destroy()  # Close the main window
                restart = False
                # sys.exit(0)  # End the program gracefully
            root = tk.Tk()  # Create a new window
            root.title("Waiting Room")
            root.protocol("WM_DELETE_WINDOW", on_root_close)  # Bind the close button to the function on_root_close


            lbl = ImageLabel(root)
            lbl.pack()
            lbl.load("important_images\ZKZg.gif")

            output_text = tk.Text(root, wrap="word", height=10, width=80)
            output_text.pack()

            # Create the custom stream redirector
            redirector = TextRedirector(output_text)

            # Redirect sys.stdout to the custom stream redirector
            sys.stdout = redirector

            def threaderz():
                threader = threading.Thread(target=normal_run)
                threader.start()

                threader.join()

                lbl.unload()
                lbl.load("important_images\checkmark.gif")
                
                "Make a button to return to the main menu"
                def return_to_main_menu():
                    global root, lbl
                    if lbl:
                        lbl.stop_animation()
                    root.destroy()

                def exit_all():
                    global restart
                    root.destroy()
                    restart = False

                "Button asking if they would like to see feature inportance image in a top window"
                def feature_importance():
                    feature_importance_window = tk.Toplevel()
                    feature_importance_window.title("Feature Importance")

                    image_path = r"temp\feature_importances.png"
                    image = Image.open(image_path)
                    photo_image = ImageTk.PhotoImage(image)

                    label = tk.Label(feature_importance_window, image=photo_image)
                    label.image = photo_image
                    label.pack()

                def confusion_matrix():
                    "Check to see if the confident confusion matrix exists"
                    
                    confident_window = tk.Toplevel()
                    confident_window.title("Confident Confusion Matrix")

                    image_path = r"temp\confident_confusion_matrix.png"
                    image = Image.open(image_path)
                    photo_image = ImageTk.PhotoImage(image)

                    label = tk.Label(confident_window, image=photo_image)
                    label.image = photo_image
                    label.pack()

                    "Not Confident"
                    not_confident_window = tk.Toplevel()
                    not_confident_window.title("Confusion Matrix")

                    image_path = r"temp\confusion_matrix.png"
                    image = Image.open(image_path)
                    photo_image = ImageTk.PhotoImage(image)

                    label = tk.Label(not_confident_window, image=photo_image)
                    label.image = photo_image
                    label.pack()

                feature_importance_button = ttk.Button(root, text="Feature Importance", command=feature_importance)
                feature_importance_button.pack()

                "Make a button next to the feature importance button on the same line that will show the confusion matrix"
                if os.path.isfile(r"temp\confident_confusion_matrix.png"):
                        "If it does, make a button to show it"
                        show_confusion_matrix_button = ttk.Button(root, text="Confusion Matrix", command=confusion_matrix)
                        show_confusion_matrix_button.pack()
                
                return_button = ttk.Button(root, text="Return to Main Menu", command=return_to_main_menu)
                return_button.pack()

                exit_button = ttk.Button(root, text="Exit", command=exit_all)
                exit_button.pack()

            threader2 = threading.Thread(target=threaderz)
            threader2.start()


            root.mainloop()
            
        run_button = ttk.Button(info_window, text="Run", command=run_button_command)
        run_button.pack()

    # # Dont touch this I will kill you
    image_path = "important_images\el_gato.png"  # Replace with the actual image file path
    image = Image.open(image_path)
    photo_image = ImageTk.PhotoImage(image=image)

    image_label = tk.Label(window, image=photo_image)
    image_label.pack(side=tk.BOTTOM, padx=10, pady=10)

    def normal_run():
        global feature_show, confusion_show
        feature_show = False
        confusion_show = False
        field_code = [["Intersection Related","intersection_related"],["Roadway Part","roadway_part"],["Road Class","road_class"],["Bridge Detail","bridge_detail"]]
        field_final = ""
        for a in field_code:
            if a[0]==selected_field:
                field_final = a[1]

        print("Field_Final: "+field_final)

        # This code will only be reached after the main window is closed
        if model_loading == "False":
            print("Training Model")
            print("Interpretive Field: "+selected_field)
            print("Data CSV: "+file_path_to_data_file)
            print("Training Data Size: "+str(sample_size))
            print("Number of Test Cases: "+str(test_size))
            print("Cross Check Value: "+str(cv_value))
            from plyer import notification
            notification.notify(
                title="Code Started for " + str(sample_size) + " with test cases "+str(test_size),
                message="Your code has started running!",
            )

            a = PipelineHandler(InterpretedField(field_final))
            a.train(sample_size)
            "check if there is a folder named temp, then Save plot to temp folder"
            a.calculate_feature_importances()
            if test_size > 0:
                a.test(test_size)
                a.plot_confusion_matrix()

            "Check to see if there is a folder called models, if not create it and save the model there"
            if not os.path.exists("models"):
                os.makedirs("models")
            a.export_to_filepath("models/"+field_final+"_"+str(sample_size)+"_cv"+str(cv_value)+".joblib")
            print("Code execution completed")
            notification.notify(
                title="Code Done for " + str(sample_size) + " with test cases "+str(test_size),
                message="Your code has finished running!",
            )
        else:
            print("Testing Model")
            print("Interpretive Field: "+selected_field)
            print("Data CSV: "+file_path_to_data_file)
            print("Number of Test Cases: "+str(test_size))
            print("Model Folder: "+model_folder)

            from plyer import notification
            notification.notify(
                title="Code Started for test value size " + str(test_size),
                message="Your code has started running!",
            )

            a = PipelineHandler(InterpretedField(field_final))
            a.import_from_filepath(model_folder)
            a.test(test_size)
            
            a.calculate_feature_importances()
            if test_size > 0:
                a.plot_confusion_matrix()

            print("Code execution completed")
            # Code execution completed
            notification.notify(
                title="Code Done for test value size " + str(test_size),
                message="Your code has finished running!",
            )

    def single_run():
        field_code = [["Intersection Related","intersection_related"],["Roadway Part","roadway_part"],["Road Class","road_class"],["Bridge Detail","bridge_detail"]]
        field_final = ""
        for a in field_code:
            if a[0]==single_field:
                field_final = a[1]
        from plyer import notification
        notification.notify(
            title="Testing singular file using model",
            message="Your code has started running!",
        )

        a = PipelineHandler(InterpretedField(field_final))
        a.import_from_filepath(single_model_file)
        a.predict_single_row(single_data_file, 0)
        # show_data = messagebox.askyesno("Show Data", "Do you want to see the data for Feature Importance and Confusion Matricies?")
        # if show_data:
        #     a.calculate_feature_importances()
        #     a.calculate_confusion_matrix()
        # Code execution completed
        notification.notify(
            title="Finished testing singular file using model",
            message="Your code has finished running!",
        )
    try:
        window.mainloop()
    except CloseError:
        print("Error occurred while closing the window")
        sys.exit(1)  # Exit the program with an error code

"If temp folder exists, delete it"

folder_path = "temp"

if os.path.exists(folder_path):
    folder_contents = os.listdir(folder_path)
    for item in folder_contents:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            os.rmdir(item_path)
    os.rmdir(folder_path)