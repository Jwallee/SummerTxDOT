def download(crash):
    import webbrowser
    import pyautogui
    import keyboard
    import time
    # Specify the URL you want to open
    url = 'https://cris.dot.state.tx.us/secure/ImageServices/DisplayImageServlet?CrashId='+crash

    # Open the URL in the default web browser
    webbrowser.open(url)

    # Delay before the mouse movement (in seconds)
    time.sleep(2)

    # Get the screen's size
    screen_width, screen_height = pyautogui.size()
    # print(screen_height, screen_width)

    # Move the mouse to the target position
    # DOWNLOAD PDF COORDS
    pyautogui.moveTo(1800, 130)

    # Click the mouse
    pyautogui.click()

    # Delay after the click (optional)
    time.sleep(0.5)

    # Type a string
    string_to_type = crash
    keyboard.write(string_to_type)
    time.sleep(0.5)

    # Press Enter key
    keyboard.press('enter')
    keyboard.release('enter')

    time.sleep(0.2)

    # CLOSE WINDOW COORDS
    pyautogui.moveTo(470, 12)
    pyautogui.click()
    time.sleep(0.2)

import pandas as pd

# Path to the Excel file
excel_file_path = 'crashes/Images Data.xlsx'

# Read the Excel file into a pandas DataFrame, skipping the first 3 rows
df = pd.read_excel(excel_file_path, skiprows=2)

# Convert the DataFrame to a 2D array
values_array = df.values.tolist()

# Print the values
for row in values_array:
    for value in row:
        download(str(value))

