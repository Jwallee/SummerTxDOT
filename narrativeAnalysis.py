import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

# Here, we read the files present in the folder path specified (crashes)
def get_file_names(folder_path):
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

folder_path = 'crashes'  # REPLACE THIS IF THE PDFs ARE STORED ELSEWHERE
file_names = get_file_names(folder_path)

narrative_array = []

# Looping through the files present in the crashes folder
for name in file_names:

    # The path to read the pdfs
    pdf_path = 'crashes/'+name

    # Convert the narrative page of the PDF to an image
    images = convert_from_path(pdf_path, first_page=2, last_page=2)

    # Save the image (whole folder)
    out_path = 'whole/'+name+'.jpg'
    images[0].save(out_path, 'JPEG')

    def crop_image(image_path, output_path, target_size):
        # Open the image using Pillow
        image = Image.open(image_path)
        
        # Get the current dimensions of the image
        current_width, current_height = image.size
        
        # Calculate the cropping coordinates
        target_width, target_height = target_size
        left = 125
        top = (target_height+1795)
        right = left + target_width
        bottom = top + target_height
        
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        
        # Save the cropped image
        cropped_image.save(output_path)


    # Example usage
    image_path = out_path  # Replace with your image path
    output_path = 'cropped/'+name+'.jpg'  # Replace with your desired output path
    target_size = (3312, 3100)  # Replace with your desired target size

    crop_image(image_path, output_path, target_size)

    # Saving Lifted Text to file

    def save_text_to_file(text, file_path):
        with open(file_path, 'w') as file:
            file.write(text)

    # Extract text using Tesseract OCR
    text = pytesseract.image_to_string(output_path)
    file_path = "liftedText/"+name+".txt"

    save_text_to_file(text, file_path)