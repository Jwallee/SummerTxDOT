### handles data preprocessing from large crash extracts in order to run ML algorithms
import os
import pandas as pd
from testing import testing
from halo import Halo
import time

# MASTER METHOD
def load_data(narratives_filepath, fields_folder_filepath, field_columns, classification_field_column):
    narrative_dictionary = read_narratives(narratives_filepath)
    fields_dictionary = read_fields(fields_folder_filepath, field_columns, classification_field_column)
    arrays_of_rows = process_data_into_arrays(combine_fields_and_narratives(narrative_dictionary, fields_dictionary))
    return arrays_of_rows[1:-1], arrays_of_rows[0], arrays_of_rows[-1] # fields, classifications, narratives

# takes file path to CSV and returns narratives processed into a dictionary
def read_narratives(filepath_to_narratives):
    narrative_dictionary = {}

    #reads all data
    narrative_raw_data = pd.read_csv(filepath_to_narratives, encoding='windows-1252', skiprows=0).values.tolist()

    #enters data into dictionary
    for entry in narrative_raw_data:
        narrative_dictionary[entry[0]] = [entry[1]]
    
    print("Loaded " + str(len(narrative_dictionary)) + " narratives.")
    
    return narrative_dictionary

# takes file path and column numbers of specified fields and returns dictionary
def read_fields(filepath_to_fields_folder, columns, classification_field_column):
    fields_dictionary = {}

    # reads each file
    for crash_file in os.listdir(filepath_to_fields_folder):
        if crash_file[-3:] == "csv":
            raw_file_data = pd.read_csv(filepath_to_fields_folder+"\\"+crash_file, skiprows=0, low_memory=False).values.tolist()

            for entry in raw_file_data:
                fields = []

                for field_column in columns:
                    fields.append(entry[field_column])
                
                fields.insert(0, entry[classification_field_column])

                fields_dictionary[entry[0]] = fields # adds fields to dictionary
    
    print("Loaded " + str(len(fields_dictionary)) + " crash fields.")

    return fields_dictionary
        
# combines narrative and fields dictionaries into one cohesive dictionary
def combine_fields_and_narratives(narrative_dictionary, fields_dictionary):
    dictionary = {}

    for crash in fields_dictionary.keys():
        if crash in narrative_dictionary:
            fields = fields_dictionary[crash]
            fields.append(narrative_dictionary[crash][0])
            dictionary[crash] = fields

    print("Loaded " + str(len(dictionary)) + " crashes into dictionary.")

    return dictionary

#returns all data in the form of a 2D array (each array is a separate field) 
def process_data_into_arrays(dictionary):
    ids = list(dictionary.keys())
    attributes = list(dictionary.values())

    # Get the number of attributes and cards
    num_attributes = len(attributes[0])
    num_cards = len(ids)

    # Create a 2D array to store the attributes
    attributes_array = [[None] * num_cards for _ in range(num_attributes)]

    # Iterate over each attribute and populate the array
    for i in range(num_attributes):
        for j in range(num_cards):
            attributes_array[i][j] = attributes[j][i]    

    return attributes_array

# samples big dictionary
def take_sample(fields_array, classifications_array, narratives_array, size):
    fields_sample_array = []

    for field in fields_array:
        fields_sample_array.append(field[0:size])

    print("Random sample of size "+str(size)+" taken successfully.")
    return fields_sample_array, classifications_array[0:size], narratives_array[0:size]

def running(narr,fields,tests,size,cv_value):
    start_time = time.time()
    spinner = Halo(text='', spinner='dots3')
    spinner.start()
    # testing
    file_path_to_narratives_file = narr
    file_path_to_fields_folder = fields

    field_columns = [14, 18, 27, 79, 83] # o (outside city limits), s (reported roadway system id), ab (toll), cb (highway sys), cf (street name)

    fields_array, classifications_array, narratives_array = load_data(file_path_to_narratives_file, file_path_to_fields_folder, field_columns, 71) # 71 is road class column

    fields_array, classifications_array, narratives_array = take_sample(fields_array, classifications_array, narratives_array, size) # Choosing 100 random tests from the narratives/classifications

    # print(fields_array, classifications_array, narratives_array)

    classes = ["blank","Interstate","US & State Highways", "Farm to Market", "County Road","City Street","Tollways","Other Roads", "Tollbridges","Non-Trafficway"]
    # Define the time interval to print elapsed time (20 seconds in this example)
    testing(narratives_array,classifications_array,fields_array,tests,classes,cv_value)
    spinner.stop()
    end_time = time.time()
    execution_time = end_time - start_time  # Calculate the difference
    print(f"Execution time: {execution_time} seconds")