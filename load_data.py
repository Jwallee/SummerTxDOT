### handles data preprocessing from large crash extracts in order to run ML algorithms
import os
import pandas as pd
import random
import time
import re 
import presets
from learn import run_model

# samples proportionally from arrays
def take_stratified_random_sample(fields_array: list[any], classifications_array: list[any], narratives_array: list[str], size: int):
    occurrences = {}
    for num in classifications_array:
        occurrences[num] = occurrences.get(num, 0) + 1

    print(occurrences)

    # Calculate the target number of occurrences for each number
    target_occurrences = size // len(occurrences.keys())

    # Stratified sampling
    sampled_list = []
    for num in occurrences.keys():
        num_occurrences = occurrences.get(num, 0)
        if num_occurrences > 0:
            sample_count = min(num_occurrences, target_occurrences)
            sampled_list.extend(random.sample([n for n in classifications_array if n == num], sample_count))

    print(sampled_list)


# samples random indices from arrays
def take_random_sample(fields_array: list[any], classifications_array: list[any], narratives_array: list[str], size: int):
    """
    Randomly samples fields, classifications, and narratives to a given sample size.
    """
    indices = list(range(len(narratives_array)))

    # Shuffle the indices randomly
    random.shuffle(indices)

    # Take the first n indices from the shuffled list
    random_indices = indices[:size]

    fields_sample_array = [[] for i in range(len(fields_array))]
    classifications_sample_array = []
    narratives_sample_array = []

    for index in random_indices:
        classifications_sample_array.append(classifications_array[index])
        narratives_sample_array.append(narratives_array[index])
        for i in range(len(fields_array)):
            fields_sample_array[i].append(fields_array[i][index])

    print("Random sample of size "+str(size)+" taken successfully.")
    return fields_sample_array, classifications_sample_array, narratives_sample_array


# MASTER METHOD
def load_data(narratives_filepath: str, 
              crash_fields_folder_filepath: str, 
              crash_field_columns: list[str], 
              classification_field_column: int, 
              should_load_property_damages: bool = False,
              damaged_property_filepath: str = "", 
              is_unit_based_interpreted_field: bool = False,
              unit_fields_filepath: str = "",
              unit_fields_columns: list[str] = []):
    """
    Takes all filepaths, booleans and column numbers as parameters and returns a 2D array of specified fields, classifications, 
    and narratives.\n
    Can process both crash-based and unit-based columns and classifications, as well as a combination so long as the classification
    field is a unit-based. This must be specified by the bool is_unit_based_interpreted_field, which is defaulted to False.\n
    This is the master data preprocessing method in which all other methods are called.
    """

    start = time.time()

    # load all the data
    narrative_dictionary = read_narratives(narratives_filepath)
    
    if is_unit_based_interpreted_field:
        fields_dictionary = read_crash_fields(crash_fields_folder_filepath, crash_field_columns)
        unit_fields_dictionary = read_unit_fields(unit_fields_filepath, unit_fields_columns, classification_field_column)
    else:
        fields_dictionary = read_crash_fields(crash_fields_folder_filepath, crash_field_columns, classification_field_column)
        unit_fields_dictionary = read_unit_fields(unit_fields_filepath, unit_fields_columns)
    
    if should_load_property_damages:
        property_damage_dictionary = read_property_damage(damaged_property_filepath)
    else: property_damage_dictionary = {}

    # combine all the data
    big_dictionary = combine_data(narrative_dictionary, fields_dictionary, property_damage_dictionary, unit_fields_dictionary)
    
    # different processing based on unit or crash
    if is_unit_based_interpreted_field:
        arrays_of_rows = process_unit_fields_into_arrays(big_dictionary, (len(crash_field_columns) == 0), should_load_property_damages)
    else:
        arrays_of_rows = process_crash_fields_into_arrays(big_dictionary, should_load_property_damages)

    print("Loaded data in", round(time.time()-start, 3), "seconds.")
    return arrays_of_rows[0], arrays_of_rows[1:-1], arrays_of_rows[-1] # classifications, fields (includes property damages), narratives


# combines narrative and fields dictionaries into one cohesive dictionary
def combine_data(narratives_dict: dict[int, str], 
                 crash_fields_dict: dict[int, list[any]], 
                 damages_dict: dict[int, list[str]],
                 unit_fields_dict: dict[int, list[any]]) -> dict[int, dict[str, list[any]]]:
    """
    \nCombines all dictionaries necessary for processing and returns a dictionary with crash IDs (str) as keys and dictionaries as values.
    \nDictionary values take on the following format:
    \n{"narrative": str, "crash fields": [any], "damages": [any], "unit fields":[any]}
    \nThis method is called immediately before process_data_into_arrays, which converts the dictionary into a 2D array for scikit-learn.
    """

    big_dictionary = {}
    empty_crash_dict = {"narrative": "", "crash fields": [], "damages": [], "unit fields":[]}

    #narratives
    for crash in narratives_dict:
        big_dictionary[crash] = empty_crash_dict.copy()
        big_dictionary[crash]["narrative"] = format_narrative(narratives_dict[crash][0].upper())

    #fields
    for crash in crash_fields_dict:
        if crash in big_dictionary:
            big_dictionary[crash]["crash fields"] = crash_fields_dict[crash]

    #damages
    for crash in damages_dict:
        if crash in big_dictionary:
            big_dictionary[crash]["damages"] = damages_dict[crash]

    #units
    for crash in unit_fields_dict:
        if crash in big_dictionary:
            big_dictionary[crash]["unit fields"] = unit_fields_dict[crash]
    
    print("Loaded " + str(len(big_dictionary)) + " crashes into dictionary.")
    return big_dictionary


# processes 2d array for CRASH analysis
def process_crash_fields_into_arrays(big_dictionary: dict[int, dict[str, list[any]]], should_process_damages: bool = False) -> list[list[any]]:
    """
    Takes a dictionary by crash ID and converts to a 2D array for CRASH-based interpreted fields. 
    Each ROW is a FIELD, and each COLUMN is a CRASH. This is an important distinction from
    process_unit_fields_into_arrays(), which processes unit-based fields.
    \nRow indices take on the following form:
    0: Classifications
    1 to -1: Fields, crash-based ONLY, followed by three indices for property damages if specified.
    -1 (last index): Narratives
    """
    
    crash_fields = [big_dictionary[id]["crash fields"] for id in big_dictionary.keys()]

    # removes all empty rows
    crash_fields = [row for row in crash_fields if any(row)]

    if should_process_damages:
        print("Loading crash fields WITH damages appended. Empty damages will be single space \" \"")

    for crash in crash_fields:
        # append DAMAGES
        if should_process_damages:
            for damage in range(len(big_dictionary[crash[0]]["damages"])):
                crash.append(big_dictionary[crash[0]]["damages"][damage]) # appends each damage in by index
            for _ in range(3 - len(big_dictionary[crash[0]]["damages"])): # ensures that three slots are used for damages regardless of true # so that narrative is the same index
                crash.append(" ")
        crash.append(big_dictionary[crash[0]]["narrative"])
        crash.pop(0)

    try:
        print("Loaded crash fields of following structure:", crash_fields[0])
    except IndexError:
        print("Error processing crash field data. Likely that is_unit_based_classification does not match column numbers.")

    #basically transposes crash_fields array and appends to big array
    big_array = []
    for row in zip(*crash_fields):
        big_array.append(list(row))
    
    return big_array


# processes 2d array for UNIT analysis
def process_unit_fields_into_arrays(big_dictionary: dict[int, dict[str, list[any]]], use_crash_fields: bool = False, should_process_damages: bool = False) -> list[list[any]]:
    """
    Takes a dictionary by crash ID and converts to a 2D array for UNIT-based interpreted fields. 
    Each ROW is a FIELD, and each COLUMN is a UNIT. This is an important distinction from
    process_crash_fields_into_arrays(), which processes crash-based fields only.
    This method still processes crash-based fields, but output indices represent singular units/vehicles.
    \nRow indices take on the following form:
    0: Classifications
    1 to -1: Fields, unit-based followed crash-based, followed by three indices for property damages if specified.
    -1 (last index): Narratives
    """
    
    units_by_crash = [big_dictionary[id]["unit fields"] for id in big_dictionary.keys()]
    units = [unit for crash in units_by_crash for unit in crash] # flattens units array to 1D
    
    if use_crash_fields:
        print("Loading unit fields WITH crash fields appended.")
    if should_process_damages:
        print("Loading unit fields WITH damages appended. Empty damages will be single space \" \"")

    # adds narratives
    for unit in units:
        # append CRASH FIELDS
        if use_crash_fields:
            for crash_field in range(len(big_dictionary[unit[0]]["crash fields"])):
                unit.append(crash_field)
        # append DAMAGES
        if should_process_damages:
            for damage in range(len(big_dictionary[unit[0]]["damages"])):
                unit.append(big_dictionary[unit[0]]["damages"][damage]) # appends each damage in by index
            for _ in range(3 - len(big_dictionary[unit[0]]["damages"])): # ensures that three slots are used for damages regardless of true # so that narrative is the same index
                unit.append(" ")
        unit.append(big_dictionary[unit[0]]["narrative"])
        unit.pop(0)

    print("If exception is thrown here it's likely that parameters are inconsistent.")
    print("Loaded crash fields of following structure:", units[0])

    # basically transposes unit_fields array and appends to big array 
    big_array = []
    for row in zip(*units):
        big_array.append(list(row))
    
    return big_array


# takes file path and returns narratives as dictionary via ID
def read_narratives(filepath_to_narratives: str) -> dict[int, str]:
    """
    Takes a filepath for a CSV file and processes narratives into a dictionary by ID.
    Assumes that crash ID is column index 0 and narratives are column index 1.
    """
    narrative_dictionary = {}

    #reads all data
    narrative_raw_data = pd.read_csv(filepath_to_narratives, encoding='windows-1252', skiprows=0).values.tolist()

    #enters data into dictionary
    for entry in narrative_raw_data:
        narrative_dictionary[entry[0]] = [entry[1]]
    
    print("Read " + str(len(narrative_dictionary)) + " narratives.")
    
    return narrative_dictionary


# takes file path and column numbers and returns dictionary via ID (classification column is FIRST INDEX)
def read_crash_fields(filepath_to_fields_folder: str, columns: list[int], classification_field_column: int = -1) -> dict[int, list[any]]:
    """
    Takes a filepath for a CSV file and processes CRASH fields into a dictionary by ID.
    Assumes that crash ID is column index 0. Classification column should be specified and columns is optional.
    """
    
    if filepath_to_fields_folder == "": 
        print("Unable to locate crash fields folder from filepath. Initializing with empty dictionary.")
        return {}
    fields_dictionary = {}

    columns.insert(0, 0)

    if classification_field_column != -1:
        columns.insert(1, classification_field_column)
        print("Indicated that classification is based on CRASH. Algorithm will use the following column:", classification_field_column)

    print("Reading crash field columns:", columns)


    csv_files = [unit_file for unit_file in os.listdir(filepath_to_fields_folder) if unit_file.endswith('.csv')]
    for crash_file in csv_files:
        raw_file_data = pd.read_csv(filepath_to_fields_folder+"\\"+crash_file, skiprows=0, low_memory=False).values.tolist()

        for row in raw_file_data:
            fields = [row[field_column] for field_column in columns]
            fields_dictionary[row[0]] = fields # adds fields to dictionary
    
    print("Read " + str(len(fields_dictionary)) + " crash fields.")
    return fields_dictionary


# takes filepath and returns 3-index list of property damages via ID
def read_property_damage(filepath_to_property_damage_folder: str) -> dict[int, list[str]]:
    """
    Takes specified filepath and returns a dictionary with crash ID as keys and property damages as values.
    Value formate is a list of three strings regardless of the number of property damages since scikit-learn requires a 
    static number of indices for its algorithm. 
    \n An example key/value pair would look as follows:
    1234567890: ['damage 1', 'damage 2', '']
    \n If less than damages exist, NAs are initialized as empty strings, such as above.
    """
    if filepath_to_property_damage_folder == "": 
        print("Unable to locate property damages folder from filepath. Initializing with empty dictionary.")
        return {}
    print("Loading property damages from specified filepath.")
    property_damage_dictionary = {}
    csv_files = [property_damage_file for property_damage_file in os.listdir(filepath_to_property_damage_folder) if property_damage_file.endswith('.csv')]

    for property_damage_file in csv_files:
        raw_file_data = pd.read_csv(filepath_to_property_damage_folder+"\\"+property_damage_file, 
                                    skiprows=0, low_memory=False).values.tolist()
        for row in raw_file_data:
            # checks if crash id is already in dictionary, appends if true
            if row[0] in property_damage_dictionary:
                property_damage_dictionary[row[0]].append(row[1])
            else: property_damage_dictionary[row[0]] = [row[1]]

    print("Read " + str(len(property_damage_dictionary)) + " crashes with property damages.")
    return property_damage_dictionary


# takes filepath and column numbers and returns dictionary via ID
def read_unit_fields(filepath_to_unit_fields_folder: str, columns: list[any], classification_field_column: int = -1) -> dict[int, list[any]]:
    """
    Takes specified filepath and returns a dictionary with crash ID as keys and unit fields as values.
    """
    
    if filepath_to_unit_fields_folder == "": 
        print("Unable to locate unit fields folder from filepath. Initializing with empty dictionary.")
        return {}
    unit_fields_dictionary = {}

    columns.insert(0, 0)

    if classification_field_column != -1:
        columns.insert(1, classification_field_column)
        print("Indicated that classification is based on UNIT. Algorithm will use the following column:", classification_field_column)

    print("Reading unit field columns:", columns)

    # finds all csv files in foler filepath
    csv_files = [unit_file for unit_file in os.listdir(filepath_to_unit_fields_folder) if unit_file.endswith('.csv')]

    for unit_file in csv_files:
        raw_file_data = pd.read_csv(filepath_to_unit_fields_folder+"\\"+unit_file, skiprows=0, low_memory=False).values.tolist()            
        for row in raw_file_data:
            # {id: [[unit_1 ],[unit_2]...,[unit_k]]}
            if row[0] in unit_fields_dictionary:
                unit_fields_dictionary[row[0]].append([row[column] for column in columns]) 
            else: unit_fields_dictionary[row[0]] = [[row[column] for column in columns]]

    print("Read " + str(len(unit_fields_dictionary)) + " crashes with units.")
    return unit_fields_dictionary


# removes unwanted characters and investigator assignments
def format_narrative(narrative: str) -> str:
    """
    Removes narratives of excess special characters "\\r" as well as anything enclosed within "{{----}}" braces, since agencies often use
    this notation for investigator assignments and are irrelevant to the crash at hand.
    """
    # Remove \r characters
    narrative = narrative.replace('\r', '').replace('\n', '')

    # Remove content within {{}}
    narrative = re.sub(r'\{\{.*?\}\}', '', narrative)

    return narrative


# tests methods
def running(file_path_to_narratives_file, file_path_to_fields_folder, file_path_to_damages_folder, file_path_to_units_folder,sample_size,test_percent,cv_val,model_folder_path,interp_field):
    """
    Tests methods in the load_data module to ensure proper functionality.
    """
    import os
    import datetime
    print("Made it to running")
    print("Narratives file path:", file_path_to_narratives_file)
    print("Fields folder path:", file_path_to_fields_folder)
    print("Damages folder path:", file_path_to_damages_folder)
    print("Units folder path:", file_path_to_units_folder)
    print("Sample size:", sample_size)
    print("Test percent:", test_percent)
    print("CV value:", cv_val)
    
    field_columns = presets.road_class_columns
    classification_column = presets.road_class_field_number
    unit_field_columns = []
    is_unit_based_classification = False
    should_load_property_damages = False

    classifications_array, fields_array, narratives_array = load_data(file_path_to_narratives_file, 
                                                                    file_path_to_fields_folder, 
                                                                    field_columns, 
                                                                    classification_column, 
                                                                    should_load_property_damages,
                                                                    file_path_to_damages_folder,
                                                                    is_unit_based_classification, 
                                                                    file_path_to_units_folder,
                                                                    unit_field_columns)

    fields_array, classifications_array, narratives_array = take_random_sample(fields_array, classifications_array, narratives_array, sample_size)
    if model_folder_path == "": 
        print("Model not provided. Creating new model.")
        "Create model folder"
        model_folder_path = "models"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        model_path = "models\\"+ str(interp_field) + "_" + str(sample_size) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".joblib"
        
        
        run_model(sample_size, test_percent, cv_val, fields_array, classifications_array, narratives_array, model_path, model_load=False)
    else:
        print("Model provided. Loading model.")
        run_model(sample_size, test_percent, cv_val, fields_array, classifications_array, narratives_array, model_folder_path, model_load=True)
