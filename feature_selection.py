from PipelineHandler import PipelineHandler
from InterpretedField import InterpretedField, is_unit_based, crash_columns, unit_columns
import random
import time

t = time.time()

data = {}

for field_name in is_unit_based: 
    print("---------------------------------", field_name, "---------------------------------")
    field = InterpretedField(field_name)
    data[field_name] = {}
    feature_columns = []

    if is_unit_based[field_name] == True:
        feature_columns = unit_columns.copy()
        t1 = r"C:\Users\aseibel\Documents\demo\crash_data\unit_data_1.csv"
        t2 = r"C:\Users\aseibel\Documents\demo\crash_data\unit_data_2.csv"
    else: 
        feature_columns = crash_columns.copy()
        t1 = r"C:\Users\aseibel\Documents\demo\crash_data\crash_data_1.csv"
        t2 = r"C:\Users\aseibel\Documents\demo\crash_data\crash_data_2.csv"
    
    if field_name == "pbcat_pedestrian" or field_name == "pbcat_pedalcyclist":
        while len(feature_columns) > 0:
            columns = []
            for _ in range(5):
                try:
                    rand = random.choice(list(feature_columns.keys()))
                    columns.append(feature_columns.pop(rand))
                except IndexError:
                    break
            field.setFeatures(columns)

            handler = PipelineHandler(field, t1, t2, 100, 100)
                
            handler.train()
            base, thresh, percent = handler.test()
            result, importances = handler.calculate_feature_importances()
            
            for feature, importance in importances.items():
                data[field_name][feature] = [importance, base, thresh, percent]


            file_path = str(field_name)+'.csv'

            # Open the file in write mode and write the feature importances
            with open(file_path, 'w') as file:
                for feature in data[field_name]:
                    line = str(feature)+","
                    for value in data[field_name][feature]:
                        line += str(value) + ","
                    line += "\n"
                    file.write(line)

            print("Feature importances written to the "+str(field_name)+" file.")
            print("----------------------------------------")

            print("Completed in " + str(round(time.time() - t, 2)) + " seconds.")

print(data)