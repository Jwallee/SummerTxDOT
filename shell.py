from MultiPipelineHandler import MultiPipelineHandler
import pandas as pd

crash_training_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_1.csv"
crash_testing_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_2.csv"

unit_training_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_1.csv"
unit_testing_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_2.csv"

save_path = r"C:\Users\aseibel\Documents\txdot_interpreted_fields\modified.xml"
sample_crash_df = pd.read_csv(r"C:\Users\aseibel\Documents\data\crash_data_small.csv", encoding='windows-1252', dtype=str)
sample_unit_df = pd.read_csv(r"C:\Users\aseibel\Documents\data\unit_data_small.csv", encoding='windows-1252', dtype=str)

a = MultiPipelineHandler()

while(True):
    lines = input("iif> ").split(" ")

    if lines[0] == ":quit":
        print("Exiting shell")
        break
    elif lines[0] == ":setdf":
        if lines[1] == "-train":
            a.set_data_frames(lines[2], "")
        elif lines[1] == "-test":
            a.set_data_frames("", lines[2])
    elif lines[0] == ":load":
        if lines[1] == "-a":
            a.import_pipelines(r"C:\Users\aseibel\Documents\pipelines")
        else:
            a.import_pipelines(r"C:\Users\aseibel\Documents\pipelines")
    elif lines[0] == ":predict":
        if lines[1] == "-s":
            sample_crash = sample_crash_df.sample(1)
            sample_unit = sample_unit_df.sample(1)
            a.predict_crash(sample_crash, save_path)
    elif lines[0] == ":test":
        if lines[1] == "-a":
            a.test_pipelines(10000)
    else:
        print("Unrecognized command.")

