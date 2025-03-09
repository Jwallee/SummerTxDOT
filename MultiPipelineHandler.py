from PipelineHandler import PipelineHandler
from InterpretedField import InterpretedField
import pandas as pd
import xml.etree.ElementTree as ET
import csv
import time

class MultiPipelineHandler:
    """Handles all interpreted field PipelineHandlers for full crash prediction."""

    def __init__(self) -> None:
        self.all_handlers: dict[str, PipelineHandler] = {
            "RoadwayRelation": None, 
            "IntersectionRelated": None, 
            "RoadClass": None, 
            "FirstHarmfulEvent": None, 
            "MannerOfCollision": None, 
            "ObjectStruck": None, 
            "PhysicalFeature1": None, 
            "PhysicalFeature2": None,
            "BridgeDetail": None,
            "OtherFactor": None,
            "RoadwayPart": None,

            "FirstHarmfulEventInvolved": None,
            "DirectionOfTravel": None,
            "AutonomousUnit": None,
            "Escooter": None,
            "PBCATPedalcyclist": None,
            "PBCATPedestrian": None,
            "PedalcyclistAction": None,
            "PedestrianAction": None,
        }

        self.directions_lookup = {
            1: "NORTH",
            2: "NORTHEAST",
            3: "EAST",
            4: "SOUTHEAST",
            5: "SOUTH",
            6: "SOUTHWEST",
            7: "WEST",
            8: "NORTHWEST",
            9: "NOT APPLICABLE",
            11: "UNKNOWN"
        }


        self.crash_training_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_1.csv"
        self.crash_testing_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_2.csv"

        self.unit_training_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_1.csv"
        self.unit_testing_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_2.csv"


    def import_pipelines(self, path: str):
        """Imports a classifier for each interpreted field's PipelineHandler from the given folder path"""
        for handler_name in self.all_handlers:
            interpreted_field = InterpretedField(str(handler_name))
            if not interpreted_field.is_unit_based:
                handler = PipelineHandler(interpreted_field, self.crash_training_data_filepath, self.crash_testing_data_filepath)
            else:
                handler = PipelineHandler(interpreted_field, self.unit_training_data_filepath, self.unit_testing_data_filepath)
            
            try:
                filepath = path + "\\" + str(handler_name) + ".joblib"
                handler.import_from_filepath(filepath)
            except FileNotFoundError:
                print("Pre-built Pipeline not found for the given interpreted field: " + str(handler_name))
                print("You must train a new Pipeline for this field before being able to test and predict crashes.")
            
            self.all_handlers[handler_name] = handler


    def set_data_frames(self, training_df: str = "", testing_df: str = ""):
        """Sets data frames for all models to the specified filepaths in order to train/test."""
        for handler_name in self.all_handlers:
            handler = self.all_handlers[handler_name]
            handler.set_data_frames(training_df, testing_df)
            # try:
            #     handler.set_data_frames(training_df, testing_df)
            # except:
            #     print("Unable to set data frames for "+handler_name)


    def train_pipelines(self, size: int):
        """Re-trains all pipelines on the following metrics:\n
        ``size``: sample size of the training data"""
        for handler_name in self.all_handlers:
            handler = self.all_handlers[handler_name]
            try:
                handler.train(size)
            except:
                print("Unable to train Pipeline for the interpreted field " + str(handler.name))


    def test_pipelines(self, size: int):
        """Tests all pipeline accuracies using the specified data size and """
        csv_file = "Test-Results-"+time.strftime("%Y-%m-%d",time.localtime(time.time()))+".csv"

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Pipeline", "Base Accuracy", "Confidence Threshold", "Percent Confident", "Confidence Accuracy"])
            for handler_name in self.all_handlers:
                handler = self.all_handlers[handler_name]
                results = handler.test(size)
                results = list(results.values())
                results.insert(0, handler_name)
                writer.writerow(results)


    def predict_crash(self, crash: pd.DataFrame, out_filepath: str) -> bool:
        """Returns classifications as an XML file from a crash report as XML"""
        try:
            tree = ET.parse('template.xml')
            root = tree.getroot()

            crash_fields = self.predict_crash_fields(crash)
            for handler_name in crash_fields.keys():
                root.find(".//"+handler_name).set("ID", crash_fields[handler_name]["Prediction"])

            current_unit_id = 1
            for unit in crash.loc["units"]:
                unit_fields = self.predict_unit_fields(unit)
                for handler_name in unit_fields.keys():
                    root.find(".//"+handler_name).set("ID", unit_fields[handler_name]["Prediction"])

            tree.write(out_filepath)
            print("Predictions exported to "+out_filepath)
            
            return True
    
        except:
            print("Unable to predict crash.")
            return False


    def predict_crash_fields(self, crash: pd.DataFrame) -> dict[str, any]: 
        """Predicts all interpreted fields for a given crash. Saves predictions to a formatted .xml"""
        predictions = {}

        for handler_name in self.all_handlers:
            handler = self.all_handlers[handler_name]
            if not handler.interpreted_field.is_unit_based:
                try:
                    predictions[handler_name] = handler.predict(crash, 0.70)
                except:
                    predictions[handler_name] = {"Prediction": 87, "Is_Confident": False}
                    print("Unable to predict crash for " + str(handler.name))
        
        return predictions


    def predict_unit_fields(self, unit: pd.DataFrame) -> dict[str, any]:
        """Predicts all interpreted fields for a given unit. Saves predictions to a formatted .xml"""
        predictions = {}

        for handler_name in self.all_handlers:
            handler = self.all_handlers[handler_name]
            if handler.interpreted_field.is_unit_based:
                try:
                    predictions[handler_name] = handler.predict(unit, 0.70)
                except:
                    predictions[handler_name] = {"Prediction": 87, "Is_Confident": False}
                    print("Unable to predict unit for " + str(handler.name))

        return predictions
    

    def export_models_to_folder(self, path: str):
        """Saves all models to a single folder with the specified path."""
        for handler_name in self.all_handlers:
            handler = self.all_handlers[handler_name]
            filepath = path + "\\" + str(handler.name) + ".joblib"
            try:
                handler.export_to_filepath(filepath)
            except:
                print("Unable to test Pipeline for the interpreted field " + str(handler.name))


crash_training_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_1.csv"
crash_testing_data_filepath = r"C:\Users\aseibel\Documents\data\crash_data_2.csv"

unit_training_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_1.csv"
unit_testing_data_filepath = r"C:\Users\aseibel\Documents\data\unit_data_2.csv"

save_path = r"C:\Users\aseibel\Documents\txdot_interpreted_fields\modified.xml"
sample_crash_df = pd.read_csv(r"C:\Users\aseibel\Documents\data\crash_data_small.csv", encoding='windows-1252', dtype=str)
sample_unit_df = pd.read_csv(r"C:\Users\aseibel\Documents\data\unit_data_small.csv", encoding='windows-1252', dtype=str)

sample_crash = sample_crash_df.sample(1)
sample_unit = sample_unit_df.sample(1)

a = MultiPipelineHandler()
a.import_pipelines(r"C:\Users\aseibel\Documents\pipelines")

# a.predict_crash(sample_crash, save_path)
a.predict_crash_fields(sample_crash, save_path)

