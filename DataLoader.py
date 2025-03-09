import os, random, re
import pandas as pd
import csv

"""This program is ONLY used for compiling data into CSV format from the appropriate sources.
It is not necessary to the function of pipelines."""

class DataLoader:
    """Handles loading data into CSV files in preparation for preprocessing via combining all data into a dictionary by crash ID. """
    def __init__(self, 
                 narratives_filepath: str,
                 crash_fields_folder_filepath: str,
                 unit_fields_folder_filepath: str = "",
                 property_damages_folder_filepath: str = ""
                 ) -> None:
        """Builds self.dictionary via reading and combining dictionaries"""
        self.narratives_filepath = narratives_filepath
        self.crash_fields_filepath = crash_fields_folder_filepath
        self.unit_fields_filepath = unit_fields_folder_filepath
        self.property_damages_filepath = property_damages_folder_filepath

        # load all the data
        narrative_dictionary = self.read_narratives(self.narratives_filepath)
        crash_fields_dictionary = self.read_crash_fields(self.crash_fields_filepath)
        unit_fields_dictionary = self.read_unit_fields(self.unit_fields_filepath)
        property_damage_dictionary = self.read_property_damage(self.property_damages_filepath)

        # combine all the data
        self.dictionary = self.combine_data(narrative_dictionary, crash_fields_dictionary, property_damage_dictionary, unit_fields_dictionary)
        

    def combine_data(self, narratives_dict: dict[int, str], 
                    crash_fields_dict: dict[int, list[any]], 
                    damages_dict: dict[int, list[str]],
                    unit_fields_dict: dict[int, list[any]]) -> dict[int, dict[str, list[any]]]:
        """Combines all dictionaries by crash ID and filters out crashes/units with missing values.
        \nDictionary values take on the following format:
        \n``{"Investigator_Narrative": str, "crash fields": [any], "damages": [any], "unit fields":[any]}``
        """

        combined_dictionary = {}
        empty_crash_dict = {"crash fields": [], "unit fields":[], "damages": [], "Investigator_Narrative": ""}

        #narratives
        for crash in narratives_dict:
            combined_dictionary[crash] = empty_crash_dict.copy()
            combined_dictionary[crash]["Investigator_Narrative"] = narratives_dict[crash]

        #fields
        for crash in crash_fields_dict:
            if crash in combined_dictionary:
                combined_dictionary[crash]["crash fields"] = crash_fields_dict[crash]

        #damages
        for crash in damages_dict:
            if crash in combined_dictionary:
                combined_dictionary[crash]["damages"] = damages_dict[crash]

        #units
        for crash in unit_fields_dict:
            if crash in combined_dictionary:
                combined_dictionary[crash]["unit fields"] = unit_fields_dict[crash]
        
        # filter crashes with missing values. ensures crashes aren't simply a narrative.
        filtered_dictionary = {}
        for crash in combined_dictionary.keys():
            if not ((len(combined_dictionary[crash]["crash fields"]) == 0 and self.unit_fields_filepath != "") or 
                (len(combined_dictionary[crash]["unit fields"]) == 0 and self.unit_fields_filepath == "")):
                filtered_dictionary[crash] = combined_dictionary[crash]
        
        print("Loaded {} crashes into filtered dictionary.".format(len(filtered_dictionary.keys())))
        return filtered_dictionary


    def read_narratives(self, filepath_to_narratives: str) -> dict[int, str]:
        """Takes a filepath for a CSV file and processes narratives into a dictionary by ID. Assumes that crash ID is column index 0 and narratives are column index 1."""
        narrative_dictionary = {}

        #reads all data
        narrative_raw_data = pd.read_csv(filepath_to_narratives, encoding='windows-1252', skiprows=0).values.tolist()

        #enters formatted narratives into dictionary
        narrative_dictionary = {entry[0]: re.sub(r'{{.*?}}', '', entry[1].replace("\n", "").replace("\r", "").upper()) for entry in narrative_raw_data}
        
        print("Read " + str(len(narrative_dictionary)) + " narratives.")
        
        return narrative_dictionary


    def read_crash_fields(self, filepath_to_fields_folder: str) -> dict[int, list[any]]:
        """Takes a filepath for a CSV file and processes CRASH fields into a dictionary by ID. Assumes that crash ID is column index 0."""        
        fields_dictionary = {}

        csv_files = [unit_file for unit_file in os.listdir(filepath_to_fields_folder) if unit_file.endswith('.csv')]
        for crash_file in csv_files:
            raw_file_data = pd.read_csv(filepath_to_fields_folder+"\\"+crash_file, skiprows=0, low_memory=False).values.tolist()

            for row in raw_file_data:
                fields = {}
                for field_column in range(len(row)):
                    fields[field_column] = row[field_column]
                fields_dictionary[row[0]] = fields # adds fields to dictionary
        
        print("Read " + str(len(fields_dictionary)) + " crash fields.")
        return fields_dictionary


    def read_property_damage(self, filepath_to_property_damage_folder: str) -> dict[int, list[str]]:
        """Takes specified filepath and returns a dictionary with crash ID as keys and three property damages as values."""
        if filepath_to_property_damage_folder == "": 
            print("Filepath to property damages is empty, initializing dictionary as \{\}.")
            return {}
        
        property_damage_dictionary = {}
        csv_files = [property_damage_file for property_damage_file in os.listdir(filepath_to_property_damage_folder) if property_damage_file.endswith('.csv')]

        for property_damage_file in csv_files:
            raw_file_data = pd.read_csv(filepath_to_property_damage_folder+"\\"+property_damage_file, skiprows=0, low_memory=False).values.tolist()
            for row in raw_file_data:
                # checks if crash id is already in dictionary, appends if true
                if row[0] in property_damage_dictionary:
                    property_damage_dictionary[row[0]].append(row[1])
                else: property_damage_dictionary[row[0]] = [row[1]]

        print("Read " + str(len(property_damage_dictionary)) + " crashes with property damages.")
        return property_damage_dictionary


    def read_unit_fields(self, filepath_to_unit_fields_folder: str) -> dict[int, list[any]]:
        """Takes specified filepath and returns a dictionary with crash ID as keys and unit fields as values."""
        if filepath_to_unit_fields_folder == "": 
            print("Filepath to unit fields is empty, initializing dictionary as \{\}.")
            return {}
        
        unit_fields_dictionary = {}

        csv_files = [unit_file for unit_file in os.listdir(filepath_to_unit_fields_folder) if unit_file.endswith('.csv')]
        for unit_file in csv_files:
            raw_file_data = pd.read_csv(filepath_to_unit_fields_folder+"\\"+unit_file, skiprows=0, low_memory=False).values.tolist()            
            for row in raw_file_data:
                if row[0] in unit_fields_dictionary:
                    unit_fields_dictionary[row[0]].append([row[column] for column in range(len(row))]) 
                else: unit_fields_dictionary[row[0]] = [[row[column] for column in range(len(row))]]

        print("Read " + str(len(unit_fields_dictionary)) + " crashes with units.")
        return unit_fields_dictionary


    def export_crashes_to_csv(self, filepath: str) -> None:
        """Exports ``self.dictionary`` to specified filepath, with one crash per row."""
        with open(filepath, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the header row (column names)
            header = []
            for col_number in crash_column_number_dict.keys():
                if str(crash_column_number_dict[col_number]) != "Investigator_Narrative":
                    header.append(str(crash_column_number_dict[col_number]))
            for i in range(1,4):
                header.append(str("Damage_"+ str(i)))
            header.append("Investigator_Narrative")
            csv_writer.writerow(header)
            
            # write all rows
            for crash in list(self.dictionary.keys()):
                crash_row = []
                for field in self.dictionary[crash]["crash fields"]:
                    if str(crash_column_number_dict[field]) != "Investigator_Narrative":
                        crash_row.append(self.dictionary[crash]["crash fields"][field])
                for i in range(3):
                    try:
                        crash_row.append(self.dictionary[crash]["damages"][i])
                    except IndexError:
                        crash_row.append("")

                crash_row.append(self.dictionary[crash]["Investigator_Narrative"])
                csv_writer.writerow(crash_row)

 
    def export_units_to_csv(self, filepath: str) -> None:
        """Exports ``self.dictionary`` to specified filepath, with one unit per row."""
        with open(filepath, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            # Write the header row (column names)
            header = []
            for col_number in unit_column_number_dict.keys():
                header.append(str(unit_column_number_dict[col_number]))
            for col_number in crash_column_number_dict.keys():
                if str(crash_column_number_dict[col_number]) != "Investigator_Narrative" or str(crash_column_number_dict[col_number]) != "Crash_ID":
                    header.append(str(crash_column_number_dict[col_number]))
            for i in range(1,4):
                header.append(str("Damage_"+ str(i)))
            header.append("Investigator_Narrative")
            csv_writer.writerow(header)
            
            # write all rows
            for crash in list(self.dictionary.keys()):
                for unit in self.dictionary[crash]["unit fields"]:
                    unit_row = []   
                    for feature in unit:
                        unit_row.append(feature)
                    for field in self.dictionary[crash]["crash fields"]:
                        if str(crash_column_number_dict[field]) != "Investigator_Narrative" and field != 0:
                            unit_row.append(self.dictionary[crash]["crash fields"][field])
                    for i in range(3):
                        try:
                            unit_row.append(self.dictionary[crash]["damages"][i])
                        except IndexError:
                            unit_row.append("")

                    unit_row.append(self.dictionary[crash]["Investigator_Narrative"])
                    # unit_row.append("END ROW")
                    csv_writer.writerow(unit_row)


crash_column_number_dict = { 0: "Crash_ID", 1: "Crash_Fatal_Fl", 2: "Cmv_Involv_Fl", 3: "Schl_Bus_Fl", 4: "Rr_Relat_Fl", 5: "Medical_Advisory_Fl", 6: "Amend_Supp_Fl", 
                            7: "Active_School_Zone_Fl", 8: "Crash_Date", 9: "Crash_Time", 10: "Case_ID", 11: "Local_Use", 12: "Rpt_CRIS_Cnty_ID", 13: "Rpt_City_ID", 
                            14: "Rpt_Outside_City_Limit_Fl", 15: "Thousand_Damage_Fl", 16: "Rpt_Latitude", 17: "Rpt_Longitude", 18: "Rpt_Rdwy_Sys_ID", 19: "Rpt_Hwy_Num", 
                            20: "Rpt_Hwy_Sfx", 21: "Rpt_Road_Part_ID", 22: "Rpt_Block_Num", 23: "Rpt_Street_Pfx", 24: "Rpt_Street_Name", 25: "Rpt_Street_Sfx", 
                            26: "Private_Dr_Fl", 27: "Toll_Road_Fl", 28: "Crash_Speed_Limit", 29: "Road_Constr_Zone_Fl", 30: "Road_Constr_Zone_Wrkr_Fl", 31: "Rpt_Street_Desc", 
                            32: "At_Intrsct_Fl", 33: "Rpt_Sec_Rdwy_Sys_ID", 34: "Rpt_Sec_Hwy_Num", 35: "Rpt_Sec_Hwy_Sfx", 36: "Rpt_Sec_Road_Part_ID", 37: "Rpt_Sec_Block_Num", 
                            38: "Rpt_Sec_Street_Pfx", 39: "Rpt_Sec_Street_Name", 40: "Rpt_Sec_Street_Sfx", 41: "Rpt_Ref_Mark_Offset_Amt", 42: "Rpt_Ref_Mark_Dist_Uom", 
                            43: "Rpt_Ref_Mark_Dir", 44: "Rpt_Ref_Mark_Nbr", 45: "Rpt_Sec_Street_Desc", 46: "Rpt_CrossingNumber", 47: "Wthr_Cond_ID", 48: "Light_Cond_ID", 
                            49: "Entr_Road_ID", 50: "Road_Type_ID", 51: "Road_Algn_ID", 52: "Surf_Cond_ID", 53: "Traffic_Cntl_ID", 54: "Investigat_Notify_Time", 
                            55: "Investigat_Notify_Meth", 56: "Investigat_Arrv_Time", 57: "Report_Date", 58: "Investigat_Comp_Fl", 59: "ORI_Number", 60: "Investigat_Agency_ID", 
                            61: "Investigat_Area_ID", 62: "Investigat_District_ID", 63: "Investigat_Region_ID", 64: "Bridge_Detail_ID", 65: "Harm_Evnt_ID", 66: "Intrsct_Relat_ID", 
                            67: "FHE_Collsn_ID", 68: "Obj_Struck_ID", 69: "Othr_Factr_ID", 70: "Road_Part_Adj_ID", 71: "Road_Cls_ID", 72: "Road_Relat_ID", 73: "Phys_Featr_1_ID", 
                            74: "Phys_Featr_2_ID", 75: "Cnty_ID", 76: "City_ID", 77: "Latitude", 78: "Longitude", 79: "Hwy_Sys", 80: "Hwy_Nbr", 81: "Hwy_Sfx", 82: "Dfo", 
                            83: "Street_Name", 84: "Street_Nbr", 85: "Control", 86: "Section", 87: "Milepoint", 88: "Ref_Mark_Nbr", 89: "Ref_Mark_Displ", 90: "Hwy_Sys_2", 
                            91: "Hwy_Nbr_2", 92: "Hwy_Sfx_2", 93: "Street_Name_2", 94: "Street_Nbr_2", 95: "Control_2", 96: "Section_2", 97: "Milepoint_2", 98: "Txdot_Rptable_Fl", 
                            99: "Onsys_Fl", 100: "Rural_Fl", 101: "Crash_Sev_ID", 102: "Pop_Group_ID", 103: "Located_Fl", 104: "Day_of_Week", 105: "Hwy_Dsgn_Lane_ID", 
                            106: "Hwy_Dsgn_Hrt_ID", 107: "Hp_Shldr_Left", 108: "Hp_Shldr_Right", 109: "Hp_Median_Width", 110: "Base_Type_ID", 111: "Nbr_Of_Lane", 
                            112: "Row_Width_Usual", 113: "Roadbed_Width", 114: "Surf_Width", 115: "Surf_Type_ID", 116: "Curb_Type_Left_ID", 117: "Curb_Type_Right_ID", 
                            118: "Shldr_Type_Left_ID", 119: "Shldr_Width_Left", 120: "Shldr_Use_Left_ID", 121: "Shldr_Type_Right_ID", 122: "Shldr_Width_Right", 
                            123: "Shldr_Use_Right_ID", 124: "Median_Type_ID", 125: "Median_Width", 126: "Rural_Urban_Type_ID", 127: "Func_Sys_ID", 128: "Adt_Curnt_Amt", 
                            129: "Adt_Curnt_Year", 130: "Adt_Adj_Curnt_Amt", 131: "Pct_Single_Trk_Adt", 132: "Pct_Combo_Trk_Adt", 133: "Trk_Aadt_Pct", 134: "Curve_Type_ID", 
                            135: "Curve_Lngth", 136: "Cd_Degr", 137: "Delta_Left_Right_ID", 138: "Dd_Degr", 139: "Feature_Crossed", 140: "Structure_Number", 
                            141: "I_R_Min_Vert_Clear", 142: "Approach_Width", 143: "Bridge_Median_ID", 144: "Bridge_Loading_Type_ID", 145: "Bridge_Loading_In_1000_Lbs", 
                            146: "Bridge_Srvc_Type_On_ID", 147: "Bridge_Srvc_Type_Under_ID", 148: "Culvert_Type_ID", 149: "Roadway_Width", 150: "Deck_Width", 
                            151: "Bridge_Dir_Of_Traffic_ID", 152: "Bridge_Rte_Struct_Func_ID", 153: "Bridge_IR_Struct_Func_ID", 154: "CrossingNumber", 155: "RRCo", 
                            156: "Poscrossing_ID", 157: "WDCode_ID", 158: "Standstop", 159: "Yield", 160: "Sus_Serious_Injry_Cnt", 161: "Nonincap_Injry_Cnt", 162: "Poss_Injry_Cnt", 
                            163: "Non_Injry_Cnt", 164: "Unkn_Injry_Cnt", 165: "Tot_Injry_Cnt", 166: "Death_Cnt", 167: "MPO_ID", 168: "Investigat_Service_ID", 
                            169: "Investigat_DA_ID", 170: "Investigator_Narrative" }

unit_column_number_dict = { 0: "Crash_ID", 1: "Unit_Nbr", 2: "Unit_Desc_ID", 3: "Veh_Parked_Fl", 4: "Veh_HNR_Fl", 5: "Veh_Lic_State_ID", 6: "VIN", 7: "Veh_Mod_Year", 
                           8: "Veh_Color_ID", 9: "Veh_Make_ID", 10: "Veh_Mod_ID", 11: "Veh_Body_Styl_ID", 12: "Emer_Respndr_Fl", 13: "Ownr_Zip", 14: "Fin_Resp_Proof_ID", 
                           15: "Fin_Resp_Type_ID", 16: "Veh_Damage_Description1_Id", 17: "Veh_Damage_Severity1_Id", 18: "Veh_Damage_Direction_Of_Force1_Id", 
                           19: "Veh_Damage_Description2_Id", 20: "Veh_Damage_Severity2_Id", 21: "Veh_Damage_Direction_Of_Force2_Id", 22: "Veh_Inventoried_Fl", 
                           23: "Veh_Transp_Name", 24: "Veh_Transp_Dest", 25: "Veh_Cmv_Fl", 26: "Cmv_Fiveton_Fl", 27: "Cmv_Hazmat_Fl", 28: "Cmv_Nine_Plus_Pass_Fl", 
                           29: "Cmv_Veh_Oper_ID", 30: "Cmv_Carrier_ID_Type_ID", 31: "Cmv_Carrier_Zip", 32: "Cmv_Veh_Type_ID", 33: "Cmv_GVWR", 34: "Cmv_RGVW", 
                           35: "Cmv_Hazmat_Rel_Fl", 36: "Hazmat_Cls_1_ID", 37: "Hazmat_IDNbr_1_ID", 38: "Hazmat_Cls_2_ID", 39: "Hazmat_IDNbr_2_ID", 40: "Cmv_Cargo_Body_ID", 
                           41: "Cmv_Evnt1_ID", 42: "Cmv_Evnt2_ID", 43: "Cmv_Evnt3_ID", 44: "Cmv_Evnt4_ID", 45: "Cmv_Tot_Axle", 46: "Cmv_Tot_Tire", 47: "Contrib_Factr_1_ID", 
                           48: "Contrib_Factr_2_ID", 49: "Contrib_Factr_3_ID", 50: "Contrib_Factr_P1_ID", 51: "Contrib_Factr_P2_ID", 52: "Veh_Dfct_1_ID", 53: "Veh_Dfct_2_ID", 
                           54: "Veh_Dfct_3_ID", 55: "Veh_Dfct_P1_ID", 56: "Veh_Dfct_P2_ID", 57: "Veh_Trvl_Dir_ID", 58: "First_Harm_Evt_Inv_ID", 59: "Sus_Serious_Injry_Cnt", 
                           60: "Nonincap_Injry_Cnt", 61: "Poss_Injry_Cnt", 62: "Non_Injry_Cnt", 63: "Unkn_Injry_Cnt", 64: "Tot_Injry_Cnt", 65: "Death_Cnt", 
                           66: "Cmv_Disabling_Damage_Fl", 67: "Cmv_Bus_Type_ID", 68: "Trlr_GVWR", 69: "Trlr_RGVW", 70: "Trlr_Type_ID", 71: "Trlr_Disabling_Dmag_ID", 
                           72: "Cmv_Intermodal_Container_Permit_Fl", 73: "CMV_Actual_Gross_Weight", 74: "Pedestrian_Action_ID", 75: "Pedalcyclist_Action_ID", 
                           76: "PBCAT_Pedestrian_ID", 77: "PBCAT_Pedalcyclist_ID", 78: "E_Scooter_ID", 79: "Autonomous_Unit_ID" }


# # EXAMPLE USAGE
# loader = DataLoader(r"C:\Users\aseibel\Documents\investigator_narrative.csv",
#                     r"C:\Users\aseibel\Documents\extract_public_2023_20230629162130066_92029_20220101-20220630Texas\crash",
#                     r"C:\Users\aseibel\Documents\extract_public_2023_20230629162130066_92029_20220101-20220630Texas\unit",
#                     r"C:\Users\aseibel\Documents\extract_public_2023_20230629162130066_92029_20220101-20220630Texas\damages")

# loader.export_units_to_csv(r"C:\Users\aseibel\Documents\test.csv")