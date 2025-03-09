class InterpretedField:
    """An object representation of an interpreted field. All class attributes are publicly accessible and are created at init.\n \n
    Initialization
    ----------
    Make sure that the input string is formatted properly and matches one of the keys in the dictionaries of this file.\n 
    Examples:\n 
    ``a = InterpretedField("IntersectionRelated")``. We can then used the InterpretedField for a pipeline via\n 
    ``b = PipelineHandler(a)`` \n 
    or simply\n 
    ``b = PipelineHandler(InterpretedField("IntersectionRelated"))``\n \n 
    Attributes
    ----------
    ``name: str`` is formatted lowercase with an underscore between words, such as "IntersectionRelated".\n 
    ``vectorized_features: list(str)`` is list of the column names in a csv that must be vectorized in preprocessing.\n 
    ``encoded_features: list(str)`` is list of the column names in a csv that must be encoded in preprocessing.\n 
    ``classification_feature: str`` is the column name of the feature being classified, such as Intrsct_Relat_ID.\n 
    ``is_unit_based: bool`` indicates whether the interpreted field is based on units or the entire crash.\n 
    """
    def __init__(self, name: str) -> None:
        self.name = name
        try:
            self.vectorized_features = vectorized_features[name]
            self.encoded_features = encoded_features[name]
            self.classification_feature = classification_feature[name]
            self.is_unit_based = is_unit_based[name]
        except KeyError:
            print("Error: key \"{}\" not found in one of the specified preset dictionaries. Initializing empty attributes.".format(name))
            self.vectorized_features = None
            self.encoded_features = None
            self.classification_feature = None
            self.is_unit_based = None
    

    def setRandomFeatures(self, feature_list: list[str]):
        import random
        self.encoded_features = []
        self.vectorized_features = []
        while len(self.encoded_features) + len(self.vectorized_features) < 5:
            column = random.choice(list(feature_list.values()))
            if column != self.classification_feature:
                if column == "Investigator_Narrative" or column == "Damage_1" or column == "Damage_2" or column == "Damage_3":
                    self.vectorized_features.append(column)
                else:
                    self.encoded_features.append(column)
        print(set(self.vectorized_features) | set(self.encoded_features))
        return set(self.vectorized_features) | set(self.encoded_features)
    
    
    def setFeatures(self, feature_list: list[str]):
        self.encoded_features = []
        self.vectorized_features = []
        for column in feature_list:
            if column != self.classification_feature:
                if column == "Investigator_Narrative" or column == "Damage_1" or column == "Damage_2" or column == "Damage_3":
                    self.vectorized_features.append(column)
                else:
                    self.encoded_features.append(column)
        print(set(self.vectorized_features) | set(self.encoded_features))
        return set(self.vectorized_features) | set(self.encoded_features)


vectorized_features = {
    "RoadwayRelation": ["Investigator_Narrative"],
    "IntersectionRelated": ["Investigator_Narrative"],
    "RoadClass": ["Investigator_Narrative"],
    "FirstHarmfulEvent": ["Investigator_Narrative"],
    "MannerOfCollision": ["Investigator_Narrative"],
    "ObjectStruck": ["Investigator_Narrative"],
    "PhysicalFeature1": ["Investigator_Narrative"],
    "PhysicalFeature2": ["Investigator_Narrative"],
    "BridgeDetail": ["Investigator_Narrative"],
    "OtherFactor": ["Investigator_Narrative"],
    "RoadwayPart": ["Investigator_Narrative"],

    "FirstHarmfulEventInvolved": ["Investigator_Narrative"],
    "DirectionOfTravel": ["Investigator_Narrative"],
    "AutonomousUnit": ["Investigator_Narrative"],
    "Escooter": ["Investigator_Narrative"],
    "PBCATPedalcyclist": ["Investigator_Narrative"],
    "PBCATPedestrian": ["Investigator_Narrative"],
    "PedalcyclistAction": ["Investigator_Narrative"],
    "PedestrianAction": ["Investigator_Narrative"],
}

encoded_features = {
    "RoadwayRelation": ["Crash_Speed_Limit"],
    "IntersectionRelated": ["Street_Name_2", "At_Intrsct_Fl"],
    "RoadClass": ["Rpt_Rdwy_Sys_ID", "Rpt_Hwy_Num", "Hwy_Sys", "Street_Name"],
    "FirstHarmfulEvent": [],
    "MannerOfCollision": [],
    "ObjectStruck": ["Damage_1"],
    "PhysicalFeature1": [],
    "PhysicalFeature2": [],
    "BridgeDetail": ["Damage_1"],
    "OtherFactor": [],
    "RoadwayPart": ["Rpt_Road_Part_ID"],

    "FirstHarmfulEventInvolved": ["Unit_Nbr", "Veh_Parked_Fl", "Veh_Damage_Description1_Id", "Veh_Damage_Direction_Of_Force1_Id"],
    "DirectionOfTravel": [],
    "AutonomousUnit": ["Veh_Make_ID"],
    "Escooter": ["Veh_Make_ID"],
    "PBCATPedalcyclist": [],
    "PBCATPedestrian": [],
    "PedalcyclistAction": [],
    "PedestrianAction": [],
}

classification_feature = {
    "RoadwayRelation": "Road_Relat_ID",
    "IntersectionRelated": "Intrsct_Relat_ID",
    "RoadClass": "Road_Cls_ID",
    "FirstHarmfulEvent": "Harm_Evnt_ID",
    "MannerOfCollision": "FHE_Collsn_ID",
    "ObjectStruck": "Obj_Struck_ID",
    "PhysicalFeature1": "Phys_Featr_1_ID",
    "PhysicalFeature2": "Phys_Featr_2_ID",
    "BridgeDetail": "Bridge_Detail_ID",
    "OtherFactor": "Othr_Factr_ID",
    "RoadwayPart": "Road_Part_Adj_ID",

    "FirstHarmfulEventInvolved": "First_Harm_Evt_Inv_ID",
    "DirectionOfTravel": "Veh_Trvl_Dir_ID",
    "AutonomousUnit": "E_Scooter_ID",
    "Escooter": "E_Scooter_ID",
    "PBCATPedalcyclist": "PBCAT_Pedalcyclist_ID",
    "PBCATPedestrian": "PBCAT_Pedestrian_ID",
    "PedalcyclistAction": "Pedalcyclist_Action_ID",
    "PedestrianAction": "Pedestrian_Action_ID",
}

is_unit_based = {
    "RoadwayRelation": False,
    "IntersectionRelated": False,
    "RoadClass": False,
    "FirstHarmfulEvent": False,
    "MannerOfCollision": False,
    "ObjectStruck": False,
    "PhysicalFeature1": False,
    "PhysicalFeature2": False,
    "BridgeDetail": False,
    "OtherFactor": False,
    "RoadwayPart": False,

    "FirstHarmfulEventInvolved": True,
    "DirectionOfTravel": True,
    "AutonomousUnit": True,
    "Escooter": True,
    "PBCATPedalcyclist": True,
    "PBCATPedestrian": True,
    "PedalcyclistAction": True,
    "PedestrianAction": True,
}

crash_columns = {
    0: 'Crash_ID', 1: 'Crash_Fatal_Fl', 2: 'Cmv_Involv_Fl', 3: 'Schl_Bus_Fl', 4: 'Rr_Relat_Fl', 5: 'Medical_Advisory_Fl', 6: 'Amend_Supp_Fl', 7: 'Active_School_Zone_Fl',
      8: 'Crash_Date', 9: 'Crash_Time', 10: 'Case_ID', 11: 'Local_Use', 12: 'Rpt_CRIS_Cnty_ID', 13: 'Rpt_City_ID', 14: 'Rpt_Outside_City_Limit_Fl', 15: 'Thousand_Damage_Fl',
        16: 'Rpt_Latitude', 17: 'Rpt_Longitude', 18: 'Rpt_Rdwy_Sys_ID', 19: 'Rpt_Hwy_Num', 20: 'Rpt_Hwy_Sfx', 21: 'Rpt_Road_Part_ID', 22: 'Rpt_Block_Num', 23: 'Rpt_Street_Pfx',
          24: 'Rpt_Street_Name', 25: 'Rpt_Street_Sfx', 26: 'Private_Dr_Fl', 27: 'Toll_Road_Fl', 28: 'Crash_Speed_Limit', 29: 'Road_Constr_Zone_Fl', 30: 'Road_Constr_Zone_Wrkr_Fl',
            31: 'Rpt_Street_Desc', 32: 'At_Intrsct_Fl', 33: 'Rpt_Sec_Rdwy_Sys_ID', 34: 'Rpt_Sec_Hwy_Num', 35: 'Rpt_Sec_Hwy_Sfx', 36: 'Rpt_Sec_Road_Part_ID', 37: 'Rpt_Sec_Block_Num',
              38: 'Rpt_Sec_Street_Pfx', 39: 'Rpt_Sec_Street_Name', 40: 'Rpt_Sec_Street_Sfx', 41: 'Rpt_Ref_Mark_Offset_Amt', 42: 'Rpt_Ref_Mark_Dist_Uom', 43: 'Rpt_Ref_Mark_Dir',
                44: 'Rpt_Ref_Mark_Nbr', 45: 'Rpt_Sec_Street_Desc', 46: 'Rpt_CrossingNumber', 47: 'Wthr_Cond_ID', 48: 'Light_Cond_ID', 49: 'Entr_Road_ID', 50: 'Road_Type_ID',
                  51: 'Road_Algn_ID', 52: 'Surf_Cond_ID', 53: 'Traffic_Cntl_ID', 54: 'Investigat_Notify_Time', 55: 'Investigat_Notify_Meth', 56: 'Investigat_Arrv_Time', 
                  57: 'Report_Date', 58: 'Investigat_Comp_Fl', 59: 'ORI_Number', 60: 'Investigat_Agency_ID', 61: 'Investigat_Area_ID', 62: 'Investigat_District_ID', 
                  63: 'Investigat_Region_ID', 64: 'Bridge_Detail_ID', 65: 'Harm_Evnt_ID', 66: 'Intrsct_Relat_ID', 67: 'FHE_Collsn_ID', 68: 'Obj_Struck_ID', 69: 'Othr_Factr_ID', 
                  70: 'Road_Part_Adj_ID', 71: 'Road_Cls_ID', 72: 'Road_Rel`at_ID', 73: 'Phys_Featr_1_ID', 74: 'Phys_Featr_2_ID', 75: 'Cnty_ID', 76: 'City_ID', 77: 'Latitude', 
                  78: 'Longitude', 79: 'Hwy_Sys', 80: 'Hwy_Nbr', 81: 'Hwy_Sfx', 82: 'Dfo', 83: 'Street_Name', 84: 'Street_Nbr', 85: 'Control', 86: 'Section', 87: 'Milepoint', 
                  88: 'Ref_Mark_Nbr', 89: 'Ref_Mark_Displ', 90: 'Hwy_Sys_2', 91: 'Hwy_Nbr_2', 92: 'Hwy_Sfx_2', 93: 'Street_Name_2', 94: 'Street_Nbr_2', 95: 'Control_2', 
                  96: 'Section_2', 97: 'Milepoint_2', 98: 'Txdot_Rptable_Fl', 99: 'Onsys_Fl', 100: 'Rural_Fl', 101: 'Crash_Sev_ID', 102: 'Pop_Group_ID', 103: 'Located_Fl', 
                  104: 'Day_of_Week', 105: 'Hwy_Dsgn_Lane_ID', 106: 'Hwy_Dsgn_Hrt_ID', 107: 'Hp_Shldr_Left', 108: 'Hp_Shldr_Right', 109: 'Hp_Median_Width', 110: 'Base_Type_ID', 
                  111: 'Nbr_Of_Lane', 112: 'Row_Width_Usual', 113: 'Roadbed_Width', 114: 'Surf_Width', 115: 'Surf_Type_ID', 116: 'Curb_Type_Left_ID', 117: 'Curb_Type_Right_ID', 
                  118: 'Shldr_Type_Left_ID', 119: 'Shldr_Width_Left', 120: 'Shldr_Use_Left_ID', 121: 'Shldr_Type_Right_ID', 122: 'Shldr_Width_Right', 123: 'Shldr_Use_Right_ID', 
                  124: 'Median_Type_ID', 125: 'Median_Width', 126: 'Rural_Urban_Type_ID', 127: 'Func_Sys_ID', 128: 'Adt_Curnt_Amt', 129: 'Adt_Curnt_Year', 130: 'Adt_Adj_Curnt_Amt', 
                  131: 'Pct_Single_Trk_Adt', 132: 'Pct_Combo_Trk_Adt', 133: 'Trk_Aadt_Pct', 134: 'Curve_Type_ID', 135: 'Curve_Lngth', 136: 'Cd_Degr', 137: 'Delta_Left_Right_ID', 
                  138: 'Dd_Degr', 139: 'Feature_Crossed', 140: 'Structure_Number', 141: 'I_R_Min_Vert_Clear', 142: 'Approach_Width', 143: 'Bridge_Median_ID', 
                  144: 'Bridge_Loading_Type_ID', 145: 'Bridge_Loading_In_1000_Lbs', 146: 'Bridge_Srvc_Type_On_ID', 147: 'Bridge_Srvc_Type_Under_ID', 148: 'Culvert_Type_ID', 
                  149: 'Roadway_Width', 150: 'Deck_Width', 151: 'Bridge_Dir_Of_Traffic_ID', 152: 'Bridge_Rte_Struct_Func_ID', 153: 'Bridge_IR_Struct_Func_ID', 154: 'CrossingNumber', 
                  155: 'RRCo', 156: 'Poscrossing_ID', 157: 'WDCode_ID', 158: 'Standstop', 159: 'Yield', 160: 'Sus_Serious_Injry_Cnt', 161: 'Nonincap_Injry_Cnt', 162: 'Poss_Injry_Cnt', 
                  163: 'Non_Injry_Cnt', 164: 'Unkn_Injry_Cnt', 165: 'Tot_Injry_Cnt', 166: 'Death_Cnt', 167: 'MPO_ID', 168: 'Investigat_Service_ID', 169: 'Damage_1', 
                  170: 'Damage_2', 171: 'Damage_3', 172: 'Investigator_Narrative'
}

unit_columns = {0: 'Crash_ID', 1: 'Unit_Nbr', 2: 'Unit_Desc_ID', 3: 'Veh_Parked_Fl', 4: 'Veh_HNR_Fl', 5: 'Veh_Lic_State_ID', 6: 'VIN', 7: 'Veh_Mod_Year', 8: 'Veh_Color_ID', 
                9: 'Veh_Make_ID', 10: 'Veh_Mod_ID', 11: 'Veh_Body_Styl_ID', 12: 'Emer_Respndr_Fl', 13: 'Ownr_Zip', 14: 'Fin_Resp_Proof_ID', 15: 'Fin_Resp_Type_ID', 
                16: 'Veh_Damage_Description1_Id', 17: 'Veh_Damage_Severity1_Id', 18: 'Veh_Damage_Direction_Of_Force1_Id', 19: 'Veh_Damage_Description2_Id', 
                20: 'Veh_Damage_Severity2_Id', 21: 'Veh_Damage_Direction_Of_Force2_Id', 22: 'Veh_Inventoried_Fl', 23: 'Veh_Transp_Name', 24: 'Veh_Transp_Dest', 25: 'Veh_Cmv_Fl', 
                26: 'Cmv_Fiveton_Fl', 27: 'Cmv_Hazmat_Fl', 28: 'Cmv_Nine_Plus_Pass_Fl', 29: 'Cmv_Veh_Oper_ID', 30: 'Cmv_Carrier_ID_Type_ID', 31: 'Cmv_Carrier_Zip', 
                32: 'Cmv_Veh_Type_ID', 33: 'Cmv_GVWR', 34: 'Cmv_RGVW', 35: 'Cmv_Hazmat_Rel_Fl', 36: 'Hazmat_Cls_1_ID', 37: 'Hazmat_IDNbr_1_ID', 38: 'Hazmat_Cls_2_ID', 
                39: 'Hazmat_IDNbr_2_ID', 40: 'Cmv_Cargo_Body_ID', 41: 'Cmv_Evnt1_ID', 42: 'Cmv_Evnt2_ID', 43: 'Cmv_Evnt3_ID', 44: 'Cmv_Evnt4_ID', 45: 'Cmv_Tot_Axle', 
                46: 'Cmv_Tot_Tire', 47: 'Contrib_Factr_1_ID', 48: 'Contrib_Factr_2_ID', 49: 'Contrib_Factr_3_ID', 50: 'Contrib_Factr_P1_ID', 51: 'Contrib_Factr_P2_ID', 
                52: 'Veh_Dfct_1_ID', 53: 'Veh_Dfct_2_ID', 54: 'Veh_Dfct_3_ID', 55: 'Veh_Dfct_P1_ID', 56: 'Veh_Dfct_P2_ID', 57: 'Veh_Trvl_Dir_ID', 58: 'First_Harm_Evt_Inv_ID', 
                59: 'Sus_Serious_Injry_Cnt', 60: 'Nonincap_Injry_Cnt', 61: 'Poss_Injry_Cnt', 62: 'Non_Injry_Cnt', 63: 'Unkn_Injry_Cnt', 64: 'Tot_Injry_Cnt', 65: 'Death_Cnt', 
                66: 'Cmv_Disabling_Damage_Fl', 67: 'Cmv_Bus_Type_ID', 68: 'Trlr_GVWR', 69: 'Trlr_RGVW', 70: 'Trlr_Type_ID', 71: 'Trlr_Disabling_Dmag_ID', 
                72: 'Cmv_Intermodal_Container_Permit_Fl', 73: 'CMV_Actual_Gross_Weight', 74: 'Pedestrian_Action_ID', 75: 'Pedalcyclist_Action_ID', 76: 'PBCAT_Pedestrian_ID', 
                77: 'PBCAT_Pedalcyclist_ID', 78: 'E_Scooter_ID', 79: 'Crash_Fatal_Fl', 80: 'Cmv_Involv_Fl', 81: 'Schl_Bus_Fl', 82: 'Rr_Relat_Fl', 83: 'Medical_Advisory_Fl', 
                84: 'Amend_Supp_Fl', 85: 'Active_School_Zone_Fl', 86: 'Crash_Date', 87: 'Crash_Time', 88: 'Case_ID', 89: 'Local_Use', 90: 'Rpt_CRIS_Cnty_ID', 91: 'Rpt_City_ID', 
                92: 'Rpt_Outside_City_Limit_Fl', 93: 'Thousand_Damage_Fl', 94: 'Rpt_Latitude', 95: 'Rpt_Longitude', 96: 'Rpt_Rdwy_Sys_ID', 97: 'Rpt_Hwy_Num', 98: 'Rpt_Hwy_Sfx', 
                99: 'Rpt_Road_Part_ID', 100: 'Rpt_Block_Num', 101: 'Rpt_Street_Pfx', 102: 'Rpt_Street_Name', 103: 'Rpt_Street_Sfx', 104: 'Private_Dr_Fl', 105: 'Toll_Road_Fl', 
                106: 'Crash_Speed_Limit', 107: 'Road_Constr_Zone_Fl', 108: 'Road_Constr_Zone_Wrkr_Fl', 109: 'Rpt_Street_Desc', 110: 'At_Intrsct_Fl', 111: 'Rpt_Sec_Rdwy_Sys_ID', 
                112: 'Rpt_Sec_Hwy_Num', 113: 'Rpt_Sec_Hwy_Sfx', 114: 'Rpt_Sec_Road_Part_ID', 115: 'Rpt_Sec_Block_Num', 116: 'Rpt_Sec_Street_Pfx', 117: 'Rpt_Sec_Street_Name', 
                118: 'Rpt_Sec_Street_Sfx', 119: 'Rpt_Ref_Mark_Offset_Amt', 120: 'Rpt_Ref_Mark_Dist_Uom', 121: 'Rpt_Ref_Mark_Dir', 122: 'Rpt_Ref_Mark_Nbr', 123: 'Rpt_Sec_Street_Desc', 
                124: 'Rpt_CrossingNumber', 125: 'Wthr_Cond_ID', 126: 'Light_Cond_ID', 127: 'Entr_Road_ID', 128: 'Road_Type_ID', 129: 'Road_Algn_ID', 130: 'Surf_Cond_ID', 
                131: 'Traffic_Cntl_ID', 132: 'Investigat_Notify_Time', 133: 'Investigat_Notify_Meth', 134: 'Investigat_Arrv_Time', 135: 'Report_Date', 136: 'Investigat_Comp_Fl', 
                137: 'ORI_Number', 138: 'Investigat_Agency_ID', 139: 'Investigat_Area_ID', 140: 'Investigat_District_ID', 141: 'Investigat_Region_ID', 142: 'Bridge_Detail_ID', 
                143: 'Harm_Evnt_ID', 144: 'Intrsct_Relat_ID', 145: 'FHE_Collsn_ID', 146: 'Obj_Struck_ID', 147: 'Othr_Factr_ID', 148: 'Road_Part_Adj_ID', 149: 'Road_Cls_ID', 
                150: 'Road_Relat_ID', 151: 'Phys_Featr_1_ID', 152: 'Phys_Featr_2_ID', 153: 'Cnty_ID', 154: 'City_ID', 155: 'Latitude', 156: 'Longitude', 157: 'Hwy_Sys', 
                158: 'Hwy_Nbr', 159: 'Hwy_Sfx', 160: 'Dfo', 161: 'Street_Name', 162: 'Street_Nbr', 163: 'Control', 164: 'Section', 165: 'Milepoint', 166: 'Ref_Mark_Nbr', 
                167: 'Ref_Mark_Displ', 168: 'Hwy_Sys_2', 169: 'Hwy_Nbr_2', 170: 'Hwy_Sfx_2', 171: 'Street_Name_2', 172: 'Street_Nbr_2', 173: 'Control_2', 174: 'Section_2', 
                175: 'Milepoint_2', 176: 'Txdot_Rptable_Fl', 177: 'Onsys_Fl', 178: 'Rural_Fl', 179: 'Crash_Sev_ID', 180: 'Pop_Group_ID', 181: 'Located_Fl', 182: 'Day_of_Week', 
                183: 'Hwy_Dsgn_Lane_ID', 184: 'Hwy_Dsgn_Hrt_ID', 185: 'Hp_Shldr_Left', 186: 'Hp_Shldr_Right', 187: 'Hp_Median_Width', 188: 'Base_Type_ID', 189: 'Nbr_Of_Lane', 
                190: 'Row_Width_Usual', 191: 'Roadbed_Width', 192: 'Surf_Width', 193: 'Surf_Type_ID', 194: 'Curb_Type_Left_ID', 195: 'Curb_Type_Right_ID', 196: 'Shldr_Type_Left_ID', 
                197: 'Shldr_Width_Left', 198: 'Shldr_Use_Left_ID', 199: 'Shldr_Type_Right_ID', 200: 'Shldr_Width_Right', 201: 'Shldr_Use_Right_ID', 202: 'Median_Type_ID', 
                203: 'Median_Width', 204: 'Rural_Urban_Type_ID', 205: 'Func_Sys_ID', 206: 'Adt_Curnt_Amt', 207: 'Adt_Curnt_Year', 208: 'Adt_Adj_Curnt_Amt', 209: 'Pct_Single_Trk_Adt', 
                210: 'Pct_Combo_Trk_Adt', 211: 'Trk_Aadt_Pct', 212: 'Curve_Type_ID', 213: 'Curve_Lngth', 214: 'Cd_Degr', 215: 'Delta_Left_Right_ID', 216: 'Dd_Degr', 
                217: 'Feature_Crossed', 218: 'Structure_Number', 219: 'I_R_Min_Vert_Clear', 220: 'Approach_Width', 221: 'Bridge_Median_ID', 222: 'Bridge_Loading_Type_ID', 
                223: 'Bridge_Loading_In_1000_Lbs', 224: 'Bridge_Srvc_Type_On_ID', 225: 'Bridge_Srvc_Type_Under_ID', 226: 'Culvert_Type_ID', 227: 'Roadway_Width', 228: 'Deck_Width', 
                229: 'Bridge_Dir_Of_Traffic_ID', 230: 'Bridge_Rte_Struct_Func_ID', 231: 'Bridge_IR_Struct_Func_ID', 232: 'CrossingNumber', 233: 'RRCo', 234: 'Poscrossing_ID', 
                235: 'WDCode_ID', 236: 'Standstop', 237: 'Yield', 238: 'Sus_Serious_Injry_Cnt', 239: 'Nonincap_Injry_Cnt', 240: 'Poss_Injry_Cnt', 241: 'Non_Injry_Cnt', 
                242: 'Unkn_Injry_Cnt', 243: 'Tot_Injry_Cnt', 244: 'Death_Cnt', 245: 'MPO_ID', 246: 'Investigat_Service_ID', 247: 'Investigat_DA_ID', 248: 'Damage_1', 
                249: 'Damage_2', 250: 'Damage_3', 251: 'Investigator_Narrative'}