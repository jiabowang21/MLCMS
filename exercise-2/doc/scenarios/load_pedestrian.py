import json
import os
import sys

def read_scenario(path):
    """
    Loads a scenario file with json
    :param path: the path of the scenario file 
    :return: the dictionary containing the scenario's data
    """
    with open(path, 'r') as f:
        data = json.load(f)
        return data

def add_pedestrian(data):
    """
    Adds a pedestrian to the data
    :param data: data containing the information of the scenario
    :return: data with an added pedestrian
    """
    pedestrian = {
        "attributes" : {
            "id" : 2,
            "shape" : {
                "x" : 0.0,
                "y" : 0.0,
                "width" : 1.0,
                "height" : 1.0,
                "type" : "RECTANGLE"
            },
            "visible" : True,
            "radius" : 0.2,
            "densityDependentSpeed" : False,
            "speedDistributionMean" : 1.34,
            "speedDistributionStandardDeviation" : 0.26,
            "minimumSpeed" : 0.5,
            "maximumSpeed" : 2.2,
            "acceleration" : 2.0,
            "footstepHistorySize" : 4,
            "searchRadius" : 1.0,
            "walkingDirectionSameIfAngleLessOrEqual" : 45.0,
            "walkingDirectionCalculation" : "BY_TARGET_CENTER"
        },
        "source" : None,
        "targetIds" : [ 4 ],
        "nextTargetListIndex" : 0,
        "isCurrentTargetAnAgent" : False,
        "position" : {
            "x" : 11.5,
            "y" : 1.5
        },
        "velocity" : {
            "x" : 0.0,
            "y" : 0.0
        },
        "freeFlowSpeed" : 0.8742984056779802,
        "followers" : [ ],
        "idAsTarget" : -1,
        "isChild" : False,
        "isLikelyInjured" : False,
        "psychologyStatus" : {
            "mostImportantStimulus" : None,
            "threatMemory" : {
                "allThreats" : [ ],
                "latestThreatUnhandled" : False
            },
            "selfCategory" : "TARGET_ORIENTED",
            "groupMembership" : "OUT_GROUP",
            "knowledgeBase" : {
                "knowledge" : [ ],
                "informationState" : "NO_INFORMATION"
            },
            "perceivedStimuli" : [ ],
            "nextPerceivedStimuli" : [ ]
        },
        "healthStatus" : None,
        "infectionStatus" : None,
        "groupIds" : [ ],
        "groupSizes" : [ ],
        "agentsInGroup" : [ ],
        "trajectory" : {
            "footSteps" : [ ]
        },
        "modelPedestrianMap" : { },
        "type" : "PEDESTRIAN"
    }
    data["name"] = "RiMEA Scenario 6 Script"
    data["scenario"]["topography"]["dynamicElements"].append(pedestrian)
    return data

def save_scenario(path, data):
    """
    Saves	the scenario in the given path
    :param path: the output path of the scenario
           scenario: the scenario with the pedestrian added
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent = 4)



path = "./scenarios/rimea6_gui.scenario"

data = read_scenario(path)

data = add_pedestrian(data)

output_path = "./scenarios/rimea6_script.scenario"

save_scenario(output_path, data)









