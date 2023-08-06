import json
import os
import random
import copy

target_purpose_to_IDs = {
    'discount': 300,
    'fruit': 301,
    'vegetable': 302,
    'bakery': 303,
    'diary': 304,
    'meat': 305,
    'seafood': 306,
    'frozen': 307,
    'canned': 308,
    'beverage': 309,
    'snack': 310,
    'convenience': 311
}

def generate_shopping_list_ids(num_items_in = (1, 10)) -> list:
        """Generates a random shopping list

        Args:
            num_items_in (tuple, optional): A tuple of minimum and maximum number
                of items for the shopping list. Defaults to (1, 10).

        Returns:
            list: A list of target IDs for the shopping list
        """
        global target_purpose_to_IDs
        targets = []
        target_possibilities = list(target_purpose_to_IDs.values())
        size = random.randint(num_items_in[0], num_items_in[1])
        targets = random.sample(target_possibilities, size)
        return targets

def add_customers(num_source = 5, 
        num_ped_per_source = 100,
        num_items_in = (1, 10), 
        time_start = (0, 60), time_end = (100, 200),
        skip_source = True):
    """Add sources to a shopping scenario

    Args:
        num_source (int, optional): The number of sources that will be generated.
        num_ped_per_source (int, optional): The number of customers that will be generated per source.
        num_items_in (tuple, optional): A tuple of minimum and maximum number of items for the shopping list.
        time_start (tuple, optional): The range of time when the source should start producing.
        time_end (tuple, optional): The range of time when the source should end producing.
        skip_source (bool, optional): The option of whether adding an extra source generating customers buying nothing at all.
    """
    source_default = {
        "id" : 100,
        "shape" : {
          "x" : 36.0,
          "y" : 40.0,
          "width" : 1.0,
          "height" : 1.0,
          "type" : "RECTANGLE"
        },
        "visible" : True,
        "targetIds" : [ 201, 202 ],
        "spawner" : {
          "type" : "org.vadere.state.attributes.spawner.AttributesRegularSpawner",
          "constraintsElementsMax" : num_ped_per_source,
          "constraintsTimeStart" : time_start[0],
          "constraintsTimeEnd" : time_end[1],
          "eventPositionRandom" : True,
          "eventPositionGridCA" : False,
          "eventPositionFreeSpace" : True,
          "eventElementCount" : 1,
          "eventElement" : None,
          "distribution" : {
            "type" : "org.vadere.state.attributes.distributions.AttributesConstantDistribution",
            "updateFrequency" : 10
          }
        },
        "groupSizeDistribution" : [ 1.0 ]
    }
    global scenario
    sources = []
    #scenario['scenario']['topography']['sources']
    if skip_source:
        sources.append(source_default)
    register = 200
    outlet = 202

    for source_id in range(num_source):
        source = copy.deepcopy(dict(source_default))
        source['id'] = source_id + 101
        source['shape']['x'] = source_id % 6 + 37
        source['shape']['y'] = int(source_id / 6) + 40
        
        source['targetIds'] = generate_shopping_list_ids(num_items_in) + [register, outlet]
        
        start = random.randint(time_start[0], time_start[1])
        end = random.randint(time_end[0], time_end[1])
        source['spawner']['constraintsElementsMax'] = num_ped_per_source
        source['spawner']['constraintsTimeStart'] = start
        source['spawner']['constraintsTimeEnd'] = end
        source['spawner']['distribution']['updateFrequency'] = (end - start) / num_ped_per_source * 10
        sources.append(source)
    
    scenario['scenario']['topography']['sources'] = sources



scenario_folder = './'
scenario_file = 'supermarket.scenario'

with open(os.path.join(scenario_folder, scenario_file)) as scen_file:
    scenario = json.load(scen_file)



num_ped_per_sourceList = [50, 25, 10]
pedPotentialPersonalSpaceWidthList = [1.0, 2.0, 3.0, 4.0, 5.0]
for pedPotentialPersonalSpaceWidth in pedPotentialPersonalSpaceWidthList:
    for num_ped_per_source in num_ped_per_sourceList:
        add_customers(num_source = 24, 
            num_ped_per_source = num_ped_per_source,
            num_items_in = (1, 12), 
            time_start = (0, 60), time_end = (100, 200),
            skip_source = True)
        scenario['scenario']['attributesModel']['org.vadere.state.attributes.models.AttributesPotentialCompactSoftshell']['pedPotentialPersonalSpaceWidth'] = pedPotentialPersonalSpaceWidth
        scenario['name'] = 'supermarket_' + str(pedPotentialPersonalSpaceWidth) + '_' + str(num_ped_per_source)
        scenario_file = scenario['name'] + '.scenario'

        with open(os.path.join(scenario_folder, scenario_file), 'w', encoding='utf-8') as f:
            json.dump(scenario, f, ensure_ascii=False, indent=2)


