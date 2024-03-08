"""
Author: Luis Ponce Pacheco
Contact: luis.poncepacheco@wur.nl
PSG, ABE group.
"""

import mesa

from robot_cow_interact.model import RobotCow
from robot_cow_interact.simpleContinuousModule import SimpleCanvas


def robot_cow_draw(agent):
    if agent.kind == "robot":
        return {"Shape": "circle", "r": 12, "Filled": "true", "Color": agent.color}
    elif agent.kind == "cow":
        return {"Shape": "circle", "r": 12, "Filled": "true", "Color": agent.color}
    elif agent.kind == "cubicle":
        return {"Shape": "rect", "w": 0.044, "h": 0.2, "Filled": "true", "Color": agent.color}
    elif agent.kind == "feeder":
        return {"Shape": "rect", "w": 0.044, "h": 0.1, "Filled": "true", "Color": agent.color}
    elif agent.kind == "drinker":
        return {"Shape": "rect", "w": 0.044, "h": 0.1, "Filled": "true", "Color": agent.color}
    elif agent.kind == "milker":
        return {"Shape": "rect", "w": 0.045, "h": 0.4, "Filled": "true", "Color": agent.color}
    elif agent.kind == "concentrate":
        return {"Shape": "rect", "w": 0.044, "h": 0.1, "Filled": "true", "Color": agent.color}
    elif agent.kind == "bound":
        return {"Shape": "rect", "w": 0.001, "h": 0.002, "Filled": "true", "Color": "black"}
    elif agent.kind == "nest":
        return {"Shape": "rect", "w": 0.044, "h": 0.2, "Filled": "true", "Color": agent.color}
    elif agent.kind == "manure":
        return {"Shape": "circle", "r": agent.radius, "Filled": "true", "Color": "Purple"}
    else:
        return None


robot_cow_canvas = SimpleCanvas(
    portrayal_method=robot_cow_draw, canvas_height=500, canvas_width=1100
)

# Plot
chart_element = mesa.visualization.ChartModule([{"Label": "Manure", "Color": "Purple"}])

# Sliders
model_params = {
    "width": 1101,
    "height": 501,
    "Cow_parameters": mesa.visualization.StaticText("Cow Parameters"),
    "cow_num": mesa.visualization.Slider(
        name="Number of Cows",
        value=30,
        min_value=0,
        max_value=30,
        step=1,
        description="Choose the number of cows",
    ),
    # "cow_step": mesa.visualization.Slider(
    #     name="Cow speed",
    #     value=5,
    #     min_value=1,
    #     max_value=20,
    #     step=1,
    #     description="How fast should the Cows move",
    # ),
    # "cow_vision": mesa.visualization.Slider(
    #     name="Vision of Cows (radius)",
    #     value=15,
    #     min_value=0,
    #     max_value=100,
    #     step=1,
    #     description="How far around should each Cows look for its neighbors",
    # ),
    # "cow_magnetism": mesa.visualization.Slider(
    #     name="Cow-Cow repulsion",
    #     value=5,
    #     min_value=1,
    #     max_value=40,
    #     step=1,
    #     description="What is the distance each Cow will attempt to keep from any other",
    # ),
    # "cow_fear": mesa.visualization.Slider(
    #     name="Cow fear",
    #     value=6,
    #     min_value=0,
    #     max_value=40,
    #     step=1,
    #     description="How much each Cow tries to avoid the Robots",
    # ),
    # "cow_health": mesa.visualization.Slider(
    #     name="Cow health",
    #     value=10,
    #     min_value=1,
    #     max_value=20,
    #     step=1,
    #     description="Number of robot collisions each Cow can support before a leg is broken",
    # ),
    "Robot_parameters": mesa.visualization.StaticText("Robot Parameters"),
    "robot_num": mesa.visualization.Slider(
        name="Number of Robots",
        value=3,
        min_value=1,
        max_value=20,
        step=1,
        description="Choose the number of Robots",
    ),
    # "robot_step": mesa.visualization.Slider(
    #     name="Robot speed",
    #     value=5,
    #     min_value=1,
    #     max_value=20,
    #     step=1,
    #     description="How fast should the Robot move",
    # ),
    # "robot_vision": mesa.visualization.Slider(
    #     name="Vision of Robot (radius)",
    #     value=10,
    #     min_value=1,
    #     max_value=50,
    #     step=1,
    #     description="How far around should each Robot look for its neighbors",
    # ),
    # "robot_caution": mesa.visualization.Slider(
    #     name="Robot cautiousness",
    #     value=2,
    #     min_value=0,
    #     max_value=20,
    #     step=1,
    #     description="What is the distance each Robot will attempt to keep from any Cow",
    # ),
    "recruit_prob": mesa.visualization.Slider(
        name="Recruit probability",
        value=0.25,
        min_value=0,
        max_value=1,
        step=0.05,
        description="The probability that one robot will be recruited by another",
    ),
    "memory_threshold": mesa.visualization.Slider(
        name="Controller memory threshold",
        value=10,
        min_value=1,
        max_value=100,
        step=1,
        description="Reset the controller memory after this threshold",
    ),
}


server = mesa.visualization.ModularServer(
    RobotCow, [robot_cow_canvas, chart_element], "Robot Cow Interactions", model_params
)
server.port = 8521
