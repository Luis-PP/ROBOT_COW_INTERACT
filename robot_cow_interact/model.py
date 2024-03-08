"""
Author: Luis Ponce Pacheco
Contact: luis.poncepacheco@wur.nl
PSG, ABE group.
"""

import mesa
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from robot_cow_interact.agents import Robot, Cow, Patch, Nest, Manure


class RobotCow(mesa.Model):
    def __init__(
        self,
        width,
        height,
        cow_num,
        # cow_step,
        # cow_vision,
        # cow_magnetism,
        # cow_fear,
        # cow_health,
        robot_num,
        # robot_step,
        # robot_vision,
        # robot_caution,
        recruit_prob,
        memory_threshold
    ) -> None:
        super().__init__()
        # self.random.seed(36)
        self.cow_num = cow_num
        self.cow_step = 2  # cow_step
        self.cow_vision = 15  # cow_vision
        self.cow_magnetism = 5  # cow_magnetism
        self.cow_fear = 5  # cow_fear
        self.cow_health = 10  # cow_health
        self.robot_num = robot_num
        self.robot_step = 5  # robot_step
        self.robot_vision = 16  # robot_vision
        self.robot_caution = 5  # robot_caution
        self.recruit_prob = recruit_prob
        self.memory_threshold = memory_threshold
        self.boundary = [(0, 0), (width, 0), (width, height), (0, height)]
        self.holes = []
        self.entrances = []
        self.schedule = mesa.time.RandomActivationByType(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)
        self.datacollector = mesa.DataCollector(
            {"Manure": lambda m: m.schedule.get_type_count(Manure)}
        )
        self.barn = self.create_barn()
        self.create_patches()
        self.create_robots()
        self.create_cows()
        self.create_manure()
        self.datacollector.collect(self)

    def create_manure(self):
        x = self.space.x_max - 2
        y = self.space.y_max - 2
        location = (x, y)
        manure_instance = Manure(0, self, 1)
        self.space.place_agent(manure_instance, location)
        self.schedule.add(manure_instance)

    def create_barn(self):
        entrance_offset = 25
        # Cubicles
        cubicles = []
        kind = "cubicle"
        color = " #935116 "
        offset = (25, 50)
        for col in range(275, self.space.width - 100, 50):
            for row in [50, 275]:
                cubicles.append(
                    (col, row, kind, color, offset, (col, row + offset[1] + entrance_offset))
                )
        cubicles.pop(21)
        cubicles.pop(22)
        # Feeders
        feeders = []
        kind = "feeder"
        color = " #229954"
        offset = (25, 25)
        for col in range(275, self.space.width - 50 - 100, 50):
            feeders.append(
                (col, 475, kind, color, offset, (col, 475 - offset[1] - entrance_offset))
            )
        # Drinkers
        drinkers = []
        kind = "drinker"
        color = " #3498db "
        offset = (25, 25)
        for col in [225, 975]:
            drinkers.append(
                (col, 475, kind, color, offset, (col, 475 - offset[1] - entrance_offset))
            )
        # Milker
        milkers = []
        kind = "milker"
        color = "#f1948a"
        offset = (25, 100)
        for row in [125, 375]:
            milkers.append((25, row, kind, color, offset, (25 + offset[0] + entrance_offset, row)))
        # Concentrate
        concentrates = []
        kind = "concentrate"
        color = "#145a32"
        offset = (25, 25)
        entrance = (225, 250 + offset[1] + entrance_offset)
        concentrates = [(225, 250, kind, color, offset, entrance)]
        # Robot Nest
        nest = []
        kind = "nest"
        color = "#5d6d7e"
        offset = (25, 50)
        entrance = (1075, 50 + offset[1] + entrance_offset)
        nest = [(1075, 50, kind, color, offset, entrance)]
        # Complete Barn
        barn = [cubicles, feeders, drinkers, milkers, concentrates, nest]
        return barn

    def create_patches(self):
        i = 0
        for areas in self.barn:
            for area in areas:
                x, y, kind, color, offset, entrance = area
                location = (x, y)
                if kind != "nest":
                    self.entrances.append(entrance)
                    patch_instance = Patch(
                        unique_id=i,
                        model=self,
                        pos=location,
                        kind=kind,
                        color=color,
                        offset=offset,
                        entrance=entrance,
                    )
                else:  # So is not a Cow target
                    patch_instance = Nest(
                        unique_id=i,
                        model=self,
                        pos=location,
                        kind=kind,
                        color=color,
                        offset=offset,
                        entrance=entrance,
                        memory_threshold=self.memory_threshold
                    )
                self.holes.append(patch_instance.env_bound)
                self.space.place_agent(patch_instance, location)
                self.schedule.add(patch_instance)
                i += 1
        self.rearrange_holes()

    def rearrange_holes(self):
        gdf = gpd.GeoDataFrame(geometry=[Polygon(hole) for hole in self.holes])
        merged = gdf.unary_union
        holes = [list(hole.exterior.coords) for hole in merged.geoms]
        for hole in holes:
            hole.pop(-1)  # The first point is NOT repeated at the end
        self.holes = holes

    def create_cows(self):
        for i in range(self.cow_num):
            x = int(1 + (self.random.random() * (self.space.x_max - 2)))
            y = int(1 + (self.random.random() * (self.space.y_max - 2)))
            location = (x, y)
            direction = np.random.random() * 2 * np.pi
            cow_instance = Cow(
                unique_id=i,
                model=self,
                location=location,
                step_size=self.cow_step,
                direction=direction,
                vision_rad=self.cow_vision,
                state=None,
                color="Black",
                fear=self.cow_fear,
                magnetism=self.cow_magnetism,
                health=self.cow_health,
                
            )
            self.space.place_agent(cow_instance, location)
            self.schedule.add(cow_instance)

    def create_robots(self):
        for i in range(self.robot_num):
            x = self.random.uniform(1050 / 1100, 1) * self.space.x_max
            y = self.random.uniform(0, 50 / 500) * self.space.y_max
            location = (x, y)
            direction = np.random.random() * 2 * np.pi
            robot_instance = Robot(
                unique_id=i,
                model=self,
                location=location,
                step_size=self.robot_step,
                direction=direction,
                vision_rad=self.robot_vision,
                state="in_nest",
                color="Orange",
                caution=self.robot_caution,
                memory=[],
            )
            self.space.place_agent(robot_instance, location)
            self.schedule.add(robot_instance)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
