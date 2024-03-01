import mesa
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from robot_cow_interact.agents import Robot, Cow, Patch, Bound


class RobotCow(mesa.Model):
    def __init__(
        self,
        width,
        height,
        cow_num,
        cow_step,
        cow_vision,
        cow_magnetism,
        cow_fear,
        cow_health,
        robot_num,
        robot_step,
        robot_vision,
        robot_caution,
    ) -> None:
        super().__init__()
        self.cow_num = cow_num
        self.cow_step = cow_step
        self.cow_vision = cow_vision
        self.cow_magnetism = cow_magnetism
        self.cow_fear = cow_fear
        self.cow_health = cow_health
        self.robot_num = robot_num
        self.robot_step = robot_step
        self.robot_vision = robot_vision
        self.robot_caution = robot_caution
        self.boundary = [(0, 0), (width, 0), (width, height), (0, height)]
        self.holes = []
        self.entrances = []
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, False)
        self.barn = self.create_barn()
        self.create_patches()
        self.create_robots()
        self.create_cows()

    def create_barn(self):
        entrance_offset = 25
        # Cubicles
        cubicles = []
        kind = "cubicle"
        color = " #935116 "
        offset = (25, 50)
        for col in range(275, self.space.width, 50):
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
        for col in range(275, self.space.width - 50, 50):
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
        # Complete Barn
        barn = [cubicles, feeders, drinkers, milkers, concentrates]
        return barn

    def create_patches(self):
        i = 0
        for areas in self.barn:
            for area in areas:
                x, y, kind, color, offset, entrance = area
                self.entrances.append(entrance)
                location = (x, y)
                patch_instance = Patch(
                    unique_id=i,
                    model=self,
                    pos=location,
                    kind=kind,
                    color=color,
                    offset=offset,
                    entrance=entrance,
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
                fear=self.cow_fear,
                magnetism=self.cow_magnetism,
                health=self.cow_health,
            )
            self.space.place_agent(cow_instance, location)
            self.schedule.add(cow_instance)

    def create_robots(self):
        for i in range(self.robot_num):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            location = (x, y)
            direction = np.random.random() * 2 * np.pi
            robot_instance = Robot(
                unique_id=i,
                model=self,
                location=location,
                step_size=self.robot_step,
                direction=direction,
                vision_rad=self.robot_vision,
                state=None,
                caution=self.robot_caution,
                memory=[],
            )
            self.space.place_agent(robot_instance, location)
            self.schedule.add(robot_instance)

    def step(self):
        self.schedule.step()
