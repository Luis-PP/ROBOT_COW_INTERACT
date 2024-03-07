import mesa
import numpy as np
from robot_cow_interact.util import *
from extremitypathfinder import PolygonEnvironment


class BaseAgent(mesa.Agent):
    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        kind: str,
        location: tuple,
        step_size: int,
        direction: float,
        vision_rad: int,
        state: str,
        color: str,
    ) -> None:
        super().__init__(unique_id, model)
        self.target = None
        self.prev_target = None
        self.kind = kind
        self.location = np.array(location)
        self.step_size = step_size
        self.direction = direction
        self.vision_rad = vision_rad
        self.state = state
        self.color = color
        self.neighbors = None
        self.environment = PolygonEnvironment()
        self.environment.store(self.model.boundary, self.model.holes)

    def check_bounds(self, location):
        return self.model.space.out_of_bounds(location)

    def random_move(self):
        x = self.model.random.random() * self.model.space.x_max
        y = self.model.random.random() * self.model.space.y_max
        random = self.model.space.get_heading(self.pos, (x, y))
        random /= np.linalg.norm(random)
        return random * 0.05

    def avoid_agent(self, kind: str):
        avoid = np.zeros(2)
        if self.neighbors:
            num_agents = 1
            for neighbor in self.neighbors:
                if neighbor.kind != kind:
                    pass
                else:
                    avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                    num_agents += 1
            avoid /= num_agents
            avoid_norm = np.linalg.norm(avoid)
            if avoid_norm != 0.0:
                avoid /= avoid_norm
            else:
                avoid /= 5e-324  # No zero division
        return avoid

    def get_path(self):
        self.closest_entrance = self.get_closest_entrance()
        self.path, length = self.environment.find_shortest_path(
            self.closest_entrance, self.target.entrance, verify=True
        )
        if self.target.kind == "feeder":  # Cows stand out of feeder while eating
            pass
        else:
            self.path.append(self.target.pos)

    def get_closest_entrance(self):
        entrances = np.array(self.model.entrances)
        dist = np.linalg.norm((entrances - self.pos), axis=1)
        min_indx = np.argmin(dist)
        return tuple(entrances[min_indx])

    def waypoint_reached(self):
        if is_close(self.pos, self.path[0], self.step_size * 2):
            if len(self.path) > 1:
                self.path.pop(0)
            else:
                pass


class Cow(BaseAgent):
    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        location: tuple,
        step_size: int,
        direction: float,
        vision_rad: int,
        state: str,
        color: str,
        fear: float,
        magnetism: float,
        health: float,
        kind: str = "cow",
    ) -> None:
        super().__init__(
            unique_id, model, kind, location, step_size, direction, vision_rad, state, color
        )
        self.fear = fear
        self.magnetism = magnetism
        self.health = health
        self.state = "moving"  # / "doing"
        self.target_reached = False
        self.time_reached = False
        self.doing_time = 0
        self.cow_time = 500
        self.manure_id = 0

    def avoid_fence(self):
        avoid = np.zeros(2)
        if self.neighbors:
            fences = 1
            for neighbor in self.neighbors:
                if neighbor.kind != "bound":  # only check Bound class
                    pass
                else:
                    # Is there a target
                    if self.target is not None:
                        # Are the Bounds part of the target? Whit this the robot can go inside of the target area.
                        if neighbor not in self.target.agents_bound:
                            # No, repulsion.
                            avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                            fences += 1
                        else:
                            # Yes, attraction.
                            avoid += self.model.space.get_heading(self.pos, self.target.pos)
                        # Are the Bounds part of the previous target? With this the robot can go out of the visited target area.
                        if self.prev_target is not None:
                            # No, repulsion.
                            if neighbor not in self.prev_target.agents_bound:
                                avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                                fences += 1
                            # Yes, attraction.
                            else:
                                avoid += self.model.space.get_heading(self.pos, self.target.pos)
                    else:
                        avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                        fences += 1
            avoid /= fences
            avoid_norm = np.linalg.norm(avoid)
            if avoid_norm != 0.0:
                avoid /= avoid_norm
            else:
                avoid /= 5e-324
        return avoid

    def update_state(self):
        if self.state == "moving":
            if self.target_reached == False:
                self.time_reached = False
                self.target = self.get_target()
                self.waypoint_reached()
                self.state = "moving"
            else:
                self.state = "doing"
        elif self.state == "doing":
            if self.time_reached == False:
                self.check_time()
                self.doing_time += 1
                self.state = "doing"
            else:
                self.target = self.get_target()
                self.time_reached = False
                self.state = "moving"
        else:
            raise (SyntaxError("Unknown state: %s" % (self.state)))

    def check_time(self):
        if self.doing_time < self.cow_time:
            pass
        else:
            self.time_reached = True
            self.doing_time = 0

    def next_move(self):
        if self.state == "moving":
            direction = self.model.space.get_heading(self.pos, self.path[0])
        elif self.state == "doing":
            if self.prev_target.kind != "feeder":
                direction = self.model.space.get_heading(self.pos, self.prev_target.pos)
            else:
                direction = self.model.space.get_heading(self.pos, self.prev_target.entrance)
        direction /= np.linalg.norm(direction)
        avoid_fence = self.avoid_fence()
        random = self.random_move()
        avoid_cows = self.avoid_agent("cow")
        avoid_robots = self.avoid_agent("robot")
        if list(avoid_robots) != [0.0, 0.0]:
            return self.pos + direction * self.step_size + avoid_robots * self.fear * self.step_size
        else:
            return (
                self.pos
                + direction * self.step_size
                + avoid_fence * self.step_size * 0.1
                + random * self.step_size
                + avoid_cows * self.magnetism * self.step_size
                + avoid_robots * self.fear * self.step_size
            )

    def get_target(self):
        if self.target is not None:
            if self.target.kind != "feeder":
                if is_close(self.pos, self.target.pos, self.step_size):
                    patches = self.model.get_agents_of_type(Patch)
                    self.prev_target = self.target
                    self.target = self.random.choice(patches)
                    self.get_path()
                    self.target_reached = True
                    return self.target
                else:
                    self.target_reached = False
                    return self.target
            else:
                if is_close(self.pos, self.target.entrance, self.step_size):
                    patches = self.model.get_agents_of_type(Patch)
                    self.prev_target = self.target
                    self.target = self.random.choice(patches)
                    self.get_path()
                    self.target_reached = True
                    return self.target
                else:
                    self.target_reached = False
                    return self.target
        else:
            patches = self.model.get_agents_of_type(Patch)
            self.target = self.random.choice(patches)
            self.get_path()
            return self.target

    def defecate(self):
        poop_probab = 0.01  # 16 / 86400
        is_defecating = np.random.choice(np.array([True, False]), p=[poop_probab, 1 - poop_probab])
        if (is_defecating and self.state == "moving") or (
            is_defecating and self.state == "doing" and self.prev_target.kind == "feeder"
        ):
            out_prev_target = False
            if self.prev_target is not None:
                out_prev_target = is_out(self.pos, self.prev_target)
            out_target = is_out(self.pos, self.target)
            if out_target and out_prev_target:
                id = float(str(self.unique_id) + "." + str(self.manure_id))
                rad = self.random.randrange(10)
                manure_instance = Manure(unique_id=id, model=self.model, radius=rad)
                self.model.space.place_agent(manure_instance, self.pos)
                self.model.schedule.add(manure_instance)
                self.manure_id += 1

    def step(self):
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision_rad, False)
        self.update_state()
        self.location = self.next_move()
        self.defecate()
        out_of_bounds = self.check_bounds(self.location)
        if out_of_bounds:
            self.model.space.move_agent(self, self.pos)
        else:
            self.model.space.move_agent(self, self.location)


class Robot(BaseAgent):
    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        location: tuple,
        step_size: int,
        direction: float,
        vision_rad: int,
        state: str,
        color: str,
        caution: float,
        memory: list,
        kind: str = "robot",
    ) -> None:
        super().__init__(
            unique_id, model, kind, location, step_size, direction, vision_rad, state, color
        )
        self.caution = caution
        self.memory = memory
        self.battery = self.random.randrange(3000, 3600)
        self.nest = self.model.get_agents_of_type(Nest)[0]
        self.target = self.nest
        self.point_to_help = None

    def avoid_fence(self):
        avoid = np.zeros(2)
        if self.neighbors:
            fences = 1
            for neighbor in self.neighbors:
                if neighbor.kind != "bound":  # only check Bound class
                    pass
                else:
                    # Is there a target
                    if self.target is not None:
                        out = is_out(self.pos, self.nest)
                        if type(self.target) == Nest or out == False:
                            avoid = 0
                        else:
                            avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                            fences += 1
                    else:
                        avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                        fences += 1
            avoid /= fences
            avoid_norm = np.linalg.norm(avoid)
            if avoid_norm != 0.0:
                avoid /= avoid_norm
            else:
                avoid /= 5e-324
        return avoid

    def update_state(self):
        if self.state == "in_nest":
            self.color = "Orange"
            if self.battery > 3600:
                self.state = "searching"
            else:
                self.state = "in_nest"

        elif self.state == "searching":
            self.color = "Orange"
            if self.battery < 1800:
                self.state = "back_to_nest"
                self.target = self.nest
                self.get_path()
            else:
                self.state = "searching"

        elif self.state == "back_to_nest":
            self.color = "Red"
            if is_out(self.pos, self.nest):
                self.waypoint_reached()
                self.state = "back_to_nest"
            else:
                self.state = "in_nest"

        elif self.state == "recruited":
            self.color = "Green"
            self.state = "otw_to_help"
            self.get_path_to_help()

        elif self.state == "otw_to_help":
            self.color = "Green"
            if is_close(self.pos, self.point_to_help, self.step_size):
                self.state = "searching"
            else:
                self.waypoint_reached()
                self.state = "otw_to_help"

    def get_path_to_help(self):
        self.closest_entrance = self.get_closest_entrance()
        self.path, length = self.environment.find_shortest_path(
            self.closest_entrance, self.point_to_help, verify=True
        )

    def next_move(self):
        self.look_around()
        # Is the target an agent or jut a point
        if self.state == "searching":
            if hasattr(self.target, "pos"):  # Is an agent
                if self.target.pos is not None:
                    direction = self.model.space.get_heading(self.pos, self.target.pos)
                else:
                    # In case a a robot just clean the manure that another was aiming
                    direction = self.model.space.get_heading(self.pos, self.nest.pos)
            else:  # Is a point
                direction = self.model.space.get_heading(self.pos, self.target)
        elif (
            self.state == "back_to_nest" or self.state == "otw_to_help" or self.state == "recruited"
        ):
            direction = self.model.space.get_heading(self.pos, self.path[0])
        elif self.state == "in_nest":
            direction = self.model.space.get_heading(self.pos, self.nest.pos)
        direction /= np.linalg.norm(direction)
        if self.state != "back_to_nest":
            avoid_fence = self.avoid_fence()
        else:
            avoid_fence = 0
        random = self.random_move()
        avoid_cows = self.avoid_agent("cow")
        avoid_robots = self.avoid_agent("robot")
        if list(avoid_cows) != [0.0, 0.0]:
            return (
                self.pos + direction * self.step_size + avoid_cows * self.step_size * self.caution
            )
        else:
            return (
                self.pos
                + direction * self.step_size
                + avoid_fence * self.step_size
                + random * self.step_size
                + avoid_cows * self.caution * self.step_size
                + avoid_robots * self.step_size
            )

    def update_battery(self):
        if self.state == "in_nest":
            self.battery += 1
        else:
            self.battery -= 1

    def agent_in_vision(self, kind, vis_multiplier):
        agents = []
        self.neighbors = self.model.space.get_neighbors(
            self.pos, self.vision_rad * vis_multiplier, False
        )
        for neighbor in self.neighbors:
            if neighbor.kind == kind:
                agents.append(neighbor)
        return agents

    def is_target_reach(self):
        if self.state == "searching":
            bounds = self.agent_in_vision("bound", 0.7)
            if hasattr(self.target, "pos"):
                if self.target.pos is not None:
                    if is_close(self.pos, self.target.pos, self.step_size) or len(bounds) > 0:
                        self.get_target()
                    else:
                        pass
                else:
                    self.get_target()
            else:
                if is_close(self.pos, self.target, self.step_size) or len(bounds) > 0:
                    self.get_target()
                else:
                    pass
        else:
            self.target = self.nest

    def look_around(self):
        if self.state == "searching" or self.state == "back_to_nest" or self.state == "otw_to_help":
            manures = self.agent_in_vision("manure", 3)
            bounds = self.agent_in_vision("bound", 0.7)
            if len(manures) > 0:
                self.get_target()
            elif len(bounds) > 0:
                self.get_target()
            else:
                pass
        else:
            pass

    def get_target(self):
        if self.state == "searching" or self.state == "back_to_nest" or self.state == "otw_to_help":
            manures = self.agent_in_vision("manure", 3)
            bounds = self.agent_in_vision("bound", 0.7)

            for bound in bounds:
                if bound in self.nest.agents_bound:
                    bounds.remove(bound)

            if len(manures) > 0:
                self.state = "searching"
                self.target = self.random.choice(manures)

            elif len(bounds) > 0:
                bound = np.random.choice(bounds)
                self.target = opposite_direction(
                    self.pos, bound.pos, self.model.space.x_max, self.model.space.y_max
                )

            else:
                x = self.model.random.random() * self.model.space.x_max
                y = self.model.random.random() * self.model.space.y_max
                self.target = (x, y)

        else:
            self.target = self.nest

    def clean(self):
        if type(self.target) != Manure:
            pass
        else:
            if is_close(self.pos, self.target.pos, self.step_size * 3):
                help_me = self.target.radius
                self.model.space.remove_agent(self.target)
                self.model.schedule.remove(self.target)
                self.target.remove()
                self.neighbors = self.model.space.get_neighbors(self.pos, self.vision_rad, False)
                self.get_target()
                self.nest.control[str(self.unique_id)] += help_me
                self.nest.recruit()

    def step(self):
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision_rad, False)
        self.update_battery()
        self.update_state()
        self.location = self.next_move()
        self.is_target_reach()
        self.clean()
        out_of_bounds = self.check_bounds(self.location)
        if out_of_bounds:
            self.model.space.move_agent(self, self.pos)
        else:
            self.model.space.move_agent(self, self.location)


class Patch(mesa.Agent):
    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        pos: tuple,
        kind: str,
        color: str,
        offset: tuple,
        entrance: tuple,
    ) -> None:
        super().__init__(unique_id, model)
        self.pos = pos
        self.kind = kind
        self.color = color
        self.offset = offset
        self.entrance = entrance
        self.create_bounds()

    def create_bounds(self):
        x, y = self.pos
        x_offset, y_offset = self.offset
        x0, x1 = x - x_offset, x + x_offset
        y0, y1 = y - y_offset, y + y_offset
        inflate = 20
        env_bound = [
            [x0 - inflate, y0 - inflate],
            [x0 - inflate, y1 + inflate],
            [x1 + inflate, y1 + inflate],
            [x1 + inflate, y0 - inflate],
        ]
        if x0 == 0:  # very left
            env_bound[0][0] = x0
            env_bound[1][0] = x0
        if x1 == self.model.space.width - 1:  # very right
            env_bound[2][0] = x1
            env_bound[3][0] = x1
        if y0 == 0:  # very top
            env_bound[0][1] = y0
            env_bound[3][1] = y0
        if y1 == self.model.space.height - 1:  # very bottom
            env_bound[1][1] = y1
            env_bound[2][1] = y1
        self.env_bound = env_bound
        top = list(get_line((x0, y0), (x1, y0), x1 - x0))
        bottom = list(get_line((x0, y1), (x1, y1), x1 - x0))
        left = list(get_line((x0, y0), (x0, y1), y1 - y0))
        right = list(get_line((x1, y0), (x1, y1), y1 - y0))
        self.bounds = [top, bottom, left, right]
        self.agents_bound = []
        i = 0
        for line in self.bounds:
            for point in line:
                bound_instance = Bound(unique_id=i, model=self)
                self.agents_bound.append(bound_instance)
                self.model.space.place_agent(bound_instance, point)
                self.model.schedule.add(bound_instance)
                i += 1


class Nest(Patch):
    def __init__(
        self,
        unique_id: int,
        model: mesa.Model,
        pos: tuple,
        kind: str,
        color: str,
        offset: tuple,
        entrance: tuple,
        memory_threshold: int,
    ) -> None:
        super().__init__(unique_id, model, pos, kind, color, offset, entrance)
        self.num_robots = len(self.model.get_agents_of_type(Robot))
        self.memory_threshold = memory_threshold
        self.counter = 0
        self.create_control()

    def create_control(self):
        self.control = {}
        for robot in range(self.model.robot_num):
            self.control[str(robot)] = 1.0

    def negotiate(self):
        try:
            k = np.array(list(self.control.keys()))
            v = np.array(list(self.control.values()))
            p = v / sum(v)
            _v = max(v) - v
            _p = _v / sum(_v)
            self.to_help = int(np.random.choice(a=k, p=p))
            self.helper = int(np.random.choice(a=k, p=_p))
            recruited = np.random.choice(
                np.array([True, False]), p=[self.model.recruit_prob, 1 - self.model.recruit_prob]
            )
            return recruited
        except:
            return False

    def recruit(self):
        self.counter += 1
        if self.counter > self.memory_threshold:
            self.create_control()
            self.counter = 0
        if self.negotiate():
            self.robot_to_help = self.model.get_agents_of_type(Robot)[self.to_help]
            self.robot_helper = self.model.get_agents_of_type(Robot)[self.helper]
            self.robot_helper.color = "Green"
            self.robot_to_help.color = "Blue"
            self.robot_helper.point_to_help = self.robot_to_help.get_closest_entrance()
            self.robot_helper.state = "recruited"
        else:
            pass

    def step(self):
        pass


class Bound(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, kind: str = "bound") -> None:
        super().__init__(unique_id, model)
        self.kind = kind


class Manure(mesa.Agent):
    def __init__(
        self, unique_id: int, model: mesa.Model, radius: int, kind: str = "manure"
    ) -> None:
        super().__init__(unique_id, model)
        self.kind = kind
        self.radius = radius
