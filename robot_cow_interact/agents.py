import mesa
import numpy as np
import math as m
from extremitypathfinder import PolygonEnvironment

COW_TM = [
    [0.03, 0.62, 0.11, 0.24, 0.0],
    [0.0, 0.03, 0.16, 0.79, 0.01],
    [0.02, 0.45, 0.10, 0.15, 0.28],
    [0.0, 0.16, 0.35, 0.08, 0.41],
    [0.90, 0.05, 0.01, 0.03, 0.01],
]

COW_TIME = [10, 10, 10, 10, 10]


def get_line(p1, p2, parts):
    return zip(
        np.linspace(p1[0], p2[0], parts, endpoint=False),
        np.linspace(p1[1], p2[1], parts, endpoint=False),
    )


def is_close(own, target, dist):
    own_x, own_y = own
    t_x, t_y = target
    if m.isclose(own_x, t_x, abs_tol=dist) and m.isclose(own_y, t_y, abs_tol=dist):
        return True
    else:
        return False


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
    ) -> None:
        super().__init__(unique_id, model)
        self.kind = kind
        self.location = np.array(location)
        self.step_size = step_size
        self.direction = direction
        self.vision_rad = vision_rad
        self.state = state
        self.neighbors = None
        self.environment = PolygonEnvironment()
        self.environment.store(self.model.boundary, self.model.holes)

    def check_bounds(self, location):
        return self.model.space.out_of_bounds(location)


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
        fear: float,
        magnetism: float,
        health: float,
        kind: str = "cow",
    ) -> None:
        super().__init__(unique_id, model, kind, location, step_size, direction, vision_rad, state)
        self.fear = fear
        self.magnetism = magnetism
        self.health = health
        self.target = None
        self.prev_target = None
        self.state = "moving"

    # def update_state(self):
    #     if self.state == "moving":
    #         if self.target.pos

    def next_move(self):
        direction = self.model.space.get_heading(self.pos, self.path[0])
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
            )

    def random_move(self):
        x = self.model.random.random() * self.model.space.x_max
        y = self.model.random.random() * self.model.space.y_max
        random = self.model.space.get_heading(self.pos, (x, y))
        random /= np.linalg.norm(random)
        return random * 0.25

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

    def avoid_fence(self):
        avoid = np.zeros(2)
        if self.neighbors:
            fences = 1
            for neighbor in self.neighbors:
                if neighbor.kind != "bound":
                    pass
                else:
                    if neighbor not in self.target.agents_bound:
                        avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                        fences += 1
                    else:
                        avoid += self.model.space.get_heading(self.pos, self.target.pos)

                    if self.prev_target is not None:
                        if neighbor not in self.prev_target.agents_bound:
                            avoid -= self.model.space.get_heading(self.pos, neighbor.pos)
                            fences += 1
                        else:
                            avoid += self.model.space.get_heading(self.pos, self.target.pos)
            avoid /= fences
            avoid_norm = np.linalg.norm(avoid)
            if avoid_norm != 0.0:
                avoid /= avoid_norm
            else:
                avoid /= 5e-324
        return avoid

    def get_target(self):
        if self.target is not None:
            if self.target.kind != "feeder":
                if is_close(self.pos, self.target.pos, self.step_size):
                    patches = self.model.get_agents_of_type(Patch)
                    self.prev_target = self.target
                    self.target = self.random.choice(patches)
                    self.get_path()
                    return self.target
                else:
                    return self.target
            else:
                if is_close(self.pos, self.target.entrance, self.step_size):
                    patches = self.model.get_agents_of_type(Patch)
                    self.prev_target = self.target
                    self.target = self.random.choice(patches)
                    self.get_path()
                    return self.target
                else:
                    return self.target
        else:
            patches = self.model.get_agents_of_type(Patch)
            self.target = self.random.choice(patches)
            self.get_path()
            return self.target

    def get_path(self):
        closest_entrance = self.get_closest_entrance()
        self.path, length = self.environment.find_shortest_path(
            closest_entrance, self.target.entrance, verify=True
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
                self.path.pop(0)  #
            else:
                pass

    def step(self):
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision_rad, False)
        self.target = self.get_target()
        self.waypoint_reached()
        self.location = self.next_move()
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
        caution: float,
        memory: list,
        kind: str = "robot",
    ) -> None:
        super().__init__(unique_id, model, kind, location, step_size, direction, vision_rad, state)
        self.caution = caution
        self.memory = memory

    def claim_area(self):
        pass

    def update_state(self):
        pass

    def cautious(self):
        pass

    def step(self):
        self.neighbors = self.model.space.get_neighbors(self.pos, self.vision_rad, False)
        self.location = np.array(self.pos) + 1 * self.step_size
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
        inflate = 15
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


class Bound(mesa.Agent):
    def __init__(self, unique_id: int, model: mesa.Model, kind: str = "bound") -> None:
        super().__init__(unique_id, model)
        self.kind = kind
