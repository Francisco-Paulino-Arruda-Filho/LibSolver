import heapq
import itertools
import pickle
import time

from abc import ABC, abstractmethod
from collections import deque
from multiprocessing import Pool
from dataclasses import dataclass, asdict
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar, Generic, Literal

import numpy as np
import pandas as pd
import plotly.express as px

from tqdm import tqdm


class Action(IntEnum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @classmethod
    def opposite(cls, action: "Action") -> "Action":
        """Returns the opposite action of the given action."""
        match action:
            case cls.UP:
                return cls.DOWN
            case cls.DOWN:
                return cls.UP
            case cls.LEFT:
                return cls.RIGHT
            case cls.RIGHT:
                return cls.LEFT
            case _:
                raise ValueError(f"Invalid action: {action}")

    @property
    def inverse(self) -> "Action":
        """Returns the opposite action."""
        return Action.opposite(self)
    
Int2DArray = np.ndarray[tuple[int, int], np.dtype[np.int64]]
IntArray = np.ndarray[tuple[int], np.dtype[np.int64]]


class Problem:
    _state: Int2DArray
    _zero_position: tuple[int, int]
    _log: IntArray

    def __init__(
        self,
        initial_state: Int2DArray | None = None,
        log: tuple[Action, ...] | None = None,
    ):
        self._state = (
            initial_state if initial_state is not None else self._make_solvable_state()
        )
        zero_pos = np.argwhere(self._state == 0)
        self._zero_position = (int(zero_pos[0][0]), int(zero_pos[0][1]))
        self._log = (
            np.array([a.value for a in log], dtype=int)
            if log is not None
            else np.array([], dtype=int)
        )

    @classmethod
    def _from_internal(
        cls, initial_state: Int2DArray, log: tuple[int, ...] | None = None
    ) -> "Problem":
        problem = cls(initial_state)
        problem._log = log if log is not None else np.array([], dtype=int)
        return problem

    @classmethod
    def _is_solvable(cls, state: IntArray | Int2DArray) -> bool:
        count = 0
        state = state[state != 0].flatten()
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] > state[j]:
                    count += 1
        return count % 2 == 0

    @classmethod
    def _make_solvable_state(
        cls,
        random_state: int | None = None,
        n_permutation: int = 20,
    ) -> Int2DArray:
        state = np.arange(9)

        rng = np.random.default_rng(seed=random_state)
        state = rng.permutation(state)

        if n_permutation % 2 != 0:
            n_permutation += 1

        for _ in range(n_permutation):
            i, j = rng.choice(len(state), size=2, replace=False)
            state[i], state[j] = state[j], state[i]

        if not cls._is_solvable(state):
            non_zero_indices = np.where(state != 0)[0]
            i, j = rng.choice(non_zero_indices, size=2, replace=False)
            state[i], state[j] = state[j], state[i]

        return state.reshape(3, 3)

    @classmethod
    def make_solvable(
        cls, random_state: int | None = None, n_permutation: int = 20
    ) -> "Problem":
        """Generates a random solvable 8-puzzle problem."""
        return cls(cls._make_solvable_state(random_state, n_permutation))

    def copy(self) -> "Problem":
        return Problem._from_internal(self._state.copy(), log=self._log)

    @property
    def state(self) -> Int2DArray:
        return self._state.copy()

    @property
    def log(self) -> tuple[Action, ...]:
        return tuple(Action(a) for a in self._log)

    @property
    def int_log(self) -> IntArray:
        return self._log.copy()

    @property
    def last_action(self) -> Action | None:
        if len(self._log) == 0:
            return None
        return Action(self._log[-1])

    @property
    def action_down(self) -> Optional["Problem"]:
        new_position = (self._zero_position[0] + 1, self._zero_position[1])

        return self._perform_action(new_position, Action.DOWN)

    @property
    def action_up(self) -> Optional["Problem"]:
        new_position = (self._zero_position[0] - 1, self._zero_position[1])

        return self._perform_action(new_position, Action.UP)

    @property
    def action_left(self) -> Optional["Problem"]:
        new_position = (self._zero_position[0], self._zero_position[1] - 1)

        return self._perform_action(new_position, Action.LEFT)

    @property
    def action_right(self) -> Optional["Problem"]:
        new_position = (self._zero_position[0], self._zero_position[1] + 1)

        return self._perform_action(new_position, Action.RIGHT)

    @property
    def all_actions(self) -> dict[Action, Optional["Problem"]]:
        return {
            Action.UP: self.action_up,
            Action.DOWN: self.action_down,
            Action.LEFT: self.action_left,
            Action.RIGHT: self.action_right,
        }

    @property
    def possible_actions(self) -> dict[Action, "Problem"]:
        return {
            action: problem
            for action, problem in self.all_actions.items()
            if problem is not None
        }

    @property
    def zero_position(self) -> tuple[int, int]:
        return self._zero_position

    @property
    def neighbors(self) -> list[tuple[Action, "Problem"]]:
        return list(self.possible_actions.items())

    def execute_log(self, log: Sequence[Action]) -> tuple["Problem", list["Problem"]]:
        curr_prob = self
        intermediate_states = [curr_prob]
        for i, action in enumerate(log):
            curr_prob = curr_prob._execute_by_action(action)
            if curr_prob is None:
                raise ValueError(f"Error when executing the {i}th action: {action}")
            intermediate_states.append(curr_prob)

        return curr_prob, intermediate_states

    def _execute_by_action(self, action: Action) -> Optional["Problem"]:
        match action:
            case Action.UP:
                return self.action_up
            case Action.DOWN:
                return self.action_down
            case Action.LEFT:
                return self.action_left
            case Action.RIGHT:
                return self.action_right
            case _:
                raise ValueError(f"Invalid action: {action}")

    def _perform_action(
        self, new_pos: tuple[int, int], action: Action
    ) -> Optional["Problem"]:
        if not self._check_new_position(new_pos):
            return None

        new_state = self.state
        new_state[self._zero_position], new_state[new_pos] = (
            new_state[new_pos],
            new_state[self._zero_position],
        )

        new_log = np.append(self._log, action.value)

        return Problem._from_internal(new_state, log=new_log)

    @property
    def is_solved(self) -> bool:
        flat = self.state.flatten()
        flat_no_zero = flat[flat != 0]
        return np.array_equal(flat_no_zero, np.array([1, 2, 3, 4, 5, 6, 7, 8]))

    def _check_new_position(self, new_position: tuple[int, int]) -> bool:
        shape = self._state.shape
        return (
            new_position[0] >= 0
            and new_position[0] < shape[0]
            and new_position[1] >= 0
            and new_position[1] < shape[1]
        )

    def __str__(self) -> str:
        state = str(self._state).replace("\n", "")
        return f"Problem(state={state}, log_size={len(self.log)})"

    def __hash__(self) -> int:
        return hash(self._state.tobytes())

    def __eq__(self, __o: object) -> bool:
        if __o is None:
            return False
        if not isinstance(__o, Problem):
            return False
        return np.array_equal(self._state, __o._state)

    def __repr__(self) -> str:
        return str(self)
    
T = TypeVar("T")

FrontierStrategy = Literal["priority", "stack", "queue"]


class Frontier(ABC, Generic[T]):
    _queue: Sequence[T]

    @abstractmethod
    def put(self, item: T, priority: float | None = None): ...

    @abstractmethod
    def pop(self) -> T: ...

    def is_empty(self) -> bool:
        return len(self._queue) == 0 if hasattr(self, "_queue") else True

    def __iter__(self):
        return iter(self._queue) if hasattr(self, "_queue") else iter([])

    def __len__(self) -> int:
        return len(self._queue) if hasattr(self, "_queue") else 0

    def __contains__(self, item: T) -> bool:
        return item in self._queue if hasattr(self, "_queue") else False

    def __str__(self) -> str:
        data = str(list(self._queue)) if hasattr(self, "_queue") else "[]"
        return f"{self.__class__.__name__}({data})"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_iterable(cls, items: Sequence[T]) -> "Frontier[T]":
        if not issubclass(cls, Frontier):
            raise TypeError(f"{cls.__name__} is not a Frontier subclass")

        queue: Frontier[T] = cls()
        for item in items:
            queue.put(item)
        return queue

    @classmethod
    def from_strategy(cls, strategy: FrontierStrategy) -> "Frontier[T]":
        match strategy:
            case "priority":
                return PriorityFrontier[T]()
            case "stack":
                return LIFOFrontier[T]()
            case "queue":
                return FIFOFrontier[T]()
            case _:
                raise ValueError(f"Invalid strategy: {strategy}")


class LIFOFrontier(Frontier[T]):
    _queue: list[T]

    def __init__(self) -> None:
        self._queue = []

    def put(self, item: T, priority: float | None = None):
        self._queue.append(item)

    def pop(self) -> T:
        return self._queue.pop()


class FIFOFrontier(Frontier[T]):
    _queue: deque[T]

    def __init__(self) -> None:
        self._queue = deque()

    def put(self, item: T, priority: float | None = None):
        self._queue.append(item)

    def pop(self) -> T:
        return self._queue.popleft()


class PriorityFrontier(Frontier[T]):
    _queue: list[tuple[float | None, int, T]]

    def __init__(self) -> None:
        self._queue = []
        self._counter = itertools.count()  # Contador incremental

    def put(self, item: T, priority: float | None = None):
        if priority is None:
            priority = 0
        count = next(self._counter)
        heapq.heappush(self._queue, (priority, count, item))

    def pop(self) -> T:
        _, _, item = heapq.heappop(self._queue)
        return item

    def __contains__(self, item: T) -> bool:
        return any(x[2] == item for x in self._queue)

    def __iter__(self):
        return iter(x[2] for x in self._queue)

    def __str__(self) -> str:
        data = list(map(lambda i: f"({i[2]}, priority={i[0]})", self._queue))
        return f"PriorityQueue({data})"
    
@dataclass
class Result:
    cost: int | None
    depth: int | None
    max_depth: int
    visited: int
    generated: int
    time: float
    initial: Problem
    final: Problem | None
    path: list[Action]
    random_state: int | None = None

HeuristicFunction = Callable[[Problem], float]
CostFunction = Callable[[Action, tuple[int, int]], float]
Algorithm = Literal[
    "bfs",
    "dfs",
    "dijkstra",
    "greedy",
    "a_star",
]


class Solver:
    _heuristic: HeuristicFunction | None
    _cost_function: CostFunction
    _queue_mode: FrontierStrategy
    _best_first: bool = False

    @staticmethod
    def _null_cf(action: Action, _: tuple[int, int]) -> float:
        """null cost function that always returns 0."""
        return 0

    @staticmethod
    def _fixed_cf(action: Action, _: tuple[int, int]) -> float:
        """fixed cost function that always returns 1."""
        return 1.0

    def __init__(
        self,
        heuristic: HeuristicFunction | None = None,
        cost_function: CostFunction | None = None,
        queue: FrontierStrategy = "queue",
        best_first: bool = False,
        greedy: bool = False,
    ):
        """
        Initializes the solver with a heuristic function, cost function, and queue mode.
        :param heuristic: A function that estimates the cost to reach the goal from a given state.
        :param cost_function: A function that returns the cost of a given action.
        :param queue: The type of queue to use for the search strategy. Options are "priority", "stack", or "queue".
        :param greedy: If True, uses a greedy search strategy, ignoring the cost of actions.
        :raises ValueError: If greedy is True and no heuristic is provided.
        :raises ValueError: If an invalid queue type is provided.
        """
        self._heuristic = heuristic
        self._queue_mode = queue
        self._cost_function = (
            cost_function if cost_function is not None else self._fixed_cf
        )
        if greedy and self._heuristic is None:
            raise ValueError("Heuristic must be provided for greedy search.")
        elif greedy:
            self._cost_function = self._null_cf

        self._best_first = best_first

    @classmethod
    def from_algorithm(
        cls,
        algorithm: Algorithm,
        heuristic: HeuristicFunction | None = None,
        cost_function: CostFunction | None = None,
    ) -> "Solver":
        match algorithm:
            case "bfs":
                return cls(None, cost_function, queue="queue", best_first=False)
            case "dfs":
                return cls(None, cost_function, queue="stack", best_first=False)
            case "dijkstra":
                return cls(None, cost_function, queue="priority", best_first=True)
            case "greedy":
                if heuristic is None:
                    raise ValueError("Heuristic must be provided for greedy search.")
                return cls(
                    heuristic, None, queue="priority", greedy=True, best_first=True
                )
            case "a_star":
                if heuristic is None:
                    raise ValueError("Heuristic must be provided for A* search.")
                return cls(heuristic, cost_function, queue="priority", best_first=True)
            case _:
                raise ValueError(f"Invalid algorithm: {algorithm}")

    def solve(
        self,
        initial: Problem,
        max_depth: int = -1,
        shuffle_actions: bool = False,
        random_state: int | None = None,
    ) -> Result:
        """
        Solves the given problem using the specified search strategy.
        Implements the `Graph Search` algorithm from AIMA 3rd edition, Russell & Norvig, section 3.3
        :param initial: The initial state of the problem to be solved.
        :param max_depth: The maximum depth to search. If -1, no limit is applied.
        :param shuffle_actions: If True, shuffles the actions before executing them.
        :param random_state: An optional random state for reproducibility.
        :return: A Result object containing the path, cost, initial state, and final state if a solution is found, or None if no solution exists.
        """
        start = time.time()
        frontier: Frontier[tuple[Problem, float]] = Frontier.from_strategy(
            self._queue_mode
        )
        frontier.put((initial, 0), priority=0)
        explored: dict[int, float] = {}
        visited = 0
        generated = 0

        rng = np.random.default_rng(seed=random_state) if shuffle_actions else None
        while not frontier.is_empty():
            state, cost = frontier.pop()

            if self._is_exausted(state, max_depth):
                continue

            state_hash = hash(state)
            if self._should_skip(state_hash, cost, explored):
                continue

            explored[state_hash] = cost
            visited += 1

            if state.is_solved:
                return Result(
                    path=state.log,
                    cost=cost,
                    final=state,
                    initial=initial,
                    visited=visited,
                    generated=generated,
                    max_depth=max_depth,
                    depth=len(state.log),
                    time=time.time() - start,
                    random_state=random_state,
                )

            neighbors = state.neighbors
            if shuffle_actions:
                neighbors = rng.permutation(neighbors)

            for action, new_state in neighbors:
                new_hash = hash(new_state)
                new_cost = cost + self._cost_function(action, new_state.zero_position)

                priority = new_cost
                if self._heuristic:
                    priority += self._heuristic(new_state)

                if self._should_expand(new_hash, new_cost, explored):
                    explored[new_hash] = new_cost
                    frontier.put((new_state, new_cost), priority=priority)
                    generated += 1

        return Result(
            path=[],
            cost=None,
            final=None,
            initial=initial,
            visited=visited,
            generated=generated,
            max_depth=max_depth,
            random_state=random_state,
            depth=None,
            time=time.time() - start,
        )

    @staticmethod
    def _is_exausted(state: Problem, max_depth: int) -> bool:
        return len(state.log) > max_depth and max_depth != -1

    def _should_expand(
        self, state_hash: int, cost: float, explored: dict[int, float]
    ) -> bool:
        if self._best_first:
            return cost < explored.get(state_hash, float("inf"))
        return state_hash not in explored

    def _should_skip(
        self, state_hash: int, cost: float, explored: dict[int, float]
    ) -> bool:
        return self._best_first and cost > explored.get(state_hash, float("inf"))

    def __str__(self):
        data = ", ".join(
            (
                f"heuristic={self._heuristic.__name__ if self._heuristic else None}",
                f"cost_function={self._cost_function.__name__}",
                f"queue_mode={self._queue_mode}",
            )
        )
        return f"Solver({data})"

    def __repr__(self):
        return str(self)

class IterativeSolver(Solver):
    def solve(
        self,
        initial: Problem,
        max_depth: int,
        shuffle_actions: bool = False,
        random_state: int | None = None,
    ) -> Result:
        """
        Solves the given problem using iterative deepening search.
        This method iteratively increases the depth limit and applies the base solver's solve method.
        :param initial: The initial state of the problem to be solved.
        :param max_depth: The maximum depth to search.
        :param shuffle_actions: If True, shuffles the actions before executing them.
        :param random_state: An optional random state for reproducibility.
        :return: A Result object containing the path, cost, initial state, and final state if a solution is found, or None if no solution exists.
        :raises ValueError: If max_depth is negative.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        current = Result(
            cost=None,
            depth=None,
            max_depth=0,
            visited=0,
            generated=0,
            time=0.0,
            initial=initial,
            final=None,
            path=[],
        )
        for depth in range(max_depth + 1):
            result = super().solve(
                initial,
                max_depth=depth,
                shuffle_actions=shuffle_actions,
                random_state=random_state,
            )
            if result.final is not None:
                result.generated += current.generated
                result.visited += current.visited
                result.time += current.time
                return result
            else:
                current.max_depth = result.max_depth
                current.visited += result.visited
                current.generated += result.generated
                current.time += result.time
        return current

def animate_8puzzle_solution(
    solution: Result, algorithm_name="Graph-Search", tile_size=100
):
    initial = solution.initial
    final = solution.final

    _, steps = initial.execute_log(solution.path)

    def step_name(step):
        if step == 0:
            return "Initial State"

        if step == len(steps) - 1:
            return "Final State"

        return f"{step}: {final.log[step].name}"

    data_for_plotly = [
        {
            "step": step_name(step_idx),
            "value": prob.state[row, col],
            "col": col,
            "row": row,
            "text_color": "black",
        }
        for step_idx, prob in enumerate(steps)
        for row in range(prob.state.shape[0])
        for col in range(prob.state.shape[1])
    ]

    df = pd.DataFrame(data_for_plotly)

    fig = px.scatter(
        df,
        x="col",
        y="row",
        animation_frame="step",
        text="value",
        range_x=[-0.5, 2.5],
        range_y=[-0.5, 2.5],
        title="8-Puzzle Animated Solution",
        subtitle=" - ".join(
            [
                f"Algorithm: {algorithm_name}",
                f"Cost: {solution.cost}",
                f"Steps: {solution.depth}",
                f"Visited: {solution.visited}",
                f"Generated: {solution.generated}",
                f"Time: {solution.time:.2f} seconds",
            ]
        ),
        size_max=80,
        color_discrete_sequence=["lightblue"],
        labels={"col": "Column", "row": "Row", "value": "Value"},
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_traces(
        mode="markers+text",
        marker=dict(
            size=tile_size * 0.8,
            sizemode="diameter",
            symbol="square",
            color="lightblue",
            line=dict(width=2, color="DarkSlateGrey"),
        ),
        textfont=dict(size=int(tile_size * 0.4), color="black"),
        textposition="middle center",
    )

    fig.update_annotations(
        textfont_size=int(tile_size * 0.4),
        textangle=0,
        showarrow=False,
        font_color="black",
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    fig.update_layout(
        plot_bgcolor="gray", xaxis_title=None, yaxis_title=None, height=600, width=600
    )

    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 500
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250

    fig.show()

All = Literal["all"]


class Experiment:
    _solvers: dict[Algorithm, list[tuple[str | None, str | None, Solver]]]
    _random_state: list[int] | None = None
    _shuffle_actions: bool = False
    _max_depth: int = -1

    def __init__(
        self,
        algorithms: Sequence[Algorithm] | Algorithm | All,
        cfs: Sequence[CostFunction] | None | All = None,
        hfs: Sequence[HeuristicFunction] | None | All = None,
        shuffle_actions: bool = False,
        random_state: int | list[int] | None = None,
        max_depth: int = -1,
    ):
        """
        Initializes the Experiment with a list of algorithms, cost functions, and heuristic functions.
        :param algorithms: A sequence of algorithms to be used in the experiment.
        :param cfs: A sequence of cost functions to be used in the experiment. If None, a default cost function is used.
        :param hfs: A sequence of heuristic functions to be used in the experiment. If None, no heuristic is used.
        :param shuffle_actions: If True, shuffles the actions before executing them.
        :param random_state: An optional random state for reproducibility.
        :param max_depth: The maximum depth to search. If -1, no limit is applied.
        :raises ValueError: If an invalid algorithm is provided.
        :raises ValueError: If an invalid cost function or heuristic function is provided.
        :raises ValueError: If a greedy search is requested without a heuristic function.
        """
        if not random_state:
            random_state = None
        elif isinstance(random_state, int):
            random_state = [random_state]

        if isinstance(algorithms, str):
            algorithms = [algorithms]

        self._solvers = self._make_solvers(algorithms, cfs, hfs)
        self._random_state = random_state
        self._shuffle_actions = shuffle_actions or bool(random_state)
        self._max_depth = max_depth

    @property
    def solvers(self) -> list[tuple[Algorithm, str, str, Solver]]:
        """
        Returns a list of tuples containing the algorithm name,
        cost function name, heuristic function name, and the Solver instance.
        """
        return [
            (algo, str(cf_name), str(hf_name), solver)
            for algo in self._solvers
            for (cf_name, hf_name, solver) in self._solvers[algo]
        ]

    @staticmethod
    def _make_solvers(
        algorithms: Sequence[Algorithm] | All,
        cfs: Sequence[CostFunction] | None | All = None,
        hfs: Sequence[HeuristicFunction] | None | All = None,
    ) -> dict[Algorithm, list[tuple[str | None, str | None, Solver]]]:
        if algorithms == "all":
            algorithms = Algorithm.__args__

        elif cfs is None:
            cfs = [None]

        elif hfs is None:
            hfs = [None]

        def name_or_none(f):
            return f.__name__ if f else None

        return {
            algo: [
                (
                    name_or_none(cf),
                    name_or_none(hf),
                    Solver.from_algorithm(algo, hf, cf),
                )
                for cf in cfs
                for hf in hfs
            ]
            for algo in algorithms
        }

    @staticmethod
    def _execute_solver(
        state: Problem,
        solver: Solver,
        algorithm: str,
        cf_name: str | None = None,
        hf_name: str | None = None,
        random_state: int | None = None,
        shuffle_actions: bool = False,
        max_depth: int = -1,
    ) -> dict:
        result = solver.solve(
            state,
            max_depth,
            shuffle_actions,
            random_state,
        )
        metadata = dict(
            algorithm=algorithm, cost_function=cf_name, heuristic=hf_name, raw=result
        )
        return metadata | asdict(result)

    def run(self, states: Sequence[Problem]) -> pd.DataFrame:
        """
        Runs the experiment on a sequence of Problem states using the defined solvers.
        :param states: A sequence of Problem instances to be solved.
        :return: A pandas DataFrame containing the results of the experiment.
        """
        results: list[dict] = []

        rs_list = self._random_state if self._random_state else [None]

        solver_tasks = [
            (algorithm, cf_name, hf_name, random_state, solver)
            for algorithm, solvers in self._solvers.items()
            for random_state in rs_list
            for cf_name, hf_name, solver in solvers
        ]

        estimated_total = len(solver_tasks) * len(states)
        with tqdm(total=estimated_total, desc="Executions", unit="exec") as pbar:
            pbar.set_description("Executing solvers")
            pbar.set_postfix({"Estimated Total": estimated_total})

            with Pool() as pool:
                for algorithm, cf_name, hf_name, random_state, solver in solver_tasks:
                    worker = partial(
                        self._execute_solver,
                        solver=solver,
                        algorithm=algorithm,
                        cf_name=cf_name,
                        hf_name=hf_name,
                        random_state=random_state,
                        shuffle_actions=self._shuffle_actions,
                        max_depth=self._max_depth,
                    )

                    partial_results = pool.map(worker, states)
                    results.extend(partial_results)
                    pbar.update(len(partial_results))

        df = pd.DataFrame(results)

        df["path"] = df["final"].apply(lambda x: tuple(x.int_log) if x else tuple())
        df["final"] = df["final"].apply(
            lambda x: tuple(x.state.flatten()) if x else None
        )
        df["initial"] = df["initial"].apply(
            lambda x: tuple(x.state.flatten()) if x else None
        )

        return df
    
def save_parquet(df: pd.DataFrame, filename: str, base_path: Path) -> None:
    """
    Saves the DataFrame to a Parquet file.
    :param df: DataFrame to be saved.
    :param filename: Name of the file to save the DataFrame to.
    """
    df["raw"] = df["raw"].apply(pickle.dumps)
    df.to_parquet(base_path / filename, index=False, compression="zstd")
    print(f"DataFrame saved to {filename}")