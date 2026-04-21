"""
Simple grid navigation environment for norm discovery.

State = (row, col), Actions = NORTH/SOUTH/EAST/WEST.
Some cells are normatively prohibited; the unconstrained planner ignores norms.
"""
from collections import deque
import numpy as np

NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
ACTIONS = [NORTH, SOUTH, EAST, WEST]
ACTION_NAMES = {NORTH: 'N', SOUTH: 'S', EAST: 'E', WEST: 'W'}
ACTION_DELTA = {NORTH: (-1, 0), SOUTH: (1, 0), EAST: (0, 1), WEST: (0, -1)}

# 10x10 grid: 0=passable, 1=wall
# Shelf pairs at rows 2, 4, 6 create cross-aisles at cols 1 and 8,
# forcing detours when central cells are prohibited.
DEFAULT_GRID = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)

# Prohibited cells: normatively forbidden to enter.
# Placed in cross-aisle centers; agents must use col-1 or col-8 passages.
DEFAULT_PROHIBITED = frozenset([
    (2, 4), (2, 5),
    (4, 4), (4, 5),
    (6, 4), (6, 5),
    (3, 4), (3, 5),
    (5, 4), (5, 5),
    (7, 4), (7, 5),
    (1, 4), (1, 5),
])


class GridNormEnv:
    def __init__(self, grid=None, prohibited_cells=None):
        self.grid = np.array(grid, dtype=np.int8) if grid is not None else DEFAULT_GRID
        self.height, self.width = self.grid.shape
        self.prohibited_cells = (
            frozenset(prohibited_cells) if prohibited_cells is not None
            else DEFAULT_PROHIBITED
        )
        self.passable = frozenset(
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if self.grid[r, c] == 0
        )
        self._dist_cache: dict = {}

    def is_passable(self, state):
        return state in self.passable

    def is_prohibited(self, state):
        return state in self.prohibited_cells

    def next_state(self, state, action):
        dr, dc = ACTION_DELTA[action]
        ns = (state[0] + dr, state[1] + dc)
        return ns if ns in self.passable else None

    def get_valid_actions(self, state):
        return [a for a in ACTIONS if self.next_state(state, a) is not None]

    # ------------------------------------------------------------------
    # BFS helpers
    # ------------------------------------------------------------------

    def _bfs_from(self, source):
        dist = {source: 0}
        q = deque([source])
        while q:
            s = q.popleft()
            for a in ACTIONS:
                ns = self.next_state(s, a)
                if ns is not None and ns not in dist:
                    dist[ns] = dist[s] + 1
                    q.append(ns)
        return dist

    def dist(self, s1, s2):
        """Unconstrained shortest-path distance (ignores norms)."""
        if s1 == s2:
            return 0
        if s1 not in self._dist_cache:
            self._dist_cache[s1] = self._bfs_from(s1)
        return self._dist_cache[s1].get(s2, float('inf'))

    def get_shortest_paths(self, start, goal, max_paths=8):
        """
        Return up to max_paths shortest paths from start to goal (ignoring norms).
        Each path is a list of (state, action) pairs such that applying each action
        in sequence from start reaches goal.
        """
        if start == goal:
            return [[]]

        d_from_start = self._bfs_from(start)
        if goal not in d_from_start:
            return []
        target = d_from_start[goal]

        d_from_goal = self._bfs_from(goal)
        paths: list = []

        def dfs(state, path, visited):
            if len(paths) >= max_paths:
                return
            if state == goal:
                paths.append(list(path))
                return
            remaining = target - len(path)
            if remaining <= 0:
                return
            for action in ACTIONS:
                ns = self.next_state(state, action)
                if ns is None or ns in visited:
                    continue
                if d_from_goal.get(ns, float('inf')) != remaining - 1:
                    continue
                visited.add(ns)
                path.append((state, action))
                dfs(ns, path, visited)
                path.pop()
                visited.remove(ns)

        dfs(start, [], {start})
        return paths

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def norm_following_path(self, start, goal):
        """BFS shortest path that avoids prohibited cells."""
        if start == goal:
            return []
        dist = {start: 0}
        prev: dict = {start: None}
        q = deque([start])
        while q:
            s = q.popleft()
            if s == goal:
                break
            for a in ACTIONS:
                ns = self.next_state(s, a)
                if ns is None or ns in dist or ns in self.prohibited_cells:
                    continue
                dist[ns] = dist[s] + 1
                prev[ns] = (s, a)
                q.append(ns)
        if goal not in prev:
            return None
        path = []
        cur = goal
        while prev[cur] is not None:
            s, a = prev[cur]
            path.append((s, a))
            cur = s
        path.reverse()
        return path
