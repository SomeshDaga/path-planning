from collections import deque
import heapq
import numpy as np


class AStar:
    def __init__(self):
        pass

    @staticmethod
    def reconstruct_path(came_from, current_vertex):
        path = deque([current_vertex])
        while current_vertex in came_from.keys():
            current_vertex = came_from[current_vertex]
            path.appendleft(current_vertex)
        return path

    @staticmethod
    def find_path(start_vertex, goal_vertex, h):
        # Initialize dictionaries to store the scores of vertices
        # and to store the best neighbour connections
        g_score = dict()
        f_score = dict()
        came_from = dict()

        # Assign the scores for the start vertex
        g_score[start_vertex] = 0
        f_score[start_vertex] = g_score[start_vertex] + h(start_vertex)

        # Maintain a priority queue (min heap) to track the vertexes to iterate
        open_set = []  # Use a priority heap to push/pop elements
        heapq.heappush(open_set, (f_score[start_vertex], start_vertex))

        while len(open_set) > 0:
            # Get vertex in our open set with the lowest cost
            vertex = heapq.heappop(open_set)[1]

            # If we've reached the goal, return the path that we took
            if vertex == goal_vertex:
                return AStar.reconstruct_path(came_from, vertex)

            for neighbour in vertex.get_neighbours():
                neighbour_g_score = g_score[vertex] + np.sqrt(vertex.calc_dist_2(neighbour))

                if neighbour not in g_score.keys() or neighbour_g_score < g_score[neighbour]:
                    came_from[neighbour] = vertex
                    g_score[neighbour] = neighbour_g_score
                    f_score[neighbour] = g_score[neighbour] + h(neighbour)
                    if neighbour not in list(n[1] for n in open_set):
                        heapq.heappush(open_set, (f_score[neighbour], neighbour))

        # If a path was not found, return None
        return None
