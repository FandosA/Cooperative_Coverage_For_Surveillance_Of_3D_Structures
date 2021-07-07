import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import matplotlib.animation as animation


class Landmark:

    def __init__(self, landmark_id=None, landmark_pos=None, landmark_normal=None, associated_agent=None):
        self.__ID = landmark_id
        self.__position = landmark_pos
        self.__normal = landmark_normal
        self.__associated_agent = associated_agent

    def get_id(self):
        return self.__ID

    def get_position(self):
        return self.__position

    def get_normal(self):
        return self.__normal

    def get_associated_agent(self):
        return self.__associated_agent

    def set_position(self, new_position):
        self.__position = new_position

    def set_normal(self, new_normal):
        self.__normal = new_normal

    def set_associated_agent(self, new_associated_agent):
        self.__associated_agent = new_associated_agent


class Agent:

    def __init__(self, agent_id=None, agent_position=None, agent_orientation=None, associated_landmarks=None):
        if associated_landmarks is None:
            associated_landmarks = []
        self.__ID = agent_id
        self.__position = agent_position
        self.__orientation = agent_orientation
        self.__associated_landmarks = associated_landmarks

    def get_id(self):
        return self.__ID

    def get_position(self):
        return self.__position

    def get_orientation(self):
        return self.__orientation

    def get_associated_landmarks(self):
        return self.__associated_landmarks

    def set_position(self, new_position):
        self.__position = new_position

    def set_orientation(self, new_orientation):
        self.__orientation = new_orientation

    def add_associated_landmark(self, new_landmark_index):
        global all_landmarks
        self.__associated_landmarks.append(new_landmark_index)
        self.__associated_landmarks.sort()
        new_landmark = all_landmarks[new_landmark_index]
        new_landmark.set_associated_agent(self.get_id())

    def remove_associated_landmark(self, index):
        self.__associated_landmarks.remove(index)

    def compute_landmark_visibility(self, landmark_vis):

        if ((self.__orientation / np.linalg.norm(self.__orientation)) @
                (landmark_vis.get_normal() / np.linalg.norm(landmark_vis.get_normal()))) < 0:
            f = 0
        else:
            inside_norm_1 = self.get_position() + D * self.get_orientation() - landmark_vis.get_position()
            f_norm_1 = np.linalg.norm(inside_norm_1)

            inside_norm_2 = self.get_position() + D * landmark_vis.get_normal() - landmark_vis.get_position()
            f_norm_2 = np.linalg.norm(inside_norm_2)

            f = - (f_norm_1 ** 2) * (ALPHA + BETA * ((self.get_orientation().T @ inside_norm_1) / f_norm_1)) \
                - GAMMA * (f_norm_2 ** 2) * (ALPHA + BETA * ((landmark_vis.get_normal().T @ inside_norm_2) / f_norm_2))

        return f

    def diff_visibility_pos_x(self, landmark_vis):
        p_s = self.get_position()
        p_s_x = p_s[0]
        u_s = self.get_orientation()
        u_s_x = u_s[0]
        p_l = landmark_vis.get_position()
        p_l_x = p_l[0]
        u_l = landmark_vis.get_normal()
        u_l_x = u_l[0]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * (p_s_x + D * u_s_x - p_l_x)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * ((u_s_x * norm_1 -
                                   (u_s.T @ inside_norm_1) * (p_s_x + D * u_s_x - p_l_x) / norm_1) / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        inside_norm_2 = p_s + D * u_l - p_l
        norm_2 = np.linalg.norm(inside_norm_2)

        second1_not_derivated = norm_2 ** 2
        second1_derivated = 2 * (p_s_x + D * u_l_x - p_l_x)

        second2_not_derivated = (ALPHA + BETA * ((u_l.T @ inside_norm_2) / norm_2))
        second2_derivated = BETA * ((u_l_x * norm_2 -
                                    (u_l.T @ inside_norm_2) * (p_s_x + D * u_l_x - p_l_x) / norm_2) / (norm_2 ** 2))

        second_term = - GAMMA * (second1_derivated * second2_not_derivated + second1_not_derivated * second2_derivated)

        return first_term + second_term

    def diff_visibility_pos_y(self, landmark_vis):
        p_s = self.get_position()
        p_s_y = p_s[1]
        u_s = self.get_orientation()
        u_s_y = u_s[1]
        p_l = landmark_vis.get_position()
        p_l_y = p_l[1]
        u_l = landmark_vis.get_normal()
        u_l_y = u_l[1]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * (p_s_y + D * u_s_y - p_l_y)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * ((u_s_y * norm_1 -
                                   (u_s.T @ inside_norm_1) * (p_s_y + D * u_s_y - p_l_y) / norm_1) / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        inside_norm_2 = p_s + D * u_l - p_l
        norm_2 = np.linalg.norm(inside_norm_2)

        second1_not_derivated = norm_2 ** 2
        second1_derivated = 2 * (p_s_y + D * u_l_y - p_l_y)

        second2_not_derivated = (ALPHA + BETA * ((u_l.T @ inside_norm_2) / norm_2))
        second2_derivated = BETA * ((u_l_y * norm_2 -
                                    (u_l.T @ inside_norm_2) * (p_s_y + D * u_l_y - p_l_y) / norm_2) / (norm_2 ** 2))

        second_term = - GAMMA * (second1_derivated * second2_not_derivated + second1_not_derivated * second2_derivated)

        return first_term + second_term

    def diff_visibility_pos_z(self, landmark_vis):
        p_s = self.get_position()
        p_s_z = p_s[2]
        u_s = self.get_orientation()
        u_s_z = u_s[2]
        p_l = landmark_vis.get_position()
        p_l_z = p_l[2]
        u_l = landmark_vis.get_normal()
        u_l_z = u_l[2]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * (p_s_z + D * u_s_z - p_l_z)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * ((u_s_z * norm_1 -
                                   (u_s.T @ inside_norm_1) * (p_s_z + D * u_s_z - p_l_z) / norm_1) / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        inside_norm_2 = p_s + D * u_l - p_l
        norm_2 = np.linalg.norm(inside_norm_2)

        second1_not_derivated = norm_2 ** 2
        second1_derivated = 2 * (p_s_z + D * u_l_z - p_l_z)

        second2_not_derivated = (ALPHA + BETA * ((u_l.T @ inside_norm_2) / norm_2))
        second2_derivated = BETA * ((u_l_z * norm_2 -
                                    (u_l.T @ inside_norm_2) * (p_s_z + D * u_l_z - p_l_z) / norm_2) / (norm_2 ** 2))

        second_term = - GAMMA * (second1_derivated * second2_not_derivated + second1_not_derivated * second2_derivated)

        return first_term + second_term

    def diff_visibility_ori_x(self, landmark_vis):
        p_s = self.get_position()
        p_s_x = p_s[0]
        u_s = self.get_orientation()
        u_s_x = u_s[0]
        p_l = landmark_vis.get_position()
        p_l_x = p_l[0]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * D * (p_s_x + D * u_s_x - p_l_x)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * (((p_s_x + 2 * D * u_s_x - p_l_x) * norm_1 -
                                    (u_s.T @ inside_norm_1) * (D * (p_s_x + D * u_s_x - p_l_x)) / norm_1)
                                   / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        second_term = 0

        return first_term + second_term

    def diff_visibility_ori_y(self, landmark_vis):
        p_s = self.get_position()
        p_s_y = p_s[1]
        u_s = self.get_orientation()
        u_s_y = u_s[1]
        p_l = landmark_vis.get_position()
        p_l_y = p_l[1]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * D * (p_s_y + D * u_s_y - p_l_y)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * (((p_s_y + 2 * D * u_s_y - p_l_y) * norm_1 -
                                    (u_s.T @ inside_norm_1) * (D * (p_s_y + D * u_s_y - p_l_y)) / norm_1)
                                   / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        second_term = 0

        return first_term + second_term

    def diff_visibility_ori_z(self, landmark_vis):
        p_s = self.get_position()
        p_s_z = p_s[2]
        u_s = self.get_orientation()
        u_s_z = u_s[2]
        p_l = landmark_vis.get_position()
        p_l_z = p_l[2]

        inside_norm_1 = p_s + D * u_s - p_l
        norm_1 = np.linalg.norm(inside_norm_1)

        first1_not_derivated = norm_1 ** 2
        first1_derivated = 2 * D * (p_s_z + D * u_s_z - p_l_z)

        first2_not_derivated = (ALPHA + BETA * ((u_s.T @ inside_norm_1) / norm_1))
        first2_derivated = BETA * (((p_s_z + 2 * D * u_s_z - p_l_z) * norm_1 -
                                    (u_s.T @ inside_norm_1) * (D * (p_s_z + D * u_s_z - p_l_z)) / norm_1)
                                   / (norm_1 ** 2))

        first_term = - (first1_derivated * first2_not_derivated + first1_not_derivated * first2_derivated)

        second_term = 0

        return first_term + second_term

    def plot_sample_visibility(self):

        global fig
        fig += 1
        figure = plt.figure(fig)

        # Compute footprints for each position of a grid
        vis = np.zeros((0, 3))
        n_grid = 200
        for i in range(n_grid):
            for j in range(n_grid):
                i_i = i / n_grid * 13 - 0.5
                j_j = j / n_grid * 5 - 2.5
                sample_landmark = Landmark(0, np.array([i_i, j_j, 0]).T, np.array([1, 0, 0]).T)
                footprint = self.compute_landmark_visibility(sample_landmark)
                vis = np.vstack((vis, np.array([i_i, j_j, footprint])))

        # Color bar
        m = plt.cm.ScalarMappable(cmap=plt.cm.rainbow)
        m.set_array(np.array([np.amax(vis[:, 2]), np.amin(vis[:, 2])]))
        figure.colorbar(m)

        df = pd.DataFrame({'X': vis[:, 0], 'Y': vis[:, 1], 'footprint': vis[:, 2]})
        plt.scatter(df.X, df.Y, c=df.footprint, cmap='rainbow', marker='.')
        plt.draw()
        plt.title('Footprint')

    def compute_individual_coverage(self):
        assigned_landmarks_indices = self.get_associated_landmarks()
        individual_coverage = 0
        for assigned_landmark_index in assigned_landmarks_indices:
            assigned_landmark = all_landmarks[assigned_landmark_index]
            individual_coverage += self.compute_landmark_visibility(assigned_landmark)

        return individual_coverage

    def diff_coverage_pos(self):
        assigned_landmarks_indices = self.get_associated_landmarks()
        diff_coverage = np.zeros(3)
        for assigned_landmark_index in assigned_landmarks_indices:
            assigned_landmark = all_landmarks[assigned_landmark_index]
            diff_coverage += np.array([self.diff_visibility_pos_x(assigned_landmark),
                                       self.diff_visibility_pos_y(assigned_landmark),
                                       self.diff_visibility_pos_z(assigned_landmark)])

        return diff_coverage.T

    def diff_coverage_ori(self):
        assigned_landmarks_indices = self.get_associated_landmarks()
        diff_coverage = np.zeros(3)
        for assigned_landmark_index in assigned_landmarks_indices:
            assigned_landmark = all_landmarks[assigned_landmark_index]
            diff_coverage += np.array([self.diff_visibility_ori_x(assigned_landmark),
                                       self.diff_visibility_ori_y(assigned_landmark),
                                       self.diff_visibility_ori_z(assigned_landmark)])

        return diff_coverage.T

    def apply_control_law(self):
        v = K_V * self.diff_coverage_pos()

        skew_ori = np.array([[0, -self.get_orientation()[2], self.get_orientation()[1]],
                             [self.get_orientation()[2], 0, -self.get_orientation()[0]],
                             [-self.get_orientation()[1], self.get_orientation()[0], 0]])

        w = -K_W * (skew_ori @ self.diff_coverage_ori())

        return v, w

    def apply_kinematics(self, v, w):
        self.set_position(self.get_position() + v)

        skew_w = np.array([[0, -w[2], w[1]],
                           [w[2], 0, -w[0]],
                           [-w[1], w[0], 0]])

        new_orientation = self.get_orientation() + skew_w @ self.get_orientation()
        self.set_orientation(new_orientation/np.linalg.norm(new_orientation))

    def distance_with_agent(self, agent_):
        return np.linalg.norm(self.get_position() - agent_.get_position())

    def communicate(self):
        global all_agents

        # Choose random neighbour
        chosen_agent = None
        valid = False
        while not valid:
            agent_communicate_id = np.random.randint(0, len(all_agents))
            if agent_communicate_id != self.get_id() and\
                    self.distance_with_agent(all_agents[agent_communicate_id]) < COMMUNICATION_RANGE:
                chosen_agent = all_agents[agent_communicate_id]
                valid = True
        # Transfer the landmarks the other agent has a better visibility of
        for transfer_landmark_index in self.get_associated_landmarks():
            transfer_landmark = all_landmarks[transfer_landmark_index]
            if self.compute_landmark_visibility(transfer_landmark)\
                    < chosen_agent.compute_landmark_visibility(transfer_landmark):
                if (len(self.get_associated_landmarks()) > 1):
                    self.remove_associated_landmark(transfer_landmark_index)
                    chosen_agent.add_associated_landmark(transfer_landmark_index)


def create_landmarks(plot_landmarks, lands_per_agent):

    landmarks_array = []

    # Spread landmarks randomly on the cylinder surface and store the points position
    # and their corresponding normals

    if plot_landmarks:
        global fig
        fig += 1
        plt.figure(fig)
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    n_agent = 0
    for l_id in range(N_LANDMARKS):

        new_landmark = Landmark(l_id)

        if l_id % lands_per_agent == 0 and l_id > 0 and (l_id < N_AGENTS * lands_per_agent):
            n_agent += 1

        new_landmark.set_associated_agent(n_agent)

        height = CYLINDER_HEIGHT * np.random.random()
        azimuth = 2 * np.pi * np.random.random()

        # Landmark position
        landmark_pos = np.array([CYLINDER_ORIGIN[0] + CYLINDER_RADIUS * np.cos(azimuth),
                                 CYLINDER_ORIGIN[1] + CYLINDER_RADIUS * np.sin(azimuth),
                                 CYLINDER_ORIGIN[2] + height])
        new_landmark.set_position(landmark_pos)

        # Landmark normal
        landmark_normal = np.array([landmark_pos[0] - CYLINDER_ORIGIN[0], landmark_pos[1] - CYLINDER_ORIGIN[1], 0])
        landmark_normal /= np.linalg.norm(landmark_normal)
        new_landmark.set_normal(landmark_normal)

        # Add to list
        landmarks_array.append(new_landmark)

        # Draw
        if plot_landmarks:
            ax.scatter(landmark_pos[0], landmark_pos[1], landmark_pos[2], marker='o', color='b')
            #ax.plot(np.array([landmark_pos[0], landmark_pos[0] + landmark_normal[0]]),
                    #np.array([landmark_pos[1], landmark_pos[1] + landmark_normal[1]]),
                    #np.array([landmark_pos[2], landmark_pos[2] + landmark_normal[2]]), color='b')
            ax.quiver(landmark_pos[0], landmark_pos[1], landmark_pos[2],
                      landmark_normal[0], landmark_normal[1], landmark_normal[2],
                      color='b', linewidth=2)
            plt.draw()

    if plot_landmarks:
        plt.title('Landmarks points in form of cylinder')

    return np.array(landmarks_array)


def compute_coverage_score_function():
    coverage_score_function = 0
    for agent in all_agents:
        coverage_score_function += agent.compute_individual_coverage()

    return coverage_score_function


def plot_coords_vs_iters():
    # X coordinate along iterations
    global N_ITERATIONS
    iters_array = np.arange(0, N_ITERATIONS)
    global fig
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_positions[:, j, 0], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_positions[0, j, 0], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_positions[-1, j, 0], marker='o', color=colours[j])
        plt.draw()
    plt.title('Trajectories of the robots in the x axis')
    plt.xlabel("Iterations")
    plt.ylabel("x-coordinate")

    # Y coordinate along iterations
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_positions[:, j, 1], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_positions[0, j, 1], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_positions[-1, j, 1], marker='o', color=colours[j])
        plt.draw()
    plt.title('Trajectories of the robots in the y axis')
    plt.xlabel("Iterations")
    plt.ylabel("y-coordinate")

    # Z coordinate along iterations
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_positions[:, j, 2], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_positions[0, j, 2], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_positions[-1, j, 2], marker='o', color=colours[j])
        plt.draw()
    plt.title('Trajectories of the robots in the z axis')
    plt.xlabel("Iterations")
    plt.ylabel("z-coordinate")


def plot_coverage_vs_iters():

    global N_ITERATIONS
    iters_array = np.arange(0, N_ITERATIONS)
    global fig
    fig += 1
    plt.figure(fig)
    list_robots = ['Coverage robot 1', 'Coverage robot 2', 'Coverage robot 3', 'Coverage robot 4', 'Total coverage']
    for j in range(N_AGENTS + 1):
        # Coverage
        plt.plot(iters_array, coverage_per_agent[:, j], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5, label=list_robots[j])
        plt.draw()
    plt.title('Coverage along iterations')
    plt.xlabel("Iterations")
    plt.ylabel("Coverage")
    plt.legend()


def plot_v_vs_iters():
    # v_x
    global N_ITERATIONS
    iters_array = np.arange(0, N_ITERATIONS)
    global fig
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_v[:, j, 0], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_v[0, j, 0], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_v[-1, j, 0], marker='o', color=colours[j])
        plt.draw()
    plt.title('Linear velocities of the robots in the x axis')
    plt.xlabel("Iterations")
    plt.ylabel("v_x")

    # v_y
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_v[:, j, 1], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_v[0, j, 1], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_v[-1, j, 1], marker='o', color=colours[j])
        plt.draw()
    plt.title('Linear velocities of the robots in the y axis')
    plt.xlabel("Iterations")
    plt.ylabel("v_y")

    # v_z
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_v[:, j, 2], color=colours[j], marker='.', linestyle='solid',
                 linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_v[0, j, 2], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_v[-1, j, 2], marker='o', color=colours[j])
        plt.draw()
    plt.title('Linear velocities of the robots in the z axis')
    plt.xlabel("Iterations")
    plt.ylabel("v_z")


def plot_w_vs_iters():
    # w_x
    global N_ITERATIONS
    iters_array = np.arange(0, N_ITERATIONS)
    global fig
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_w[:, j, 0],
                 color=colours[j], marker='.', linestyle='solid', linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_w[0, j, 0], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_w[-1, j, 0], marker='o', color=colours[j])
        plt.draw()
    plt.title('Angular velocities of the robots in the x axis')
    plt.xlabel("Iterations")
    plt.ylabel("w_x")

    # w_y
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_w[:, j, 1],
                 color=colours[j], marker='.', linestyle='solid', linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_w[0, j, 1], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_w[-1, j, 1], marker='o', color=colours[j])
        plt.draw()
    plt.title('Angular velocities of the robots in the y axis')
    plt.xlabel("Iterations")
    plt.ylabel("w_y")

    # w_z
    fig += 1
    plt.figure(fig)
    for j in range(N_AGENTS):
        # Trajectory
        plt.plot(iters_array, agents_w[:, j, 2],
                 color=colours[j], marker='.', linestyle='solid', linewidth=2, markersize=5)
        # Special markers for the first and last values
        plt.plot(iters_array[0], agents_w[0, j, 2], marker='x', color=colours[j])
        plt.plot(iters_array[-1], agents_w[-1, j, 2], marker='o', color=colours[j])
        plt.draw()
    plt.title('Angular velocities of the robots in the z axis')
    plt.xlabel("Iterations")
    plt.ylabel("w_z")


def plot_v_w():
    # v
    global fig
    fig += 1
    plt.figure(fig)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for j in range(N_AGENTS):
        plt.plot(agents_v[:, j, 0], agents_v[:, j, 1], agents_v[:, j, 2],
                 color=colours[j], linestyle='solid', linewidth=1)
        plt.plot(agents_v[0, j, 0], agents_v[0, j, 1], agents_v[0, j, 2],
                 marker='x', color=colours[j])
        plt.plot(agents_v[-1, j, 0], agents_v[-1, j, 1], agents_v[-1, j, 2],
                 marker='o', color=colours[j])
        plt.draw()
    plt.title('Linear velocity v')

    # w
    fig += 1
    plt.figure(fig)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for j in range(N_AGENTS):
        plt.plot(agents_w[:, j, 0], agents_w[:, j, 1], agents_w[:, j, 2],
                 color=colours[j], linestyle='solid', linewidth=1)
        plt.plot(agents_w[0, j, 0], agents_w[0, j, 1], agents_w[0, j, 2],
                 marker='x', color=colours[j])
        plt.plot(agents_w[-1, j, 0], agents_w[-1, j, 1], agents_w[-1, j, 2],
                 marker='o', color=colours[j])
        plt.draw()
    plt.title('Angular velocity w')


def plot_trajectories():

    global fig
    fig += 1
    plt.figure(fig)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for landmark in all_landmarks:
        plt.plot(landmark.get_position()[0],
                 landmark.get_position()[1],
                 landmark.get_position()[2], marker='o', color=colours[landmark.get_associated_agent()])
        plt.draw()

    for j in range(N_AGENTS):
        plt.plot(agents_positions[:, j, 0], agents_positions[:, j, 1], agents_positions[:, j, 2],
                 color=colours[j], linestyle='solid', linewidth=1)
        plt.plot(agents_positions[0, j, 0], agents_positions[0, j, 1], agents_positions[0, j, 2],
                 marker='x', color=colours[j])
        plt.plot(agents_positions[-1, j, 0], agents_positions[-1, j, 1], agents_positions[-1, j, 2],
                 marker='P', color=colours[j])
        ax.quiver(agents_positions[-1, j, 0], agents_positions[-1, j, 1], agents_positions[-1, j, 2],
                  agents_orientations[-1, j, 0], agents_orientations[-1, j, 1], agents_orientations[-1, j, 2],
                  color=colours[j], linewidth=1)
        plt.draw()
    plt.title('Trajectories with landmarks')

    fig += 1
    plt.figure(fig)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for j in range(N_AGENTS):
        plt.plot(agents_positions[:, j, 0], agents_positions[:, j, 1], agents_positions[:, j, 2],
                 color=colours[j], linestyle='solid', linewidth=1)
        plt.plot(agents_positions[0, j, 0], agents_positions[0, j, 1], agents_positions[0, j, 2],
                 marker='x', color=colours[j])
        plt.plot(agents_positions[-1, j, 0], agents_positions[-1, j, 1], agents_positions[-1, j, 2],
                 marker='P', color=colours[j])
        ax.quiver(agents_positions[-1, j, 0], agents_positions[-1, j, 1], agents_positions[-1, j, 2],
                  agents_orientations[-1, j, 0], agents_orientations[-1, j, 1], agents_orientations[-1, j, 2],
                  color=colours[j], linewidth=1)
        plt.draw()
    plt.title('Trajectories without landmarks')


if __name__ == '__main__':

    # Parameters
    N_ITERATIONS = 300
    N_LANDMARKS = 150  # 152
    N_AGENTS = 4
    CYLINDER_ORIGIN = np.array([0, 0, 0])
    CYLINDER_HEIGHT = 5
    CYLINDER_RADIUS = 1
    COMMUNICATION_RANGE = 1000.0
    D = 1.3  # 1.3
    ALPHA = 0.515  # 0.515
    BETA = 0.485  # 0.485
    GAMMA = 0.5  # 0.5
    K_V = 0.005   # 0.00005
    K_W = 0.0025  # 0.000025
    PLOT_LANDMARKS = False
    PLOT_FOOTPRINT = False
    PLOT_INITIAL_POS = False
    LANDMARKS_REASSIGNMENT = False

    # Variables
    fig = 0
    landmarks_per_agent = math.floor(N_LANDMARKS / N_AGENTS)

    # Create landmarks
    all_landmarks = create_landmarks(PLOT_LANDMARKS, landmarks_per_agent)

    # Create agents
    all_agents = []
    for id_ in range(N_AGENTS):
        a_position = np.array([CYLINDER_ORIGIN[0] + 2 * CYLINDER_RADIUS,
                               CYLINDER_ORIGIN[1] - 2 * CYLINDER_RADIUS + id_ * 4 * CYLINDER_RADIUS / (N_AGENTS - 1),
                               CYLINDER_ORIGIN[2]])

        if id_ == N_AGENTS - 1:
            new_agent = Agent(id_, a_position.T, np.array([-1, 0, 0]).T,
                              list(range(id_ * landmarks_per_agent, N_LANDMARKS)))
        else:
            new_agent = Agent(id_, a_position.T, np.array([-1, 0, 0]).T,
                              list(range(id_ * landmarks_per_agent, id_ * landmarks_per_agent + landmarks_per_agent)))

        all_agents.append(new_agent)


    if PLOT_INITIAL_POS:
        fig += 1
        plt.figure(fig)
        colours = ['b', 'g', 'r', 'y', 'k']
        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for landmark in all_landmarks:
            ax.scatter(landmark.get_position()[0], landmark.get_position()[1], landmark.get_position()[2],
                       marker='o', color=colours[landmark.get_associated_agent()])
            plt.draw()
        for j, agent in enumerate(all_agents):
            ax.scatter(agent.get_position()[0], agent.get_position()[1], agent.get_position()[2],
                       marker='P', color=colours[j])
            ax.quiver(agent.get_position()[0], agent.get_position()[1], agent.get_position()[2],
                      agent.get_orientation()[0], agent.get_orientation()[1], agent.get_orientation()[2],
                      color=colours[j], linewidth=2)
            plt.draw()
        plt.title('Landmarks and robots in initial position')
        plt.xlim(-2, 2)

    # Plot footprint for a simple case
    if PLOT_FOOTPRINT:
        sample_agent = Agent(0, np.array([0, 0, 0]).T, np.array([1, 0, 0]).T)
        sample_agent.plot_sample_visibility()


    # colours = ['g', 'r', 'k']
    colours = ['b', 'g', 'r', 'y', 'k']
    # colours = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'brown', 'k']

    agents_positions = np.zeros((N_ITERATIONS, N_AGENTS, 3))
    agents_orientations = np.zeros((N_ITERATIONS, N_AGENTS, 3))
    agents_v = np.zeros((N_ITERATIONS, N_AGENTS, 3))
    agents_w = np.zeros((N_ITERATIONS, N_AGENTS, 3))
    coverage_per_agent = np.zeros((N_ITERATIONS, N_AGENTS + 1))

    fig += 1
    plt.figure(fig)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for i in range(N_ITERATIONS):

        if i == 2500 and LANDMARKS_REASSIGNMENT:
            # Batidora
            new_list = list(range(N_LANDMARKS))
            random.shuffle(new_list)
            for a_idx, agent_ in enumerate(all_agents):
                landmarks_agent = agent_.get_associated_landmarks()
                for l_ in landmarks_agent:
                    agent_.remove_associated_landmark(l_)
                for l_index in new_list[a_idx * int(N_LANDMARKS / N_AGENTS):a_idx * int(N_LANDMARKS / N_AGENTS) + int(N_LANDMARKS / N_AGENTS)]:
                    agent_.add_associated_landmark(l_index)

        total_coverage = 0

        for idx, agent in enumerate(all_agents):

            coverage_per_agent[i, idx] = agent.compute_individual_coverage()
            total_coverage += coverage_per_agent[i, idx]

            linear_velocity, angular_velocity = agent.apply_control_law()
            agent.apply_kinematics(linear_velocity, angular_velocity)
            agent.communicate()

            agents_positions[i, idx, :] = agent.get_position()
            agents_orientations[i, idx, :] = agent.get_orientation()
            agents_v[i, idx, :] = linear_velocity
            agents_w[i, idx, :] = angular_velocity

            if i == N_ITERATIONS - 1:

                for landmark in all_landmarks:
                    ax.scatter(landmark.get_position()[0],
                               landmark.get_position()[1],
                               landmark.get_position()[2], marker='o', color=colours[landmark.get_associated_agent()])
                    plt.draw()

                ax.scatter(agent.get_position()[0], agent.get_position()[1], agent.get_position()[2],
                           marker='P', color=colours[idx])

                ax.plot(np.array([agent.get_position()[0], agent.get_position()[0] + agent.get_orientation()[0]]),
                        np.array([agent.get_position()[1], agent.get_position()[1] + agent.get_orientation()[1]]),
                        np.array([agent.get_position()[2], agent.get_position()[2] + agent.get_orientation()[2]]),
                        color=colours[idx], linestyle='solid', linewidth=0.5)

                ax.quiver(agent.get_position()[0], agent.get_position()[1], agent.get_position()[2],
                          agent.get_orientation()[0], agent.get_orientation()[1], agent.get_orientation()[2],
                          color=colours[idx], linewidth=1)

                plt.title('Final positions and orientations of the robots')
                plt.draw()

        coverage_per_agent[i, N_AGENTS] = total_coverage

    #plot_coords_vs_iters()
    #plot_v_vs_iters()
    plot_w_vs_iters()
    # plot_v_w()
    plot_coverage_vs_iters()
    plot_trajectories()
    plt.show()
