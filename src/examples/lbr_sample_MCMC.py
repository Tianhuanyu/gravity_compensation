#!/usr/bin/python3
import pybullet as p
import pybullet_data
import time
import csv
import os
import xacro
from ament_index_python import get_package_share_directory
import rclpy
from rclpy import qos
from rclpy.node import Node
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TrajGeneration import TrajGenerationUsrPath
from trajsimulation import TrajectoryConductionSim
from regression import Estimator,traj_filter,compare_traj
from utility_math import csv_save
from MetaGen import combine_input_output, csv_saveCreate

import rospkg
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from MetaGen import generate_excitation_sat_path,process_regression_data,plot_params,process_data_with_given_params



def main(args=None):

    # define parameters
    rclpy.init(args=args)
    gravity_vec = [0.0, 0.0, -9.81] #[4.905, 0.0, -8.496]

    file_name = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "urdf",
            "med",
            "med7dock.urdf"
        )
    paths = [
        os.path.join(
            get_package_share_directory("med7_dock_description")
        ),
        os.path.join(
            get_package_share_directory("lbr_description")
        ),
        os.path.join(
            get_package_share_directory("gravity_compensation"),
            "urdf",
            "med"
        )
    ]

    path_arm = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "urdf",
            "med",
            "med7dock.urdf.xacro",
        )


    # Test

    # Generate a excitation trajectory



    # Init a simulation environment
    instance = TrajectoryConductionSim(file_name, paths,is_traj_from_path=False,traj_data=None,gravity_vector=gravity_vec)


    #regression simplication
    """111"""
    values_list = generate_excitation_sat_path(path_arm, gravity_vec)
    instance.import_traj_fromlist(values_list)
    instance.set_friction_params()


    # Identify the parameters
    prefix = "/home/thy/test/"
    paraEstimator = Estimator(gravity_vec=gravity_vec)
    """222"""
    data = instance.run_sim_to_list()
    estimate_pam, ref_pam = process_regression_data(data, prefix, "data_spt.csv", paraEstimator)


    # positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentList(data)
    # efforts_f = efforts
    # ##filter the noise in vel and efforts
    # # velocities=traj_filter(velocities)
    # # efforts_f=traj_filter(efforts)
    # estimate_pam,ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    # tau_ests, tau_exts =paraEstimator.testWithEstimatedParaIDyn(positions, velocities, estimate_pam,ref_pam)
    # data = combine_input_output(positions[1:], velocities[1:], tau_exts, tau_ests)

    # for d in data:
    #     csv_saveCreate(prefix+"/data_spt.csv", 
    #             d
    #             )
        
    data = instance.run_sim_in_workspace()
    _estimate_pam, _ref_pam = process_data_with_given_params(data, prefix, "data_qry.csv", paraEstimator, 
                                                             estimate_pam,ref_pam,'MC_test')

    # data = instance.run_sim_in_workspace()
    # positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentListZeroVel(data)
    # efforts_f = efforts
    # estimate_pam,ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    # tau_ests, tau_exts =paraEstimator.testWithEstimatedParaIDyn(positions, velocities, estimate_pam,ref_pam)
    # data = combine_input_output(positions[1:], velocities[1:], tau_exts, tau_ests)

    # for d in data:
    #     csv_saveCreate(prefix+"/data_qry.csv", 
    #             d
    #             )
        
    # print("estimate_pam = ",estimate_pam)
    # print("ref_pam = ",ref_pam)
    # print("e_para = ",np.array(ref_pam)-np.array(estimate_pam))
    
    """333"""
    ref_pam_list = ref_pam.flatten().tolist()
    estimate_pam_list = estimate_pam.full().flatten().tolist()
    plot_params(ref_pam_list, estimate_pam_list)


    ref_pam_list = _ref_pam.flatten().tolist()
    estimate_pam_list = _estimate_pam.full().flatten().tolist()

    plot_params(ref_pam_list, estimate_pam_list)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(ref_pam_list)), ref_pam_list, label='List 1', marker='o')
    # plt.scatter(range(len(estimate_pam_list)), estimate_pam_list, label='List 2', marker='x')
    # plt.title('Scatter Plot Comparison')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()
    
    rclpy.shutdown()



if __name__ == "__main__":
    main()
