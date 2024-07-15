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


def main(args=None):

    # define parameters
    rclpy.init(args=args)
    gravity_vec = [0.0, 0.0, -9.81] #[4.905, 0.0, -8.496]
    theta1 = 0.0
    theta2 = 0.0

    Ff = 0.1
    sampling_rate = 100.0
    sampling_rate_inoptimization = 20.0

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

    path_xarm = os.path.join(
            get_package_share_directory("gravity_compensation"),
            "urdf",
            "med",
            "med7dock.urdf.xacro",
        )


    # Test

    # instance = TrajectoryConductionSim(file_name, paths,is_traj_from_path=False,traj_data=None,gravity_vector=gravity_vec)
    # data = instance.run_sim_in_workspace()
    # raise ValueError("run to here")


    # Generate a excitation trajectory
    conditional_num = 100000.0
    while(conditional_num>400):
        traj_instance = TrajGenerationUsrPath(path=path_xarm, gravity_vector=gravity_vec)
        S,lbg,ubg,fc = traj_instance.get_optimization_problem(Ff = Ff,sampling_rate = sampling_rate_inoptimization, bias = [theta1, theta2, 0.0, 0.0, 0.0, 0.0, 0.0])
        a,b = traj_instance.find_optimal_point_with_randomstart(S,lbg,ubg, Rank=5)
        print("a = {0} \n b = {1}".format(a,b))
        eigenvalues, eigenvectors = np.linalg.eig(fc(a,b))
        conditional_num = np.sqrt(eigenvalues[0]/eigenvalues[-1])
        values_list,keys = traj_instance.generateToList(a,b,Ff = Ff,sampling_rate=sampling_rate)

        print("conditional_num = ",conditional_num)
    # raise ValueError("run tohere11")



    # Init a simulation environment
    instance = TrajectoryConductionSim(file_name, paths,is_traj_from_path=False,traj_data=None,gravity_vector=gravity_vec)
    instance.import_traj_fromlist(values_list)
    instance.set_friction_params()
    data = instance.run_sim_to_list()


    # Identify the parameters

    prefix = "/home/thy/test/"
    paraEstimator = Estimator(gravity_vec=gravity_vec)
    positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentList(data)
    efforts_f = efforts
    ##filter the noise in vel and efforts
    # velocities=traj_filter(velocities)
    # efforts_f=traj_filter(efforts)
    estimate_pam,ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    tau_ests, tau_exts =paraEstimator.testWithEstimatedParaIDyn(positions, velocities, estimate_pam,ref_pam)
    data = combine_input_output(positions[1:], velocities[1:], tau_exts, tau_ests)

    print("tau_ests = len", len(tau_ests))
    print("tau_exts = len", len(tau_exts))
    # for i in range(1,300,1):
    # print("conditional_num_best = ",conditional_num)

    for d in data:
        csv_saveCreate(prefix+"/data_spt.csv", 
                d
                )
        
    # raise ValueError("111")


    # Test Part via MCMC

    # instance = TrajectoryConductionSim(file_name, paths,is_traj_from_path=False,traj_data=None,gravity_vector=gravity_vec)
    
    # data = instance.run_sim_in_workspace()

    # # a_n, b_n = traj_instance.trajectory_with_random()

    # Traj GEn for meta
    # a_n,b_n = traj_instance.find_optimal_point_with_randomstart(S,lbg,ubg, Rank=5)
    # values_list,keys = traj_instance.generateToList(a_n,b_n,Ff = Ff,sampling_rate=sampling_rate)
    # instance.import_traj_fromlist(values_list)
    # instance.set_friction_params()


    data = instance.run_sim_in_workspace()
    positions, velocities, efforts = paraEstimator.ExtractFromMeasurmentListZeroVel(data)
    efforts_f = efforts
    # velocities=traj_filter(velocities)
    # efforts_f=traj_filter(efforts)
    # tau_exts, es =paraEstimator.testWithEstimatedParaIDyn(positions, velocities, estimate_pam,ref_pam)
    # data = combine_input_output(positions[1:], velocities[1:], efforts[1:], tau_exts)


    estimate_pam,ref_pam = paraEstimator.timer_cb_regressor_physical_con(positions, velocities, efforts_f)
    tau_ests, tau_exts =paraEstimator.testWithEstimatedParaIDyn(positions, velocities, estimate_pam,ref_pam)
    data = combine_input_output(positions[1:], velocities[1:], tau_exts, tau_ests)

    for d in data:
        csv_saveCreate(prefix+"/data_qry.csv", 
                d
                )
        
    print("estimate_pam = ",estimate_pam)
    print("ref_pam = ",ref_pam)
    print("e_para = ",np.array(ref_pam)-np.array(estimate_pam))
    
    # nn_cp = er_state_compensator("/home/thy/learningDynModel/vae_simple_1_VAE_CNN3.pth")

    # outputs =[]
    # errs =[]
    # for p,v,e,ep in zip(positions, velocities, efforts_f, tau_exts):
    #     data_input = p+v+e
    #     output = nn_cp.predict(data_input)

    #     err = np.asarray(ep)-np.asarray(e)
    #     outputs.append(output)
    #     errs.append(err.tolist())

    

        # print("output = ",output)
        # print("err = ",err.tolist())


        # for d in data:
        #     csv_saveCreate(prefix+"{0}".format(i)+"/data.csv", 
        #             d
        #             )
    # paraEstimator.saveEstimatedPara(estimate_pam)
    # print("tau_exts = ",tau_exts)
    # compare_traj(outputs, errs)







    rclpy.shutdown()

if __name__ == "__main__":
    main()
