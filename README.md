Smart cities aim to integrate multiple sectors such as traffic, energy, healthcare, and governance
to enhance urban living. However, the escalating number of vehicles on roads without an equal
increase in road capacity leads to congestion, accidents, environmental problems, and a decline
in quality of life. Addressing these challenges requires effective traffic management, notably
through dynamic scheduling of traffic lights, to mitigate these concerns and ensure smoother
urban mobility.The manual scheduling of traffic lights, relying on static timing, fails to handle
unpredictable traffic conditions, leading to wasted time, energy, and economic drawbacks.
Object detection for traffic monitoring offers a solution. Hence, this project proposes a
framework for adaptive traffic light system. To do so, the project outlines the following
contributions: Firstly, it presents a dataset comprising various classes of vehicles. Secondly, it
conducts a comparative analysis among five commonly used computer vision models: YOLOv8,
YOLOv3, YOLOv9, Faster R-CNN, and Rt-Detr. Thirdly, it introduces a simulation environment
aimed at comparing and determining the most effective scheduling technique for traffic lights
based on Round-robin scheduling and Reinforcement Learning using the DQN algorithm. The
results indicate that YOLOv8 achieves the highest mean Average Precision (mAP) at 90.7%,
followed by YOLOv9 at 90.5%. These models were combined into an Ensemble Model to
enhance detection performance. Additionally, the simulation environment is evaluated using
synthetic data of different scenarios to assess traffic scheduling. The primary result
demonstrates a 33.47% reduction in time compared to static systems for traffic lights based on
Round-robin scheduling. Furthermore, the DQN model applied to the SUMO simulation
environment shows a 53.66% reduction in time compared to static systems.
