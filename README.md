# Hanglok_Summary
Document, code, working environment, and all related stuff for work in Hanglok

The code are segmented into several parts. 
- Data preprocessing can be viewed [here](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Data%20Preprocessing)
- Some literature review can be viewed [here](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Literature%20Review)
- For simulation, there are [ROS approach](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Simulation/ROS), implemented in Ubuntu, and [Webots approach](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Simulation/Webots), implemented in Windows.
- Toy classification and generation model can be viewed [here](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Toy%20Classification%20and%20Generation%20Model), and the simulation can be done using Webots, using code [here](https://github.com/randulfzhao/Hanglok_Summary/tree/main/Toy%20Classification%20and%20Generation%20Model/controllers). The training data is self-recorded videos. The toy model for recognition and generation are both LSTM.
    - `util` include the packages and function to implement the code
    - `pre-train` includes code to use the webcam to record video and train the model.