# ROS2 Node for Vision Subsystem

check coordinates/src/main.py for pipeline of image -> X, Y
TODO: integrate X,Y -> Z as new class
TODO: write the node in main.py and define a service: request Image, response: X, Y, Z

### Managing the Unet submodule:

1. if the path of the unet folder changes, fix the path in .gitmodules
2. to sync changes with UNet repo:
```bash
git submodule update
```
This pulls changes from whatever branch it is attached to on the UNet repo and detaches from that branch - by default, keep this branch as 'local'
3. to make changes to UNet repo
    1. make changes in src/unet
    2. add and commit 
    3. push changes to the 'local' branch:
    ```bash
    git push origin HEAD:local
    ```
    4. cd into src directory and run
    ```bash
    git submodule update
    ```
