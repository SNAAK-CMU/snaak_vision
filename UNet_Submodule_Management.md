# Managing a submodule

## Meaning of Different Terms:
1. Parent Repository: This repository
2. Submodule: The src/segmentation/UNet folder in this repo
3. Module: The original repository of the submodule

## Management Instructions:

1. If the path of the UNet folder changes, fix the path in .gitmodules
2. To sync changes to the latest pushed state of the submodule in this repo:

    ```bash
    git submodule update
    ```

    This pulls changes from whatever detached HEAD the submodule is currently in, **not the latest state of the original module**

3. To checkout to a specific branch in the original module's repo and get its latest state:

    ```bash
    git checkout "branch"
    git pull
    ```

    To update this submodule in the parent repository to the latest state of the original module as pulled above:

    ```bash
    cd ../
    git add <submodule_folder>
    git commit -m "updated <submodule> from <branch name> branch"
    git pull
    git push
    ```

3. To make changes to the original module from the submodule directory (avoid):
    1. Navigate to submodule folder and update its state as per desired baseline, following steps in **2**.
    2. Make changes, git add and git commit
    3. Connect to <branch> and push changes:

        ```bash
        git push origin HEAD:<branch>
        ```
        
        example:

        ```bash
        git push origin HEAD:local
        ```

        This will push changes to 'local' branch in submodule repo and detach

    4. Update this submodule in the parent repository to the latest state of the original module as pushed above:

        ```bash
        cd ../
        git add <submodule_folder>
        git commit -m "updated <submodule> from <branch name> branch"
        git pull
        git push
        ```

        Now the submodule in this parent repository will be connected to the branch of the module that the changes were pushed to 
    
    5. Update the submodule wherever the parent repository was cloned:

        ```bash
        git submodule update
        ```
