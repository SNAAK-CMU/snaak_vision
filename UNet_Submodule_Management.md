# Managing a submodule

1. if the path of the unet folder changes, fix the path in .gitmodules
2. to sync changes to the latest pushed state of the submodule in this repo:

    ```bash
    git submodule update
    ```

    This pulls changes from whatever detached HEAD the submodule is currently in, **not the latest state of the original module**

3. To checkout to a specific branch in the original module's repo and get its latest state:

    ```bash
    git checkout "branch"
    git pull
    ```

    To update this submodule to the latest state of the original module as pulled above:

    ```bash
    cd ../
    git submodule update
    git add <submodule_folder>
    git commit -m "updated <submodule> from <branch>"
    git pull
    git push
    ```

3. To make changes to the original module:
    1. cd to submodule folder and update its state as desired, following steps in **2**.
    2. make changes, git add and git commit
    3. connect to <branch> and push changes:

        ```bash
        git push origin HEAD:<branch>
        ```
        
        example:

        ```bash
        git push origin HEAD:local
        ```

        This will push changes to 'local' branch in submodule repo and detach
    4. cd outside the submodule directory and run:

        ```bash
        git submodule update
        ```

    5. Now push the submodule update in the parent repo:

        ```bash
        cd ..
        git commit -m "updated submodule to track latest changes to 'local' branch in submodule repo"
        git pull && git push
        ```

    Now the submodule is detached and the latest pushed state is saved
