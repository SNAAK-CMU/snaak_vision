# Managing a submodule

1. if the path of the unet folder changes, fix the path in .gitmodules
2. to sync changes to the latest pushed state of the submodule in this repo:

    ```bash
    git submodule update
    ```

    This pulls changes from whatever detached HEAD the submodule is currently in

    To checkout to a specific branch in the submodule's original repo:

    ```bash
    git checkout "branch"
    ```

3. to make changes to the UNet repo
    1. make changes in src/unet
    2. add and commit
    3. connect to local branch and push changes:

        ```bash
        git push origin HEAD:local
        ```

        This will push changes to 'local' branch in submodule repo and detach
    4. cd into src directory and run

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
