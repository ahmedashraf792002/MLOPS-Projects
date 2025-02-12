<<<<<<< HEAD
# 02--ML-Exp-with-DVC-CML
02- ML Exp with DVC &amp; CML
=======
## `Liver Detection with use of CML & DVC Tools `
    * Using different approaches.
    * Using different Algorithms also.
-------------------
### Note
> `liver.yaml` file is attached to this directory. You can put it in `.github/workflows/liver.yaml` as usual.

-----------------------------------------------------------------
## `DVC Installation`
``` bash
pip install dvc
pip install dvc-gdrive  # for using gdrive
```

### Git
``` bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/ahmedashraf792002/MLflow-Projects.git
git push
git status
```

### DVC
``` bash
dvc init
dvc add ./data/ ./models/
dvc remote add --default myremote gdrive://1WrMzHCLh0BBaa7XrnfHpBYepZlzNtVUO # if using gdrive
dvc push
dvc status
dvc doctor
```


``` bash
# find the default.json file credentials
C:\Users\YourUsername\AppData\Local
```
>>>>>>> ae1bc72 (initial commit)
