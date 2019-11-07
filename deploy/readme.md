```sh
docker run --rm -it gcr.io/google.com/cloudsdktool/cloud-sdk
apt-get install nano
cd /home
nano cors-json-file.json
gcloud auth login
gcloud projects list
gsutil cors get gs://abstract-strategy-games-data
gsutil cors set cors-json-file.json gs://abstract-strategy-games-data
```

https://storage.googleapis.com/abstract-strategy-games-data/four-row/model/alpha-14/model.json