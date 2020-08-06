adduser igor
usermod -aG sudo igor
su - igor
sudo ls -la /root

sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt-get install nvidia-driver-440
sudo reboot

nvidia-smi

sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker
sudo usermod -aG docker ${USER}

su - ${USER}
id -nG
docker run hello-world

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

docker run --gpus all -ti --rm nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 bash

cd ~
apt-get update
apt-get install curl nano
curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
bash nodesource_setup.sh
apt install nodejs
nodejs -v
npm -v
apt install build-essential
npm install @tensorflow/tfjs-node-gpu

