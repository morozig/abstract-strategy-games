```bash
adduser igor
usermod -aG sudo igor
su - igor
sudo apt-get update
sudo apt-get install -y nano
mkdir ~/.ssh
nano ~/.ssh/authorized_keys

sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt-get install -y nvidia-driver-440
sudo apt install -y nvidia-cuda-toolkit
sudo reboot

or variant B
# sudo apt-get install -y build-essential gcc-multilib dkms
sudo apt-get update
sudo apt-get install -y linux-headers-4.15.0-112-generic
# https://www.nvidia.com/Download/index.aspx
curl -O https://us.download.nvidia.com/XFree86/Linux-x86_64/440.100/NVIDIA-Linux-x86_64-440.100.run
# curl -O https://us.download.nvidia.com/XFree86/Linux-x86_64/450.66/NVIDIA-Linux-x86_64-450.66.run
sudo bash ./NVIDIA-Linux-x86_64-440.100.run
# sudo bash ./NVIDIA-Linux-x86_64-450.66.run

nvidia-smi

sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
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
docker run --gpus all nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 nvidia-smi
```

```bash
sudo apt-get install -y git
mkdir app
cd app
git clone https://github.com/morozig/abstract-strategy-games.git

```

```bash
cd abstract-strategy-games
cd cuda
docker build -t tfjs-node-gpu .
cd ..

```

```bash
docker run --gpus all \
--rm \
-it \
-v $(pwd):/app \
-w=/app \
tfjs-node-gpu \
npm run cuda

docker run --gpus all \
-u $(id -u):$(id -g) \
--rm \
-it \
-v $(pwd):/app \
-w=/app \
tfjs-node-gpu \
npm run cuda

```




