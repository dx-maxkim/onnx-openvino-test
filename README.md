## install OpenVINO with APT on x86_64 Ubuntu system 
- Reference: https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html
```
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-get install gnupg
# for Ubuntu 24.04
echo "deb https://apt.repos.intel.com/openvino ubuntu24 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
# for Ubutnu 22.04
# echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
# for Ubutnu 20.04
# echo "deb https://apt.repos.intel.com/openvino ubuntu20 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update
# apt-cache search openvino
sudo apt install openvino
/usr/share/openvino/samples/cpp/build_samples.sh
```

## ONNX install for OpenVINO
- Reference: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
```
pip install --upgrade onnxruntime-openvino

```
