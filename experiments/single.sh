python3 ../main.py --network ConvNet --dataset MNIST --ipc 1 &&
python3 ../main.py --network ConvNet --dataset MNIST --ipc 10 &&
python3 ../main.py --network ConvNet --dataset MNIST --ipc 50 &&

python3 ../main.py --network MLP --dataset MNIST --ipc 1 --syn_lr 0.01 &&
python3 ../main.py --network MLP --dataset MNIST --ipc 10 --syn_lr 0.01 &&
python3 ../main.py --network MLP --dataset MNIST --ipc 50 --syn_lr 0.01 &&

python3 ../main.py --network AlexNet --dataset MNIST --ipc 1 &&
python3 ../main.py --network AlexNet --dataset MNIST --ipc 10 &&
python3 ../main.py --network AlexNet --dataset MNIST --ipc 50 &&

python3 ../main.py --network LeNet --dataset MNIST --ipc 1 &&
python3 ../main.py --network LeNet --dataset MNIST --ipc 10 &&
python3 ../main.py --network LeNet --dataset MNIST --ipc 50 &&


python3 ../main.py --network MLP --dataset SVHN --ipc 1 &&
python3 ../main.py --network MLP --dataset SVHN --ipc 10 &&
python3 ../main.py --network MLP --dataset SVHN --ipc 50 &&

python3 ../main.py --network AlexNet --dataset SVHN --ipc 1 &&
python3 ../main.py --network AlexNet --dataset SVHN --ipc 10 &&
python3 ../main.py --network AlexNet --dataset SVHN --ipc 50 &&

python3 ../main.py --network LeNet --dataset SVHN --ipc 1 &&
python3 ../main.py --network LeNet --dataset SVHN --ipc 10 &&
python3 ../main.py --network LeNet --dataset SVHN --ipc 50 &&

python3 ../main.py --network ConvNet --dataset SVHN --ipc 1 &&
python3 ../main.py --network ConvNet --dataset SVHN --ipc 10 &&
python3 ../main.py --network ConvNet --dataset SVHN --ipc 50 &&


python3 ../main.py --network MLP --dataset FashionMNIST --ipc 1 &&
python3 ../main.py --network MLP --dataset FashionMNIST --ipc 10 &&
python3 ../main.py --network MLP --dataset FashionMNIST --ipc 50 &&

python3 ../main.py --network AlexNet --dataset FashionMNIST --ipc 1 &&
python3 ../main.py --network AlexNet --dataset FashionMNIST --ipc 10 &&
python3 ../main.py --network AlexNet --dataset FashionMNIST --ipc 50 &&

python3 ../main.py --network LeNet --dataset FashionMNIST --ipc 1 &&
python3 ../main.py --network LeNet --dataset FashionMNIST --ipc 10 &&
python3 ../main.py --network LeNet --dataset FashionMNIST --ipc 50 &&

python3 ../main.py --network ConvNet --dataset FashionMNIST --ipc 1 &&
python3 ../main.py --network ConvNet --dataset FashionMNIST --ipc 10 &&
python3 ../main.py --network ConvNet --dataset FashionMNIST --ipc 50 &&


python3 ../main.py --network MLP --dataset CIFAR10 --ipc 1 &&
python3 ../main.py --network MLP --dataset CIFAR10 --ipc 10 &&
python3 ../main.py --network MLP --dataset CIFAR10 --ipc 50 &&

python3 ../main.py --network AlexNet --dataset CIFAR10 --ipc 1 &&
python3 ../main.py --network AlexNet --dataset CIFAR10 --ipc 10 &&
python3 ../main.py --network AlexNet --dataset CIFAR10 --ipc 50 &&

python3 ../main.py --network LeNet --dataset CIFAR10 --ipc 1 &&
python3 ../main.py --network LeNet --dataset CIFAR10 --ipc 10 &&
python3 ../main.py --network LeNet --dataset CIFAR10 --ipc 50 &&

python3 ../main.py --network ConvNet --dataset CIFAR10 --ipc 1 &&
python3 ../main.py --network ConvNet --dataset CIFAR10 --ipc 10 &&
python3 ../main.py --network ConvNet --dataset CIFAR10 --ipc 50