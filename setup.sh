#!/bin/bash

read -p "Do you want to create a new conda environment? (y/n): " create_env

if [[ $create_env == "y" ]]; then
    read -p "Enter the name for the new conda environment: " env_name
    conda create -n $env_name python=3.9
    conda activate $env_name
elif [[ $create_env == "n" ]]; then
    read -p "Do you want to install packages in the current environment? (y/n): " install_packages
    if [[ $install_packages == "y" ]]; then
        echo "Installing packages in the current environment."
        echo ""
    elif [[ $install_packages == "n" ]]; then
        echo "Aborting installation."
        exit 0
    else
        echo "Invalid input. Aborting installation."
        exit 1
    fi
else
    echo "Invalid input. Aborting installation."
    exit 1
fi

pip install -r requirements.txt

read -p "Do you want to download the outlier dataset? (y/n): " download_file

download_file=${download_file,,}  # Convert input to lowercase

if [[ $download_file == "y" ]]; then
    echo "Downloading file..."
    wget -O data/Imagenet.tar.gz https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
    
    echo "Unpacking file..."
    tar -xvf data/Imagenet.tar.gz -C data/
    
    echo "File downloaded and unpacked successfully."
elif [[ $download_file == "n" ]]; then
    echo "Skipping downloading outlier dataset"
else
    echo "Invalid input"
fi