# ML on MCU: Keyword Spotting Project

This repository contains the project developed for the **Machine Learning on Microcontrollers** course at ETH Zürich (Spring semester 2025).

## Project Overview

The goal of this project is to develop a classification model for **keyword spotting** using the [Google Speech Commands v2 dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). The trained model is then deployed on an **STM32 L4** microcontroller board.

## Main Steps

1. **Data Preparation:** Designing an efficient and customizable data loader.
2. **Model Development:** Designing and training a neural network for keyword spotting.
3. **Deployment:** Converting and deploying the trained model onto the STM32 L4 board for on-edge inference.