# Federated Trust Evaluation for Smart Rooms

This project implements a privacy-preserving smart room system using:

- Raspberry Pi clients
- Jetson Nano fog server
- Cloud monitoring server

The system detects unusual user behavior using federated learning.

## Architecture

Sensors → Raspberry Pi → Jetson Nano → Cloud

## Technologies

- Python
- PyTorch
- Flower Federated Learning
- FastAPI