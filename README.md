# Light Detection

This project leverages the `YOLOv8n` model to detect traffic lights in real-time. The model is trained to recognize four states of traffic lights: red, green, yellow, and no light. It can process video feeds at approximately 30 frames per second (fps) and is capable of working with both local and IP cameras, making it suitable for a variety of real-time traffic light detection applications.

## Training

To train the model with your own dataset, follow these steps:

1. **Prepare your dataset**: Organize your dataset in the following structure: dataset/ ├── images/ │ ├── train/ │ └── val/ ├── labels/ │ ├── train/ │ └── val/
2. **Create a dataset configuration file**: Create a `dataset.yaml` file with the following structure:
````yaml
train: path/to/your/dataset/images/train
val: path/to/your/dataset/images/val

nc: 4
names: ['red', 'green', 'yellow', 'no_light']
````

3. Train the model:
## Usage

### Real-time Detection

You can use the trained model to perform real-time detection. This can be done using a local camera or an IP camera.
