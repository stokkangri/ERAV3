# MNIST CNN Training with Flask Server

## Requirements
- Python 3.x
- PyTorch
- Flask
- torchvision
- matplotlib
- Chart.js (included in HTML)

## Setup
1. Clone the repository or create the directory structure as shown above.
2. Install the required packages:
   ```bash
   pip install torch torchvision flask matplotlib
   ```

## Training the Model
1. Run the training script:
   ```bash
   python app.py
   ```
   This will train the CNN on the MNIST dataset and log the loss values.

## Viewing Training Logs
1. In a new terminal, run the Flask server:
   ```bash
   python server.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/` to view the training loss curve.

## Displaying Model Results
To display model results on random images, you can extend the `app.py` file to include a function that picks random images from the MNIST dataset and displays the predictions.

Additional Notes
Ensure you have CUDA installed and configured properly to utilize GPU training.
You may want to add functionality to display model results on random images after training, which can be done by extending the app.py file.
Feel free to ask if you need any modifications or additional features!