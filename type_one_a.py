import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from astropy.io import fits
from tkinter import *

# Function to classify supernovae Type Ia using neural networks
def classify_snia(fits_data):
  # Load the fits data into a numpy array
  data = fits.getdata(fits_data)

  # Extract the features and labels from the data
  X = data[:, :-1]
  y = data[:, -1]

  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create a neural network classifier
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

  # Train the classifier on the training data
  clf.fit(X_train, y_train)

  # Test the classifier on the test data
  accuracy = clf.score(X_test, y_test)

  # Print the accuracy of the classifier
  print("Accuracy:", accuracy)

# Function to create the GUI
def create_gui():
  # Create the main window
  window = Tk()
  window.title("SN Ia Classifier")

  # Create a label and text entry for the fits file
  Label(window, text="Fits file:").grid(row=0, column=0)
  fits_entry = Entry(window)
  fits_entry.grid(row=0, column=1)

  # Create a button to classify the supernovae
  Button(window, text="Classify", command=lambda: classify_snia(fits_entry.get())).grid(row=1, column=0)

  # Run the main loop
  window.mainloop()

# Create the GUI
create_gui()
