// Software: Training MLP for MNIST database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Final update: October 4th, 2015
// Copyright 2015 (c) Hy Truong Son. All rights reserved. Only use for academic purposes.

import java.io.*;
import MyLib.*;

public class training_mnist {
	
	static String TrainingImageFileName = "mnist-database/train-images.idx3-ubyte";
	static String TrainingLabelFileName = "mnist-database/train-labels.idx1-ubyte";
	static String ReportFileName        = "training-report.txt";
	static String WeightsFileName       = "MLP-785-128-10.dat";
	
	static int widthImage   = 28;
	static int heightImage  = 28;
	
	static int nInput   = widthImage * heightImage + 1; // +1: Bias
	static int nHidden  = 128;
	static int nOutput  = 10;
	
	static MLP myNet    = new MLP(nInput, nHidden, nOutput); // Create an object of the class MLP with 3 layers
	
	static int Epochs           = 1000;     // Number of iterations
	static double LearningRate  = 1e-3;     // Learning rate (good only for stochastic learning process)
	static double Momentum      = 0.9;      // Momentum for stochastic learning process
	static double Lambda        = 0.0;      // Regularization parameter for batch learning process, no regularization 
	static double Epsilon       = 1e-3;     // Epsilon for stopping stochastic learning process
	
	static int nSamples     = 60000;
	static int image[][]    = new int [widthImage][heightImage];
	static double input[]   = new double [nInput];
	static double output[]  = new double [nOutput];
	
	static BufferedInputStream imageFile, labelFile;
	static PrintWriter reportFile;
	
	private static void getImage() throws IOException {
	    for (int i = 0; i < heightImage; ++i) {
	        for (int j = 0; j < widthImage; ++j) {
	            image[i][j] = 255 - imageFile.read();
	        }
	    }
	}
	
	private static void getInput() throws IOException {
	    getImage();
	    
	    for (int i = 0; i < heightImage; ++i) {
	        for (int j = 0; j < widthImage; ++j) {
	            if (image[i][j] > 0) {
	                input[j + i * widthImage] = 1.0;
	            } else {
	                input[j + i * widthImage] = 0.0;
	            }
	        }    
	    }
	    input[widthImage * heightImage] = 1.0; // bias
	}
	
	private static int getOutput() throws IOException {
		int label = labelFile.read();
		for (int i = 0; i < nOutput; ++i){
			output[i] = 0.0;
		}
		output[label] = 1.0;
		return label;
	}
	
	public static void main(String args[]) throws IOException {	
	    // Files opening
		imageFile   = new BufferedInputStream(new FileInputStream(TrainingImageFileName));
		labelFile   = new BufferedInputStream(new FileInputStream(TrainingLabelFileName));
		reportFile  = new PrintWriter(new FileWriter(ReportFileName));
		
		for (int i = 0; i < 16; ++i) {
		    imageFile.read();
		}
		
		for (int i = 0; i < 8; ++i) {
		    labelFile.read();
		}
		
		// MLP setups
		myNet.setEpochs(Epochs);
		myNet.setLearningRate(LearningRate);
		myNet.setMomentum(Momentum);
		myNet.setRegularizationParameter(Lambda);
		myNet.setEpsilon(Epsilon);
		
		// Training process
		for (int sample = 1; sample <= nSamples; ++sample){
			getInput();
			int label = getOutput();
			
			// Learning through Back-propagation
			double error = myNet.StochasticLearning(input, output);
			
			System.out.format("Sample %d: %.9f\n", sample, error);
			reportFile.println("Sample "+ Integer.toString(sample) + ": " + Double.toString(error));
			
			// Saving weights to file after training every 100 samples
			if (sample % 100 == 0) {
			    System.out.println("Saving weights to file " + WeightsFileName);
			    myNet.writeWeights(WeightsFileName);
			}
		}
		
		myNet.writeWeights(WeightsFileName);
		
		imageFile.close();
		labelFile.close();
		reportFile.close();
	}
	
}
