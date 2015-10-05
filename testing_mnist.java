// Software: Testing MLP for MNIST database
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

public class testing_mnist {
	
	static String TestingImageFileName  = "mnist-database/t10k-images.idx3-ubyte";
	static String TestingLabelFileName  = "mnist-database/t10k-labels.idx1-ubyte";
	static String ReportFileName        = "testing-report.txt";
	static String WeightsFileName       = "MLP-785-128-10.dat";
	
	static int widthImage   = 28;
	static int heightImage  = 28;
	
	static int nInput   = widthImage * heightImage + 1; // +1: Bias
	static int nHidden  = 128;
	static int nOutput  = 10;
	
	static MLP myNet    = new MLP(nInput, nHidden, nOutput); // Create an object of the class MLP with 3 layers
	
	static int nSamples     = 10000;
	static int image[][]    = new int [widthImage][heightImage];
	static double input[]   = new double [nInput];
	static double output[]  = new double [nOutput];
	
	static BufferedInputStream imageFile, labelFile;
	static PrintWriter reportFile;
	
	private static void getImage() throws IOException {
	    for (int i = 0; i < heightImage; ++i) {
	        for (int j = 0; j < widthImage; ++j) {
	            image[i][j] = imageFile.read();
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
	
	private static int getOutput(double output[]) {
	    int predict = 0;
	    for (int i = 1; i < nOutput; ++i) {
	        if (output[i] > output[predict]) {
	            predict = i;
	        }
	    }
	    return predict;
	}
	
	public static void main(String args[]) throws IOException {	
	    // Files opening
		imageFile   = new BufferedInputStream(new FileInputStream(TestingImageFileName));
		labelFile   = new BufferedInputStream(new FileInputStream(TestingLabelFileName));
		reportFile  = new PrintWriter(new FileWriter(ReportFileName));
		
		for (int i = 0; i < 16; ++i) {
		    imageFile.read();
		}
		
		for (int i = 0; i < 8; ++i) {
		    labelFile.read();
		}
		
		// MLP setups
		myNet.setWeights(WeightsFileName);
		
		// Testing process
		int nCorrect = 0;
		for (int sample = 1; sample <= nSamples; ++sample){
			getInput();
            int label = labelFile.read();
			
			// Perceptron prediction
			myNet.Predict(input, output);
			int predict = getOutput(output);
			
            if (label == predict) {
                ++nCorrect;
                System.out.println("Sample " + Integer.toString(sample) + ": YES");
			    reportFile.println("Sample " + Integer.toString(sample) + ": YES");
			} else {
			    System.out.println("Sample " + Integer.toString(sample) + ": NO");
			    reportFile.println("Sample " + Integer.toString(sample) + ": NO");
			}
		}
		
		double percent = (double)(nCorrect) / nSamples * 100.0;
		
		System.out.println("Number of correct samples: " + Integer.toString(nCorrect) + " / " + Integer.toString(nSamples));
		System.out.println("Accuracy: " + Double.toString(percent));
		
		reportFile.println("Number of correct samples: " + Integer.toString(nCorrect) + " / " + Integer.toString(nSamples));
		reportFile.println("Accuracy: " + Double.toString(percent));
		
		imageFile.close();
		labelFile.close();
		reportFile.close();
	}
	
}
