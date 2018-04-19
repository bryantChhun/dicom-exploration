# Part 2:

Using your code from Phase 1, add the parsing of the o-contours into your pipeline. Note that there are only half the number of o-contour files as there are i-contour files, so not every i-contour will have a corresponding o-contour.

After building the pipeline, please discuss any changes that you made to the pipeline you built in Phase 1, and why you made those changes.

	- Because my part 1 approach already included flags to parse o-contours, there was not much additional coding needed
	- only minor changes to load_data method were made, and a few other comments clarified
	
	
Let’s assume that you want to create a system to outline the boundary of the blood pool (i-contours), and you already know the outer border of the heart muscle (o-contours). Compare the differences in pixel intensities inside the blood pool (inside the i-contour) to those inside the heart muscle (between the i-contours and o-contours); could you use a simple thresholding scheme to automatically create the i-contours, given the o-contours? Why or why not? Show figures that help justify your answer.

Do you think that any other heuristic (non-machine learning)-based approaches, besides simple thresholding, would work in this case? Explain.

	 - Please see the two ipython notebooks in the repo for detailed approach to these questions


========================================================================================

# Part 1


# dicoms

This repo contains methods for parsing MRI Dicom files, corresponding segmentation masks, and methods to stream this data into a DL training pipeline

# dependencies

	- keras, csv, dicom, PIL


# My method:

The parsing.py file was already provided, but it did not include a method for parsing the link.csv file.  I added this method and created two more .py files: load_data and feed_data

parsing:

	- included a new method, parse_patient_csv, which parses the link.csv file to 
	match patient dicoms with patient contours 

load_data:

	Call "load_all_patients" to return a list of images and a list of masks
    Currently this returns ONLY images that have matching masks, but it is simple
    to allow all images to pass, with None if there is no matching mask
        
	- contains a Patient class that is called every time a new patient is parsed
	- each patient and his/her corresponding contour is searched, parsed, and loaded
	- each patient instance has its own properties: labeled_dicoms, or all_dicoms
	- load_all_patients returns a concatenated list of images from all
	patients in directory
	- write_all_patients is executed at cmdline and will save all images as .npy arrays

feed_data:

	contains a simple generator that uses the Keras Image Data Generator to stream data
    
	- basic image preprocessing necessary for training DL models is included:
		- channels last
		- images are zero mean-centered, and converted to float32
		- masks are one-hot encoded, and converted to uint8
		- keras's 'flow' method has 'shuffle' flag to randomize batch images


# Answers to questions:
# Part 1: Parse the DICOM images and Contour Files


- How did you verify that you are parsing the contours correctly?
      
    I manually verified that the parse_patient_csv and Patient class were matching contours with appropriate images.  For a couple images/pairs, I plotted overlays using matplotlib.pyplot
    
    I also manually verified that all images are 256x256.  The code always generates masks that are matched in size to its corresponding images, but the final feed_data generator must have consistent size, which is not accounted for in this code.
    
    Finally, I wrote tests to check outputs of the parsing code.  The tests check the following: Boolean mask? 256x256 shape? Same number of images as masks?
      

- What changes did you make to the code, if any, in order to integrate it into our production code base? 

	I added another method to the parsing.py file: parse_patient_csv
    
    I also added the ability to run load_data from command line with two flags: data directory, output directory.  This saves the matched image/mask numpy arrays to disk as .npy files.  I did this because much of the "load_data" method can be considered image preprocessing.  The feed_data can be modified to load these .npy files, speeding up downstream iterations on training.
	
	

# Part 2: Model training pipeline

- Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

	- No major changes to the pipeline, however I did write the pipeline with the idea of including outer-contour masks in the long run.  Currently the load_all_patients returns ONLY inner-contour masks, but it is trivial to add one more list to that output.

- How do you/did you verify that the pipeline was working correctly?

	- I wrote a lot of tests!  The feed_data method builds on all of the previous ones and returns expected results.  Additionally there are unit tests for all upstream modules.

- Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

	- The model training pipeline currently is not splitting the data into train/validation, but could do so easily depending on the task.
    
	- As mentioned above, I could add flags for including outer contour.  The code already has a check for 'inner' vs 'outer' but is hard coded for 'inner' right now.
    
    - Also as mentioned above, I need checks and methods to make sure all images coming out of the generator are the same dimensionality.

	- The pipeline is not robust to subtle effects like: blank images, images with strange artifacts/aberrations, images with saturation effects, mismatched masks/dicoms from the csv file.  I could try generating data to test this, and then have appropriate checks/filters in the main code.
