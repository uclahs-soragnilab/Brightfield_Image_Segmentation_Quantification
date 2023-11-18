# brightfield_image_segmentation
 Repository for scripts to use U-Net to quantify Celigo brighfield images of Maxi Rings (24-well plates).

# Instructions for use in a Docker Container
1. Install Docker on your host machine (https://docs.docker.com/engine/install/)
2. Pull code from GitHub to your host machine.
3. In the repository, add the images you want to segment to the "Image-Folder" directory. Be sure to keep any file naming conventions consistent with the output from the Nexcelom Celigo Instrument for proper quantification. The segmentation process does not require any specific naming.
4. In a terminal, navigate to the repository containing the Image-Folder folder and the run_segmentation_pipeline.py script.
5. The Docker image can be built from the Dockerfile in the directory
```
docker build -t bf_image_quant .
```
or be pulled from Docker hub:
```
docker pull soragnilab/brightfield_image_segmentation
```
6. Once the Docker image has been pulled, run the image while mounting the repository into the app directory of the Docker container
```
docker run -it -v {hostpath}BF_Image_Segmentation_Quantification:/app bf_image_quant
```
7. The terminal will show that it is working in the /app directory within the Docker container. Proceed to running the run_segmentation_pipeline.py script to segment the images in the "Image-Folder" on the host machine
```
python3 run_segmentation_pipeline.py --mask
```
8. Once the images have been segmented, run the same script with the "--quantify" argument to track changes in segmentation across images.
```
python3 run_segmentation_pipeline.py --quantify
```
9. Exit the Docker container
```
exit
```
10. The segmented and quantified images will be available on the host machine in the Image-Folder directory.


# Instructions for use on an Arch Linux OS (Alternative method to Docker for Linux users)
# Set up virtual environment (only need to do this once)
 1. Open terminal
 2. Install the virtual environment package (if not already installed)
 ```
 sudo pacman -S python-virtualenv
 ```
 3. Create the virtual environment in the desired directory. This does not have to be the same directory as the code being run.
 ```
 python3 -m venv myenv
 ```
 Replace "myenv" with the desired name for the virtual environment.
 4. Activate the virtual environment
 ```
 source myenv/bin/activate
 ```
 5. Navigate to the directory containing the script and run the dependencies bash file to install the required packages. The dependencies.sh file can be viewed in a text or code editor to view the specific package versions required for operation.
 ```
 cd {File path}/BF_Image_Segmentation_Quantification
 bash dependencies.sh
 ```
 6. The required dependencies will be installed automatically.
 
# Segment
 1. Move the images to be processed into the "Image-Folder" directory within the "BF_Image_Segmentation_Quantification" folder. Preserve any existing subfolders and file naming conventions to indicate that images were taken on different days or at different Z-planes within the well.
 2. Open terminal 
 3. Activate the virtual environment
 ```
 cd {File path}/myenv
 source bin/activate
 ```
 {myenv} should appear in front of the computer name
 4. Navigate to the segmentation script and run it with the --mask argument to segment each of the images in the Image-Folder directory
 ```
 cd {File path}/BF_Image_Segmentation_Quantification
 python3 run_segmentation_pipeline.py --mask
 ```
 
The model trained in the Soragni Lab will be used to perform the segmentation. The model is stored as a .pkl file in the BF_Image_Segmentation_Quantification folder.
 	
# Quantify segmented area
 1. Leave all segmented images (..._mask.png) in their original output locations within the Image-Folder. All files must be named with the following format:
 Well_B2_Ch1_1um_mask.png
 This naming convention allows for the aggregation of replicates across different days based on the well and channel in which the image was taken. This naming convention is followed when images are exported from the Nexcelom Celigo software. The script will throw an error if the format is not followed.
 2. In terminal, activate the virtual environment (if it is not already active)
 ```
 cd {File path}/myenv
 source bin/activate
 ```
 {myenv should appear in front of the computer name}
 3. Navigate to the segmentation script and run it with the --quantify argument to quantify the organoid area in each masked image.
 ```
 cd {File path}/BF_Image_Segmentation_Quantification
 python3 run_segmentation_pipeline.py --quantify
 ```
 4. A csv file will be output in the Image-Folder directory with the results of the quantification.
 

 

