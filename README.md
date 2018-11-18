# Super Dobble!
### Your best cheat for looking for pairs in Dobble

This is the application that can find pairs between dobble cards on the image using OpenCV2 and SIFT in Python.

Result:
![Screenshot](data/output/easy_4_output.jpg)

Steps:
Take a photo:
![Screenshot](data/processing/1.JPG)
Count average light and set threshold:
```
avgLight>140 threshold = 140 + (avgLight-140) * 3/4
else threshold = 140
```
![Screenshot](data/processing/2.JPG)
Find conturous on the image with threshold
![Screenshot](data/processing/3.JPG)
Add mask for the clear image and cut circles
![Screenshot](data/processing/4.JPG)
Load every photo with animal and compary with every card eg.
![Screenshot](data/processing/5231.JPG)
Find center circle and best pair with between center circle and every other circle. Write it on clear image
![Screenshot](data/processing/6.JPG)