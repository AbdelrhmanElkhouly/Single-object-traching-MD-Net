# MDnet
i have worked on google colab "https://colab.research.google.com/drive/10w2ILePl9iFdsnW1pjhCg-HgDTK-LcpN?usp=sharing"
this note book contains the command to run "run_project_single_object_tracking"s
MDnet visual tracking algorithm implementation version 3. A trainded model mdnetv3.pt which is trained on vot2015 and one result, vot2015/crossing are also uploaded, in which blue windows are groundtruth boundingboxes while green ones are tracking results. Better performance with respect to fps, precision and success has achieved


How to run:
		from notebook "single object tracking"

		python srcv3.py online0
		
		the program asks you to input a video name and you need to download and prepare vot2015 datasets

Files:

		libv3.py: contains all classes and most functions
  
		options.py: contains all parameters we need to modify
  
		srcv3.py: offline_training, online_tracking


Folder paths organization:

	-vot2015/
	
	-mdnet/
		
		-libv3.py
		
		-srcv3.py
		
		-options.py
		
		-vot2015.txt
		
		-results/
		
		-trained_nets/
		
			-mdnetv3.pt


Dependencies:

		(1)python3.5
  
		(2)opencv,numpy
  
		(3)pytorch
  
		(4)scikit-learn
 

  

 
