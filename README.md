# MDnet
![image](https://user-images.githubusercontent.com/85629136/163677255-8c8ad9ab-3684-4375-84db-a879b62aa5a0.png)

![image](https://user-images.githubusercontent.com/85629136/163677261-cf663e1a-f66b-4169-8109-50b612979134.png)

![image](https://user-images.githubusercontent.com/85629136/163677266-844e2f7a-ec39-478d-be66-d8f189caa841.png)
![image](https://user-images.githubusercontent.com/85629136/163677273-636e0ea5-fff5-4865-bded-273f827ee1b0.png)

![image](https://user-images.githubusercontent.com/85629136/163677279-7d0e0a02-d2ab-409d-ae7a-b6e36ce6c468.png)

![image](https://user-images.githubusercontent.com/85629136/163677283-33b02aaf-55c5-4619-b867-28291193358e.png)
![image](https://user-images.githubusercontent.com/85629136/163677288-92cd4fe9-beab-42d6-ba06-e808ea40a614.png)

![image](https://user-images.githubusercontent.com/85629136/163677298-2e38a2e9-f0e1-4f26-a5d4-339da3c2532f.png)
![image](https://user-images.githubusercontent.com/85629136/163677303-319b6a86-84cd-4ea3-8f66-4965539fd58b.png)

 
![image](https://user-images.githubusercontent.com/85629136/163677310-1438475c-76fc-4265-82f5-207ccd21cbf5.png)

![image](https://user-images.githubusercontent.com/85629136/163677334-f82075d2-6ad5-4c0c-925b-b836b06887b2.png)
![image](https://user-images.githubusercontent.com/85629136/163677341-dc792edf-3e66-419b-bb97-f53c4043a1c7.png)
![image](https://user-images.githubusercontent.com/85629136/163677345-7cc1a5d0-c99c-46a0-ada8-d382a197389c.png)

![image](https://user-images.githubusercontent.com/85629136/163677398-cf39443e-0c93-4e1a-8ecf-00271f4dfc75.png)
![image](https://user-images.githubusercontent.com/85629136/163677409-258e347a-ed58-4553-912b-ef23fb8c6c81.png)

![image](https://user-images.githubusercontent.com/85629136/163677415-3f0a4c69-b937-4f18-afff-3a8115bb7582.png)

![image](https://user-images.githubusercontent.com/85629136/163677417-66c44d1f-2153-4241-9bff-42f55124c0af.png)

![image](https://user-images.githubusercontent.com/85629136/163677419-7eaa0eb1-8c27-49bf-aa84-d3c3c81bbfd4.png)
 
![image](https://user-images.githubusercontent.com/85629136/163677426-9fb4c11c-4182-4805-8b73-7058f352108b.png)

![image](https://user-images.githubusercontent.com/85629136/163677435-442c6e85-ca1b-4193-8cd4-43ef0b89d077.png)

![image](https://user-images.githubusercontent.com/85629136/163677431-c196917f-3a6f-4af0-b847-2f6adef9989a.png)
![image](https://user-images.githubusercontent.com/85629136/163677442-a4dec3ab-2e5e-41c2-98d8-807edf2877d6.png)
Results of offline training :
Offline training results are recorded in log file contain of:


      ![image](https://user-images.githubusercontent.com/85629136/163677459-b680cb37-fe5d-41f9-892d-0fb9799be723.png)






Column no.1 is sequence name 
Column no.2 is frame no.
Column no.3 is frame per second 
Column no.4 is binary loss 
![image](https://user-images.githubusercontent.com/85629136/163677454-72c05287-294f-4efa-b655-31ba0c5d9fba.png)

![image](https://user-images.githubusercontent.com/85629136/163677480-9c51d4b9-fdbf-4270-a416-258cce280a6b.png)

![image](https://user-images.githubusercontent.com/85629136/163677485-254fc163-1f36-4836-a77e-991e0a25ea1f.png)

 
![image](https://user-images.githubusercontent.com/85629136/163677490-93849a85-467d-47b2-8f3b-84de6de4f1a2.png)

![image](https://user-images.githubusercontent.com/85629136/163677506-ba9a37f0-b2a9-46a7-a6f3-94b4475d2dc3.png)

![image](https://user-images.githubusercontent.com/85629136/163677513-a491d4cf-3bf0-4e0b-8964-5fd91f6f4cdc.png)

![image](https://user-images.githubusercontent.com/85629136/163677521-c4dfe1be-bcd6-4b68-b7c9-87efd525fb39.png)

![image](https://user-images.githubusercontent.com/85629136/163677527-785ed326-f549-4dd1-af1d-0e734ab2c93f.png)



![image](https://user-images.githubusercontent.com/85629136/163677533-d97b1976-940f-4d82-8ec3-a86a897ca97d.png)



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
 

  

 
